"""
Micro Opus: a 7,440-weight decoder-only transformer, in plain numpy.

A language model is two things:
  1. a dictionary of named weight matrices
  2. functions that push activations through them:
       forward()  - the whole sequence at once (what training uses)
       step()     - one new token, reusing a key/value cache (what generation uses)
       backward() - the chain rule, run in reverse, giving one gradient per weight

Only train.py changes weights. Everything here just reads them.
"""
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).parent / "weights"


def rmsnorm(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * g


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def gelu(z):
    return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def causal_mask(T):
    return np.triu(np.full((T, T), -np.inf), k=1)


def split_heads(t, H):
    *lead, T, D = t.shape
    return t.reshape(*lead, T, H, D // H).swapaxes(-2, -3)


def merge_heads(t):
    *lead, H, T, dh = t.shape
    return t.swapaxes(-2, -3).reshape(*lead, T, H * dh)


class MicroOpus:
    def __init__(self, name="words", W=None, cfg=None, vocab=None):
        if W is None:
            blob = json.loads((WEIGHTS / f"{name}.json").read_text())
            W, cfg, vocab = blob["weights"], blob["config"], blob["vocab"]
        self.cfg, self.vocab = cfg, vocab
        self.W = {name: np.array(t, dtype=float) for name, t in W.items()}

    def tokenize(self, text):
        pieces = list(text) if self.cfg.get("chars") else text.split()
        return [0] + [self.vocab.index(p) for p in pieces]

    # ------------------------------------------------------------ forward
    def forward(self, ids, tr=None):
        W, cfg = self.W, self.cfg
        ids = np.asarray(ids)
        T, H = ids.shape[-1], cfg["H"]
        dh = cfg["D"] // H
        tr = {} if tr is None else tr

        emb = W["W_E"][ids]
        pos = W["W_P"][:T]
        x = emb + pos
        tr.update(ids=ids, emb=emb, pos=pos, x0=x, layers=[])

        for l in range(cfg["L"]):
            P = lambda name: W[f"L{l}.{name}"]
            xin = x
            h1 = rmsnorm(x, P("g1"))
            q = h1 @ P("W_Q")
            k = h1 @ P("W_K")
            v = h1 @ P("W_V")
            qh, kh, vh = split_heads(q, H), split_heads(k, H), split_heads(v, H)
            scores = qh @ kh.swapaxes(-1, -2) / np.sqrt(dh) + causal_mask(T)
            att = softmax(scores)
            mix = merge_heads(att @ vh)
            attn_out = mix @ P("W_O")
            x = x + attn_out
            xmid = x
            h2 = rmsnorm(x, P("g2"))
            up = h2 @ P("W_up")
            act = gelu(up)
            down = act @ P("W_down")
            x = x + down
            tr["layers"].append(dict(
                xin=xin, h1=h1, q=q, k=k, v=v, scores=scores, att=att, mix=mix,
                attnOut=attn_out, xmid=xmid, h2=h2, up=up, act=act, down=down, xout=x))

        hf = rmsnorm(x, W["g_f"])
        logits = hf @ W["W_U"]
        probs = softmax(logits)
        tr.update(hf=hf, logits=logits, probs=probs)
        return probs

    # ------------------------------------------------------------ loss
    @staticmethod
    def cross_entropy(probs, targets):
        p = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
        return float(-np.mean(np.log(p + 1e-12)))

    # ------------------------------------------------------------ backward
    def backward(self, tr, targets):
        """Chain rule in reverse. Returns d(loss)/d(weight) for every named weight."""
        W, cfg = self.W, self.cfg
        H, dh = cfg["H"], cfg["D"] // cfg["H"]
        G = {name: np.zeros_like(w) for name, w in W.items()}
        sum_rows = lambda a, b: a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

        dlogits = tr["probs"].copy()
        np.put_along_axis(dlogits, targets[..., None], np.take_along_axis(dlogits, targets[..., None], -1) - 1, -1)
        dlogits /= targets.size
        G["W_U"] = sum_rows(tr["hf"], dlogits)
        dhf = dlogits @ W["W_U"].T
        dx, G["g_f"] = rmsnorm_backward(dhf, tr["layers"][-1]["xout"], W["g_f"])

        for l in reversed(range(cfg["L"])):
            Ly, P = tr["layers"][l], lambda name: W[f"L{l}.{name}"]
            G[f"L{l}.W_down"] = sum_rows(Ly["act"], dx)
            dact = dx @ P("W_down").T
            dup = dact * gelu_backward(Ly["up"])
            G[f"L{l}.W_up"] = sum_rows(Ly["h2"], dup)
            dh2 = dup @ P("W_up").T
            dxm, G[f"L{l}.g2"] = rmsnorm_backward(dh2, Ly["xmid"], P("g2"))
            dx = dx + dxm

            G[f"L{l}.W_O"] = sum_rows(Ly["mix"], dx)
            dmix = split_heads(dx @ P("W_O").T, H)
            att = Ly["att"]
            qh, kh, vh = split_heads(Ly["q"], H), split_heads(Ly["k"], H), split_heads(Ly["v"], H)
            datt = dmix @ vh.swapaxes(-1, -2)
            dvh = att.swapaxes(-1, -2) @ dmix
            dscores = att * (datt - np.sum(datt * att, axis=-1, keepdims=True)) / np.sqrt(dh)
            dq, dk, dv = merge_heads(dscores @ kh), merge_heads(dscores.swapaxes(-1, -2) @ qh), merge_heads(dvh)
            G[f"L{l}.W_Q"] = sum_rows(Ly["h1"], dq)
            G[f"L{l}.W_K"] = sum_rows(Ly["h1"], dk)
            G[f"L{l}.W_V"] = sum_rows(Ly["h1"], dv)
            dh1 = dq @ P("W_Q").T + dk @ P("W_K").T + dv @ P("W_V").T
            dxi, G[f"L{l}.g1"] = rmsnorm_backward(dh1, Ly["xin"], P("g1"))
            dx = dx + dxi

        T = dx.shape[-2]
        G["W_P"][:T] = dx.reshape(-1, T, dx.shape[-1]).sum(0)
        np.add.at(G["W_E"], tr["ids"].reshape(-1), dx.reshape(-1, dx.shape[-1]))
        return G

    # ------------------------------------------------------------ generation with a KV cache
    def step(self, token, pos, cache):
        """Push ONE new token through, appending its key and value to the cache."""
        W, cfg = self.W, self.cfg
        H, dh = cfg["H"], cfg["D"] // cfg["H"]
        x = W["W_E"][token] + W["W_P"][pos]
        for l in range(cfg["L"]):
            P = lambda name: W[f"L{l}.{name}"]
            h1 = rmsnorm(x, P("g1"))
            cache[l]["k"].append(h1 @ P("W_K"))
            cache[l]["v"].append(h1 @ P("W_V"))
            q = (h1 @ P("W_Q")).reshape(H, dh)
            K = np.array(cache[l]["k"]).reshape(-1, H, dh).swapaxes(0, 1)
            V = np.array(cache[l]["v"]).reshape(-1, H, dh).swapaxes(0, 1)
            att = softmax(np.einsum("hd,htd->ht", q, K) / np.sqrt(dh))
            x = x + np.einsum("ht,htd->hd", att, V).reshape(-1) @ P("W_O")
            x = x + gelu(rmsnorm(x, P("g2")) @ P("W_up")) @ P("W_down")
        return softmax(rmsnorm(x, W["g_f"]) @ W["W_U"])


def rmsnorm_backward(dy, x, g):
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    dg = np.sum((dy * x * r).reshape(-1, x.shape[-1]), axis=0)
    dxhat = dy * g
    dx = r * dxhat - x * r ** 3 * np.mean(dxhat * x, axis=-1, keepdims=True)
    return dx, dg


def gelu_backward(z):
    c = 0.7978845608
    t = np.tanh(c * (z + 0.044715 * z ** 3))
    return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * c * (1 + 3 * 0.044715 * z * z)


if __name__ == "__main__":
    m = MicroOpus("words")
    p = m.forward(m.tokenize("the bird flew over the"))[-1]
    for i in np.argsort(-p)[:3]:
        print(f"{m.vocab[i]:>6}  {p[i]:.3f}")
