"""
Trains Micro Opus from random noise. This is the only code that ever changes a weight.

    python train.py words     the toy sentence world, one word per token
    python train.py names     32k real first names, one character per token

One training step is four moves:
    1. forward   push a batch of text through the current weights
    2. loss      how surprised was the model by the real next token?
    3. backward  chain rule in reverse: one gradient per weight
    4. update    nudge every weight against its gradient (Adam)
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from model import MicroOpus

ROOT = Path(__file__).parent
CONFIG = dict(D=16, H=2, L=2, F=64, T=16)


def words_corpus(rng):
    subj = {"cat": ["sat", "slept"], "dog": ["ran", "slept"], "bird": ["flew", "sang"],
            "fish": ["swam"], "kid": ["ran", "sat", "sang"]}
    place = {"sat": ["on", ["mat", "bed", "chair"]], "slept": ["on", ["mat", "bed"]],
             "ran": ["to", ["park", "house", "store"]], "flew": ["over", ["tree", "house"]],
             "sang": ["in", ["tree", "park"]], "swam": ["in", ["pond", "lake"]]}
    adjs = ["big", "small", "happy"]
    vocab = ["<s>", "the", ".", *adjs, *subj, *place, "on", "to", "over", "in"]
    for _, objs in place.values():
        vocab += [o for o in objs if o not in vocab]

    def sentence():
        s = rng.choice(list(subj))
        v = rng.choice(subj[s])
        p, objs = place[v]
        words = ["the"] + ([rng.choice(adjs)] if rng.random() < 0.35 else [])
        return words + [s, v, p, "the", rng.choice(objs), "."]

    def stream(n_tokens):
        out = ["<s>"]
        while len(out) < n_tokens:
            out += sentence()
        return [vocab.index(w) for w in out]

    return vocab, stream(30000), stream(3000), False


def names_corpus(rng):
    names = (ROOT / "data" / "names.txt").read_text().split()
    rng.shuffle(names)
    vocab = ["<s>"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    enc = lambda ns: [i for n in ns for i in [0] + [vocab.index(ch) for ch in n]] + [0]
    return vocab, enc(names[:-1000]), enc(names[-1000:]), True


def init_weights(cfg, V, rng):
    D, F, T = cfg["D"], cfg["F"], cfg["T"]
    r = lambda *s: rng.normal(0, 0.08, s)
    W = {"W_E": r(V, D), "W_P": r(T, D), "g_f": np.ones(D), "W_U": r(D, V)}
    for l in range(cfg["L"]):
        W |= {f"L{l}.g1": np.ones(D), f"L{l}.W_Q": r(D, D), f"L{l}.W_K": r(D, D), f"L{l}.W_V": r(D, D),
              f"L{l}.W_O": r(D, D), f"L{l}.g2": np.ones(D), f"L{l}.W_up": r(D, F), f"L{l}.W_down": r(F, D) * 0.5}
    return W


def batch(stream, T, B, rng):
    starts = rng.integers(0, len(stream) - T - 1, B)
    chunk = np.array([stream[s:s + T + 1] for s in starts])
    return chunk[:, :-1], chunk[:, 1:]


def train(kind, steps, B=32, lr=0.01, seed=7):
    rng = np.random.default_rng(seed)
    vocab, stream, held_out, chars = (words_corpus if kind == "words" else names_corpus)(random.Random(seed))
    cfg = dict(CONFIG, V=len(vocab), chars=chars)
    model = MicroOpus(W=init_weights(cfg, len(vocab), rng), cfg=cfg, vocab=vocab)
    m = {k: np.zeros_like(w) for k, w in model.W.items()}
    v = {k: np.zeros_like(w) for k, w in model.W.items()}
    b1, b2 = 0.9, 0.99
    curve, t0 = [], time.time()
    xe, ye = batch(held_out, cfg["T"], 256, np.random.default_rng(1))

    for step in range(1, steps + 1):
        xb, yb = batch(stream, cfg["T"], B, rng)
        tr = {}
        probs = model.forward(xb, tr)
        loss = model.cross_entropy(probs, yb)
        grads = model.backward(tr, yb)
        lr_t = lr * min(1, step / 100) * (0.1 + 0.9 * (1 - step / steps))
        for k, g in grads.items():
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            model.W[k] -= lr_t * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + 1e-8)
        if step == 1 or step % 100 == 0:
            held = model.cross_entropy(model.forward(xe), ye)
            curve.append([step, round(loss, 4), round(held, 4)])
            print(f"step {step:5d}  loss {loss:.3f}  held-out {held:.3f}  ({time.time() - t0:.0f}s)")

    R = lambda a: np.round(a, 4).tolist()
    out = {
        "config": cfg, "vocab": vocab, "loss_curve": curve,
        "weights": {k: R(w) for k, w in model.W.items()},
        "eval": {"x": xe[:48].tolist(), "y": ye[:48].tolist()},
        "corpus": stream[:40000],
    }
    path = ROOT / "weights" / f"{kind}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path}  ({sum(w.size for w in model.W.values())} weights)")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "words"
    train(kind, steps=3000 if kind == "words" else 6000)
