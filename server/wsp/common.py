"""Shared pieces for weight-space programming: task examples, scoring, which
matrices can be tuned, a low-rank delta that can be added to any of them and
removed again exactly, and how far a change moves the model on general text.

An example is one line of text. "prompt => target" scores only the target
tokens given the prompt; a line without "=>" scores every token after the
first. Scores are always teacher-forced: no sampling, so they are exact and
repeatable.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from ..hub.acts import find_blocks

SEP = "=>"
MAX_TOKENS = 64


# ------------------------------------------------------------------ examples
def parse_examples(text_or_lines):
    lines = text_or_lines.splitlines() if isinstance(text_or_lines, str) else list(text_or_lines)
    out = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if SEP in s:
            p, t = s.split(SEP, 1)
            p, t = p.rstrip(), t.strip()
            if p and t:
                out.append({"prompt": p, "target": " " + t})
        else:
            out.append({"prompt": "", "target": s})
    return out


def split(examples, held_out=None, frac=0.25, seed=0):
    """(practice, held_out). A separate held-out list wins; otherwise a fixed share is set aside."""
    if held_out:
        return list(examples), list(held_out)
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    k = max(1, int(round(len(ex) * frac)))
    if len(ex) < 4:
        raise ValueError("Give at least 4 examples (or a separate held-out list) so some can be kept unseen.")
    return ex[k:], ex[:k]


def _ids(tok, text):
    ids = tok(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode(tok, examples):
    """Token ids and a mask of the positions that are scored, one per example."""
    lim = min(MAX_TOKENS, getattr(tok, "max_len", MAX_TOKENS) or MAX_TOKENS)
    rows = []
    for e in examples:
        if e["prompt"]:
            p = _ids(tok, e["prompt"])
            f = _ids(tok, e["prompt"] + e["target"])[:lim]
            n = len(p) if f[:len(p)] == p else len(_ids(tok, e["prompt"]))
            mask = [0] * len(f)
            for i in range(max(1, n), len(f)):
                mask[i] = 1
        else:
            f = _ids(tok, e["target"])[:lim]
            mask = [0] + [1] * (len(f) - 1)
        if sum(mask) == 0 or len(f) < 2:
            continue
        rows.append((f, mask))
    if not rows:
        raise ValueError("None of the examples gives a token to score with this model's vocabulary.")
    return rows


def _batch(rows, device, pad=0):
    w = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [pad] * (w - len(f)) for f, _ in rows], device=device)
    att = torch.tensor([[1] * len(f) + [0] * (w - len(f)) for f, _ in rows], device=device)
    msk = torch.tensor([m + [0] * (w - len(m)) for _, m in rows], device=device, dtype=torch.float32)
    return ids, att, msk


def model_device(model):
    return next(model.parameters()).device


def token_loss(model, rows, batch=16, grad=False):
    """Mean negative log-probability per scored token, and the share of scored tokens
    that are the model's top prediction. With grad=True returns the loss tensor."""
    dev = model_device(model)
    tot, n, hits, losses = 0.0, 0, 0, []
    for i in range(0, len(rows), batch):
        ids, att, msk = _batch(rows[i:i + batch], dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lg = model(input_ids=ids, attention_mask=att).logits.float()
            lp = lg[:, :-1].log_softmax(-1)
            tgt = ids[:, 1:]
            m = msk[:, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            if grad:
                losses.append((nll * m).sum())
            tot += float((nll * m).sum().detach())
            hits += int(((lp.argmax(-1) == tgt).float() * m).sum().detach())
            n += int(m.sum())
    if grad:
        return torch.stack(losses).sum() / max(n, 1)
    return {"loss": tot / max(n, 1), "acc": hits / max(n, 1), "tokens": n}


# ------------------------------------------------------------------ general text, for drift
def own_texts(tok, model=None, n=200):
    """Texts in the model's own language: its calibration texts, or for a Micro Opus model, sentences
    (or names) cut from its own training stream at <s> and at full stops."""
    cal = getattr(tok, "calibration_texts", None)
    if cal:
        return list(cal[:n])
    spec = getattr(model, "spec", None)
    if spec and hasattr(model, "mo_cfg"):
        from .. import microopus

        blob = microopus.weights_blob(spec)
        vocab, stream = blob["vocab"], blob.get("corpus") or []
        stop = vocab.index(".") if "." in vocab else None
        sep = "" if blob["config"].get("chars") else " "
        out, cur = [], []
        for i in stream:
            if i == 0:
                if cur:
                    out.append(cur)
                cur = []
                continue
            cur.append(i)
            if i == stop:
                out.append(cur)
                cur = []
            if len(out) >= n:
                break
        return [sep.join(vocab[i] for i in t) for t in out if t]
    return []


def general_texts(tok, n=24, model=None):
    own = own_texts(tok, model, n)
    if own:
        return own[:n]
    from ..hub.corpus import CORPUS

    return CORPUS[:n]


def next_token_logp(model, tok, texts):
    rows = [(f, [1] * len(f)) for f in (_ids(tok, t)[:MAX_TOKENS] for t in texts) if len(f) >= 2]
    dev = model_device(model)
    outs = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            ids, att, _ = _batch(rows[i:i + 16], dev)
            lg = model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
            for j, (f, _) in enumerate(rows[i:i + 16]):
                outs.append(lg[j, :len(f)].cpu())
    return outs


def drift(before, after):
    """Mean KL(before || after) per token on general text: how much else the change moved."""
    kl, n = 0.0, 0
    for a, b in zip(before, after):
        kl += float((a.exp() * (a - b)).sum())
        n += a.shape[0]
    return kl / max(n, 1)


# ------------------------------------------------------------------ which matrices
def tunable(model):
    """Every 2-D weight inside the transformer blocks (attention and MLP matrices),
    with its owning module and attribute, so a delta can be attached to it."""
    blocks = find_blocks(model)
    if blocks is None:
        raise ValueError("Could not find this model's transformer blocks.")
    out = []
    names = {id(m): n for n, m in model.named_modules()}
    for li, b in enumerate(blocks):
        for mn, m in b.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if p.ndim != 2 or "norm" in (mn + pn).lower():
                    continue
                full = f"{names[id(b)]}.{mn + '.' if mn else ''}{pn}"
                out.append({"name": full, "layer": li, "module": m, "attr": pn, "shape": tuple(p.shape),
                            "kind": _kind(mn + "." + pn)})
    return out


def _kind(n):
    n = n.lower()
    if any(k in n for k in ("mlp", "ffn", "feed_forward", ".fc", "up_proj", "down_proj", "gate_proj", "dense_h_to_4h", "dense_4h_to_h")):
        return "mlp"
    if any(k in n for k in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "query", "c_attn")):
        return "attention"
    return "other"


class LowRank(nn.Module):
    """W -> W + scale * U V^T. U starts at zero, so the model is unchanged until trained."""

    def __init__(self, shape, rank, alpha=None, seed=0, device="cpu"):
        super().__init__()
        m, n = shape
        g = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(torch.zeros(m, rank, device=device))
        self.V = nn.Parameter((torch.randn(n, rank, generator=g) / math.sqrt(n)).to(device))
        self.scale = float(alpha if alpha is not None else 1.0)

    def forward(self, W):
        return W + self.scale * (self.U @ self.V.T).to(W.dtype)


def lowrank_params(shape, rank):
    return rank * (shape[0] + shape[1])


class Deltas:
    """Attach low-rank deltas to chosen matrices; remove them and put the exact
    original weights back afterwards, even if training fails."""

    def __init__(self, model):
        self.model, self.items, self.frozen, self.orders = model, [], [], {}

    def __enter__(self):
        for p in self.model.parameters():
            if p.requires_grad:
                self.frozen.append(p)
                p.requires_grad_(False)
        return self

    def add(self, entry, rank, seed=0):
        mod = entry["module"]
        self.orders.setdefault(id(mod), (mod, list(mod._parameters.keys())))
        p = getattr(mod, entry["attr"])
        lr = LowRank(tuple(p.shape), rank, seed=seed, device=p.device)
        P.register_parametrization(entry["module"], entry["attr"], lr)
        self.items.append((entry, lr))
        return lr

    def parameters(self):
        for _, lr in self.items:
            yield lr.U
            yield lr.V

    def __exit__(self, *exc):
        for entry, _ in reversed(self.items):
            if P.is_parametrized(entry["module"], entry["attr"]):
                P.remove_parametrizations(entry["module"], entry["attr"], leave_parametrized=False)
        for mod, keys in self.orders.values():          # put parameters back in their original order
            params = mod._parameters
            ordered = {k: params[k] for k in keys if k in params}
            ordered.update({k: v for k, v in params.items() if k not in ordered})
            params.clear()
            params.update(ordered)
        self.orders = {}
        for p in self.frozen:
            p.requires_grad_(True)
        self.items = []
        return False


def checksum(model):
    """Name -> (sum, sum of squares) of every parameter: equal before and after means restored (and in order)."""
    with torch.no_grad():
        return [(n, float(p.detach().double().sum()), float(p.detach().double().pow(2).sum())) for n, p in model.named_parameters()]


def checksum_diff(a, b):
    if [x[0] for x in a] != [x[0] for x in b]:
        return float("inf")
    return max((max(abs(x[1] - y[1]), abs(x[2] - y[2])) for x, y in zip(a, b)), default=0.0)
