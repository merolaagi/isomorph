"""Surgery: damage chosen weights and watch behaviour change.

Edits are applied in place to the loaded model, measured, and always restored
(the original values are kept and written back even if something fails), so a
cached model is never left damaged.

An edit is {tensor, target, index, op, amount, seed}:
  target  all | row | col      index  one number or [first, last]
  op      zero | scale | noise | negate
Quick edits (heads, neurons, token rows) are expanded into plain edits by
`expand`, using each architecture's own layout.
"""
import math

import torch
import torch.nn.functional as F

from ..hub.acts import find_blocks
from ..hub.load import load_lm
from .anatomy import _child, _linear_roles, ROLE, MLP_ROLE


def params_info(model):
    alias = {}
    if hasattr(model, "mo_cfg"):
        from .. import microopus

        alias = {k: v[0] for k, v in microopus.param_map(model.mo_cfg["L"]).items()}
    out = []
    for n, p in model.named_parameters():
        out.append({"name": n, "shape": list(p.shape), "params": p.numel(), "alias": alias.get(n)})
    cfg = model.config
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    blocks = find_blocks(model)
    return {"params": out, "layers": len(blocks) if blocks is not None else None, "heads": heads,
            "d_model": getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None),
            "d_mlp": getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None)}


def _prefix_of(model, module):
    for n, m in model.named_modules():
        if m is module:
            return n
    return None


def expand(model, tok, quick):
    """Turn a quick edit (remove a head, a neuron, a token) into tensor edits."""
    kind = quick["kind"]
    blocks = find_blocks(model)
    if kind == "token":
        emb = model.get_input_embeddings()
        ids = tok(quick["token"], return_tensors=None)["input_ids"]
        from .. import microopus

        if isinstance(tok, microopus.Tokenizer):
            ids = ids[1:]  # Micro Opus starts every text with <s>
        if not ids:
            raise ValueError(f"'{quick['token']}' is not in this model's vocabulary.")
        name = _prefix_of(model, emb) + ".weight"
        return [{"tensor": name, "target": "row", "index": int(i), "op": "zero"} for i in ids[:4]], \
               f"remove the embedding of {', '.join(repr(tok.decode([i])) for i in ids[:4])}"
    layer = int(quick["layer"])
    b = blocks[layer]
    if kind == "head":
        an, attn = _child(b, ("attention", "attn", "self_attn", "self_attention"), ("attention",))
        out = next((l for l in _linear_roles(attn, ROLE) if l["role"] == "Output projection"), None)
        if out is None:
            raise ValueError("Could not find this model's attention output projection.")
        mod = getattr(attn, out["module"])
        heads = getattr(model.config, "num_attention_heads", None) or getattr(model.config, "n_head")
        W = mod.weight
        conv = type(mod).__name__ == "Conv1D"  # GPT-2 stores (in, out)
        width = (W.shape[0] if conv else W.shape[1])
        dh = width // heads
        h = int(quick["index"])
        name = _prefix_of(model, mod) + ".weight"
        return [{"tensor": name, "target": "row" if conv else "col", "index": [h * dh, (h + 1) * dh - 1], "op": "zero"}], \
               f"remove attention head {h} of layer {layer}"
    if kind == "neuron":
        mn, mlp = _child(b, ("mlp", "feed_forward", "ffn"), ("mlp", "feedforward"))
        roles = {l["role"]: l for l in _linear_roles(mlp, MLP_ROLE)}
        down = roles.get("Down projection")
        if down is None:
            raise ValueError("Could not find this model's MLP down projection.")
        mod = getattr(mlp, down["module"])
        conv = type(mod).__name__ == "Conv1D"
        n = int(quick["index"])
        name = _prefix_of(model, mod) + ".weight"
        return [{"tensor": name, "target": "row" if conv else "col", "index": n, "op": "zero"}], \
               f"silence MLP neuron {n} of layer {layer}"
    raise ValueError(f"Unknown quick edit {kind}.")


def _slice(p, target, index):
    if target == "all":
        return (slice(None),) * p.ndim
    lo, hi = (index, index) if isinstance(index, int) else (int(index[0]), int(index[1]))
    if target == "row":
        return (slice(lo, hi + 1),) + (slice(None),) * (p.ndim - 1)
    if target == "col":
        if p.ndim < 2:
            raise ValueError("A one-dimensional tensor has no columns.")
        return (slice(None), slice(lo, hi + 1))
    raise ValueError(f"Unknown target {target}.")


def _apply(model, edits):
    params = dict(model.named_parameters())
    backups = []
    with torch.no_grad():
        for e in edits:
            p = params.get(e["tensor"])
            if p is None:
                raise ValueError(f"No tensor called {e['tensor']}.")
            sl = _slice(p, e.get("target", "all"), e.get("index", 0))
            backups.append((p, sl, p[sl].clone()))
            v = p[sl]
            op, amt = e.get("op", "zero"), float(e.get("amount", 1.0))
            if op == "zero":
                v.zero_()
            elif op == "scale":
                v.mul_(amt)
            elif op == "negate":
                v.neg_()
            elif op == "noise":
                g = torch.Generator().manual_seed(int(e.get("seed", 1)))
                sd = float(p.detach().float().std()) or 1.0
                v.add_((torch.randn(v.shape, generator=g) * amt * sd).to(v.device, v.dtype))
            else:
                raise ValueError(f"Unknown operation {op}.")
    return backups


def _restore(backups):
    with torch.no_grad():
        for p, sl, old in reversed(backups):
            p[sl] = old


def _measure(model, tok, ids):
    from .trace import final_norm

    x = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True)
    lg = out.logits[0].float()
    loss = float(F.cross_entropy(lg[:-1], x[0, 1:])) if len(ids) > 1 else None
    p = lg[-1].softmax(-1)
    top = p.topk(5)
    norm, head = final_norm(model), model.get_output_embeddings()
    lens = []
    hs = out.hidden_states
    with torch.no_grad():
        for i, h in enumerate(hs):
            if i == len(hs) - 1:
                l = out.logits[0, -1].float()  # the last hidden state is already normed
            else:
                z = h[0, -1:]
                l = head(norm(z) if norm is not None else z)[0].float()
            j = int(l.argmax())
            lens.append({"t": tok.decode([j]), "p": float(l.softmax(-1)[j])})
    return {"logp": lg[-1].log_softmax(-1).cpu(), "loss": loss, "hidden": [h[0].float().cpu() for h in hs],
            "top": [{"t": tok.decode([int(i)]), "p": float(v)} for v, i in zip(top.values, top.indices)], "lens": lens}


def surgery_report(spec, prompt, edits=None, quick=None, loader=load_lm):
    tok, model, causal = loader(spec)
    if not causal:
        raise ValueError("Surgery needs a model that predicts the next token.")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
    ids = enc["input_ids"][0].tolist()
    if len(ids) < 2:
        raise ValueError("The prompt needs at least one word the model knows.")
    plan, labels = list(edits or []), []
    for q in quick or []:
        e, label = expand(model, tok, q)
        plan += e
        labels.append(label)
    for e in edits or []:
        labels.append(f"{e.get('op', 'zero')} {e.get('target', 'all')}{'' if e.get('target', 'all') == 'all' else ' ' + str(e.get('index'))} of {e['tensor']}" +
                      (f" (×{e.get('amount')})" if e.get("op") in ("scale", "noise") else ""))
    if not plan:
        raise ValueError("Add at least one edit.")
    before = _measure(model, tok, ids)
    backups = _apply(model, plan)
    try:
        touched = sum(int(old.numel()) for _, _, old in backups)
        after = _measure(model, tok, ids)
    finally:
        _restore(backups)
    check = _measure(model, tok, ids)
    restored = float((check["logp"] - before["logp"]).abs().max())
    kl = float((before["logp"].exp() * (before["logp"] - after["logp"])).sum())
    changes = [float(((a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-9)).mean()) for a, b in zip(after["hidden"], before["hidden"])]
    total = sum(p.numel() for p in model.parameters())
    pred = before["top"][0]["t"]
    idx = int(before["logp"].argmax())
    return {"spec": spec, "prompt": prompt, "tokens": [tok.decode([i]) for i in ids], "edits": plan, "labels": labels,
            "weights_touched": touched, "weights_total": total,
            "before": {k: before[k] for k in ("loss", "top", "lens")}, "after": {k: after[k] for k in ("loss", "top", "lens")},
            "kl": kl, "p_orig_before": float(before["logp"][idx].exp()), "p_orig_after": float(after["logp"][idx].exp()),
            "prediction": pred, "flipped": after["top"][0]["t"] != pred, "layer_change": changes, "restored_diff": restored}
