"""Profile one model into the pattern library.

A profile holds four kinds of pattern, each measured the same way for every
model, and each also measured on a randomly initialised copy of the same
architecture. The random copy is the baseline: a pattern only counts if the
trained model has it and the random copy does not.

  fingerprint  per-layer similarity structure over a fixed anchor set
               (centred Gram matrices; comparable across any two models)
  circuits     per-head scores for known attention mechanisms, plus a causal
               test: does removing the induction heads break in-context copying?
  concepts     per-layer linear readability of plural, tense, negation,
               sentiment and number magnitude, with a shuffled-label control
  weights      per-matrix spectral statistics (effective rank, heavy tail)
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from .. import metrics as M
from ..hub.acts import find_blocks
from ..hub.load import load_lm
from ..hub.weights import _kind, _layer_of
from .probes import ANCHOR_VERSION, anchor_texts, concepts

BATCH = 16


def profile_id(spec):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", spec).strip("-")[-60:]
    return f"{slug}-{hashlib.sha1(spec.encode()).hexdigest()[:6]}"


# ------------------------------------------------------------------ running text through a model
def _encode(tok, texts, max_len=64):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.convert_ids_to_tokens(0)
    tok.padding_side = "right"
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)


def hidden_pooled(tok, model, texts, mode="mean"):
    """Per layer: (N, d) with mean over tokens (skipping the first) or the last token."""
    dev = next(model.parameters()).device
    out = None
    for i in range(0, len(texts), BATCH):
        enc = _encode(tok, texts[i:i + BATCH])
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        with torch.no_grad():
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        lens = mask.sum(1)
        rows = []
        for h in hs:
            h = h.float()
            if mode == "last":
                v = h[torch.arange(h.shape[0]), lens - 1]
            else:
                m = mask.clone().float()
                m[:, 0] = torch.where(lens > 1, 0.0, 1.0)
                v = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            rows.append(v.cpu())
        out = rows if out is None else [torch.cat([a, b]) for a, b in zip(out, rows)]
    return out


def centred_gram(x):
    x = x.double()
    x = x - x.mean(0, keepdim=True)
    k = x @ x.T
    return (k / (k.norm() + 1e-12)).float()


# ------------------------------------------------------------------ circuits
def _attn_out_proj(block):
    for name in ("attention", "attn", "self_attn", "self_attention"):
        a = getattr(block, name, None)
        if a is not None:
            for pn in ("dense", "o_proj", "c_proj", "out_proj", "wo"):
                m = getattr(a, pn, None)
                if isinstance(m, torch.nn.Module):
                    return m
    return None


def circuits(tok, model, seed=0, T=40, B=4):
    """Head scores on random token sequences, and an ablation test for induction."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    V = min(model.get_input_embeddings().num_embeddings, len(tok)) if hasattr(tok, "__len__") else model.get_input_embeddings().num_embeddings
    lo = min(100, V // 4)
    rnd = torch.randint(lo, V - 1, (B, T), generator=g)
    rep = torch.cat([rnd, rnd], 1)
    with torch.no_grad():
        att_rep = model(input_ids=rep.to(dev), output_attentions=True).attentions
        att_one = model(input_ids=torch.randint(lo, V - 1, (B, 2 * T), generator=g).to(dev), output_attentions=True).attentions
    if att_rep is None or att_rep[0] is None:
        return None
    L, H = len(att_rep), att_rep[0].shape[1]
    t2 = torch.arange(T + 1, 2 * T)
    scores = {"induction": [], "previous": [], "duplicate": [], "sink": []}
    for l in range(L):
        a = att_rep[l].float().cpu()
        o = att_one[l].float().cpu()
        scores["induction"].append(a[:, :, t2, t2 - T + 1].mean((0, 2)).tolist())
        scores["duplicate"].append(a[:, :, t2, t2 - T].mean((0, 2)).tolist())
        tt = torch.arange(1, 2 * T)
        scores["previous"].append(o[:, :, tt, tt - 1].mean((0, 2)).tolist())
        scores["sink"].append(o[:, :, 2:, 0].mean((0, 2)).tolist())

    # Causal test: remove the top induction heads and measure in-context copying loss.
    blocks = find_blocks(model)
    ind = torch.tensor(scores["induction"])
    flat = ind.flatten()
    k = int(max(1, min(8, (flat > 0.3).sum().item())))
    top = [divmod(int(i), H) for i in flat.topk(k).indices]

    def copy_loss(heads):
        handles = []
        if heads:
            by_layer = {}
            for l, h in heads:
                by_layer.setdefault(l, []).append(h)
            for l, hs in by_layer.items():
                proj = _attn_out_proj(blocks[l]) if blocks is not None else None
                if proj is None:
                    return None

                def pre(_m, args, hs=hs):
                    x = args[0].clone()
                    dh = x.shape[-1] // H
                    for h in hs:
                        x[..., h * dh:(h + 1) * dh] = 0
                    return (x,) + tuple(args[1:])
                handles.append(proj.register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                lg = model(input_ids=rep.to(dev)).logits.float()
        finally:
            for hd in handles:
                hd.remove()
        tgt = rep[:, T + 1:].to(dev)
        return float(torch.nn.functional.cross_entropy(lg[:, T:-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1)))

    base = copy_loss([])
    abl = copy_loss(top)
    ctrl = []
    others = [(l, h) for l in range(L) for h in range(H) if (l, h) not in top]
    for r in range(3):
        rr = torch.Generator().manual_seed(100 + r)
        pick = [others[int(i)] for i in torch.randperm(len(others), generator=rr)[:k]]
        c = copy_loss(pick)
        if c is not None:
            ctrl.append(c)
    first_half = None
    with torch.no_grad():
        lg = model(input_ids=rep.to(dev)).logits.float()
        first_half = float(torch.nn.functional.cross_entropy(lg[:, :T - 1].reshape(-1, lg.shape[-1]), rep[:, 1:T].to(dev).reshape(-1)))
    return {
        "heads": H, "layers": L, "scores": scores,
        "ablation": {"heads": [list(x) for x in top], "loss_base": base, "loss_ablated": abl,
                     "loss_random_heads": (sum(ctrl) / len(ctrl)) if ctrl else None, "loss_first_half": first_half},
    }


# ------------------------------------------------------------------ concepts
def _ridge_cv(x, y, classify, folds=5, lam=1.0, seed=0):
    """k-fold cross-validated ridge probe. Accuracy for classes, R² for regression."""
    n = x.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    x = x.double()
    y = y.double()
    preds = torch.zeros(n, dtype=torch.float64)
    for f in range(folds):
        te = idx[f::folds]
        tr = torch.tensor([i for i in idx.tolist() if i not in set(te.tolist())])
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
        ym = y[tr].mean()
        # dual form: cheap when features outnumber samples
        k = xtr @ xtr.T
        alpha = torch.linalg.solve(k + lam * xtr.shape[1] * 1e-2 * torch.eye(len(tr), dtype=k.dtype), y[tr] - ym)
        preds[te] = xte @ (xtr.T @ alpha) + ym
    if classify:
        return float(((preds > 0.5).double() == y).double().mean())
    ss = float(((y - y.mean()) ** 2).sum())
    return max(-1.0, 1 - float(((preds - y) ** 2).sum()) / (ss + 1e-12))


def concept_scores(tok, model):
    out = {}
    for name, c in concepts().items():
        texts = [t for t, _ in c["items"]]
        y = torch.tensor([float(v) for _, v in c["items"]])
        classify = c["kind"] == "class"
        if not classify:
            y = torch.log10(y)
        hs = hidden_pooled(tok, model, texts, mode="last")
        acc = [_ridge_cv(h, y, classify) for h in hs]
        yp = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(1))]
        control = [_ridge_cv(h, yp, classify) for h in hs]
        out[name] = {"kind": c["kind"], "about": c["about"], "score": acc, "control": control, "n": len(texts)}
    return out


# ------------------------------------------------------------------ weights
def weight_signature(model):
    rows = []
    blocks = find_blocks(model)
    L = len(blocks) if blocks is not None else 1
    for n, p in model.named_parameters():
        if p.ndim != 2 or min(p.shape) < 16 or min(p.shape) > 4096 or _kind(n) in ("buffer", "norm"):
            continue
        w = p.detach().cpu().double()
        g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
        ev = torch.linalg.eigvalsh(g).clamp_min(1e-12).flip(0)
        s = ev.sqrt()
        k = max(5, int(0.1 * len(ev)))
        tail = ev[:k]
        alpha = 1 + k / float(torch.log(tail / tail[-1]).sum() + 1e-12)
        lay = _layer_of(n)
        rows.append({"name": n, "kind": _kind(n), "depth": (lay + 0.5) / L if lay is not None else None,
                     "erank_ratio": M.effective_rank(s) / len(s), "alpha": alpha, "shape": list(p.shape)})
    return rows


# ------------------------------------------------------------------ the profile
def random_twin(model):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    try:
        twin = AutoModelForCausalLM.from_config(model.config, attn_implementation="eager")
    except (TypeError, ValueError):
        twin = AutoModelForCausalLM.from_config(model.config)
    return twin.eval().to(next(model.parameters()).device)


def profile_model(spec, resolved, atlas_dir: Path, progress=lambda f, m: None, loader=load_lm, baseline=True):
    progress(0.02, f"Loading {spec}")
    tok, model, causal = loader(resolved)
    if not causal:
        raise ValueError("The pattern library profiles language models that predict the next token. Use the weight comparison for other models.")
    anchors = anchor_texts()
    blocks = find_blocks(model)
    L = len(blocks) if blocks is not None else model.config.num_hidden_layers

    def one(m, tag, f0, f1):
        span = f1 - f0
        progress(f0, f"{tag}: fingerprint over {len(anchors)} anchor texts")
        hs = hidden_pooled(tok, m, anchors)
        grams = [centred_gram(h) for h in hs]
        progress(f0 + 0.35 * span, f"{tag}: attention circuits")
        circ = circuits(tok, m)
        progress(f0 + 0.55 * span, f"{tag}: concept probes")
        conc = concept_scores(tok, m)
        return grams, circ, conc

    t0 = time.time()
    g_real, c_real, k_real = one(model, "Trained model", 0.08, 0.5)
    progress(0.52, "Weight signature")
    wsig = weight_signature(model)
    g_rand = c_rand = k_rand = wsig_rand = None
    if baseline:
        progress(0.58, "Building a randomly initialised copy as the baseline")
        twin = random_twin(model)
        g_rand, c_rand, k_rand = one(twin, "Random baseline", 0.6, 0.95)
        progress(0.96, "Random baseline: weight signature")
        wsig_rand = weight_signature(twin)
        del twin

    pid = profile_id(spec)
    d = atlas_dir / pid
    d.mkdir(parents=True, exist_ok=True)
    arrays = {f"real_{i}": g.numpy().astype(np.float16) for i, g in enumerate(g_real)}
    if g_rand:
        arrays |= {f"rand_{i}": g.numpy().astype(np.float16) for i, g in enumerate(g_rand)}
    np.savez_compressed(d / "kernels.npz", **arrays)
    cfg = model.config
    meta = {
        "id": pid, "spec": spec, "resolved": str(resolved), "created": time.time(), "seconds": round(time.time() - t0, 1),
        "anchor_version": ANCHOR_VERSION, "anchors": len(anchors),
        "arch": (getattr(cfg, "architectures", None) or [type(model).__name__])[0], "model_type": cfg.model_type,
        "layers": L, "d_model": getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None),
        "params": sum(p.numel() for p in model.parameters()),
        "circuits": c_real, "circuits_random": c_rand, "concepts": k_real, "concepts_random": k_rand,
        "weights": wsig, "weights_random": wsig_rand, "has_baseline": bool(g_rand),
    }
    from ..jsonsafe import clean

    (d / "profile.json").write_text(json.dumps(clean(meta)))
    progress(1.0, "Profile saved")
    return meta
