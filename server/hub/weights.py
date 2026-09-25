"""Architecture-agnostic weight comparison.

Works on any pair of checkpoints that share tensor names, including custom
architectures that transformers cannot load (for example Laya). Answers:
  * Are the raw weights close? (only meaningful when the models share an init)
  * Do matching matrices have the same singular-value shape? (invariant to
    rotations and permutations, so meaningful even across seeds)
  * When they are close, is the difference low-rank? (if so, one model is the
    other plus a cheap update, which is where compute savings live)
"""
import re

import torch

from .. import metrics as M

MAX_SVD_SIDE = 4096


def _svals(w):
    w = w.detach().cpu().double()
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    r, c = w.shape
    if min(r, c) > MAX_SVD_SIDE:
        return None
    if max(r, c) > 4 * min(r, c):
        g = w.T @ w if c < r else w @ w.T
        ev = torch.linalg.eigvalsh(g).clamp_min(0).flip(0)
        return ev.sqrt()
    return torch.linalg.svdvals(w)


def _layer_of(name):
    m = re.search(r"\.(\d+)\.", "." + name + ".")
    return int(m.group(1)) if m else None


def _kind(name):
    n = name.lower()
    for key, label in (("lm_head", "unembedding"), ("embed_out", "unembedding"), ("embed", "embedding"),
                       ("wte", "embedding"), ("wpe", "embedding"), ("norm", "norm"), ("ln", "norm"), ("attn", "attention"),
                       ("attention", "attention"), ("query", "attention"), ("key", "attention"), ("value", "attention"),
                       ("qkv", "attention"), ("mlp", "mlp"), ("intermediate", "mlp"), ("dense_h_to_4h", "mlp"),
                       ("dense_4h_to_h", "mlp"), ("fc", "mlp"), ("head", "head")):
        if key in n:
            return label
    return "other"


def weight_report(wa, wb, pairs, progress=lambda f, m: None):
    rows = []
    tot_ab = tot_aa = tot_bb = 0.0
    dnum = 0.0
    n_common = 0
    for i, (ka, kb) in enumerate(pairs):
        a, b = wa[ka], wb[kb]
        if a.shape != b.shape:
            continue
        if i % 8 == 0:
            progress(i / max(1, len(pairs)), f"Comparing {ka}")
        a64, b64 = a.detach().cpu().double().flatten(), b.detach().cpu().double().flatten()
        ab, aa, bb = float(a64 @ b64), float(a64 @ a64), float(b64 @ b64)
        diff = float((b64 - a64).norm())
        n_common += a.numel()
        if a.ndim >= 2:
            # Overall numbers use weight matrices only: norm gains start at 1.0 in
            # every model and would make unrelated models look similar.
            tot_ab, tot_aa, tot_bb = tot_ab + ab, tot_aa + aa, tot_bb + bb
            dnum += diff ** 2
        row = {
            "name": ka, "shape": list(a.shape), "numel": a.numel(), "kind": _kind(ka), "layer": _layer_of(ka),
            "cosine": round(ab / ((aa * bb) ** 0.5 + 1e-30), 4),
            "rel_diff": round(diff / (aa ** 0.5 + 1e-30), 4),
        }
        if a.ndim >= 2 and min(a.shape[0], a[0].numel()) >= 8:
            sa, sb = _svals(a), _svals(b)
            if sa is not None:
                sd = _svals(b - a)
                row.update({
                    "spec_dist": round(M.spectrum_distance(sa, sb), 4),
                    "erank_a": round(M.effective_rank(sa), 1), "erank_b": round(M.effective_rank(sb), 1),
                    "sv_a": [round(v, 4) for v in M.downsample(sa, 48)],
                    "sv_b": [round(v, 4) for v in M.downsample(sb, 48)],
                    "delta_rank90": M.energy_rank(sd, 0.9), "full_rank": int(min(len(sa), len(sb))),
                    "delta_erank": round(M.effective_rank(sd), 1),
                })
                row["delta_rank_frac"] = round(row["delta_rank90"] / max(1, row["full_rank"]), 4)
        rows.append(row)

    overall_cos = tot_ab / ((tot_aa * tot_bb) ** 0.5 + 1e-30)
    overall_rel = (dnum ** 0.5) / (tot_aa ** 0.5 + 1e-30)
    mats = [r for r in rows if "spec_dist" in r]
    spec = sorted(r["spec_dist"] for r in mats)
    dfrac = sorted(r["delta_rank_frac"] for r in mats)
    med = lambda v: v[len(v) // 2] if v else None

    layers = {}
    for r in rows:
        if r["layer"] is None:
            continue
        L = layers.setdefault(r["layer"], {"layer": r["layer"], "cos": [], "rel": [], "spec": []})
        L["cos"].append((r["cosine"], r["numel"]))
        L["rel"].append((r["rel_diff"], r["numel"]))
        if "spec_dist" in r:
            L["spec"].append(r["spec_dist"])
    by_layer = []
    for k in sorted(layers):
        L = layers[k]
        w = sum(n for _, n in L["cos"]) or 1
        by_layer.append({
            "layer": k,
            "cosine": round(sum(c * n for c, n in L["cos"]) / w, 4),
            "rel_diff": round(sum(c * n for c, n in L["rel"]) / w, 4),
            "spec_dist": round(sum(L["spec"]) / len(L["spec"]), 4) if L["spec"] else None,
        })

    if overall_cos > 0.9:
        relation = "shared-init"
        verdict = ("The raw weights are close, so these models share an initialisation or one was fine-tuned from the other. "
                   f"The update is compact: in the median matrix, {med(dfrac) * 100:.1f}% of the directions carry 90% of the change.")
    elif overall_cos > 0.3:
        relation = "partial"
        verdict = "The raw weights are partly aligned. Look at the layer chart: the parts that stayed close were probably frozen or barely trained."
    else:
        relation = "independent"
        verdict = ("The raw weights are unrelated, which is expected for independent random inits: neuron order and residual basis differ. "
                   "Compare the singular-value shapes instead, which ignore those symmetries, or run the activation comparison.")

    return {
        "summary": {
            "tensors_a": len(wa), "tensors_b": len(wb), "paired": len(rows),
            "params_compared": n_common,
            "overall_cosine": round(overall_cos, 4), "overall_rel_diff": round(overall_rel, 4),
            "median_spec_dist": med(spec), "median_delta_rank_frac": med(dfrac),
            "relation": relation, "verdict": verdict,
        },
        "by_layer": by_layer,
        "tensors": rows,
    }
