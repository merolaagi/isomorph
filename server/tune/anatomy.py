"""Where did the fine-tune go? An adapter read as a weight-space object.

For every matrix W with an adapter, the change is dW = (alpha/r) U V^T. Read:

  size        ||dW|| / ||W||, and the share of the whole change each matrix,
              block and kind (attention, MLP) carries
  rank used   how many directions hold 90% of dW's energy, out of the rank given
  new or old  intruder directions (Shuttleworth et al., "LoRA vs Full
              Fine-tuning: An Illusion of Equivalence", 2024): top singular
              vectors of W + dW that match none of W's own singular vectors
              (best |cosine| below 0.6). Compared with a random change of the
              same size and rank, so "LoRA wrote new directions" means more than
              a random change would.
              Also the share of dW's output directions that lie in W's own top
              singular subspace (amplifying what is there) against the share a
              random direction would have.
  laws        heavy-tail exponent alpha and eigenvalues beyond the
              Marchenko-Pastur edge, before and after, from the Laws tab.

compare(a, b) asks the Isomorph question of two fine-tunes of the same base:
do they write into the same directions? Subspace overlap of their changes,
matrix by matrix, against the overlap of random subspaces of the same sizes.
"""
import numpy as np
import torch

from . import lora

INTRUDER_COS = 0.6
TOP_K = 10


def _svd_lowrank(U, V, s):
    """Singular values and left/right singular vectors of s U V^T without forming it."""
    Qu, Ru = torch.linalg.qr(U.double())
    Qv, Rv = torch.linalg.qr(V.double())
    a, sv, bt = torch.linalg.svd(s * Ru @ Rv.T)
    return sv, Qu @ a, Qv @ bt.T


def _intruders(W, dW, k, Uw=None):
    """How many of the top-k left singular vectors of W + dW have best |cos| < INTRUDER_COS with W's."""
    if Uw is None:
        Uw = torch.linalg.svd(W, full_matrices=False)[0]
    Un = torch.linalg.svd(W + dW, full_matrices=False)[0][:, :k]
    best = (Un.T @ Uw).abs().max(-1).values
    return int((best < INTRUDER_COS).sum()), Uw


def _random_like(shape, r, norm, g):
    A = torch.randn(shape[0], r, generator=g, dtype=torch.float64)
    B = torch.randn(shape[1], r, generator=g, dtype=torch.float64)
    D = A @ B.T
    return D * (norm / (D.norm() + 1e-30))


def matrix_reading(name, W, U, V, s, laws=True, seed=0):
    W = W.double()
    sv, L, R = _svd_lowrank(U, V, s)
    e = sv ** 2
    tot = float(e.sum())
    r = U.shape[1]
    out = {"name": name, "shape": list(W.shape), "rank": int(r), "norm": float(tot ** 0.5), "w_norm": float(W.norm()),
           "rel": float(tot ** 0.5 / (float(W.norm()) + 1e-30))}
    if tot <= 0:
        out.update(rank90=0, intruders=0, intruders_random=0.0, in_top=None, in_top_random=None, sv=[])
        return out
    out["rank90"] = int((e.cumsum(0) / tot < 0.9).sum()) + 1
    out["sv"] = [float(x) for x in sv[:min(r, 16)]]
    k = min(TOP_K, min(W.shape))
    dW = (L * sv) @ R.T
    n_int, Uw = _intruders(W, dW, k)
    g = torch.Generator().manual_seed(seed)
    rnd = [_intruders(W, _random_like(W.shape, max(1, out["rank90"]), out["norm"], g), k, Uw)[0] for _ in range(3)]
    out["intruders"], out["intruders_random"] = n_int, float(np.mean(rnd))
    # share of the change's output directions inside W's top singular subspace (top 10% of directions)
    m = max(1, int(0.1 * min(W.shape)))
    top = Uw[:, :m]
    w8 = e / tot
    out["in_top"] = float(((top.T @ L).pow(2).sum(0) * w8).sum())
    out["in_top_random"] = m / W.shape[0]
    if laws:
        from ..laws import _sub, matrix_measures

        rng = np.random.default_rng(0)
        a = W.float().numpy()
        b = (W + dW).float().numpy()
        if min(a.shape) >= 8:
            ma, mb = matrix_measures(_sub(a, rng)), matrix_measures(_sub(b, np.random.default_rng(0)))
            out["laws"] = {"alpha_before": ma.get("alpha"), "alpha_after": mb.get("alpha"),
                           "spikes_before": ma.get("spikes"), "spikes_after": mb.get("spikes")}
    return out


def anatomy(svc, name, laws=True):
    meta, fac = lora.load_adapter(name)
    _, base, _ = svc.load_lm(meta["base"])
    mats = []
    for n, (U, V, s) in fac.items():
        W = lora._param(base, n).detach().double().cpu()
        rd = matrix_reading(n, W, U, V, s, laws=laws)
        mats.append(rd)
    from ..wsp.common import _kind

    layer_of = {}
    for c in lora.select(base, True, True, False):
        layer_of[c["name"]] = c["layer"]
    tot = sum(m["norm"] ** 2 for m in mats) or 1.0
    by_layer, by_kind = {}, {}
    for m in mats:
        share = m["norm"] ** 2 / tot
        m["share"] = share
        m["layer"] = layer_of.get(m["name"], -1)
        m["kind"] = _kind(m["name"])
        by_layer[m["layer"]] = by_layer.get(m["layer"], 0.0) + share
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0.0) + share
    live = [m for m in mats if m["norm"] > 0]
    ints = sum(m["intruders"] for m in live)
    ints_r = sum(m["intruders_random"] for m in live)
    used = [m["rank90"] / max(m["rank"], 1) for m in live]
    in_top = [m["in_top"] for m in live if m.get("in_top") is not None]
    in_top_r = [m["in_top_random"] for m in live if m.get("in_top_random") is not None]
    la = [m["laws"] for m in live if m.get("laws")]
    summary = {
        "matrices": len(mats), "changed": len(live), "rank": max((m["rank"] for m in mats), default=0),
        "median_rel": float(np.median([m["rel"] for m in live])) if live else 0.0,
        "median_rank90": float(np.median([m["rank90"] for m in live])) if live else 0.0,
        "rank_used": float(np.mean(used)) if used else 0.0,
        "intruders": ints, "intruders_random": ints_r,
        "in_top": float(np.mean(in_top)) if in_top else None, "in_top_random": float(np.mean(in_top_r)) if in_top_r else None,
        "by_layer": [{"layer": k, "share": v} for k, v in sorted(by_layer.items())],
        "by_kind": by_kind,
        "alpha_shift": float(np.median([x["alpha_after"] - x["alpha_before"] for x in la
                                        if x.get("alpha_after") is not None and x.get("alpha_before") is not None])) if la else None,
        "spikes_added": int(sum((x.get("spikes_after") or 0) - (x.get("spikes_before") or 0) for x in la)) if la else None,
    }
    summary["lines"] = _lines(summary, mats)
    out = {"adapter": meta["name"], "base": meta["base"], "summary": summary, "matrices": mats}
    try:
        import json

        from ..jsonsafe import clean

        (lora.adapter_dir(name) / "anatomy.json").write_text(json.dumps(clean(out)))
    except Exception:
        pass
    return out


def _lines(s, mats):
    out = []
    if not s["changed"]:
        return ["The adapter is still zero: nothing was learned."]
    top = max(s["by_layer"], key=lambda x: x["share"]) if s["by_layer"] else None
    kinds = sorted(s["by_kind"].items(), key=lambda kv: -kv[1])
    if top:
        out.append(f"Block {top['layer']} carries {top['share'] * 100:.0f}% of the change; by kind, "
                   + ", ".join(f"{k} {v * 100:.0f}%" for k, v in kinds) + ".")
    out.append(f"The median matrix moved {s['median_rel'] * 100:.2f}% of its own size and used {s['median_rank90']:.0f} of "
               f"{s['rank']} directions for 90% of its change" + (" (rank to spare: a smaller rank would do)." if s["rank_used"] < 0.5 else "."))
    if s["intruders"] > 1.5 * s["intruders_random"] + 1:
        out.append(f"It made {s['intruders']} intruder directions (new top directions that the base matrix lacked), against "
                   f"{s['intruders_random']:.1f} for a random change of the same size and rank: the fine-tune wrote new directions.")
    elif s["intruders"] < 0.5 * s["intruders_random"]:
        out.append(f"It made {s['intruders']} intruder directions against {s['intruders_random']:.1f} for a random change of the "
                   f"same size: it kept to the base matrices' own directions.")
    else:
        out.append(f"It made {s['intruders']} intruder directions, about as many as a random change of the same size "
                   f"({s['intruders_random']:.1f}).")
    if s["in_top"] is not None:
        ratio = s["in_top"] / max(s["in_top_random"], 1e-9)
        verb = "amplified directions the base already used most" if ratio > 2 else \
            "mostly wrote outside the base's main directions" if ratio < 0.7 else "wrote about as much into the base's main directions as chance"
        out.append(f"{s['in_top'] * 100:.0f}% of the change's output lies in each matrix's top 10% of directions, against "
                   f"{s['in_top_random'] * 100:.0f}% by chance: it {verb}.")
    if s["alpha_shift"] is not None:
        out.append(f"Laws: the heavy-tail exponent moved by {s['alpha_shift']:+.2f} in the median matrix and "
                   f"{s['spikes_added']:+d} eigenvalues crossed the noise edge.")
    return out


def compare(svc, a, b):
    ma, fa = lora.load_adapter(a)
    mb, fb = lora.load_adapter(b)
    if ma["base"] != mb["base"]:
        raise ValueError(f"These adapters tune different base models ({ma['base']} and {mb['base']}); compare fine-tunes of one base.")
    rows = []
    g = torch.Generator().manual_seed(0)
    for n in fa:
        if n not in fb:
            continue
        sa, La, Ra = _svd_lowrank(*fa[n])
        sb, Lb, Rb = _svd_lowrank(*fb[n])
        if float(sa.sum()) == 0 or float(sb.sum()) == 0:
            continue
        ka = int(((sa ** 2).cumsum(0) / (sa ** 2).sum() < 0.9).sum()) + 1
        kb = int(((sb ** 2).cumsum(0) / (sb ** 2).sum() < 0.9).sum()) + 1
        k = min(ka, kb)
        m = La.shape[0]
        ov = float((La[:, :ka].T @ Lb[:, :kb]).pow(2).sum() / k)
        ov_in = float((Ra[:, :ka].T @ Rb[:, :kb]).pow(2).sum() / k)
        rnd = []
        for _ in range(5):
            Q = torch.linalg.qr(torch.randn(m, kb, generator=g, dtype=torch.float64))[0]
            rnd.append(float((La[:, :ka].T @ Q).pow(2).sum() / k))
        da, db = lora.delta(*fa[n]).double(), lora.delta(*fb[n]).double()
        cos = float((da * db).sum() / (da.norm() * db.norm() + 1e-30))
        rows.append({"name": n, "k_a": ka, "k_b": kb, "overlap_out": ov, "overlap_in": ov_in, "random": float(np.mean(rnd)),
                     "cosine": cos})
    if not rows:
        raise ValueError("The two adapters share no trained matrix.")
    o = float(np.mean([r["overlap_out"] for r in rows]))
    rn = float(np.mean([r["random"] for r in rows]))
    c = float(np.mean([r["cosine"] for r in rows]))
    ratio = o / max(rn, 1e-9)
    if ratio > 3:
        v = (f"The two fine-tunes write into largely the same directions: {o * 100:.0f}% subspace overlap against "
             f"{rn * 100:.1f}% for random subspaces of the same size ({ratio:.0f}×).")
    elif ratio > 1.5:
        v = f"Some shared directions: {o * 100:.0f}% overlap against {rn * 100:.1f}% by chance ({ratio:.1f}×)."
    else:
        v = f"No more shared directions than chance: {o * 100:.0f}% overlap against {rn * 100:.1f}% for random subspaces."
    v += f" Mean cosine between the two changes: {c:+.2f}" + (" (they point the same way)." if c > 0.3 else
                                                               " (opposite ways)." if c < -0.3 else ".")
    return {"a": ma["name"], "b": mb["name"], "base": ma["base"], "rows": rows, "overlap": o, "random": rn, "cosine": c,
            "verdict": v}
