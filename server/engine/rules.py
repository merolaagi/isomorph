"""Rule miner: turns measurements into candidate rules.

Every rule states what it claims, which measurements support it, which
contradict it, how big the effect is compared with random baselines, and which
experiment could test it. Nothing here is written by hand about specific
models; the rules come only from what is stored in the library and the lab.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..atlas import library as LB

MIN_MODELS = 2


def rid(text):
    return "R-" + hashlib.sha1(text.encode()).hexdigest()[:6]


def _conf(n, counter, effect):
    if n < 2:
        return "anecdote"
    if counter == 0 and n >= 5 and effect >= 0.2:
        return "strong"
    if counter == 0 and n >= 3:
        return "moderate"
    return "weak"


def _rule(kind, statement, support, counter, effect, evidence, lever=None, about="", unit=""):
    n = len(support) + len(counter)
    return {"id": rid(kind + statement), "kind": kind, "statement": statement, "about": about,
            "support": support, "counter": counter, "n": n, "effect": effect, "unit": unit,
            "confidence": _conf(n, len(counter), abs(effect) if effect is not None else 0),
            "evidence": evidence, "lever": lever}


def _spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def mine_library(atlas_dir: Path):
    metas = [m for m in LB.list_profiles(atlas_dir) if m.get("anchor_version") == LB.ANCHOR_VERSION]
    rules = []
    if not metas:
        return rules, metas
    name = {m["id"]: m["spec"] for m in metas}
    pats = {m["id"]: {p["id"]: p for p in LB.patterns_for(m)} for m in metas}
    ids = [m["id"] for m in metas]

    # 1. Invariants and near-invariants: a pattern present (or absent) everywhere.
    all_pids = sorted({k for v in pats.values() for k in v})
    for pid in all_pids:
        have = [i for i in ids if pid in pats[i] and pats[i][pid]["present"]]
        lack = [i for i in ids if pid in pats[i] and not pats[i][pid]["present"]]
        if not have and not lack:
            continue
        label = next(pats[i][pid]["label"] for i in ids if pid in pats[i])
        eff = float(np.mean([pats[i][pid]["value"] - (pats[i][pid]["baseline"] or 0) for i in have])) if have else 0.0
        ev = [{"model": name[i], "value": pats[i][pid]["value"], "baseline": pats[i][pid]["baseline"], "present": pats[i][pid]["present"]} for i in have + lack]
        if len(have) >= max(MIN_MODELS, 1) and len(have) >= 0.75 * (len(have) + len(lack)):
            lever = {"seed": "shared_pattern"} if pid.startswith("concept_") or pid in ("induction", "previous") else None
            rules.append(_rule("invariant" if not lack else "near-invariant",
                               f"{label} appears in {'every' if not lack else 'almost every'} trained model",
                               [name[i] for i in have], [name[i] for i in lack], eff, ev, lever,
                               about=pats[ids[0]].get(pid, {}).get("about", "")))
        elif len(lack) >= MIN_MODELS and not have:
            rules.append(_rule("absence", f"{label} is not found in any model so far", [name[i] for i in lack], [], 0.0, ev,
                               about="Either the models are too small or the probe does not capture it."))

    # 2. Depth ordering: concept A becomes readable before concept B in every model with both.
    cons = sorted({k for v in pats.values() for k in v if k.startswith("concept_")})
    for a in cons:
        for b in cons:
            if a >= b:
                continue
            before, after, ev = [], [], []
            for i in ids:
                pa, pb = pats[i].get(a), pats[i].get(b)
                if not pa or not pb or pa.get("first_depth") is None or pb.get("first_depth") is None:
                    continue
                da, db = pa["first_depth"], pb["first_depth"]
                ev.append({"model": name[i], a: da, b: db})
                if da < db:
                    before.append(name[i])
                elif db < da:
                    after.append(name[i])
            if len(before) + len(after) >= MIN_MODELS and (not before or not after):
                first, second = (a, b) if before else (b, a)
                sup = before or after
                rules.append(_rule("ordering", f"{first.replace('concept_', '')} becomes readable at a shallower depth than {second.replace('concept_', '')}",
                                   sup, [], 0.0, ev, about="The order in which models make concepts linearly readable."))

    # 3. Where circuits live.
    for key, label in (("induction", "The strongest induction head"), ("previous", "The strongest previous-token head")):
        depths, ev = [], []
        for i in ids:
            p = pats[i].get(key)
            if p and p["present"] and p.get("depth") is not None:
                depths.append(p["depth"])
                ev.append({"model": name[i], "depth": p["depth"]})
        if len(depths) >= MIN_MODELS:
            lo, hi = min(depths), max(depths)
            half = "first half" if hi < 0.5 else "second half" if lo >= 0.5 else None
            if half:
                rules.append(_rule("location", f"{label} sits in the {half} of the network",
                                   [e["model"] for e in ev], [], 0.0, ev, about=f"Relative depths {lo:.2f} to {hi:.2f}."))

    # 4. Redundancy: neighbouring layers with nearly identical representations.
    red_sup, red_ev = [], []
    for m in metas:
        real, _ = LB.kernels(atlas_dir, m["id"])
        if len(real) < 4:
            continue
        adj = [LB.cka(real[i], real[i + 1]) for i in range(1, len(real) - 1)]
        frac = float(np.mean([a >= 0.9 for a in adj]))
        red_ev.append({"model": m["spec"], "adjacent_cka_mean": float(np.mean(adj)), "share_above_0_9": frac})
        if frac >= 0.3:
            red_sup.append(m["spec"])
    if len(red_ev) >= MIN_MODELS:
        cnt = [e["model"] for e in red_ev if e["model"] not in red_sup]
        eff = float(np.mean([e["share_above_0_9"] for e in red_ev]))
        if len(red_sup) >= 0.5 * len(red_ev):
            rules.append(_rule("redundancy", "Many neighbouring layers compute nearly the same representation (CKA ≥ 0.9)",
                               red_sup, cnt, eff, red_ev, lever={"lever": "share"}, unit="share of layer pairs",
                               about="If layers repeat each other, a model could reuse one set of weights across several depths."))

    # 5. Sparsity: few neurons carry most of the activity.
    sp_sup, sp_ev = [], []
    for m in metas:
        u, ur = m.get("mlp_share"), m.get("mlp_share_random")
        if not u:
            continue
        mu = float(np.mean(u))
        mr = float(np.mean(ur)) if ur else None
        sp_ev.append({"model": m["spec"], "share_for_90pct": mu, "random": mr})
        if mu <= 0.5 and (mr is None or mu < mr - 0.05):
            sp_sup.append(m["spec"])
    if len(sp_ev) >= MIN_MODELS:
        med = float(np.median([e["share_for_90pct"] for e in sp_ev]))
        cnt = [e["model"] for e in sp_ev if e["model"] not in sp_sup]
        if len(sp_sup) > len(sp_ev) / 2:
            rules.append(_rule("sparsity", f"About {med * 100:.0f}% of MLP neurons carry 90% of the activity for a token",
                               sp_sup, cnt, 1 - med, sp_ev, lever={"lever": "topk", "k": round(max(0.1, min(0.6, med + 0.05)), 2)},
                               unit="share of neurons", about="The brain runs with a few percent of neurons active; a model that already concentrates its work may not need the rest."))

    # 6. Low effective rank of weights.
    lr_sup, lr_ev = [], []
    for m in metas:
        w, wr = m.get("weights"), m.get("weights_random")
        if not w:
            continue
        e = float(np.median([r["erank_ratio"] for r in w]))
        er = float(np.median([r["erank_ratio"] for r in wr])) if wr else None
        lr_ev.append({"model": m["spec"], "erank_ratio": e, "random": er})
        if er is not None and e < er - 0.1:
            lr_sup.append(m["spec"])
    if len(lr_ev) >= MIN_MODELS and len(lr_sup) > len(lr_ev) / 2:
        med = float(np.median([x["erank_ratio"] for x in lr_ev]))
        rules.append(_rule("low-rank", f"Trained weight matrices use about {med * 100:.0f}% of their possible rank, well below random matrices",
                           lr_sup, [x["model"] for x in lr_ev if x["model"] not in lr_sup], 1 - med, lr_ev,
                           lever={"lever": "rank", "frac": round(max(0.125, min(0.5, med / 2)), 3)}, unit="rank ratio",
                           about="If trained weights only use part of their rank, thin factored matrices may do the same job."))

    # 7. Convergence with training: agreement with the consensus above random copies.
    lib = LB.library_report(atlas_dir) if len(metas) >= 2 else None
    if lib:
        gaps, ev = [], []
        for c, m in zip(lib["consensus"], lib["models"]):
            a = [x for x in c["agreement"] if x is not None]
            r = [x for x in c["random"] if x is not None]
            if a and r:
                gap = float(np.mean(a) - np.mean(r))
                gaps.append(gap)
                ev.append({"model": m["spec"], "agreement": float(np.mean(a)), "random": float(np.mean(r))})
        if len(gaps) >= MIN_MODELS:
            pos = [e["model"] for e, gp in zip(ev, gaps) if gp > 0.02]
            neg = [e["model"] for e, gp in zip(ev, gaps) if gp <= 0.02]
            if len(pos) > len(neg):
                rules.append(_rule("convergence", "Trained models organise text more like each other than their random copies do",
                                   pos, neg, float(np.mean(gaps)), ev, lever={"seed": "shared_pattern"}, unit="CKA gap",
                                   about="Evidence for a shared, learnable structure: the raw material for seeding new models."))

        # 8. Scaling: pattern strength vs size (needs 4+ models of clearly different sizes,
        # and the pattern must actually be present somewhere, or it is noise).
        sizes_raw = [m["params"] for m in metas]
        if len(metas) >= 4 and max(sizes_raw) >= 3 * min(sizes_raw):
            sizes = np.log10(sizes_raw)
            for pid in all_pids:
                vals = [pats[m["id"]][pid]["value"] if pid in pats[m["id"]] else None for m in metas]
                if any(v is None for v in vals) or not any(pats[m["id"]][pid]["present"] for m in metas):
                    continue
                rho = _spearman(sizes, np.array(vals))
                if abs(rho) >= 0.8:
                    label = pats[metas[0]["id"]][pid]["label"]
                    rules.append(_rule("scaling", f"{label}: the measured strength {'rises' if rho > 0 else 'falls'} as models get larger",
                                       [m["spec"] for m in metas], [], rho, [{"model": m["spec"], "params": m["params"], "value": v} for m, v in zip(metas, vals)],
                                       unit="Spearman ρ"))
    return rules, metas


def mine_lab(lab_dir: Path):
    """Rules from pattern-seeded training runs in the Ground-truth lab."""
    rules = []
    by_cond = {}
    for f in lab_dir.glob("*/seeded_*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        for s in r.get("summary", []):
            if s["condition"] == "random" or s.get("step_saving") is None:
                continue
            by_cond.setdefault(s["condition"], []).append({"run": f.parent.name, "saving": s["step_saving"], "seeds": s["runs"], "label": s["label"]})
        sh = next((s for s in r.get("summary", []) if s["condition"] == "shuffled"), None)
        if sh:
            by_cond.setdefault("_shuffled", []).append({"run": f.parent.name, "reached": sh["reached"], "runs": sh["runs"]})
    for c, rows in by_cond.items():
        if c == "_shuffled":
            continue
        sav = [x["saving"] for x in rows]
        sup = [x["run"] for x in rows if x["saving"] > 0.2]
        cnt = [x["run"] for x in rows if x["saving"] <= 0.2]
        rules.append(_rule("lab", f"Starting from {rows[0]['label'].lower()} cuts training steps by about {np.median(sav) * 100:.0f}% on modular addition",
                           sup, cnt, float(np.median(sav)), rows, lever={"lever": "seed"}, unit="share of steps saved",
                           about="Measured in the Ground-truth lab against a random start with the same data and seeds."))
    sh = by_cond.get("_shuffled", [])
    if sh:
        never = [x["run"] for x in sh if x["reached"] == 0]
        rules.append(_rule("lab", "Shuffling a transplanted embedding's rows destroys its benefit: the arrangement is what helps",
                           never, [x["run"] for x in sh if x["reached"] > 0], 1.0 if never else 0.0, sh,
                           about="Control condition of the seeding experiment."))
    return rules
