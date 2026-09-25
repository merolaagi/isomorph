"""The pattern library: every profile, compared with every other.

Pattern verdicts always carry their baseline. A trained model "has" a pattern
only if it is clearly stronger than in a randomly initialised copy of the
same architecture, because the input text alone already gives even a random
network some structure.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .probes import ANCHOR_VERSION

BINS = 10
THRESH = {"induction": 0.3, "previous": 0.5, "duplicate": 0.3, "sink_share": 0.25}
CONCEPT_MIN = {"class": 0.15, "reg": 0.2}


def list_profiles(atlas_dir: Path):
    out = []
    for d in sorted(atlas_dir.glob("*/profile.json")):
        try:
            m = json.loads(d.read_text())
        except Exception:
            continue
        out.append(m)
    out.sort(key=lambda m: m.get("created", 0))
    return out


@lru_cache(maxsize=64)
def _kernels(path: str, mtime: float):
    z = np.load(path)
    real = [z[f"real_{i}"].astype(np.float64) for i in range(sum(1 for k in z.files if k.startswith("real_")))]
    rand = [z[f"rand_{i}"].astype(np.float64) for i in range(sum(1 for k in z.files if k.startswith("rand_")))]
    return real, rand


def kernels(atlas_dir: Path, pid: str):
    p = atlas_dir / pid / "kernels.npz"
    return _kernels(str(p), p.stat().st_mtime)


def cka(ka, kb):
    return float((ka * kb).sum() / (np.linalg.norm(ka) * np.linalg.norm(kb) + 1e-12))


def layer_matrix(ka_list, kb_list):
    return [[cka(a, b) for b in kb_list] for a in ka_list]


def fingerprint_similarity(ka_list, kb_list):
    """Symmetric best-match CKA over layers, skipping the raw embedding layer."""
    A, B = ka_list[1:] or ka_list, kb_list[1:] or kb_list
    m = np.array(layer_matrix(A, B))
    return float((m.max(1).mean() + m.max(0).mean()) / 2)


def _bin_layer(n_layers, b):
    """Layer index (1..L) for relative-depth bin b of BINS."""
    return max(1, min(n_layers, round((b + 0.5) / BINS * n_layers)))


def _mds(dist):
    n = len(dist)
    if n < 2:
        return [[0.0, 0.0]] * n
    d2 = np.array(dist) ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:2]
    coords = v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
    return coords.tolist()


# ------------------------------------------------------------------ pattern verdicts
def _circuit_patterns(c, cr):
    if not c:
        return []
    s = {k: np.array(v) for k, v in c["scores"].items()}
    sr = {k: np.array(v) for k, v in cr["scores"].items()} if cr else None
    L = s["induction"].shape[0]
    out = []

    def head_pattern(key, label, about):
        best = float(s[key].max())
        l, h = np.unravel_index(int(s[key].argmax()), s[key].shape)
        n = int((s[key] >= THRESH[key]).sum())
        base = float(sr[key].max()) if sr is not None else None
        present = best >= THRESH[key] and (base is None or best >= 2 * base)
        return {"id": key, "group": "circuit", "label": label, "about": about, "present": bool(present),
                "value": best, "baseline": base, "detail": f"{n} head{'s' if n != 1 else ''} above {THRESH[key]}; strongest is layer {l}, head {h}",
                "depth": (l + 0.5) / L}

    out.append(head_pattern("induction", "Induction heads", "Copy what followed a token the last time it appeared: the core of in-context learning."))
    out.append(head_pattern("previous", "Previous-token heads", "Attend to the token just before: they feed induction heads."))
    out.append(head_pattern("duplicate", "Duplicate-token heads", "Attend to earlier copies of the current token."))
    sink = s["sink"]
    share = float((sink >= 0.5).mean())
    base = float((sr["sink"] >= 0.5).mean()) if sr is not None else None
    out.append({"id": "sink", "group": "circuit", "label": "Attention sinks", "about": "Heads that park attention on the first token when they have nothing to do.",
                "present": bool(share >= THRESH["sink_share"] and (base is None or share > base + 0.1)), "value": share, "baseline": base,
                "detail": f"{share * 100:.0f}% of heads send most attention to the first token"})
    ab = c.get("ablation") or {}
    if ab.get("loss_base") is not None and ab.get("loss_ablated") is not None:
        d_ind = ab["loss_ablated"] - ab["loss_base"]
        d_rnd = (ab["loss_random_heads"] - ab["loss_base"]) if ab.get("loss_random_heads") is not None else None
        out.append({"id": "induction_necessary", "group": "circuit", "label": "Induction heads are necessary",
                    "about": "Removing the top induction heads should break copying of a repeated sequence far more than removing the same number of other heads.",
                    "present": bool(d_rnd is not None and d_ind > 0.5 and d_ind > 3 * max(d_rnd, 0.05)),
                    "value": d_ind, "baseline": d_rnd,
                    "detail": f"copying loss {ab['loss_base']:.2f} → {ab['loss_ablated']:.2f} without {len(ab['heads'])} induction heads"
                              + (f"; → {ab['loss_random_heads']:.2f} without random heads" if ab.get("loss_random_heads") is not None else "")
                              + (f". Without context the loss is {ab['loss_first_half']:.2f}." if ab.get("loss_first_half") is not None else "")})
    return out


def _concept_patterns(k, kr):
    out = []
    for name, c in (k or {}).items():
        sc = np.array(c["score"])
        ctl = np.array(c["control"])
        rnd = np.array(kr[name]["score"]) if kr and name in kr else np.zeros_like(sc)
        n = min(len(sc), len(rnd))
        sel = sc[:n] - np.maximum(ctl[:n], rnd[:n])
        best = int(sel.argmax())
        thr = CONCEPT_MIN[c["kind"]]
        first = next((i for i, v in enumerate(sel) if v >= thr), None)
        metric = "accuracy" if c["kind"] == "class" else "R²"
        out.append({"id": f"concept_{name}", "group": "concept", "label": f"Concept: {name}", "about": c["about"],
                    "present": bool(sel[best] >= thr), "value": float(sel[best]), "baseline": float(max(ctl[best], rnd[best])),
                    "depth": best / max(1, n - 1), "first_depth": (first / max(1, n - 1)) if first is not None else None,
                    "detail": f"best at layer {best}: {metric} {sc[best]:.2f} vs {max(ctl[best], rnd[best]):.2f} for the random copy or shuffled labels"})
    return out


def _weight_pattern(w, wr):
    if not w:
        return []
    a = np.median([r["alpha"] for r in w if r.get("alpha")])
    ar = np.median([r["alpha"] for r in wr if r.get("alpha")]) if wr else None
    e = np.median([r["erank_ratio"] for r in w])
    er = np.median([r["erank_ratio"] for r in wr]) if wr else None
    return [{"id": "heavy_tail", "group": "weights", "label": "Heavy-tailed weight spectra",
             "about": "Training concentrates each matrix's energy into a few strong directions. Lower tail exponent means more structure.",
             "present": bool(ar is not None and a < 0.8 * ar), "value": float(a), "baseline": float(ar) if ar is not None else None,
             "detail": f"median tail exponent {a:.2f} vs {ar:.2f} random; effective rank {e * 100:.0f}% vs {er * 100:.0f}% random of full" if ar is not None else f"median tail exponent {a:.2f}"}]


def patterns_for(meta):
    return (_circuit_patterns(meta.get("circuits"), meta.get("circuits_random"))
            + _concept_patterns(meta.get("concepts"), meta.get("concepts_random"))
            + _weight_pattern(meta.get("weights"), meta.get("weights_random")))


# ------------------------------------------------------------------ the whole library
def library_report(atlas_dir: Path):
    metas = [m for m in list_profiles(atlas_dir) if m.get("anchor_version") == ANCHOR_VERSION]
    n = len(metas)
    ker = {m["id"]: kernels(atlas_dir, m["id"]) for m in metas}
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = fingerprint_similarity(ker[metas[i]["id"]][0], ker[metas[j]["id"]][0])
    self_rand = []
    for m in metas:
        real, rand = ker[m["id"]]
        self_rand.append(fingerprint_similarity(real, rand) if rand else None)

    # Consensus by relative depth, leave-one-out, and each model's agreement with it.
    consensus = []
    for i, m in enumerate(metas):
        real, rand = ker[m["id"]]
        row, row_r = [], []
        for b in range(BINS):
            others = []
            for j, o in enumerate(metas):
                if j == i:
                    continue
                oreal = ker[o["id"]][0]
                k = oreal[_bin_layer(len(oreal) - 1, b)]
                others.append(k / (np.linalg.norm(k) + 1e-12))
            if not others:
                row.append(None)
                row_r.append(None)
                continue
            cons = np.mean(others, 0)
            row.append(cka(real[_bin_layer(len(real) - 1, b)], cons))
            row_r.append(cka(rand[_bin_layer(len(rand) - 1, b)], cons) if rand else None)
        consensus.append({"id": m["id"], "agreement": row, "random": row_r})

    pats = {m["id"]: patterns_for(m) for m in metas}
    prevalence = {}
    for pid, ps in pats.items():
        for p in ps:
            e = prevalence.setdefault(p["id"], {"id": p["id"], "label": p["label"], "group": p["group"], "about": p["about"], "have": 0, "of": 0})
            e["of"] += 1
            e["have"] += int(p["present"])
    return {
        "models": [{k: m.get(k) for k in ("id", "spec", "arch", "model_type", "layers", "d_model", "params", "created", "seconds", "has_baseline")} for m in metas],
        "similarity": sim.tolist(), "self_random": self_rand, "map": _mds((1 - sim).clip(0).tolist()),
        "consensus": consensus, "bins": BINS, "patterns": pats, "prevalence": list(prevalence.values()),
        "anchor_version": ANCHOR_VERSION,
    }


def model_report(atlas_dir: Path, pid: str):
    """One model against the rest of the library."""
    metas = {m["id"]: m for m in list_profiles(atlas_dir) if m.get("anchor_version") == ANCHOR_VERSION}
    if pid not in metas:
        raise KeyError(pid)
    me = metas[pid]
    real, rand = kernels(atlas_dir, pid)
    near = []
    for oid, o in metas.items():
        if oid == pid:
            continue
        oreal = kernels(atlas_dir, oid)[0]
        near.append({"id": oid, "spec": o["spec"], "similarity": fingerprint_similarity(real, oreal),
                     "layers": layer_matrix(real, oreal)})
    near.sort(key=lambda x: -x["similarity"])
    mine = patterns_for(me)
    lib = {}
    for oid, o in metas.items():
        if oid == pid:
            continue
        for p in patterns_for(o):
            e = lib.setdefault(p["id"], [0, 0])
            e[0] += int(p["present"])
            e[1] += 1
    for p in mine:
        have, of = lib.get(p["id"], [0, 0])
        p["library_have"], p["library_of"] = have, of
        if of == 0:
            p["verdict"] = "first of its kind in the library" if p["present"] else "absent"
        elif p["present"] and have / of >= 0.5:
            p["verdict"] = "shared: most models in the library have it"
        elif p["present"]:
            p["verdict"] = "unusual: present here but in few other models"
        elif have / of >= 0.5:
            p["verdict"] = "missing: most models in the library have it"
        else:
            p["verdict"] = "absent here and in most models"
    return {"model": {k: me.get(k) for k in ("id", "spec", "arch", "model_type", "layers", "d_model", "params", "created")},
            "self_random": fingerprint_similarity(real, rand) if rand else None,
            "self_random_layers": [cka(a, b) for a, b in zip(real, rand)] if rand else None,
            "nearest": near, "patterns": mine, "concepts": me.get("concepts"), "concepts_random": me.get("concepts_random"),
            "circuits": me.get("circuits"), "circuits_random": me.get("circuits_random")}
