"""Physics- and maths-informed laws, tested on any model's weights.

Each law is a statement that should hold (or not) whatever the architecture.
Every measure is computed on the real weight matrices and on two nulls:

  gaussian  random matrices with the same shape and the same standard deviation
            (random-matrix theory's reference point)
  shuffled  the same numbers, positions scrambled (keeps the value distribution,
            destroys all structure)

Laws come in two kinds. An emergent law is something training creates: it holds
only if the trained value clearly differs from the nulls. A preserved law is a
property random matrices already have (or a quantity a symmetry conserves); it
holds if training leaves it intact.

Matrix laws (eigenvalues of W^T W / N, N = longer side)
  heavy_tail     power-law tail of the eigenvalue spectrum (exponent alpha);
                 Martin & Mahoney's heavy-tailed self-regularisation
  spikes         eigenvalues above the Marchenko-Pastur edge sigma^2 (1 + sqrt q)^2:
                 directions that random noise cannot produce
  level_spacing  mean ratio of consecutive eigenvalue gaps in the bulk:
                 0.531 for random-matrix (GOE) level repulsion, 0.386 for
                 uncorrelated (Poisson) levels
  sv_powerlaw    singular values decay as a power law i^-gamma (scale-free)
  benford        first significant digits follow log10(1 + 1/d)
  kurtosis       excess kurtosis of the entries (0 for a Gaussian)

Model laws
  balance        rescaling symmetry: for each MLP neuron, squared norm of its
                 incoming weights tracks squared norm of its outgoing weights
                 (conserved by gradient flow in homogeneous networks)
  shift          softmax is unchanged by adding a constant to every logit, so
                 gradient descent leaves the vocabulary-mean of the output
                 vectors where it started (a Noether-style conservation law)
  qk_symmetry    share of each attention head's bilinear form W_Q W_K^T that is
                 symmetric (0.5 for random)
  equipartition  how evenly squared weight norm is spread across layers (Gini)
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from .hub.weights import _kind, _layer_of

MAX_SIDE = 1024
BENFORD = np.log10(1 + 1 / np.arange(1, 10))
GOE_R, POISSON_R = 0.5307, 0.3863

LAWS = {
    "heavy_tail": {"kind": "emergent", "label": "Heavy-tailed eigenvalue spectrum", "field": "Random-matrix theory",
                   "formula": "ρ(λ) ∝ λ^(−α) in the tail of the spectrum of WᵀW",
                   "expect": "Well-trained layers show α between 2 and 6; random matrices have no power-law tail (α very large)."},
    "spikes": {"kind": "emergent", "label": "Eigenvalues beyond the Marchenko–Pastur edge", "field": "Random-matrix theory",
               "formula": "λ > σ²(1 + √q)², q = short side / long side",
               "expect": "Pure noise stays inside the edge. Each eigenvalue outside it is a direction the matrix learned."},
    "level_spacing": {"kind": "preserved", "label": "Level repulsion in the spectral bulk", "field": "Quantum chaos / random-matrix theory",
                      "formula": "⟨r⟩ = ⟨min(sᵢ, sᵢ₊₁) / max(sᵢ, sᵢ₊₁)⟩, sᵢ = λᵢ₊₁ − λᵢ",
                      "expect": "0.531 means eigenvalues repel like a random (GOE) system; 0.386 means they are independent (Poisson)."},
    "sv_powerlaw": {"kind": "emergent", "label": "Scale-free singular values", "field": "Scaling laws",
                    "formula": "σᵢ ∝ i^(−γ), fitted on log–log axes",
                    "expect": "A straight log–log line (high R²) means no preferred scale; γ says how fast importance falls off."},
    "benford": {"kind": "preserved", "label": "Benford's law of first digits", "field": "Statistics",
                "formula": "P(d) = log₁₀(1 + 1/d), d = 1…9",
                "expect": "Mean absolute deviation below 0.012 is conformity (Nigrini's thresholds)."},
    "kurtosis": {"kind": "emergent", "label": "Heavy-tailed weight values", "field": "Statistics",
                 "formula": "excess kurtosis = E[(w − μ)⁴]/σ⁴ − 3",
                 "expect": "0 for Gaussian values; large positive values mean a few weights are much bigger than the rest."},
    "balance": {"kind": "emergent", "label": "Neuron balance (rescaling symmetry)", "field": "Symmetry and conservation (Noether)",
                "formula": "corr( log‖w_in,j‖², log‖w_out,j‖² ) across MLP neurons j",
                "expect": "Gradient flow conserves ‖w_in‖² − ‖w_out‖² per neuron in homogeneous networks, so trained neurons stay balanced."},
    "shift": {"kind": "preserved", "label": "Softmax-shift conservation", "field": "Symmetry and conservation (Noether)",
              "formula": "‖mean over vocabulary of output vectors‖ / mean ‖output vector‖",
              "expect": "Adding a constant to every logit changes nothing, so the vocabulary-mean of the output layer never receives gradient: it stays at its random starting size (about 1/√V)."},
    "qk_symmetry": {"kind": "emergent", "label": "Symmetry of attention's bilinear form", "field": "Linear algebra",
                    "formula": "‖½(B + Bᵀ)‖² / ‖B‖², B = W_Q W_Kᵀ per head",
                    "expect": "0.5 for random. Above 0.5 means heads mostly match tokens to similar tokens; below means directed relations."},
    "equipartition": {"kind": "preserved", "label": "Equipartition of weight energy across layers", "field": "Statistical physics",
                      "formula": "Gini coefficient of ‖W‖² summed per layer",
                      "expect": "0 means every layer carries the same energy; near 1 means one layer holds almost all of it."},
}


# ------------------------------------------------------------------ loading
def load_tensors(spec, lab_dir: Path):
    """Named weight tensors for any model reference, including lab:<run>:<seed>."""
    if spec.startswith("lab:"):
        _, run, seed = spec.split(":")
        blob = torch.load(lab_dir / run / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        out = {}
        for k, v in blob["state"].items():
            v = v.float()
            if v.ndim == 3 and k in ("W_Q", "W_K", "W_V"):
                v = v.permute(1, 0, 2).reshape(v.shape[1], -1)  # (heads, d, dh) -> (d, heads*dh)
            elif v.ndim == 3 and k == "W_O":
                v = v.reshape(-1, v.shape[-1])  # (heads, dh, d) -> (heads*dh, d)
            out[k] = v
        return out, {"family": "lab", "heads": blob["config"]["n_heads"]}
    from .hub.load import fetch_weights

    t = fetch_weights(spec)
    return t, {"family": "micro" if spec.startswith("microopus:") else "hf"}


# ------------------------------------------------------------------ matrix measures
def _sub(w, rng):
    if w.shape[0] > MAX_SIDE:
        w = w[rng.choice(w.shape[0], MAX_SIDE, replace=False)]
    if w.shape[1] > MAX_SIDE:
        w = w[:, rng.choice(w.shape[1], MAX_SIDE, replace=False)]
    return w


def _eigs(w):
    n, m = w.shape
    if n < m:
        w = w.T
        n, m = m, n
    return np.clip(np.linalg.eigvalsh(w.T @ w / n), 0, None)[::-1], m / n


def matrix_measures(w, with_hist=False):
    w = np.asarray(w, dtype=np.float64)
    sigma2 = float(w.var()) or 1e-12
    ev, q = _eigs(w)
    m = len(ev)
    out = {}
    k = max(5, int(0.1 * m))
    tail = ev[:k]
    out["alpha"] = float(1 + k / (np.log(tail / max(tail[-1], 1e-30)).sum() + 1e-12))
    edge = sigma2 * (1 + math.sqrt(q)) ** 2
    out["spike_frac"] = float((ev > edge * 1.05).mean())
    out["spikes"] = int((ev > edge * 1.05).sum())
    bulk = np.sort(ev[ev <= edge])
    lo, hi = int(0.1 * len(bulk)), int(0.9 * len(bulk))
    b = bulk[lo:hi]
    if len(b) >= 12:
        s = np.diff(b)
        s = s[s > 0]
        r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
        out["r"] = float(r.mean()) if len(r) else None
    else:
        out["r"] = None
    sv = np.sqrt(ev[ev > 1e-12 * ev[0]]) if ev[0] > 0 else ev
    if len(sv) >= 8:
        i = np.arange(1, len(sv) + 1)
        sel = (i >= 2) & (i <= max(4, len(sv) // 2))
        x, y = np.log(i[sel]), np.log(sv[sel])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss = float(((y - y.mean()) ** 2).sum()) or 1e-12
        out["gamma"] = float(-coef[0])
        out["gamma_r2"] = float(1 - ((y - pred) ** 2).sum() / ss)
    a = np.abs(w[w != 0]).ravel()
    if len(a):
        d = np.floor(a / 10 ** np.floor(np.log10(a))).astype(int)
        d = d[(d >= 1) & (d <= 9)]
        f = np.bincount(d, minlength=10)[1:10] / max(1, len(d))
        out["benford_mad"] = float(np.abs(f - BENFORD).mean())
        out["benford_freq"] = f.round(4).tolist()
    z = (w - w.mean()) / (w.std() + 1e-12)
    out["kurtosis"] = float((z ** 4).mean() - 3)
    if with_hist:
        top = float(np.quantile(ev, 0.999))
        h, e = np.histogram(ev, bins=40, range=(0, top))
        lam_m, lam_p = sigma2 * (1 - math.sqrt(q)) ** 2, edge
        xs = (e[:-1] + e[1:]) / 2
        mp = np.where((xs > lam_m) & (xs < lam_p), np.sqrt(np.clip((lam_p - xs) * (xs - lam_m), 0, None)) / (2 * math.pi * sigma2 * q * xs + 1e-30), 0)
        out["esd"] = {"edges": e.round(6).tolist(), "counts": h.tolist(), "mp": (mp * m * (e[1] - e[0])).round(3).tolist(),
                      "edge": edge, "max": float(ev[0]), "q": q}
    return out


# ------------------------------------------------------------------ model-level laws
def _neuron_axis(shape, f):
    return 0 if shape[0] == f and shape[1] != f else 1 if shape[1] == f else None


MLP_PAIRS = [("dense_h_to_4h", "dense_4h_to_h"), ("c_fc", "mlp.c_proj"), ("up_proj", "down_proj"), ("fc1", "fc2"),
             ("W_up", "W_down"), ("W_in", "W_out"), ("fc_in", "fc_out")]


def balance(T):
    rows = []
    for up_key, down_key in MLP_PAIRS:
        ups = {(_layer_of(k) if _layer_of(k) is not None else -1): k for k in T if k.endswith(up_key + ".weight") or k.endswith(up_key) and T[k].ndim == 2}
        for l, uk in ups.items():
            dk = next((k for k in T if (k.endswith(down_key + ".weight") or k.endswith(down_key)) and T[k].ndim == 2 and
                       (_layer_of(k) if _layer_of(k) is not None else -1) == l), None)
            if not dk:
                continue
            U, D = T[uk].double().numpy(), T[dk].double().numpy()
            common = set(U.shape) & set(D.shape)
            f = max(common, key=lambda s: s) if common else None
            if f is None:
                continue
            au, ad = _neuron_axis(U.shape, f), _neuron_axis(D.shape, f)
            if au is None or ad is None:
                continue
            a = (U ** 2).sum(1 - au)
            b = (D ** 2).sum(1 - ad)
            keep = (a > 0) & (b > 0)
            if keep.sum() < 8:
                continue
            la, lb = np.log(a[keep]), np.log(b[keep])
            corr = float(np.corrcoef(la, lb)[0, 1])
            rng = np.random.default_rng(0)
            null = float(np.corrcoef(la, rng.permutation(lb))[0, 1])
            rows.append({"layer": l, "up": uk, "down": dk, "corr": corr, "null": null, "log_ratio_sd": float(np.std(la - lb))})
        if rows:
            break
    return rows


def shift(T):
    cand = [k for k in T if T[k].ndim == 2 and _kind(k) == "unembedding"] or \
           [k for k in T if T[k].ndim == 2 and _kind(k) == "embedding" and max(T[k].shape) >= 20 and "pos" not in k.lower() and not k.endswith("W_P") and "wpe" not in k]
    if not cand:
        return None
    k = cand[0]
    W = T[k].double().numpy()
    V = max(W.shape)
    vec = W if W.shape[0] == V else W.T  # one row per vocabulary entry
    mean = np.linalg.norm(vec.mean(0))
    typical = float(np.linalg.norm(vec, axis=1).mean())
    rel = float(mean / (typical + 1e-12))
    rng = np.random.default_rng(0)
    g = rng.normal(0, vec.std(), vec.shape)
    null = float(np.linalg.norm(g.mean(0)) / np.linalg.norm(g, axis=1).mean())
    return {"tensor": k, "tied_or_input": _kind(k) == "embedding", "vocab": V, "rel_mean": rel, "null": null, "ratio": rel / (null + 1e-12)}


def qk_symmetry(T, heads_hint=None):
    out = []
    keys = list(T)
    # separate Q and K matrices
    for qk_pat in (("q_proj", "k_proj"), ("W_Q", "W_K"), ("query", "key")):
        qs = [k for k in keys if k.endswith(qk_pat[0] + ".weight") or k.endswith(qk_pat[0])]
        for qkey in qs:
            kkey = qkey.replace(qk_pat[0], qk_pat[1])
            if kkey not in T:
                continue
            Q, K = T[qkey].double().numpy(), T[kkey].double().numpy()
            if Q.shape != K.shape:
                continue
            # make rows = input dimension
            if qkey.endswith(".weight") and not qkey.startswith("L"):
                Q, K = Q.T, K.T  # torch Linear stores (out, in)
            out.append(_sym(Q, K, heads_hint, qkey))
        if out:
            return out
    # fused: GPT-NeoX query_key_value (3d, d) interleaved per head; GPT-2 c_attn (d, 3d)
    for k in keys:
        W = T[k].double().numpy()
        if k.endswith("query_key_value.weight") and W.ndim == 2 and heads_hint:
            d = W.shape[1]
            h = heads_hint
            dh = d // h
            Wr = W.reshape(h, 3, dh, d)
            Q = Wr[:, 0].reshape(d, d).T
            K = Wr[:, 1].reshape(d, d).T
            out.append(_sym(Q, K, h, k))
        elif k.endswith("attn.c_attn.weight") and W.ndim == 2 and W.shape[1] == 3 * W.shape[0]:
            d = W.shape[0]
            out.append(_sym(W[:, :d], W[:, d:2 * d], heads_hint, k))
    return out


def _sym(Q, K, heads, name):
    d = Q.shape[0]
    h = heads or 1
    dh = Q.shape[1] // h
    vals = []
    for i in range(h):
        B = Q[:, i * dh:(i + 1) * dh] @ K[:, i * dh:(i + 1) * dh].T
        S = (B + B.T) / 2
        vals.append(float((S ** 2).sum() / ((B ** 2).sum() + 1e-30)))
    return {"tensor": name, "heads": h, "sym": float(np.mean(vals)), "per_head": [round(v, 3) for v in vals]}


def equipartition(T):
    by = {}
    for k, v in T.items():
        l = _layer_of(k)
        if l is not None and v.ndim == 2 and _kind(k) != "buffer":
            by[l] = by.get(l, 0.0) + float((v.double() ** 2).sum())
    if len(by) < 2:
        return None
    x = np.sort(np.array(list(by.values())))
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    return {"gini": gini, "layers": [{"layer": k, "energy": v} for k, v in sorted(by.items())]}


# ------------------------------------------------------------------ the battery
def _heads_from(spec, meta):
    if meta.get("heads"):
        return meta["heads"]
    if spec.startswith("microopus:"):
        from . import microopus

        return microopus.weights_blob(spec)["config"]["H"]
    try:
        from .hub.load import parse_spec
        from huggingface_hub import hf_hub_download
        import os

        if os.path.isdir(spec):
            cfg = json.loads((Path(spec) / "config.json").read_text())
        else:
            repo, sub, rev = parse_spec(spec)
            cfg = json.loads(Path(hf_hub_download(repo, (sub + "/" if sub else "") + "config.json", revision=rev)).read_text())
        return cfg.get("num_attention_heads") or cfg.get("n_head")
    except Exception:
        return None


def run_battery(spec, resolved, lab_dir: Path, progress=lambda f, m: None):
    progress(0.02, f"Loading weights of {spec}")
    T, meta = load_tensors(resolved, lab_dir)
    T = {k: v for k, v in T.items() if isinstance(v, torch.Tensor) and v.is_floating_point() and _kind(k) != "buffer"}
    heads = _heads_from(resolved, meta)
    rng = np.random.default_rng(0)
    mats = [(k, v) for k, v in T.items() if v.ndim == 2 and min(v.shape) >= 16]
    rows = []
    hist_for = set()
    by_kind = {}
    for k, _ in mats:
        by_kind.setdefault(_kind(k), []).append(k)
    for kind, ks in by_kind.items():
        hist_for.add(max(ks, key=lambda x: T[x].numel()))
    for i, (k, v) in enumerate(mats):
        progress(0.05 + 0.8 * i / max(1, len(mats)), f"Spectra of {k}")
        w = _sub(v.double().numpy(), rng)
        real = matrix_measures(w, with_hist=k in hist_for)
        gauss = matrix_measures(rng.normal(0, w.std() or 1e-6, w.shape))
        shuf = matrix_measures(rng.permutation(w.ravel()).reshape(w.shape))
        rows.append({"name": k, "kind": _kind(k), "layer": _layer_of(k), "shape": list(v.shape), "sampled": list(w.shape),
                     "real": real, "gaussian": gauss, "shuffled": shuf})
    progress(0.9, "Symmetry and conservation laws")
    model = {"balance": balance(T), "shift": shift(T), "qk": qk_symmetry(T, heads), "equipartition": equipartition(T)}
    progress(1.0, "Done")
    return {"spec": spec, "resolved": str(resolved), "created": time.time(), "params": int(sum(v.numel() for v in T.values())),
            "matrices": rows, "model": model, "heads": heads, "family": meta.get("family")}


# ------------------------------------------------------------------ verdicts
def _med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else None


def verdicts(b):
    M = b["matrices"]
    g = lambda which, key: _med([m[which].get(key) for m in M])
    out = {}
    a, ag, ash = g("real", "alpha"), g("gaussian", "alpha"), g("shuffled", "alpha")
    out["heavy_tail"] = {"value": a, "null": ag, "shuffled": ash, "holds": a is not None and 2 <= a <= 6 and (ag is None or a < 0.7 * ag),
                         "unit": "median α"}
    s, sg, ss = g("real", "spike_frac"), g("gaussian", "spike_frac"), g("shuffled", "spike_frac")
    out["spikes"] = {"value": s, "null": sg, "shuffled": ss, "holds": s is not None and s > max(sg or 0, ss or 0) + 0.005,
                     "unit": "median share of eigenvalues past the edge"}
    r, rg, rs = g("real", "r"), g("gaussian", "r"), g("shuffled", "r")
    out["level_spacing"] = {"value": r, "null": rg, "shuffled": rs, "holds": r is not None and abs(r - GOE_R) < 0.04,
                            "unit": "median ⟨r⟩ (GOE 0.531, Poisson 0.386)"}
    gm, gr2 = g("real", "gamma"), g("real", "gamma_r2")
    gmg = g("gaussian", "gamma")
    out["sv_powerlaw"] = {"value": gm, "null": gmg, "shuffled": g("shuffled", "gamma"), "r2": gr2,
                          "holds": gr2 is not None and gr2 > 0.9 and gm is not None and (gmg is None or gm > 1.5 * gmg), "unit": "median γ"}
    bm, bg = g("real", "benford_mad"), g("gaussian", "benford_mad")
    out["benford"] = {"value": bm, "null": bg, "shuffled": g("shuffled", "benford_mad"), "holds": bm is not None and bm < 0.012,
                      "unit": "median MAD from Benford"}
    k, kg = g("real", "kurtosis"), g("gaussian", "kurtosis")
    out["kurtosis"] = {"value": k, "null": kg, "shuffled": g("shuffled", "kurtosis"), "holds": k is not None and k > 0.5 and (kg is None or k > kg + 0.5),
                       "unit": "median excess kurtosis"}
    bal = b["model"]["balance"]
    if bal:
        c, cn = _med([x["corr"] for x in bal]), _med([x["null"] for x in bal])
        out["balance"] = {"value": c, "null": cn, "holds": c is not None and c > 0.5 and c > cn + 0.3, "unit": "median correlation across layers"}
    sh = b["model"]["shift"]
    if sh:
        out["shift"] = {"value": sh["rel_mean"], "null": sh["null"], "holds": sh["ratio"] < 3,
                        "unit": f"relative mean of {'input/tied embedding' if sh['tied_or_input'] else 'output'} vectors", "ratio": sh["ratio"],
                        "note": "Only the output layer is protected by this symmetry; a tied or input embedding also receives other gradients." if sh["tied_or_input"] else ""}
    qk = b["model"]["qk"]
    if qk:
        v = _med([x["sym"] for x in qk])
        out["qk_symmetry"] = {"value": v, "null": 0.5, "holds": v is not None and abs(v - 0.5) > 0.05, "unit": "median symmetric share",
                              "direction": "symmetric" if v and v > 0.5 else "directed"}
    eq = b["model"]["equipartition"]
    if eq:
        out["equipartition"] = {"value": eq["gini"], "null": 0.0, "holds": eq["gini"] < 0.2, "unit": "Gini of layer energy"}
    return out


def summary(laws_dir: Path):
    res = []
    for f in sorted(laws_dir.glob("*.json")):
        try:
            b = json.loads(f.read_text())
        except Exception:
            continue
        res.append({"id": f.stem, "spec": b["spec"], "params": b["params"], "created": b["created"], "family": b.get("family"),
                    "verdicts": verdicts(b)})
    uni = {}
    for law in LAWS:
        rows = [r for r in res if law in r["verdicts"]]
        uni[law] = {"have": sum(1 for r in rows if r["verdicts"][law]["holds"]), "of": len(rows)}
    return {"laws": LAWS, "models": res, "universality": uni}


def law_id(spec):
    return re.sub(r"[^A-Za-z0-9]+", "-", spec).strip("-")[-70:]
