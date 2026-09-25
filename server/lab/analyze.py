"""Analyses for a family of tiny modular-addition transformers.

The lab is the ground truth: we know what a correct solution looks like
(Fourier "clock" features), so every comparison tool can be checked here
before it is trusted on real language models.
"""
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import metrics as M
from .model import load_model, make_dataset

SITES = ["embed_ab", "attn_out", "resid_mid", "mlp_post", "resid_post", "logits"]
SITE_LABELS = {
    "embed_ab": "Token embeddings",
    "attn_out": "Attention output",
    "resid_mid": "Residual after attention",
    "mlp_post": "MLP neurons",
    "resid_post": "Residual after MLP",
    "logits": "Output logits",
}


# ---------------------------------------------------------------- loading
class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.cfg = json.loads((run_dir / "config.json").read_text())
        hp = run_dir / "history.json"
        self.history = json.loads(hp.read_text()) if hp.exists() else {}
        self.models = {}
        for f in sorted(run_dir.glob("seed_*.pt"), key=lambda f: int(f.stem.split("_")[1])):
            self.models[int(f.stem.split("_")[1])] = load_model(f)
        self.p = int(self.cfg["p"])
        self.x, self.y, self.tr, self.te = make_dataset(self.p, float(self.cfg["train_frac"]), int(self.cfg.get("data_seed", 0)))
        self._cache = {}

    def acts(self, seed):
        if seed not in self._cache:
            c = {}
            with torch.no_grad():
                self.models[seed](self.x, c)
            c["embed_ab"] = c["resid_pre"][:, :2].reshape(len(self.x), -1)
            self._cache[seed] = c
        return self._cache[seed]


def evaluate(model, x, y, idx=None):
    with torch.no_grad():
        lg = model(x)
    if idx is not None:
        lg, y = lg[idx], y[idx]
    return float(F.cross_entropy(lg, y)), float((lg.argmax(-1) == y).float().mean())


# ---------------------------------------------------------------- Fourier
def fourier_basis(p):
    x = torch.arange(p, dtype=torch.float64)
    rows, names = [torch.ones(p, dtype=torch.float64)], ["const"]
    for k in range(1, p // 2 + 1):
        rows.append(torch.cos(2 * math.pi * k * x / p))
        rows.append(torch.sin(2 * math.pi * k * x / p))
        names += [f"cos{k}", f"sin{k}"]
    B = torch.stack(rows)
    return B / B.norm(dim=1, keepdim=True), names


def freq_energy(W, p):
    """Fraction of embedding energy per frequency k = 1..p//2 (constant excluded)."""
    B, _ = fourier_basis(p)
    c = (B @ W[:p].double()).pow(2).sum(1)
    e = torch.stack([c[2 * k - 1] + c[2 * k] for k in range(1, p // 2 + 1)])
    return (e / e.sum()).tolist()


def key_freqs(energy, cover=0.8, cap=10):
    order = sorted(range(len(energy)), key=lambda i: -energy[i])
    out, tot = [], 0.0
    for i in order:
        out.append(i + 1)
        tot += energy[i]
        if tot >= cover or len(out) >= cap:
            break
    return sorted(out)


def neuron_freqs(post, p):
    """Dominant input frequency for each MLP neuron (0 = dead or flat)."""
    a = post.reshape(p, p, -1).double()
    a = a - a.mean((0, 1), keepdim=True)
    fa = torch.fft.rfft(a, dim=0).abs().pow(2).sum(1)  # (p//2+1, n)
    fb = torch.fft.rfft(a, dim=1).abs().pow(2).sum(0)
    pw = (fa + fb)[1:]
    out = pw.argmax(0) + 1
    out[a.abs().amax((0, 1)) < 1e-6] = 0
    return out


def compress_to_freqs(model, freqs, p):
    """Project the number embeddings onto the key-frequency subspace only."""
    B, names = fourier_basis(p)
    keep = [0] + [i for i, n in enumerate(names) if n != "const" and int(n[3:]) in freqs]
    Bk = B[keep].float()
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.W_E[:p] = Bk.T @ (Bk @ m.W_E[:p])
    return m, len(keep)


# ---------------------------------------------------------------- alignment
def align(run: Run, sa: int, sb: int):
    """Express model B in model A's coordinates using exact symmetries only:
    one orthogonal rotation of the residual stream, a permutation of heads, an
    orthogonal change of basis inside each head, and a permutation of MLP
    neurons. B's outputs must not change; we check that."""
    A, B = run.models[sa], run.models[sb]
    ca, cb = run.acts(sa), run.acts(sb)
    d, h = A.d, A.h

    def stack(c):
        return torch.cat([c["resid_pre"].reshape(-1, d), c["resid_mid"], c["resid_post"]], 0)

    R = M.procrustes(stack(cb), stack(ca))
    Bp = copy.deepcopy(B)
    with torch.no_grad():
        Bp.W_E.copy_(B.W_E @ R)
        Bp.W_pos.copy_(B.W_pos @ R)
        Bp.W_Q.copy_(torch.einsum("de,hef->hdf", R.T, B.W_Q))
        Bp.W_K.copy_(torch.einsum("de,hef->hdf", R.T, B.W_K))
        Bp.W_V.copy_(torch.einsum("de,hef->hdf", R.T, B.W_V))
        Bp.W_O.copy_(B.W_O @ R)
        Bp.W_in.copy_(R.T @ B.W_in)
        Bp.W_out.copy_(B.W_out @ R)
        Bp.b_out.copy_(B.b_out @ R)
        Bp.W_U.copy_(R.T @ B.W_U)

        # Heads: compare basis-free circuits QK = W_Q W_K^T and OV = W_V W_O.
        def circuits(m):
            qk = [m.W_Q[i] @ m.W_K[i].T for i in range(h)]
            ov = [m.W_V[i] @ m.W_O[i] for i in range(h)]
            return qk, ov

        qa, oa = circuits(A)
        qb, ob = circuits(Bp)
        score = torch.tensor([[M.cosine(qa[i], qb[j]) + M.cosine(oa[i], ob[j]) for j in range(h)] for i in range(h)])
        hperm, _ = M.match_units(score)
        for name in ("W_Q", "W_K", "W_V", "W_O"):
            getattr(Bp, name).copy_(getattr(Bp, name)[hperm])
        heads = []
        for i in range(h):
            N = M.procrustes(torch.cat([Bp.W_Q[i], Bp.W_K[i]]), torch.cat([A.W_Q[i], A.W_K[i]]))
            Bp.W_Q[i] = Bp.W_Q[i] @ N
            Bp.W_K[i] = Bp.W_K[i] @ N
            Mv = M.procrustes(torch.cat([Bp.W_V[i], Bp.W_O[i].T]), torch.cat([A.W_V[i], A.W_O[i].T]))
            Bp.W_V[i] = Bp.W_V[i] @ Mv
            Bp.W_O[i] = Mv.T @ Bp.W_O[i]
            heads.append({
                "a": i, "b": int(hperm[i]),
                "qk": round(M.cosine(qa[i], qb[int(hperm[i])]), 3),
                "ov": round(M.cosine(oa[i], ob[int(hperm[i])]), 3),
            })

        # MLP neurons: match by activation correlation over every input.
        corr = M.correlation_matrix(ca["mlp_post"], cb["mlp_post"])
        nperm, matched = M.match_units(corr)
        Bp.W_in.copy_(Bp.W_in[:, nperm])
        Bp.b_in.copy_(Bp.b_in[nperm])
        Bp.W_out.copy_(Bp.W_out[nperm])

        # Invariance check: the re-expressed B must compute exactly what B computes.
        drift = float((Bp(run.x) - B(run.x)).abs().max())
    return Bp, {"R": R, "heads": heads, "neuron_perm": nperm, "neuron_corr": matched, "corr": corr, "drift": drift}


def interpolate(A, B, x, y, n=11):
    out = []
    sa, sb = A.state_dict(), B.state_dict()
    m = copy.deepcopy(A)
    for i in range(n):
        t = i / (n - 1)
        m.load_state_dict({k: (1 - t) * sa[k] + t * sb[k] for k in sa})
        loss, acc = evaluate(m, x, y)
        out.append({"t": round(t, 3), "loss": round(loss, 4), "acc": round(acc, 4)})
    ends = (out[0]["loss"], out[-1]["loss"])
    barrier = max(o["loss"] - ((1 - o["t"]) * ends[0] + o["t"] * ends[1]) for o in out)
    return out, round(barrier, 4)


# ---------------------------------------------------------------- task symmetry
def relabel_index(p, u):
    """Row order that feeds B the inputs (u*a, u*b) when A sees (a, b)."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return ((u * a) % p) * p + (u * b) % p


def circuit_match(run: Run, sa: int, sb: int, key_a, key_b):
    """Modular addition has its own symmetry: multiplying every number by a unit u
    (mod p) maps a correct algorithm to another correct algorithm, and moves a
    clock circuit at frequency k to frequency k*u. So two models that use
    different frequencies may still be built from the same parts.

    For each pair of key frequencies (kA, kB) we relabel B's inputs by the u that
    sends kB to kA, then match A's kA-neurons with B's kB-neurons."""
    p = run.p
    pa, pb = run.acts(sa)["mlp_post"], run.acts(sb)["mlp_post"]
    fa, fb = neuron_freqs(pa, p), neuron_freqs(pb, p)
    cells = []
    mat = torch.zeros(len(key_a), len(key_b))
    for i, ka in enumerate(key_a):
        na = fa == ka
        for j, kb in enumerate(key_b):
            nb = fb == kb
            if na.sum() < 2 or nb.sum() < 2:
                continue
            raw_c = M.correlation_matrix(pa[:, na], pb[:, nb])
            _, raw_m = M.match_units(raw_c)
            best = (-1.0, None)
            for sign in (1, -1):
                u = (sign * ka * pow(int(kb), -1, p)) % p
                c = M.correlation_matrix(pa[:, na], pb[relabel_index(p, u)][:, nb])
                _, m = M.match_units(c)
                if float(m.mean()) > best[0]:
                    best = (float(m.mean()), u)
            # Null: the same procedure with relabellings that should not line up.
            g = torch.Generator().manual_seed(int(ka) * 1000 + int(kb))
            wrong = [u for u in torch.randperm(p - 1, generator=g).add(1).tolist()
                     if u not in ((ka * pow(int(kb), -1, p)) % p, (-ka * pow(int(kb), -1, p)) % p)][:3]
            null = []
            for u in wrong:
                _, m = M.match_units(M.correlation_matrix(pa[:, na], pb[relabel_index(p, u)][:, nb]))
                null.append(float(m.mean()))
            mat[i, j] = best[0]
            cells.append({"ka": int(ka), "kb": int(kb), "u": int(best[1]), "n_a": int(na.sum()), "n_b": int(nb.sum()),
                          "relabeled": round(best[0], 4), "raw": round(float(raw_m.mean()), 4),
                          "null": round(sum(null) / len(null), 4)})
    pairs = []
    if cells:
        perm, _ = M.match_units(mat) if len(key_a) <= len(key_b) else (None, None)
        if perm is None:
            pt, _ = M.match_units(mat.T)
            idx = [(int(pt[j]), j) for j in range(len(key_b))]
        else:
            idx = [(i, int(perm[i])) for i in range(len(key_a))]
        lookup = {(c["ka"], c["kb"]): c for c in cells}
        for i, j in idx:
            c = lookup.get((int(key_a[i]), int(key_b[j])))
            if c:
                pairs.append(c)
    # Best single relabelling for the whole model (one u for every circuit).
    glob = []
    for u in range(1, p):
        glob.append((M.linear_cka(pa, pb[relabel_index(p, u)]), u))
    glob.sort(reverse=True)
    return {"pairs": pairs, "cells": cells, "key_a": list(key_a), "key_b": list(key_b),
            "global_best_u": glob[0][1], "global_best_cka": round(glob[0][0], 4),
            "global_identity_cka": round(M.linear_cka(pa, pb), 4)}


# ---------------------------------------------------------------- reports
def family_report(run: Run):
    seeds = list(run.models)
    per = []
    energies = {}
    for s in seeds:
        m = run.models[s]
        _, tr_acc = evaluate(m, run.x, run.y, run.tr)
        te_loss, te_acc = evaluate(m, run.x, run.y, run.te)
        e = freq_energy(m.W_E.detach(), run.p)
        energies[s] = e
        kf = key_freqs(e)
        cm, dims = compress_to_freqs(m, kf, run.p)
        _, c_acc = evaluate(cm, run.x, run.y, run.te)
        nf = neuron_freqs(run.acts(s)["mlp_post"], run.p)
        live = nf[nf > 0]
        in_key = float(sum(int(f) in kf for f in live.tolist()) / max(1, len(live)))
        per.append({
            "seed": s, "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4), "test_loss": round(te_loss, 4),
            "key_freqs": kf, "energy": [round(v, 5) for v in e],
            "compressed_acc": round(c_acc, 4), "compressed_dims": dims, "full_dims": run.p,
            "live_neurons": int(len(live)), "neurons_on_key_freqs": round(in_key, 3),
        })
    n = len(seeds)
    kfs = {r["seed"]: set(r["key_freqs"]) for r in per}
    jac = [[round(len(kfs[a] & kfs[b]) / max(1, len(kfs[a] | kfs[b])), 3) for b in seeds] for a in seeds]
    cka = {}
    for site in ("resid_post", "mlp_post", "logits"):
        cka[site] = [[round(M.linear_cka(run.acts(a)[site], run.acts(b)[site]), 3) if a != b else 1.0 for b in seeds] for a in seeds]
    common = set.intersection(*kfs.values()) if kfs else set()
    union = set.union(*kfs.values()) if kfs else set()
    return {
        "config": run.cfg, "seeds": seeds, "models": per, "history": run.history,
        "key_freq_jaccard": jac, "cka": cka,
        "shared_freqs": sorted(common), "all_freqs": sorted(union),
        "n_models": n,
    }


def pair_report(run: Run, sa: int, sb: int):
    A, B = run.models[sa], run.models[sb]
    Bp, info = align(run, sa, sb)

    tensors = []
    for name, _ in A.named_parameters():
        tensors.append({
            "name": name, "shape": list(getattr(A, name).shape),
            "raw": round(M.cosine(getattr(A, name), getattr(B, name)), 4),
            "aligned": round(M.cosine(getattr(A, name), getattr(Bp, name)), 4),
        })
    flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])
    overall = {"raw": round(M.cosine(flat(A), flat(B)), 4), "aligned": round(M.cosine(flat(A), flat(Bp)), 4)}

    # Neuron matching: compare against a random pairing baseline.
    matched = info["neuron_corr"]
    rand = info["corr"][torch.arange(A.dmlp), torch.randperm(A.dmlp)]
    hist_edges = [i / 10 for i in range(-10, 11)]

    def hist(v):
        c = torch.histc(v.clamp(-1, 1), bins=20, min=-1, max=1)
        return [int(i) for i in c.tolist()]

    ca, cb = run.acts(sa), run.acts(sb)
    fa = neuron_freqs(ca["mlp_post"], run.p)
    fb = neuron_freqs(cb["mlp_post"], run.p)[info["neuron_perm"]]
    live = (fa > 0) & (fb > 0)
    strong = live & (matched > 0.8)
    freq_agree = float((fa[strong] == fb[strong]).float().mean()) if strong.any() else 0.0

    naive, naive_barrier = interpolate(A, B, run.x, run.y)
    aligned, aligned_barrier = interpolate(A, Bp, run.x, run.y)

    cka = [[round(M.linear_cka(ca[s1], cb[s2]), 3) for s2 in SITES] for s1 in SITES]
    knn = {s: round(M.mutual_knn(ca[s][::3], cb[s][::3], k=10), 3) for s in ("mlp_post", "resid_post", "logits")}

    # Stitching: map A's state into B's space at a site, let B finish the job.
    stitch = []
    tr, te = run.tr, run.te
    _, acc_a = evaluate(A, run.x, run.y, te)
    _, acc_b = evaluate(B, run.x, run.y, te)
    for site in ("resid_pre", "resid_mid", "resid_post"):
        sa_act, sb_act = ca[site], cb[site]
        row = {"site": site}
        with torch.no_grad():
            if site == "resid_pre":
                src_tr, dst_tr = sa_act[tr].reshape(-1, A.d), sb_act[tr].reshape(-1, B.d)
                shape = lambda z, idx: z.reshape(len(idx), A.n_ctx, B.d)
                src_te = sa_act[te].reshape(-1, A.d)
            else:
                src_tr, dst_tr, src_te = sa_act[tr], sb_act[tr], sa_act[te]
                shape = lambda z, idx: z
            for kind in ("identity", "orthogonal", "affine"):
                if kind == "identity":
                    mapped = src_te
                elif kind == "orthogonal":
                    mapped = src_te @ M.procrustes(src_tr, dst_tr)
                else:
                    mapped = M.apply_affine(src_te, M.affine_fit(src_tr, dst_tr))
                lg = B.run_from(site, shape(mapped, te))
                row[kind] = round(float((lg.argmax(-1) == run.y[te]).float().mean()), 4)
        stitch.append(row)

    ea, eb = freq_energy(A.W_E.detach(), run.p), freq_energy(B.W_E.detach(), run.p)
    circuits = circuit_match(run, sa, sb, key_freqs(ea), key_freqs(eb))
    return {
        "circuits": circuits,
        "a": sa, "b": sb,
        "drift": info["drift"],
        "overall": overall, "tensors": tensors, "heads": info["heads"],
        "neurons": {
            "matched_mean": round(float(matched.mean()), 4),
            "random_mean": round(float(rand.mean()), 4),
            "above_0_8": round(float((matched > 0.8).float().mean()), 4),
            "hist_matched": hist(matched), "hist_random": hist(rand), "edges": hist_edges,
            "freq_agreement": round(freq_agree, 4), "strong_pairs": int(strong.sum()),
        },
        "interp": {"naive": naive, "aligned": aligned, "naive_barrier": naive_barrier, "aligned_barrier": aligned_barrier},
        "cka": {"sites": SITES, "labels": [SITE_LABELS[s] for s in SITES], "matrix": cka},
        "mutual_knn": knn,
        "stitch": {"rows": stitch, "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4)},
        "fourier": {"a": [round(v, 5) for v in ea], "b": [round(v, 5) for v in eb], "key_a": key_freqs(ea), "key_b": key_freqs(eb)},
    }
