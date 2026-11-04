"""Fine-tune planner: decide from the weights which matrices to tune, then check
the decision against the obvious alternatives at the same budget.

Two ways of choosing, both made before any training:

  weights   from the weights alone. Each matrix's heavy-tail exponent alpha
            (the eigenvalue spectrum of W^T W): a large alpha means a spectrum
            close to random, a matrix training has shaped least, so the most
            room to change. This is the WeightWatcher reading of a spectrum,
            used here as a hypothesis, not as a fact.
  gradient  from the weights and your examples. One backward pass over the
            practice examples: matrices whose gradient is largest relative to
            their own size are chosen, and each one's low-rank delta gets the
            rank that holds 90% of its gradient's energy.

Each plan becomes a set of low-rank deltas (W + U V^T, U starting at zero) with
the same number of trainable values, within a few percent. The controls get the same
budget: every matrix at one shared rank, the last blocks first, and matrices
in a random order. All plans train with the same steps, learning rate,
batches and seeds, and are scored on held-out examples they never saw, plus
how far they moved the model on general text (mean KL per token). The model's
own weights are restored exactly afterwards and the restore is checked.
"""
import math
import random
import time

import numpy as np
import torch

from ..hub.load import load_lm
from .common import (Deltas, checksum, checksum_diff, drift, encode, general_texts, lowrank_params, next_token_logp,
                     parse_examples, split, token_loss, tunable)

PLANS = {
    "gradient": "Chosen from the gradient on your examples",
    "weights": "Chosen from the weights alone (spectrum)",
    "all": "Every matrix, one shared rank",
    "last": "The last blocks first",
    "random": "Matrices in a random order",
}
CHOSEN = ("gradient", "weights")
CONTROLS = ("all", "last", "random")


def _alpha(W, rng):
    from ..laws import _sub, matrix_measures

    w = _sub(W.detach().float().cpu().numpy(), rng)
    if min(w.shape) < 8:
        return None
    return matrix_measures(w).get("alpha")


def read_weights(cands):
    rng = np.random.default_rng(0)
    out = []
    for c in cands:
        W = getattr(c["module"], c["attr"])
        s = torch.linalg.svdvals(W.detach().float().cpu())
        stable = float((s ** 2).sum() / (s[0] ** 2 + 1e-12))
        out.append({"alpha": _alpha(W, rng), "stable_rank": stable, "norm": float(W.detach().float().norm())})
    return out


def read_gradient(model, cands, rows, batch=16):
    """Gradient of the practice loss for every candidate matrix, from one pass over all of them."""
    params = [getattr(c["module"], c["attr"]) for c in cands]
    keep = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    try:
        n_tok = token_loss(model, rows)["tokens"]
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            frac = sum(sum(m) - (m[0] if m else 0) for _, m in part) / max(n_tok, 1)
            loss = token_loss(model, part, batch=batch, grad=True) * frac
            loss.backward()
        out = []
        for p in params:
            G = p.grad.detach().float().cpu()
            s = torch.linalg.svdvals(G)
            e = (s ** 2).cumsum(0) / ((s ** 2).sum() + 1e-30)
            r90 = int((e < 0.9).sum()) + 1
            out.append({"grad_rel": float(G.norm() / (p.detach().float().norm() + 1e-12)), "grad_rank90": r90,
                        "grad_top": float(s[0] ** 2 / ((s ** 2).sum() + 1e-30))})
        return out
    finally:
        for p in params:
            p.grad = None
        for p, k in zip(model.parameters(), keep):
            p.requires_grad_(k)


def _fill(order, cands, rank_of, budget):
    """Take matrices in order, each at its wanted rank or the largest rank that still fits,
    until the budget is spent."""
    pick, used = [], 0
    for i in order:
        left = budget - used
        per = cands[i]["shape"][0] + cands[i]["shape"][1]
        r = min(rank_of(i), min(cands[i]["shape"]), left // per)
        if r < 1:
            continue
        pick.append((i, int(r)))
        used += lowrank_params(cands[i]["shape"], int(r))
        if budget - used < min(c["shape"][0] + c["shape"][1] for c in cands):
            break
    return pick, used


def make_plans(cands, wread, gread, budget, uniform_rank=4, seed=0, max_rank=16):
    n = len(cands)
    grad_order = sorted(range(n), key=lambda i: -gread[i]["grad_rel"])
    wkey = [w["alpha"] if w["alpha"] is not None else -1.0 for w in wread]
    weight_order = sorted(range(n), key=lambda i: -wkey[i])
    plans = {}
    plans["gradient"] = _fill(grad_order, cands, lambda i: min(max_rank, gread[i]["grad_rank90"]), budget)
    plans["weights"] = _fill(weight_order, cands, lambda i: uniform_rank, budget)
    per = sum(c["shape"][0] + c["shape"][1] for c in cands)
    r_all = max(1, int(budget // per))
    plans["all"] = ([(i, min(r_all, min(cands[i]["shape"]))) for i in range(n)],
                    sum(lowrank_params(cands[i]["shape"], min(r_all, min(cands[i]["shape"]))) for i in range(n)))
    last_order = sorted(range(n), key=lambda i: (-cands[i]["layer"], i))
    plans["last"] = _fill(last_order, cands, lambda i: uniform_rank, budget)
    rnd = list(range(n))
    random.Random(1000 + seed).shuffle(rnd)
    plans["random"] = _fill(rnd, cands, lambda i: uniform_rank, budget)
    return plans


def train_plan(model, cands, pick, rows, steps, lr, seed, batch=16, evals=(), on_eval=None, final=None):
    """Train low-rank deltas on the chosen matrices; measure while they are attached; then remove them."""
    with Deltas(model) as D:
        for i, r in pick:
            D.add(cands[i], r, seed=seed * 7919 + i)
        opt = torch.optim.Adam(list(D.parameters()), lr=lr)
        g = random.Random(seed)
        curve = []
        for step in range(1, steps + 1):
            part = [rows[g.randrange(len(rows))] for _ in range(min(batch, len(rows)))]
            opt.zero_grad(set_to_none=True)
            loss = token_loss(model, part, batch=batch, grad=True)
            loss.backward()
            opt.step()
            if step in evals and on_eval:
                curve.append((step, on_eval()))
        return curve, final() if final else None


def _mean_spread(xs):
    return float(np.mean(xs)), float((max(xs) - min(xs)) / 2) if len(xs) > 1 else 0.0


def verdict(res):
    """One honest sentence per chosen plan: ahead of every control beyond the seeds' spread, or not."""
    def band(k):
        r = res[k]["held"]
        return r["loss"] - r["loss_spread"], r["loss"] + r["loss_spread"]

    lines = []
    for k in CHOSEN:
        lo, hi = band(k)
        beat = [c for c in CONTROLS if hi < band(c)[0]]
        lost = [c for c in CONTROLS if band(c)[1] < lo]
        name = "The gradient plan" if k == "gradient" else "The weights-only plan"
        best = min(CONTROLS, key=lambda c: res[c]["held"]["loss"])
        if len(beat) == len(CONTROLS):
            lines.append(f"{name} beat every control on held-out loss ({res[k]['held']['loss']:.4f} against the best control, "
                         f"{best}, at {res[best]['held']['loss']:.4f}), beyond the seeds' spread.")
        elif lost:
            lines.append(f"{name} was worse than {', '.join(lost)} beyond the seeds' spread ({res[k]['held']['loss']:.4f} against "
                         f"{best} at {res[best]['held']['loss']:.4f}); here the choice did not help.")
        else:
            lines.append(f"{name} was within the seeds' spread of {', '.join(c for c in CONTROLS if c not in beat)} "
                         f"({res[k]['held']['loss']:.4f} against {best} at {res[best]['held']['loss']:.4f}): no clear gain from choosing.")
    return " ".join(lines)


def plan_report(spec, examples, held_out=None, budget_frac=0.01, steps=150, lr=2e-3, seeds=2, uniform_rank=4,
                plans=None, progress=lambda f, m: None, should_stop=lambda: False, loader=load_lm):
    t0 = time.time()
    tok, model, causal = loader(spec)
    if not causal:
        raise ValueError("The fine-tune planner needs a model that predicts the next token.")
    ex = parse_examples(examples) if isinstance(examples, str) else examples
    ho = parse_examples(held_out) if isinstance(held_out, str) and held_out.strip() else (held_out or None)
    practice, held = split(ex, ho)
    prow, hrow = encode(tok, practice), encode(tok, held)
    gen = general_texts(tok, model=model)
    model.eval()
    before_sum = checksum(model)
    base_gen = next_token_logp(model, tok, gen)
    base = {"practice": token_loss(model, prow), "held": token_loss(model, hrow)}

    cands = tunable(model)
    if not cands:
        raise ValueError("No tunable matrices found in this model's blocks.")
    progress(0.03, f"Reading the spectra of {len(cands)} matrices")
    wread = read_weights(cands)
    progress(0.08, "One backward pass over the practice examples")
    gread = read_gradient(model, cands, prow)
    tunable_total = sum(c["shape"][0] * c["shape"][1] for c in cands)
    # never below one rank on every matrix, so the every-matrix control fits the same budget
    budget = max(int(budget_frac * tunable_total), sum(lowrank_params(c["shape"], 1) for c in cands))
    want = [p for p in (plans or list(PLANS)) if p in PLANS]
    for k in CHOSEN:
        if k not in want:
            want.insert(0, k)

    matrices = []
    for c, w, g in zip(cands, wread, gread):
        matrices.append({"name": c["name"], "layer": c["layer"], "kind": c["kind"], "shape": list(c["shape"]),
                         **w, **g})

    out = {}
    evals = sorted({max(1, steps // 4), max(1, steps // 2), steps})
    total_runs = len(want) * seeds
    done = 0
    for k in want:
        held_l, held_a, prac_l, kl, curves, used_list, picks = [], [], [], [], [], [], None
        for s in range(seeds):
            if should_stop():
                raise InterruptedError("Stopped.")
            plans_s = make_plans(cands, wread, gread, budget, uniform_rank, seed=s)
            pick, used = plans_s[k]
            picks = picks or pick
            progress(0.1 + 0.88 * done / total_runs, f"Training the {k} plan, seed {s + 1} of {seeds}")
            def final():
                h = token_loss(model, hrow)
                return {"held": h, "practice": token_loss(model, prow)["loss"],
                        "kl": drift(base_gen, next_token_logp(model, tok, gen))}

            curve, f = train_plan(model, cands, pick, prow, steps, lr, s, evals=evals,
                                  on_eval=lambda: token_loss(model, hrow)["loss"], final=final)
            held_l.append(f["held"]["loss"])
            held_a.append(f["held"]["acc"])
            prac_l.append(f["practice"])
            kl.append(f["kl"])
            curves.append([[st, v] for st, v in curve])
            used_list.append(used)
            done += 1
        m, sp = _mean_spread(held_l)
        out[k] = {"label": PLANS[k], "params": int(np.mean(used_list)),
                  "matrices": [{"name": cands[i]["name"], "rank": r} for i, r in picks],
                  "held": {"loss": m, "loss_spread": sp, "acc": float(np.mean(held_a))},
                  "practice": {"loss": float(np.mean(prac_l))},
                  "drift": {"kl": float(np.mean(kl)), "kl_spread": _mean_spread(kl)[1]},
                  "curves": curves}
    after_sum = checksum(model)
    restored = checksum_diff(before_sum, after_sum)
    if restored > 1e-6:
        raise RuntimeError(f"The model's weights were not restored exactly (difference {restored:.3g}).")
    return {"spec": spec, "kind": "plan", "created": time.time(), "seconds": time.time() - t0,
            "examples": {"practice": len(practice), "held_out": len(held), "practice_tokens": base["practice"]["tokens"],
                         "held_tokens": base["held"]["tokens"]},
            "settings": {"budget_frac": budget_frac, "budget": budget, "steps": steps, "lr": lr, "seeds": seeds,
                         "uniform_rank": uniform_rank},
            "model_params": sum(p.numel() for p in model.parameters()), "tunable_params": tunable_total,
            "base": base, "matrices": matrices, "plans": out, "verdict": verdict(out), "restored_diff": restored}
