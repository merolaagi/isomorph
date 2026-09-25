"""Pattern-seeded training: does starting with a known pattern save compute?

Every condition trains on the same data split, with the same seeds, and stops
once the model generalises (test accuracy >= target for three evaluations in a
row). The cost of a run is the number of steps, and the compute estimate
counts a forward pass over all parameters plus a backward pass over the
trainable ones.

Conditions
  random            ordinary initialisation (the control)
  fourier           number embeddings start as cos/sin waves at a few random
                    frequencies: the pattern predicted by the task's symmetry,
                    no trained model needed
  fourier_frozen    same, and the embeddings are never updated
  transplant        number embeddings copied from a trained member of the family
  transplant_frozen same, and never updated
  shuffled          the donor's embedding with its rows shuffled: same numbers,
                    pattern destroyed (controls for "any trained-looking init")
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .analyze import key_freqs, freq_energy
from .model import TinyTransformer, load_model, make_dataset

CONDITIONS = {
    "random": "Ordinary random start",
    "fourier": "Waves from the task's symmetry",
    "fourier_frozen": "Waves, frozen",
    "transplant": "Embedding from a trained model",
    "transplant_frozen": "Transplant, frozen",
    "shuffled": "Transplant with rows shuffled (control)",
}


def fourier_embedding(p, d, k, seed, scale):
    g = torch.Generator().manual_seed(10_000 + seed)
    freqs = (torch.randperm(p // 2, generator=g)[:k] + 1).tolist()
    x = torch.arange(p, dtype=torch.float64)
    cols = []
    for f in freqs:
        cols += [torch.cos(2 * math.pi * f * x / p), torch.sin(2 * math.pi * f * x / p)]
    Fm = torch.stack(cols, 1)
    Fm = Fm / Fm.norm(dim=0, keepdim=True)
    q, _ = torch.linalg.qr(torch.randn(d, 2 * k, generator=g, dtype=torch.float64))
    W = Fm @ q.T  # p x d, rank 2k
    W = W / W.norm(dim=1, keepdim=True).mean() * scale
    return W.float(), freqs


def _count(m, frozen):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for n, p in m.named_parameters() if not (frozen and n == "W_E"))
    return total, train


def run_condition(cond, seed, cfg, donor, data, max_steps, target, progress, should_stop):
    x, y, tr, te = data
    p = int(cfg["p"])
    torch.manual_seed(seed)
    m = TinyTransformer(p, int(cfg["d_model"]), int(cfg["n_heads"]), int(cfg["d_mlp"]))
    info = {}
    scale = float(m.W_E.detach()[:p].norm(dim=1).mean())
    with torch.no_grad():
        if cond.startswith("fourier"):
            k = len(key_freqs(freq_energy(donor.W_E.detach(), p))) if donor is not None else 5
            W, freqs = fourier_embedding(p, m.d, k, seed, scale)
            m.W_E[:p] = W
            info["freqs"] = freqs
        elif cond.startswith("transplant") or cond == "shuffled":
            if donor is None:
                raise ValueError("This condition needs a trained donor model.")
            W = donor.W_E.detach()[:p].clone()
            if cond == "shuffled":
                W = W[torch.randperm(p, generator=torch.Generator().manual_seed(seed))]
            m.W_E[:p] = W
            m.W_pos.copy_(donor.W_pos.detach())
    frozen = cond.endswith("_frozen")
    if frozen:
        m.W_E.requires_grad_(False)
    params = [q for q in m.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]), betas=(0.9, 0.98))
    total, trainable = _count(m, frozen)
    tokens = len(tr) * 3
    hist = {"step": [], "test_acc": [], "train_acc": []}
    every = 25
    reached, streak, step = None, 0, 0
    for step in range(max_steps + 1):
        if should_stop():
            raise InterruptedError("Stopped.")
        if step % every == 0:
            with torch.no_grad():
                lg = m(x)
                ta = float((lg[te].argmax(-1) == y[te]).float().mean())
                tra = float((lg[tr].argmax(-1) == y[tr]).float().mean())
            hist["step"].append(step)
            hist["test_acc"].append(round(ta, 4))
            hist["train_acc"].append(round(tra, 4))
            streak = streak + 1 if ta >= target else 0
            if streak == 1:
                reached = step
            if streak == 0:
                reached = None
            if streak >= 3:
                break
            progress(step / max_steps, f"{CONDITIONS[cond]}, seed {seed}: step {step}, test accuracy {ta:.2f}")
        if step == max_steps:
            break
        loss = F.cross_entropy(m(x[tr]), y[tr])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    flops = step * tokens * (2 * total + 4 * trainable)
    return {"condition": cond, "seed": seed, "steps_to_target": reached, "steps_run": step,
            "final_test_acc": hist["test_acc"][-1], "params": total, "trainable": trainable,
            "flops": flops, "history": hist} | info


def seeded_experiment(run_dir: Path, conditions, seeds, max_steps, target=0.95, donor_seed=None,
                      progress=lambda f, m, x=None: None, should_stop=lambda: False):
    cfg = json.loads((run_dir / "config.json").read_text())
    donor = None
    donors = sorted(run_dir.glob("seed_*.pt"))
    if donor_seed is not None and (run_dir / f"seed_{donor_seed}.pt").exists():
        donor = load_model(run_dir / f"seed_{donor_seed}.pt")
    elif donors:
        donor = load_model(donors[0])
        donor_seed = int(donors[0].stem.split("_")[1])
    data = make_dataset(int(cfg["p"]), float(cfg["train_frac"]), int(cfg.get("data_seed", 0)))
    # Fresh seeds that the donor family did not use, so no run starts from a donor's own init.
    used = set(int(s) for s in cfg.get("seeds", []))
    seeds = [s for s in seeds if s not in used] or [max(used or {0}) + 100 + i for i in range(len(seeds))]
    results = []
    jobs = [(c, s) for c in conditions for s in seeds]
    t0 = time.time()
    for i, (c, s) in enumerate(jobs):
        def pr(f, msg, i=i):
            progress((i + f) / len(jobs), msg, {"done": results})
        results.append(run_condition(c, s, cfg, donor, data, max_steps, target, pr, should_stop))
    summary = []
    base = [r["steps_to_target"] for r in results if r["condition"] == "random"]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    base_ok = [b for b in base if b is not None]
    base_med = med(base_ok) if base_ok else None
    base_flops = med([r["flops"] for r in results if r["condition"] == "random"])
    for c in conditions:
        rs = [r for r in results if r["condition"] == c]
        ok = [r["steps_to_target"] for r in rs if r["steps_to_target"] is not None]
        m_steps = med(ok)
        m_flops = med([r["flops"] for r in rs])
        summary.append({"condition": c, "label": CONDITIONS[c], "runs": len(rs), "reached": len(ok),
                        "median_steps": m_steps, "median_flops": m_flops,
                        "step_saving": (1 - m_steps / base_med) if (m_steps is not None and base_med) else None,
                        "compute_saving": (1 - m_flops / base_flops) if (m_flops and base_flops and len(ok) == len(rs)) else None})
    out = {"created": time.time(), "seconds": round(time.time() - t0, 1), "target": target, "max_steps": max_steps,
           "donor_seed": donor_seed, "seeds": seeds, "conditions": conditions, "summary": summary, "results": results,
           "config": cfg}
    (run_dir / f"seeded_{int(time.time())}.json").write_text(json.dumps(out))
    return out
