"""Train a family of tiny transformers that differ only in their random seed."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TinyTransformer, make_dataset


def train_family(run_dir: Path, cfg: dict, progress=lambda f, msg, extra=None: None, should_stop=lambda: False):
    run_dir.mkdir(parents=True, exist_ok=True)
    p = int(cfg["p"])
    seeds = [int(s) for s in cfg["seeds"]]
    steps = int(cfg["steps"])
    eval_every = max(10, steps // 120)
    # One shared data split: the only thing that differs between models is the init seed.
    x, y, tr, te = make_dataset(p, float(cfg["train_frac"]), int(cfg.get("data_seed", 0)))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    histories = {}
    total = len(seeds) * steps
    done = 0
    t0 = time.time()
    for s in seeds:
        torch.manual_seed(s)
        m = TinyTransformer(p, int(cfg["d_model"]), int(cfg["n_heads"]), int(cfg["d_mlp"]))
        opt = torch.optim.AdamW(m.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]), betas=(0.9, 0.98))
        hist = {"step": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        for step in range(steps + 1):
            if should_stop():
                raise InterruptedError("Training stopped.")
            if step % eval_every == 0 or step == steps:
                with torch.no_grad():
                    lg = m(x)
                    for split, idx in (("train", tr), ("test", te)):
                        hist[f"{split}_loss"].append(round(F.cross_entropy(lg[idx], y[idx]).item(), 5))
                        hist[f"{split}_acc"].append(round((lg[idx].argmax(-1) == y[idx]).float().mean().item(), 4))
                    hist["step"].append(step)
                el = time.time() - t0
                eta = el / max(done, 1) * (total - done)
                progress(done / total, f"Seed {s}: step {step}/{steps}, test accuracy {hist['test_acc'][-1]:.3f}, about {eta:.0f}s left",
                         {"seed": s, "history": hist})
            if step == steps:
                break
            loss = F.cross_entropy(m(x[tr]), y[tr])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
        torch.save({"config": m.config(), "state": m.state_dict(), "seed": s}, run_dir / f"seed_{s}.pt")
        histories[str(s)] = hist
        (run_dir / "history.json").write_text(json.dumps(histories))
    progress(1.0, "Training finished.", None)
    return histories
