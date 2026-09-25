"""Smoke test for the lab: train two tiny models, check alignment is exact."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from server.lab.train import train_family
from server.lab.analyze import Run, align, family_report, pair_report


def test_lab_pipeline():
    d = pathlib.Path(tempfile.mkdtemp()) / "run"
    cfg = dict(p=13, seeds=[1, 2], steps=150, train_frac=0.5, d_model=32, n_heads=4, d_mlp=64, lr=1e-3, wd=1.0)
    train_family(d, cfg)
    run = Run(d)
    Bp, info = align(run, 1, 2)
    assert info["drift"] < 1e-3, info["drift"]          # symmetries must not change B's function
    # self-alignment must recover identical weights
    Ap, _ = align(run, 1, 1)
    for n, p in run.models[1].named_parameters():
        assert torch.allclose(p, getattr(Ap, n), atol=1e-4), n
    f = family_report(run); p = pair_report(run, 1, 2)
    assert len(f["models"]) == 2 and p["overall"]["aligned"] >= p["overall"]["raw"] - 1e-6
    print("lab ok", p["overall"], "drift", info["drift"])

if __name__ == "__main__":
    test_lab_pipeline()
