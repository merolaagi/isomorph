"""Micro Opus adapter: numerics match model.py, and every Isomorph view accepts it."""
import sys, pathlib, tempfile, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import microopus as mo


def test_microopus():
    for spec in ("microopus:words", "microopus:names"):
        d = mo.check_against_numpy(spec)
        assert d < 1e-5, (spec, d)
    from server.trace.trace import trace_report
    from server.trace.anatomy import anatomy_report
    from server.trace.storage import storage_report
    r = trace_report("microopus:words", "microopus:names", "the big dog ran to the")
    assert r["A"]["layers"] == 2 and r["learning"]["a"]["loss_after"] <= r["learning"]["a"]["loss_before"] + 1e-6
    a = anatomy_report("microopus:words", "the big dog ran to the")
    blk = [n for n in a["nodes"] if n["id"] == "block"][0]
    attn = [c for c in blk["children"] if c["id"] == "attn"][0]
    assert len(attn["code"]["entries"]) == 4 and "model.py" in attn["code"]["file"]
    s = storage_report("microopus:names")
    assert s["format"] == "json" and s["total_params"] > 7000
    mo.set_data_dir(pathlib.Path(tempfile.mkdtemp()))
    spec = mo.train_seed("words", 3, steps=200)
    assert mo.check_against_numpy(spec) < 1e-5
    json.dumps(r); json.dumps(a); json.dumps(s)
    print("microopus ok", spec)


if __name__ == "__main__":
    test_microopus()
