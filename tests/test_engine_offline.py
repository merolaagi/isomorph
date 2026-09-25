"""Offline test of the engine: mine, run a lever experiment, build a blueprint, write a digest."""
import sys, pathlib, tempfile, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server.engine.ledger import mine, run_lever, build_blueprint, load
from server.engine.narrator import narrate


def test_engine():
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "atlas").mkdir(); (d / "lab").mkdir()
    L = mine(d / "engine", d / "atlas", d / "lab")
    assert L["rules"] == {}
    e = run_lever(d / "engine", d, "copy", "share", {"n": 1}, [1], 300)
    assert e["verdict"] in ("holds", "rejected", "inconclusive") and e["summary"]["param_saving"] > 0.5
    bp = build_blueprint(d / "engine", d, {"share": {"n": 1}, "topk": {"k": 0.5}}, ["copy"], [1], 0.1)
    assert bp["results"][0]["params_bp"] < bp["results"][0]["params_base"]
    L = mine(d / "engine", d / "atlas", d / "lab")
    assert any(r["kind"] == "lever" for r in L["rules"].values())
    n = narrate(L)
    assert "## What seems to be true" in n["text"] and not n["unknown_citations"]
    json.dumps(load(d / "engine"))
    print("engine ok", e["verdict"], e["reason"])


if __name__ == "__main__":
    test_engine()


def test_autopilot_resume():
    import time
    from server.jobs import JobQueue
    from server.engine.autopilot import Autopilot
    d = pathlib.Path(tempfile.mkdtemp())
    dirs = {"atlas": d / "atlas", "lab": d / "lab", "engine": d / "engine"}
    for x in dirs.values():
        x.mkdir(parents=True)
    ap = Autopilot(d, JobQueue(), dirs)
    s = ap.state()
    s.update(status="running", current={"type": "mine", "started": time.time()}, queue=[{"type": "plan"}], done=[], deadline=time.time() + 3600,
             snapshot={"rules": {}, "experiments": [], "blueprints": []}, config={"claude": False, "max_experiments": 1, "seeds": [1], "scale": 0.05, "tinystories": False, "blueprint": False})
    ap._save(s)
    ap2 = Autopilot(d, JobQueue(), dirs)
    ap2.boot()
    for _ in range(300):
        if ap2.state()["status"] != "running":
            break
        time.sleep(1)
    st = ap2.state()
    assert st["status"] == "done" and st["report"] and st["done"][0]["type"] == "mine", st["status"]
    print("autopilot ok", [(x["type"], x["ok"]) for x in st["done"]])


if __name__ == "__main__":
    test_autopilot_resume()
