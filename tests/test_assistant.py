"""Assistant: the checklist follows what exists, answers work without a key, and actions never start a job."""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import assistant as AS


def dirs():
    root = pathlib.Path(tempfile.mkdtemp())
    d = {k: root / k for k in ("lab", "atlas", "engine", "laws", "wsp", "trace")}
    for p in d.values():
        p.mkdir()
    return root, d


def test_checklist_follows_what_exists():
    root, d = dirs()
    s = AS.state(root, d["lab"], d["atlas"], d["engine"], d["laws"], False, "0.15.0")
    st = AS.steps(s)
    ids = [x["id"] for x in st["steps"]]
    assert ids[:3] == ["tour", "trace", "plan"] and "key" in ids and "read" not in ids      # Claude steps only once a key exists
    assert st["next"] == "trace" and st["done"] == 0
    # do some things: a trace, a plan, two lab models, three laws runs, the tour and the playground in this browser
    (d["trace"] / "t1.json").write_text("{}")
    (d["wsp"] / "20261104-000000-aaaaaa.json").write_text(json.dumps({"kind": "plan", "spec": "microopus:words", "verdict": "x"}))
    run = d["lab"] / "r1"
    run.mkdir()
    (run / "config.json").write_text("{}")
    for i in (1, 2):
        (run / f"seed_{i}.pt").write_text("")
    for i in range(3):
        (d["laws"] / f"m{i}.json").write_text(json.dumps({"spec": f"m{i}"}))
    s = AS.state(root, d["lab"], d["atlas"], d["engine"], d["laws"], True, "0.15.0", seen=["tour", "write"])
    st = AS.steps(s)
    done = {x["id"] for x in st["steps"] if x["done"]}
    assert {"tour", "trace", "plan", "lab", "laws", "write", "key"} <= done, done
    assert st["next"] == "read" and "control" in {x["id"] for x in st["steps"]}               # no random model among the laws runs
    assert all(AS.safe_action(x["action"]) for x in st["steps"])                              # every built-in step is a safe action
    json.dumps(st)


def test_actions_are_page_only():
    ok = [{"type": "go", "view": "wsp", "mode": "write", "fill": {"ww-pattern": "disc"}, "press": "ww-crafted"}, {"type": "tour", "tour": "laws"}]
    bad = [{"type": "go", "view": "wsp", "press": "ws-plan button"}, {"type": "go", "view": "nowhere"}, {"type": "run", "job": "laws"},
           {"type": "go", "view": "laws", "fill": {"lw-go": "x"}}, {"type": "go", "view": "wsp", "mode": "delete"}, None]
    assert all(AS.safe_action(a) for a in ok) and not any(AS.safe_action(a) for a in bad)


def test_answers_without_and_with_a_key():
    root, d = dirs()
    s = AS.state(root, d["lab"], d["atlas"], d["engine"], d["laws"], False, "0.15.0")
    r = AS.offline("what does the drift KL mean in the fine-tune plan?", s, {"view": "wsp"})
    assert r["source"] == "manual" and "plan a fine-tune" in r["reply"].lower() and r["actions"]

    seen = {}

    def call(system, messages, tools):
        seen["system"], seen["messages"] = system, messages
        return {"content": [{"type": "text", "text": "Press Example task, then Plan and check."},
                            {"type": "tool_use", "id": "t", "name": "suggest", "input": {"actions": [
                                {"label": "Open the planner", "action": {"type": "go", "view": "wsp", "mode": "plan", "press": "ws-plan-demo"}},
                                {"label": "Run it for me", "action": {"type": "go", "view": "wsp", "press": "ws-plan button[type=submit]"}}]}}]}
    r = AS.ask("How do I start?", [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}], s,
               {"view": "wsp", "screen": "Held-out loss 0.53"}, call)
    assert r["source"] == "claude" and r["reply"].startswith("Press Example task")
    assert [a["label"] for a in r["actions"]] == ["Open the planner"]                         # the job-starting one is dropped
    assert "Held-out loss 0.53" in seen["system"] and "MANUAL" in seen["system"] and len(seen["messages"]) == 3


def test_every_page_has_its_help():
    root, d = dirs()
    s = AS.state(root, d["lab"], d["atlas"], d["engine"], d["laws"], True, "0.16.0")
    for view in ("trace", "atlas", "laws", "wsp", "engine", "lab", "hub", "about"):
        p = AS.page(view, None, s)
        assert p["title"] and p["intro"] and p["prompts"], view
        assert all(AS.safe_action(a["action"]) for a in p["actions"]), view
    for m in ("plan", "read", "fix", "write"):
        p = AS.page("wsp", m, s)
        assert p["key"] == f"wsp-{m}" and m.capitalize()[:4] in p["title"] or m == "plan", (m, p["title"])
        assert p["intro"] == AS.MANUAL[f"wsp-{m}"][1] and all(AS.safe_action(a["action"]) for a in p["actions"])
    assert [x["id"] for x in AS.page("wsp", "write", s)["steps"]] == ["write", "key", "write-claude"]
    assert AS.page("nowhere")["key"] == "trace"
    seen = {}
    AS.ask("What is broken here?", [], s, {"view": "wsp", "mode": "write", "screen": "Agreement 51.2%"},
           lambda system, messages, tools: seen.update(system=system) or {"content": [{"type": "text", "text": "v[0]."}]})
    assert "THE USER IS ON: Weight space · Write" in seen["system"] and "Agreement 51.2%" in seen["system"]
    r = AS.offline("explain", s, {"view": "laws"})
    assert r["reply"].startswith("**Laws.**")


if __name__ == "__main__":
    test_checklist_follows_what_exists()
    test_actions_are_page_only()
    test_answers_without_and_with_a_key()
    test_every_page_has_its_help()
    print("assistant ok")
