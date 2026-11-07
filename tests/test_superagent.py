"""The superagent: skills, playbooks, missions (plan, approve, run, fail, retry, skip, each-step
approval, stop), Claude planning and replanning (scripted stand-in), the board and its alerts.

Starts a real server on a free port with its own data folder; Claude is replaced
by scripted responses, so the harness is tested; how well Claude plans is not.
"""
import json
import os
import pathlib
import socket
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ISOMORPH_DATA"] = tempfile.mkdtemp()
os.environ.setdefault("ISOMORPH_DEVICE", "cpu")

from server.assistant import safe_action
from server.superagent import playbooks
from server.superagent.skills import SKILLS, normalise, tool_schema


def test_skills_and_playbooks():
    assert normalise("tune_sl", {})["spec"] == "microopus:words"
    assert normalise("lab_train", {"seeds": "1, 2"})["seeds"] == [1, 2]
    for bad in ({"reward": "nope"}, {"steps": "many"}, {"bogus": 1}):
        try:
            normalise("tune_rl", bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        normalise("tune_compare", {"a": "x"})
        raise AssertionError("missing b accepted")
    except ValueError as e:
        assert "required" in str(e)
    assert all("script" not in n and "delete" not in n for n in SKILLS)          # no code, no deleting
    names = {t["name"] for t in tool_schema(False)}
    assert "wsp_read" not in names and "tune_sl" in names                         # key-only skills hidden without a key
    for pid, p in playbooks.PLAYBOOKS.items():
        for sk, args, why in p["steps"]:
            assert sk in SKILLS, (pid, sk)
    assert playbooks.match("fine-tune with a reward and check what it breaks", False)[0] == "reward-tune"
    assert playbooks.match("do two fine-tunes agree", False)[0] == "seeds-agree"
    assert safe_action({"type": "mission", "goal": "tune twice"}) and not safe_action({"type": "mission"})


def start_server():
    import uvicorn

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    from server import app as appmod

    srv = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    return appmod, f"http://127.0.0.1:{port}", srv


def call(base, method, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read()).get("detail"))


def wait(base, mid, until=("done", "paused", "waiting", "stopped"), t=300):
    t0 = time.time()
    while time.time() - t0 < t:
        m = call(base, "GET", f"/api/superagent/mission/{mid}")
        if m["status"] in until:
            return m
        time.sleep(0.4)
    raise AssertionError(f"mission stuck: {m['status']}")


def test_missions():
    appmod, base, srv = start_server()
    SA = appmod.SUPER
    call(base, "GET", "/api/health")                      # the first request teaches it its own address
    assert SA.base_url == base

    # a playbook, approved as a whole, with references between steps
    m = call(base, "POST", "/api/superagent/plan", {"playbook": "plan-then-tune"})
    assert m["status"] == "draft" and [s["skill"] for s in m["steps"]] == ["wsp_plan", "tune_sl", "tune_anatomy"]
    m = call(base, "POST", f"/api/superagent/mission/{m['id']}/drop", {"step": 1})          # edit the draft: drop the planner
    assert [s["i"] for s in m["steps"]] == [1, 2] and m["steps"][1]["args"]["name"] == "$1.adapter"   # references follow
    try:
        call(base, "POST", f"/api/superagent/mission/{m['id']}/drop", {"step": 1})
        raise AssertionError("dropped a step another one needs")
    except RuntimeError as e:
        assert "uses the result" in str(e)
    m = call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
    m = wait(base, m["id"])
    assert m["status"] == "done", m
    assert m["steps"][0]["out"]["adapter"].startswith("planned-") and m["steps"][1]["resolved"]["name"] == m["steps"][0]["out"]["adapter"]
    assert "held-out loss" in m["steps"][0]["summary"] and m["report"] and m["steps"][0]["open"]["type"] == "tune"

    # each-step approval: waits before every step
    m = call(base, "POST", "/api/superagent/plan", {"playbook": "reward-tune", "approve": "each"})
    m = call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
    m = wait(base, m["id"], until=("waiting",))
    assert m["steps"][0]["status"] == "done" and m["steps"][1]["status"] == "waiting"
    board = call(base, "GET", "/api/superagent")["board"]
    assert any(a["id"].startswith(f"wait-{m['id']}") for a in board["alerts"])
    m = call(base, "POST", f"/api/superagent/mission/{m['id']}/stop")
    assert m["status"] == "stopped"

    # a failing step pauses the mission; retry fails again; skip finishes it; the failed job raises an alert
    m = SA.from_playbook("trace-pair")
    SA.missions[m["id"]]["steps"].insert(0, {"i": 0, "skill": "tune_sl", "label": "Fine-tune", "area": "Tune", "args": {"spec": "microopus:nope"},
                                             "why": "", "cost": "seconds", "downloads": False, "status": "pending"})
    for i, s in enumerate(SA.missions[m["id"]]["steps"]):
        s["i"] = i + 1
    call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
    m = wait(base, m["id"])
    assert m["status"] == "paused" and m["steps"][0]["status"] == "failed" and "nope" in m["steps"][0]["error"].lower()
    board = call(base, "GET", "/api/superagent")["board"]
    errs = [a for a in board["alerts"] if a["level"] == "error"]
    assert errs and any(a["action"] and a["action"].get("mission") == m["id"] for a in errs)
    call(base, "POST", "/api/superagent/dismiss", {"id": errs[0]["id"]})
    assert errs[0]["id"] not in {a["id"] for a in call(base, "GET", "/api/superagent")["board"]["alerts"]}
    call(base, "POST", f"/api/superagent/mission/{m['id']}/retry")
    m = wait(base, m["id"])
    assert m["steps"][0]["status"] == "failed"
    m = call(base, "POST", f"/api/superagent/mission/{m['id']}/skip")
    m = wait(base, m["id"])
    assert m["status"] == "done" and m["steps"][1]["status"] == "done" and "different tokens" in m["steps"][1]["summary"], json.dumps(m["steps"])[:1500]

    # stop in the middle of a job: the job is stopped too
    m = call(base, "POST", "/api/superagent/plan", {"playbook": "plan"})
    SA.missions[m["id"]]["steps"][0]["args"].update(steps=3000, seeds=3)
    call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
    t0 = time.time()
    while not SA.missions[m["id"]]["steps"][0].get("job") and time.time() - t0 < 30:
        time.sleep(0.2)
    job = SA.missions[m["id"]]["steps"][0]["job"]
    call(base, "POST", f"/api/superagent/mission/{m['id']}/stop")
    t0 = time.time()
    while appmod.jobs.jobs[job].state in ("queued", "running") and time.time() - t0 < 120:
        time.sleep(0.3)
    assert appmod.jobs.jobs[job].state == "stopped" and SA.missions[m["id"]]["status"] == "stopped"

    # Claude plans from words (scripted), and replans after a failure
    plans = [
        {"title": "Tune and read", "note": "Uses Micro Opus only.", "steps": [
            {"skill": "tune_sl", "args": {"steps": 30, "name": "claude-plan"}, "why": "teach it"},
            {"skill": "tune_compare", "args": {"a": "$1.adapter", "b": "no-such-adapter"}, "why": "compare"}]},
        {"title": "Revised", "steps": [{"skill": "tune_anatomy", "args": {"name": "claude-plan"}, "why": "read it instead"}]},
    ]
    seen = []

    def fake(system, messages, tools):
        seen.append((system, messages, tools))
        if tools:
            return {"content": [{"type": "tool_use", "name": "propose_plan", "input": plans.pop(0)}]}
        return {"content": [{"type": "text", "text": "Report: the fine-tune learned the rule (step 1)."}]}
    SA.call_claude, real_key = fake, SA.key
    SA.key = lambda: "test-key"
    try:
        m = call(base, "POST", "/api/superagent/plan", {"goal": "teach micro opus a rule and read the change", "view": "tune"})
        assert m["source"] == "claude" and m["note"] == "Uses Micro Opus only." and "GOAL: teach micro opus" in seen[0][1][0]["content"]
        assert "tune_sl" in seen[0][0] and "wsp_read" in seen[0][0]                     # the skills are in the prompt
        call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
        m = wait(base, m["id"])
        assert m["status"] == "paused" and m["steps"][1]["status"] == "failed"
        m = call(base, "POST", f"/api/superagent/mission/{m['id']}/replan")
        assert m["status"] == "draft" and [s["status"] for s in m["steps"]] == ["done", "skipped", "pending"]
        assert "STEPS ALREADY RUN" in seen[-1][1][0]["content"]
        call(base, "POST", f"/api/superagent/mission/{m['id']}/approve")
        m = wait(base, m["id"])
        assert m["status"] == "done" and m["report"].startswith("Report:")
    finally:
        SA.call_claude, SA.key = fake, real_key

    # board controls are one-step missions approved by the click
    sess = appmod.TUNE.create("microopus:words", rank=2)
    m = call(base, "POST", "/api/superagent/skill", {"skill": "tune_close", "args": {"id": sess.id}})
    m = wait(base, m["id"])
    assert m["status"] == "done" and sess.closed
    ov = call(base, "GET", "/api/superagent?view=tune")
    assert ov["playbooks"][0]["here"] and {"jobs", "autopilot", "sessions", "claude", "missions", "alerts"} <= set(ov["board"])
    srv.should_exit = True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
