"""Missions, the runner that carries them out, and the board that supervises every agent.

A mission is a goal and a plan: steps, each one skill call. It is drafted (by a
playbook, or by Claude from your words), approved by you (the whole plan, or each
step as it comes), then run one step at a time in a background thread. A step
that starts a job waits for it like the page would; its result is digested and
can be opened in the page. If a step fails, the mission pauses: retry it, skip
it, or (with a key) let Claude revise the rest of the plan. At the end, Claude (or,
without a key, the digests) writes the report.

Skills are called over this server's own API, with the same validation and the
same password as the page, so the superagent can do nothing the page cannot.
"""
import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..jsonsafe import clean
from . import playbooks
from .skills import SKILLS, normalise, tool_schema

REF = re.compile(r"\$(\d+)\.([A-Za-z_]+)")
FINAL = ("done", "failed", "stopped")
IDLE_SESSION_MIN = 20
LONG_JOB_MIN = 90


class Superagent:
    def __init__(self, data_dir: Path, jobs, password, key, lab_dir: Path, autopilot=None, tune=None, version=""):
        self.dir = Path(data_dir) / "superagent"
        (self.dir / "missions").mkdir(parents=True, exist_ok=True)
        self.data, self.jobs, self.password, self.key, self.lab = Path(data_dir), jobs, password, key, Path(lab_dir)
        self.autopilot, self.tune, self.version = autopilot, tune, version
        self.base_url = None
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.missions = {}
        self.dismissed = set(self._read(self.dir / "dismissed.json") or [])
        self.call_claude = None                      # set by the API layer: (system, messages, tools) -> response
        from . import skills as _sk

        _sk.CTX["data"] = self.data
        for f in (self.dir / "missions").glob("*.json"):
            m = self._read(f)
            if not m:
                continue
            if m["status"] in ("running", "approved"):
                m["status"] = "paused"
                for s in m["steps"]:
                    if s["status"] == "running":
                        s["status"] = "pending"
                self._log(m, "The server restarted; press Resume to continue from this step.")
            self.missions[m["id"]] = m
        threading.Thread(target=self._loop, daemon=True).start()

    # -------------------------------------------------------------- files
    @staticmethod
    def _read(f):
        try:
            return json.loads(Path(f).read_text())
        except Exception:
            return None

    def _save(self, m):
        m["updated"] = time.time()
        (self.dir / "missions" / f"{m['id']}.json").write_text(json.dumps(clean(m)))

    def _log(self, m, text, kind="note"):
        m.setdefault("log", []).append({"t": time.time(), "kind": kind, "text": text[:2000]})
        m["log"] = m["log"][-200:]

    # -------------------------------------------------------------- drafting
    def _new(self, title, goal, source, steps, approve="plan"):
        mid = uuid.uuid4().hex[:6]
        out = []
        for i, (sk, args, why) in enumerate(steps):
            args = json.loads(json.dumps(args).replace("{m}", mid))
            out.append({"i": i + 1, "skill": sk, "label": SKILLS[sk]["label"], "area": SKILLS[sk]["area"], "args": args, "why": why,
                        "cost": SKILLS[sk]["cost"], "downloads": SKILLS[sk]["downloads"], "status": "pending"})
        m = {"id": mid, "title": title, "goal": goal, "source": source, "status": "draft", "approve": approve, "steps": out,
             "log": [], "report": None, "created": time.time(), "updated": time.time()}
        self._check(m)
        with self.lock:
            self.missions[mid] = m
            self._save(m)
        return m

    def _check(self, m):
        """Validate every step now (references to earlier results are checked when they run)."""
        for s in m["steps"]:
            if s["skill"] not in SKILLS:
                raise ValueError(f"Step {s['i']}: unknown skill {s['skill']}.")
            if SKILLS[s["skill"]]["needs_key"] and not self.key():
                raise ValueError(f"Step {s['i']} ({s['label']}) needs an Anthropic key: save one under Engine.")
            probe = json.loads(REF.sub("0", json.dumps(s["args"])))
            normalise(s["skill"], probe)

    def from_playbook(self, pid, approve="plan"):
        p = playbooks.PLAYBOOKS.get(pid)
        if not p:
            raise ValueError(f"No playbook {pid}.")
        return self._new(p["title"], p["why"], f"playbook:{pid}", p["steps"], approve)

    def single(self, skill, args, why=""):
        """One step the user asked for directly (a board button): approved by the click."""
        m = self._new(SKILLS[skill]["label"], why or SKILLS[skill]["label"], "button", [(skill, args, why)])
        return self.act(m["id"], "approve")

    def plan(self, goal, view=None, approve="plan"):
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("Say what you want done.")
        if self.key() and self.call_claude:
            return self._plan_claude(goal, view, approve)
        picks = playbooks.match(goal, bool(self.key()))
        if not picks:
            raise ValueError("Without an Anthropic key the superagent runs ready-made playbooks; none matched. "
                             "Pick one from the list, or save a key under Engine so Claude can plan from your words.")
        m = self.from_playbook(picks[0], approve)
        m["goal"] = goal
        m["alternatives"] = [{"id": p, "title": playbooks.PLAYBOOKS[p]["title"]} for p in picks[1:]]
        self._log(m, f"Matched your goal to the playbook '{m['title']}' (no key, so no free planning).")
        self._save(m)
        return m

    # -------------------------------------------------------------- Claude: plan, replan, report
    def _context(self, state=None):
        board = self.board()
        return {"state": state, "adapters": [a["name"] for a in (self.tune_adapters() or [])][:20],
                "lab_runs": self._lab_runs()[:5], "running_jobs": [j["title"] for j in board["jobs"] if j["state"] in ("queued", "running")],
                "autopilot": board["autopilot"].get("status")}

    SYSTEM = """You are the superagent of Isomorph, a local research bench on the user's Mac for comparing neural networks and reading their weights. You turn the user's goal into a plan of skill calls that this app will carry out after the user approves it.

Rules:
- Use only the skills listed. Fill arguments; leave out ones whose default fits.
- To use a value from an earlier step's result, write $N.key (N is the step number): tune_sl and tune_rl give $N.adapter and $N.spec (tune:<name>); lab_train gives $N.run_id; microopus_seed gives $N.spec. Ground-truth lab models are lab:<run_id>:<seed>, e.g. lab:$1.run_id:1.
- Prefer models that need no download (microopus:words, microopus:names, lab models, tune adapters) unless the goal needs real models; say so in the plan when it downloads.
- Every claim needs a control: add the step that gives one (a random model, a second seed, a baseline) when the goal is a question.
- Keep plans short: 1 to 6 steps. Long campaigns go to autopilot_start.
- If the goal is not something these skills can do, return no steps and say what the user can do instead."""

    def _plan_claude(self, goal, view, approve, done_steps=None):
        tools = [{"name": "propose_plan", "description": "The plan to show the user for approval.",
                  "input_schema": {"type": "object", "properties": {
                      "title": {"type": "string", "description": "a short title"},
                      "note": {"type": "string", "description": "one or two sentences: what the plan will show, and anything to know first"},
                      "steps": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
                          "skill": {"type": "string", "enum": list(SKILLS)}, "args": {"type": "object"}, "why": {"type": "string"}},
                          "required": ["skill", "why"]}}}, "required": ["title", "steps"]}}]
        skills = json.dumps(tool_schema(bool(self.key())))
        ctx = json.dumps(self._context(), default=str)[:4000]
        msg = f"GOAL: {goal}\nTHE USER IS ON THE {view or 'unknown'} TAB."
        if done_steps:
            msg += "\n\nSTEPS ALREADY RUN (with results; plan only what remains, numbering continues after them):\n" + json.dumps(done_steps, default=str)[:6000]
        resp = self.call_claude(self.SYSTEM + "\n\nSKILLS:\n" + skills + "\n\nTHIS MAC NOW:\n" + ctx,
                                [{"role": "user", "content": msg}], tools)
        plan, text = None, []
        for b in resp.get("content") or []:
            if b.get("type") == "tool_use" and b.get("name") == "propose_plan":
                plan = b.get("input") or {}
            elif b.get("type") == "text":
                text.append(b["text"])
        if not plan or not plan.get("steps"):
            raise ValueError(("Claude did not propose a plan. " + " ".join(text)).strip()[:600])
        steps = [(s["skill"], s.get("args") or {}, s.get("why", "")) for s in plan["steps"] if s.get("skill") in SKILLS]
        if done_steps is not None:
            return plan, steps
        m = self._new(plan.get("title") or goal[:60], goal, "claude", steps, approve)
        if plan.get("note"):
            self._log(m, plan["note"], "plan")
            m["note"] = plan["note"]
            self._save(m)
        return m

    def _report(self, m):
        done = [{"step": s["i"], "skill": s["label"], "args": s["args"], "result": s.get("summary"), "status": s["status"]} for s in m["steps"]]
        plain = "\n".join(f"{d['step']}. {d['skill']}: {d['result'] or d['status']}" for d in done)
        if not (self.key() and self.call_claude):
            return plain
        try:
            resp = self.call_claude(
                "You write the closing report of a mission run by Isomorph's superagent. Use only the step results given; never invent a number. "
                "Three to six sentences: what was found, against which control, what is uncertain, and the one next step worth taking. Cite steps as (step N).",
                [{"role": "user", "content": f"GOAL: {m['goal']}\n\nRESULTS:\n{json.dumps(done, default=str)[:9000]}"}], [])
            text = "".join(b.get("text", "") for b in resp.get("content") or [] if b.get("type") == "text").strip()
            return text or plain
        except Exception as e:
            return plain + f"\n\n(Claude could not write the report: {e})"

    # -------------------------------------------------------------- control
    def act(self, mid, action, step=None):
        with self.lock:
            m = self.missions.get(mid)
            if not m:
                raise KeyError(f"No mission {mid}.")
            st = m["status"]
            if action in ("approve", "run"):
                if st not in ("draft", "paused", "waiting"):
                    raise ValueError(f"The mission is {st}.")
                m["status"] = "approved"
                if m["approve"] == "each":
                    s = self._current(m)
                    if s:
                        s["ok"] = True
                self._log(m, "Approved." if st == "draft" else "Continuing.")
            elif action == "each":
                m["approve"] = "each" if m["approve"] == "plan" else "plan"
            elif action == "pause":
                if st in ("approved", "running", "waiting"):
                    m["status"] = "paused" if st != "running" else "pausing"
                    self._log(m, "Pausing after the current step." if st == "running" else "Paused.")
            elif action == "resume":
                if st in ("paused", "pausing"):
                    m["status"] = "approved"
                    self._log(m, "Resumed.")
            elif action == "stop":
                if st not in FINAL:
                    cur = self._current(m)
                    if cur and cur.get("job"):
                        self.jobs.stop(cur["job"])
                    m["status"] = "stopped"
                    self._log(m, "Stopped.")
            elif action in ("retry", "skip"):
                s = self._current(m)
                if not s or st not in ("paused", "failed"):
                    raise ValueError("Nothing to retry or skip.")
                if action == "retry":
                    s.update(status="pending", error=None, job=None)
                else:
                    s["status"] = "skipped"
                    self._log(m, f"Skipped step {s['i']}.")
                m["status"] = "approved"
            elif action == "drop":
                if st != "draft":
                    raise ValueError("Steps can be removed only before the plan is approved.")
                users = [s["i"] for s in m["steps"] if s["i"] != step and any(int(n) == step for n, _ in REF.findall(json.dumps(s["args"])))]
                if users:
                    raise ValueError(f"Step {users[0]} uses the result of step {step}; remove that one first.")

                def shift(v):
                    return REF.sub(lambda mm: f"${int(mm.group(1)) - (1 if int(mm.group(1)) > step else 0)}.{mm.group(2)}", v)
                m["steps"] = [s for s in m["steps"] if s["i"] != step]
                for i, s in enumerate(m["steps"]):
                    s["i"] = i + 1
                    s["args"] = json.loads(shift(json.dumps(s["args"])))
            elif action == "replan":
                if not (self.key() and self.call_claude):
                    raise ValueError("Replanning needs an Anthropic key.")
                done = [{"step": s["i"], "skill": s["skill"], "args": s["args"], "status": s["status"], "result": s.get("summary") or s.get("error")}
                        for s in m["steps"] if s["status"] in ("done", "failed", "skipped")]
                plan, steps = self._plan_claude(m["goal"], None, m["approve"], done_steps=done)
                keep = [s for s in m["steps"] if s["status"] in ("done", "skipped")]
                for s in m["steps"]:
                    if s["status"] == "failed":
                        s["status"] = "skipped"
                        keep.append(s)
                keep.sort(key=lambda s: s["i"])
                n = len(keep)
                new = [{"i": n + j + 1, "skill": sk, "label": SKILLS[sk]["label"], "area": SKILLS[sk]["area"], "args": a, "why": w,
                        "cost": SKILLS[sk]["cost"], "downloads": SKILLS[sk]["downloads"], "status": "pending"} for j, (sk, a, w) in enumerate(steps)]
                m["steps"] = keep + new
                self._log(m, f"Claude revised the rest of the plan: {plan.get('note') or plan.get('title')}", "plan")
                m["status"] = "draft"
            else:
                raise ValueError(f"Unknown action {action}.")
            self._save(m)
        self.wake.set()
        return m

    def delete(self, mid):
        with self.lock:
            m = self.missions.get(mid)
            if m and m["status"] in ("running", "approved", "pausing"):
                raise ValueError("Stop the mission first.")
            self.missions.pop(mid, None)
            f = self.dir / "missions" / f"{mid}.json"
            if f.exists():
                f.unlink()

    # -------------------------------------------------------------- running
    @staticmethod
    def _current(m):
        return next((s for s in m["steps"] if s["status"] in ("pending", "waiting", "running", "failed")), None)

    def _loop(self):
        while True:
            self.wake.wait(1.0)
            self.wake.clear()
            try:
                with self.lock:
                    m = next((x for x in sorted(self.missions.values(), key=lambda x: x["created"])
                              if x["status"] in ("approved", "running")), None)
                if m:
                    self._advance(m)
            except Exception as e:  # never let the supervisor die
                import traceback

                traceback.print_exc()
                if m:
                    with self.lock:
                        m["status"] = "paused"
                        self._log(m, f"Internal error: {e}", "error")
                        self._save(m)

    def _advance(self, m):
        with self.lock:
            s = self._current(m)
            if s is None:
                m["status"] = "running"
                self._log(m, "All steps done; writing the report.")
                self._save(m)
            elif m["approve"] == "each" and not s.get("ok"):
                s["status"] = "waiting"
                m["status"] = "waiting"
                self._log(m, f"Waiting for your go-ahead on step {s['i']}: {s['label']}.")
                self._save(m)
                return
            else:
                s["status"] = "running"
                s["t0"] = time.time()
                m["status"] = "running"
                self._save(m)
        if s is None:
            rep = self._report(m)
            with self.lock:
                m["report"] = rep
                m["status"] = "done"
                self._save(m)
            return
        try:
            out = self._run_step(m, s)
            with self.lock:
                s.update(status="done", t1=time.time(), **out)
                self._log(m, f"Step {s['i']} done: {out.get('summary', '')}", "result")
                if m["status"] == "pausing":
                    m["status"] = "paused"
                elif m["status"] == "running":
                    m["status"] = "approved"
                self._save(m)
        except InterruptedError:
            with self.lock:
                s.update(status="pending" if m["status"] != "stopped" else "skipped", job=None)
                if m["status"] != "stopped":
                    m["status"] = "paused"
                self._save(m)
        except Exception as e:
            with self.lock:
                s.update(status="failed", error=str(e)[:1500], t1=time.time())
                m["status"] = "paused"
                self._log(m, f"Step {s['i']} failed: {e}", "error")
                self._save(m)
        self.wake.set()

    def _resolve(self, m, args):
        results = {s["i"]: s.get("out") or {} for s in m["steps"] if s["status"] == "done"}

        def sub(v):
            if isinstance(v, str):
                whole = REF.fullmatch(v)
                if whole:
                    n, k = int(whole.group(1)), whole.group(2)
                    if k not in results.get(n, {}):
                        raise ValueError(f"Step {n} has no {k} to use (did it run?).")
                    return results[n][k]
                def rep(mm):
                    n, k = int(mm.group(1)), mm.group(2)
                    if k not in results.get(n, {}):
                        raise ValueError(f"Step {n} has no {k} to use (did it run?).")
                    return str(results[n][k])
                v = REF.sub(rep, v)
                if "lab:latest:" in v or v == "latest":
                    runs = self._lab_runs()
                    if not runs:
                        return None
                    v = v.replace("lab:latest:", f"lab:{runs[0]}:") if v != "latest" else runs[0]
                return v
            if isinstance(v, list):
                return [x for x in (sub(y) for y in v) if x is not None]
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            return v
        return sub(args)

    def _lab_runs(self):
        if not self.lab.exists():
            return []
        return [d.name for d in sorted(self.lab.iterdir(), key=lambda d: d.name, reverse=True) if (d / "config.json").exists()]

    def _http(self, method, path, body):
        if not self.base_url:
            raise RuntimeError("The superagent does not know this server's address yet; reload the page.")
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Content-Type": "application/json", "X-Isomorph-Agent": "superagent"})
        pw = self.password()
        if pw:
            req.add_header("Authorization", "Basic " + base64.b64encode(f"superagent:{pw}".encode()).decode())
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("detail")
            except Exception:
                msg = str(e)
            raise RuntimeError(msg if isinstance(msg, str) else json.dumps(msg)) from None

    def _run_step(self, m, s):
        args = normalise(s["skill"], self._resolve(m, s["args"]))
        s["resolved"] = args
        (method, path, body), digest = SKILLS[s["skill"]]["build"](args)
        r = self._http(method, path, body)
        if isinstance(r, dict) and "job" in r and len(r) <= 2:
            s["job"] = r["job"]
            with self.lock:
                self._save(m)
            job = self.jobs.jobs.get(r["job"])
            while job and job.state in ("queued", "running"):
                if m["status"] == "stopped":
                    self.jobs.stop(job.id)
                    raise InterruptedError()
                s["progress"], s["message"] = job.progress, job.message
                time.sleep(0.7)
            if job is None:
                raise RuntimeError("The job disappeared (did the server restart?).")
            if job.state != "done":
                if m["status"] == "stopped":
                    raise InterruptedError()
                raise RuntimeError(job.error or f"The job {job.state}.")
            r = job.result
        elif isinstance(r, dict) and "result" in r and len(r) == 1:
            r = r["result"]
        d = digest(r if r is not None else {})
        return {"summary": d.get("summary", ""), "out": d.get("out") or {}, "open": d.get("open"), "progress": 1.0, "message": None}

    # -------------------------------------------------------------- the board
    def tune_adapters(self):
        try:
            from ..tune.lora import list_adapters

            return list_adapters()
        except Exception:
            return []

    def board(self):
        now = time.time()
        js = sorted(self.jobs.jobs.values(), key=lambda j: -j.created)
        jobs = [j.public(with_result=False) for j in js if j.state in ("queued", "running")] + \
               [j.public(with_result=False) for j in js if j.state not in ("queued", "running")][:6]
        ap = {}
        if self.autopilot is not None:
            try:
                st = self.autopilot.state()
                ap = {"status": st.get("status"), "queued": len(st.get("queue", [])), "done": len(st.get("done", [])),
                      "log": [x.get("msg") for x in st.get("log", [])[-3:]], "report": bool(st.get("report")),
                      "current": (st.get("current") or (st.get("queue") or [{}])[0]).get("type") if (st.get("current") or st.get("queue")) else None}
            except Exception as e:
                ap = {"status": "unknown", "error": str(e)}
        sessions = []
        if self.tune is not None:
            for x in self.tune.listing():
                if x.get("live"):
                    sessions.append({"id": x["id"], "name": x.get("name"), "spec": x.get("spec"), "step": x.get("step"), "source": x.get("source"),
                                     "idle_min": round((now - (x.get("last") or now)) / 60, 1)})
        claude = []
        wsp = self.data / "wsp"
        for f in sorted(wsp.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:6] if wsp.exists() else []:
            r = self._read(f) or {}
            if r.get("kind") in ("read", "fix", "write"):
                g = r.get("grade") or {}
                claude.append({"id": r.get("id"), "kind": r["kind"], "task": r.get("task"), "spec": r.get("spec"), "created": r.get("created"),
                               "line": ("correct" if g.get("correct") else "not correct" if g.get("correct") is False else "") if r["kind"] == "read" else
                               (f"{r['agreement'] * 100:.0f}% agreement" if r.get("agreement") is not None else "") if r["kind"] == "write" else
                               (f"repaired {g.get('repaired_share', 0) * 100:.0f}%" if r.get("task") == "planted" else "")})
        missions = [self.public(x, brief=True) for x in sorted(self.missions.values(), key=lambda x: -x["created"])[:12]]
        alerts = []

        def alert(aid, level, text, action=None):
            if aid not in self.dismissed:
                alerts.append({"id": aid, "level": level, "text": text, "action": action})
        for j in js[:20]:
            if j.state == "error" and j.finished and now - j.finished < 86400:
                alert(f"job-{j.id}", "error", f"{j.title} failed: {(j.error or '')[:240]}", {"label": "Ask why", "ask": f"The job '{j.title}' failed with: {j.error}. Why, and what should I do?"})
            if j.state == "running" and now - j.created > LONG_JOB_MIN * 60:
                alert(f"long-{j.id}", "warn", f"{j.title} has run for {int((now - j.created) / 60)} minutes.",
                      {"label": "Stop it", "skill": "job_stop", "args": {"id": j.id}})
        for s in sessions:
            if s["idle_min"] > IDLE_SESSION_MIN:
                alert(f"idle-{s['id']}", "warn", f"Tune session {s['name']} has been idle {int(s['idle_min'])} minutes and still holds its model in memory.",
                      {"label": "Close it", "skill": "tune_close", "args": {"id": s["id"]}})
        if ap.get("status") in ("done", "finished") and ap.get("report"):
            alert("autopilot-report", "info", "Autopilot finished its campaign; the report is ready.", {"label": "Open the Engine", "open": {"type": "go", "view": "engine"}})
        if ap.get("status") == "error":
            alert("autopilot-error", "error", "Autopilot stopped with an error.", {"label": "Open the Engine", "open": {"type": "go", "view": "engine"}})
        for x in self.missions.values():
            if x["status"] == "waiting":
                alert(f"wait-{x['id']}-{len(x['log'])}", "info", f"Mission '{x['title']}' is waiting for your go-ahead.", {"label": "Review", "mission": x["id"]})
            if x["status"] == "paused" and any(s["status"] == "failed" for s in x["steps"]):
                alert(f"fail-{x['id']}-{len(x['log'])}", "error", f"Mission '{x['title']}' paused: a step failed.", {"label": "Review", "mission": x["id"]})
        return {"jobs": jobs, "autopilot": ap, "sessions": sessions, "claude": claude, "missions": missions, "alerts": alerts}

    def dismiss(self, aid):
        self.dismissed.add(aid)
        (self.dir / "dismissed.json").write_text(json.dumps(sorted(self.dismissed)[-500:]))

    def public(self, m, brief=False):
        if brief:
            done = sum(1 for s in m["steps"] if s["status"] in ("done", "skipped"))
            cur = self._current(m)
            return {"id": m["id"], "title": m["title"], "status": m["status"], "source": m["source"], "done": done, "of": len(m["steps"]),
                    "current": cur["label"] if cur and m["status"] not in FINAL else None, "created": m["created"], "updated": m.get("updated")}
        out = json.loads(json.dumps(clean(m)))
        cur = self._current(m)
        if cur and cur.get("job"):
            j = self.jobs.jobs.get(cur["job"])
            if j:
                for s in out["steps"]:
                    if s["i"] == cur["i"]:
                        s["progress"], s["message"] = j.progress, j.message
        return out
