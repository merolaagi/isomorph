"""Autopilot: run the whole research loop unattended.

A campaign is a queue of steps: profile models, mine rules, test the most
informative levers, build a blueprint from what held, write the world view.
Each step runs as its own job in the shared queue, so manual work can slot in
between steps. State is saved after every step; if the server restarts
mid-campaign, the interrupted step is re-queued and the campaign continues.
While a campaign runs on macOS, `caffeinate` keeps the machine from idling
to sleep.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CURATED = [
    {"spec": "EleutherAI/pythia-14m", "note": "Smallest Pythia", "default": True},
    {"spec": "EleutherAI/pythia-70m", "note": "Pythia 70M", "default": True},
    {"spec": "EleutherAI/pythia-70m-seed1", "note": "Same recipe, another seed", "default": True},
    {"spec": "EleutherAI/pythia-160m", "note": "Bigger sibling", "default": True},
    {"spec": "openai-community/gpt2", "note": "GPT-2 small, a different family", "default": True},
    {"spec": "distilbert/distilgpt2", "note": "Distilled from GPT-2", "default": True},
    {"spec": "HuggingFaceTB/SmolLM2-135M", "note": "Modern small model, Llama-style", "default": True},
    {"spec": "EleutherAI/pythia-410m", "note": "Larger; slower to profile", "default": False},
]
TRAJECTORY = ["EleutherAI/pythia-70m@step1000", "EleutherAI/pythia-70m@step8000", "EleutherAI/pythia-70m@step33000"]
LEVER_TASKS = {"share": ["sort", "tinystories"], "topk": ["sort", "tinystories"], "rank": ["sort", "tinystories"], "seed": ["modadd"]}
DEFAULT_ARGS = {"share": {"n": 2}, "topk": {"k": 0.25}, "rank": {"frac": 0.25}, "seed": {}}

_lock = threading.RLock()
_caffeinate = None


class Autopilot:
    def __init__(self, data: Path, jobs, dirs):
        self.data, self.jobs, self.dirs = data, jobs, dirs
        self.path = data / "engine" / "autopilot.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ state
    def state(self):
        with _lock:
            if self.path.exists():
                try:
                    return json.loads(self.path.read_text())
                except Exception:
                    pass
            return {"status": "idle", "queue": [], "done": [], "log": [], "report": None, "reports": []}

    def _save(self, s):
        from ..jsonsafe import clean

        with _lock:
            self.path.write_text(json.dumps(clean(s)))

    def _log(self, s, msg):
        s["log"] = (s.get("log", []) + [{"t": time.time(), "msg": msg}])[-200:]

    def public(self):
        s = self.state()
        from ..atlas.library import list_profiles

        have = {m["spec"] for m in list_profiles(self.dirs["atlas"])}
        s["curated"] = [c | {"profiled": c["spec"] in have} for c in CURATED]
        s["trajectory"] = [{"spec": t, "profiled": t in have} for t in TRAJECTORY]
        s["levers_tasks"] = LEVER_TASKS
        return s

    # ------------------------------------------------------------ control
    def start(self, cfg):
        from ..atlas.library import list_profiles
        from .ledger import load

        s = self.state()
        if s["status"] in ("running", "paused"):
            raise ValueError("A campaign is already in progress. Stop it first.")
        have = {m["spec"] for m in list_profiles(self.dirs["atlas"])}
        models = [m.strip() for m in cfg.get("models", []) if m.strip()]
        if cfg.get("trajectory"):
            models += [t for t in TRAJECTORY if t not in models]
        queue = [{"type": "profile", "spec": m} for m in models if m not in have or cfg.get("reprofile")]
        queue.append({"type": "mine"})
        queue.append({"type": "plan"})
        L = load(self.dirs["engine"])
        snap = {"rules": {k: r.get("status") for k, r in L["rules"].items()}, "experiments": list(L["experiments"]),
                "blueprints": list(L["blueprints"])}
        s = {"status": "running", "config": cfg, "started": time.time(), "deadline": time.time() + float(cfg.get("hours", 8)) * 3600,
             "queue": queue, "done": [], "current": None, "log": [], "snapshot": snap, "report": None,
             "reports": s.get("reports", [])[:10], "skipped_profiles": [m for m in models if m in have and not cfg.get("reprofile")]}
        self._log(s, f"Campaign started: {len(queue)} steps queued, budget {cfg.get('hours', 8)} h.")
        self._save(s)
        self._awake(True)
        self._advance()
        return self.public()

    def pause(self):
        s = self.state()
        if s["status"] == "running":
            s["status"] = "paused"
            self._log(s, "Paused. The current step will finish first.")
            self._save(s)
            self._awake(False)
        return self.public()

    def resume(self):
        s = self.state()
        if s["status"] == "paused":
            s["status"] = "running"
            self._log(s, "Resumed.")
            self._save(s)
            self._awake(True)
            if not s.get("current"):
                self._advance()
        return self.public()

    def stop(self):
        s = self.state()
        if s["status"] in ("running", "paused"):
            s["status"] = "stopping" if s.get("current") else "stopped"
            s["queue"] = []
            self._log(s, "Stopped by you. Writing the report.")
            self._save(s)
            if not s.get("current"):
                self._finish("stopped")
        return self.public()

    def boot(self):
        """Called at server start: resume a campaign that was running."""
        s = self.state()
        if s["status"] in ("running", "stopping"):
            if s.get("current"):
                s["queue"] = [s["current"]] + s["queue"]
                self._log(s, f"Server restarted during '{self._label(s['current'])}'; running it again.")
                s["current"] = None
            if s["status"] == "stopping":
                s["status"] = "stopped"
                self._save(s)
                self._finish("stopped")
                return
            self._save(s)
            self._awake(True)
            self._advance()

    # ------------------------------------------------------------ execution
    @staticmethod
    def _label(step):
        t = step["type"]
        if t == "profile":
            return f"Profile {step['spec']}"
        if t == "experiment":
            return f"Test {step['lever']} on {step['task']}" + (f" (rule {step['rule']})" if step.get("rule") else "")
        return {"mine": "Mine rules", "plan": "Plan experiments", "blueprint": "Build blueprint", "narrate": "Write world view"}.get(t, t)

    def _advance(self):
        s = self.state()
        if s["status"] != "running" or s.get("current"):
            return
        if not s["queue"]:
            return self._finish("done")
        if time.time() > s["deadline"]:
            tail = [q for q in s["queue"] if q["type"] in ("mine", "blueprint", "narrate")]
            if len(tail) < len(s["queue"]):
                self._log(s, "Time budget reached; skipping remaining tests and wrapping up.")
                s["queue"] = tail
            if not s["queue"]:
                self._save(s)
                return self._finish("done")
        step = s["queue"].pop(0)
        s["current"] = step | {"started": time.time()}
        self._save(s)
        self.jobs.submit("autopilot", f"Autopilot: {self._label(step)}", lambda update, stopped, step=step: self._run(step, update, stopped))

    def _run(self, step, update, stopped):
        ok, info = True, ""
        try:
            info = self._do(step, update, stopped)
        except InterruptedError:
            ok, info = False, "stopped"
        except Exception as e:  # recorded, the campaign continues
            ok, info = False, f"{type(e).__name__}: {e}"
        s = self.state()
        s["done"].append(step | {"finished": time.time(), "ok": ok, "info": info, "started": (s.get("current") or {}).get("started")})
        s["current"] = None
        self._log(s, f"{'Done' if ok else 'Failed'}: {self._label(step)}" + (f". {info}" if info else ""))
        self._save(s)
        if s["status"] == "stopping":
            self._finish("stopped")
        else:
            self._advance()
        return {"ok": ok, "info": info}

    def _do(self, step, update, stopped):
        from ..atlas.profile import profile_model
        from ..hub import catalog
        from .ledger import build_blueprint, load, mine, run_lever, save

        d = self.dirs
        t = step["type"]
        if t == "profile":
            resolved = catalog.resolve(step["spec"], self.data, update)
            meta = profile_model(step["spec"], resolved, d["atlas"], update)
            catalog.recent_add(self.data, step["spec"])
            n = meta["params"]
            size = f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"
            return f"{meta['layers']} layers, {size} parameters, profiled in {meta['seconds']} s"
        if t == "mine":
            L = mine(d["engine"], d["atlas"], d["lab"])
            return f"{len(L['rules'])} rules"
        if t == "plan":
            return self._plan()
        if t == "experiment":
            e = run_lever(d["engine"], self.data, step["task"], step["lever"], step.get("args", {}), step.get("seeds", [1, 2, 3]),
                          step.get("steps"), step.get("rule"), update, stopped)
            return f"{e['verdict']}: {e['reason']}"
        if t == "blueprint":
            L = mine(d["engine"], d["atlas"], d["lab"])
            held = {}
            for r in L["rules"].values():
                if r["kind"] == "lever" and r.get("status") == "holds" and r.get("lever"):
                    lv = r["lever"]["lever"]
                    held.setdefault(lv, {k: v for k, v in r["lever"].items() if k != "lever"})
            if not held:
                return "No lever has held yet, so there is nothing to combine."
            cfg = self.state().get("config", {})
            tasks = ["sort", "copy", "modadd"] + (["tinystories"] if cfg.get("tinystories", True) else [])
            bp = build_blueprint(d["engine"], self.data, held, tasks, [1, 2], float(cfg.get("scale", 1.0)), update, stopped)
            return f"{bp['id']} with {', '.join(held)}"
        if t == "narrate":
            from .narrator import get_key, narrate

            L = mine(d["engine"], d["atlas"], d["lab"])
            cfg = self.state().get("config", {})
            key = get_key(self.data / "settings.json") if cfg.get("claude", True) else None
            try:
                n = narrate(L, key)
            except Exception as e:
                n = narrate(L, None)
                n["note"] = f"Claude failed ({e}); built-in digest used instead."
            L = load(d["engine"])
            L["narratives"] = ([n] + L.get("narratives", []))[:10]
            save(d["engine"], L)
            return f"written by {n['source']}"
        raise ValueError(f"Unknown step {t}")

    def _plan(self):
        """Choose the experiments that should teach the most, within the cap."""
        from .ledger import load

        s = self.state()
        cfg = s.get("config", {})
        cap = int(cfg.get("max_experiments", 8))
        seeds = cfg.get("seeds", [1, 2, 3])
        scale = float(cfg.get("scale", 1.0))
        from .nano import DEFAULT_OPT
        tasks_ok = set(["sort", "copy", "modadd"] + (["tinystories"] if cfg.get("tinystories", True) else []))
        L = load(self.dirs["engine"])
        tested = {(e["lever"], e["task"]) for e in L["experiments"].values()}
        cands = []
        kind_lever = {"redundancy": "share", "sparsity": "topk", "low-rank": "rank", "lab": "seed"}
        for r in L["rules"].values():
            lv = (r.get("lever") or {}).get("lever") or kind_lever.get(r["kind"])
            if r["kind"] in ("absence", "lever") or not lv or lv not in LEVER_TASKS:
                continue
            if r.get("status") in ("holds", "rejected"):
                continue
            args = {k: v for k, v in (r.get("lever") or {}).items() if k != "lever"} or DEFAULT_ARGS[lv]
            conf_gain = {"anecdote": 3, "weak": 2, "moderate": 1, "strong": 0}.get(r.get("confidence"), 1)
            for task in LEVER_TASKS[lv]:
                if task not in tasks_ok:
                    continue
                cost = 4 if task == "tinystories" else 1
                novelty = 2 if (lv, task) not in tested else 0
                cands.append({"score": (conf_gain + novelty + abs(r.get("effect") or 0)) / cost, "type": "experiment",
                              "task": task, "lever": lv, "args": args, "rule": r["id"], "seeds": seeds,
                              "steps": int(DEFAULT_OPT[task]["steps"] * scale)})
        # Coverage: every lever on every task at least once, even without a rule pointing at it.
        for lv, tasks in LEVER_TASKS.items():
            for task in tasks:
                if task in tasks_ok and (lv, task) not in tested:
                    cands.append({"score": 1.5 / (4 if task == "tinystories" else 1), "type": "experiment", "task": task, "lever": lv,
                                  "args": DEFAULT_ARGS[lv], "rule": None, "seeds": seeds, "steps": int(DEFAULT_OPT[task]["steps"] * scale)})
        seen, plan = set(), []
        for c in sorted(cands, key=lambda c: -c["score"]):
            key = (c["lever"], c["task"])
            if key in seen:
                continue
            seen.add(key)
            plan.append({k: v for k, v in c.items() if k != "score"})
            if len(plan) >= cap:
                break
        s = self.state()
        s["queue"] = plan + [{"type": "mine"}] + ([{"type": "blueprint"}] if cfg.get("blueprint", True) else []) + [{"type": "narrate"}] + s["queue"]
        self._save(s)
        return f"{len(plan)} experiments chosen from {len(cands)} candidates"

    # ------------------------------------------------------------ report
    def _finish(self, how):
        from .ledger import load

        s = self.state()
        L = load(self.dirs["engine"])
        snap = s.get("snapshot", {"rules": {}, "experiments": [], "blueprints": []})
        new_rules = [{"id": k, "statement": r["statement"], "status": r.get("status")} for k, r in L["rules"].items() if k not in snap["rules"]]
        changed = [{"id": k, "statement": r["statement"], "before": snap["rules"][k], "after": r.get("status")}
                   for k, r in L["rules"].items() if k in snap["rules"] and snap["rules"][k] != r.get("status")]
        exps = [{"id": e["id"], "label": f"{e['lever_label']} on {e['task_label']}", "verdict": e["verdict"], "reason": e["reason"]}
                for e in L["experiments"].values() if e["id"] not in snap["experiments"]]
        bps = [{"id": b["id"], "results": [{"task": r["task_label"], "cpc": r.get("capability_per_compute"), "params": [r["params_bp"], r["params_base"]]} for r in b["results"]]}
               for b in L["blueprints"].values() if b["id"] not in snap["blueprints"]]
        failed = [{"step": self._label(d_), "info": d_["info"]} for d_ in s["done"] if not d_["ok"]]
        profiled = [d_["spec"] for d_ in s["done"] if d_["type"] == "profile" and d_["ok"]]
        rep = {"finished": time.time(), "started": s.get("started"), "how": how, "hours": round((time.time() - (s.get("started") or time.time())) / 3600, 2),
               "profiled": profiled, "skipped_profiles": s.get("skipped_profiles", []), "new_rules": new_rules, "changed": changed,
               "experiments": exps, "blueprints": bps, "failed": failed,
               "holds": sum(1 for e in exps if e["verdict"] == "holds"), "rejected": sum(1 for e in exps if e["verdict"] == "rejected"),
               "world_view": (L.get("narratives") or [{}])[0].get("id") if L.get("narratives") else None}
        s["status"] = how
        s["report"] = rep
        s["reports"] = ([rep] + s.get("reports", []))[:10]
        s["current"] = None
        self._log(s, f"Campaign {how}. Report ready.")
        self._save(s)
        self._awake(False)

    # ------------------------------------------------------------ keep the Mac awake
    def _awake(self, on):
        global _caffeinate
        if sys.platform != "darwin":
            return
        if on and (_caffeinate is None or _caffeinate.poll() is not None):
            try:
                _caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except Exception:
                _caffeinate = None
        elif not on and _caffeinate is not None:
            _caffeinate.terminate()
            _caffeinate = None
