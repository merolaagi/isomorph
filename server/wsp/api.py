"""HTTP routes for weight-space programming (/api/wsp/...)."""
import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..jsonsafe import clean
from .agent import AGENT_MODEL
from .finetune import PLANS
from .tasks import FIX_TASKS, READ_TASKS


class PlanReq(BaseModel):
    spec: str
    examples: str
    held_out: str = ""
    budget_frac: float = 0.01
    steps: int = 150
    lr: float = 2e-3
    seeds: int = 2


class ReadReq(BaseModel):
    task: str
    spec: str
    question: str = ""
    forward_budget: int | None = None
    model: str | None = None
    max_turns: int = 40


class FixReq(BaseModel):
    task: str
    spec: str
    goal: str = ""
    examples: str = ""
    held_out: str = ""
    forward_budget: int = 25
    model: str | None = None
    max_turns: int = 40


def _summary(r):
    g = r.get("grade") or {}
    s = {"id": r["id"], "kind": r["kind"], "task": r.get("task"), "spec": r["spec"], "created": r.get("created"),
         "seconds": r.get("seconds")}
    if r["kind"] == "plan":
        s["line"] = r.get("verdict", "")[:160]
    elif r["kind"] == "read":
        s["line"] = ("correct" if g.get("correct") else "not correct" if g.get("correct") is False else "not graded") + (
            f", score {g['score']:.2f}" if isinstance(g.get("score"), (int, float)) else "")
    else:
        if r.get("task") == "planted" and isinstance(g.get("repaired_share"), (int, float)):
            s["line"] = f"repaired {g['repaired_share'] * 100:.0f}% of the damage" + ("" if r.get("kept") else " (no edit kept)")
        else:
            s["line"] = (f"held-out loss {r['base']['held']['loss']:.4f} → {r['edited']['held']['loss']:.4f}" if r.get("base") else "")
    return s


def make_router(jobs, data_dir: Path, lab_dir: Path, resolve):
    """resolve(spec, progress) -> a spec the loaders understand (downloads links, checks folders)."""
    from . import imports

    WSP = data_dir / "wsp"
    WSP.mkdir(parents=True, exist_ok=True)
    imports.set_dir(WSP / "imports")
    r = APIRouter()

    def key():
        from ..engine.narrator import get_key

        return get_key(data_dir / "settings.json")

    def save(res):
        res["id"] = res.get("id") or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        res = clean(res)
        (WSP / f"{res['id']}.json").write_text(json.dumps(res, allow_nan=False, default=str))
        return res

    def spec_of(s, progress):
        s = s.strip()
        if not s:
            raise ValueError("Choose a model.")
        return s if s.startswith(("lab:", "import:")) else resolve(s, progress)

    @r.get("/api/wsp/meta")
    def meta():
        return {"read_tasks": READ_TASKS, "fix_tasks": FIX_TASKS, "plans": PLANS, "has_key": bool(key()), "agent_model": AGENT_MODEL,
                "imports": imports.list_imports()[:30]}

    @r.post("/api/wsp/plan")
    def plan(req: PlanReq):
        from .finetune import plan_report

        if not req.examples.strip():
            raise HTTPException(400, "Paste some examples: one per line, 'prompt => target' or plain text.")
        if not (1 <= req.steps <= 3000 and 1 <= req.seeds <= 5 and 0 < req.budget_frac <= 0.2):
            raise HTTPException(400, "Steps 1 to 3,000, seeds 1 to 5, budget up to 20% of the tunable weights.")

        def work(update, stopped):
            spec = spec_of(req.spec, update)
            res = plan_report(spec, req.examples, req.held_out, req.budget_frac, req.steps, req.lr, req.seeds,
                              progress=update, should_stop=stopped)
            return save(res)
        return {"job": jobs.submit("wsp-plan", f"Fine-tune plan for {req.spec}", work).id}

    def agent_call(model):
        from .agent import api_call

        k = key()
        if not k:
            raise HTTPException(400, "Reading and fixing weights needs Claude. Save an Anthropic API key under Engine first.")
        return api_call(k, model or AGENT_MODEL)

    @r.post("/api/wsp/read")
    def read(req: ReadReq):
        from .tasks import run_read

        if req.task not in READ_TASKS:
            raise HTTPException(400, f"Unknown task {req.task}.")
        call = agent_call(req.model)

        def work(update, stopped):
            spec = spec_of(req.spec, update)
            res = run_read(req.task, spec, call, req.question, req.forward_budget, lab_dir, max_turns=req.max_turns,
                           progress=update, should_stop=stopped)
            res["agent_model"] = req.model or AGENT_MODEL
            return save(res)
        return {"job": jobs.submit("wsp-read", f"Read the weights of {req.spec}", work).id}

    @r.post("/api/wsp/fix")
    def fix(req: FixReq):
        from .tasks import run_fix

        if req.task not in FIX_TASKS:
            raise HTTPException(400, f"Unknown task {req.task}.")
        if not 1 <= req.forward_budget <= 200:
            raise HTTPException(400, "Forward-pass budget 1 to 200.")
        call = agent_call(req.model)
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

        def work(update, stopped):
            spec = spec_of(req.spec, update)
            res = run_fix(req.task, spec, call, req.goal, req.examples, req.held_out, req.forward_budget,
                          max_turns=req.max_turns, progress=update, should_stop=stopped, export_dir=WSP / rid)
            res["id"], res["agent_model"] = rid, req.model or AGENT_MODEL
            return save(res)
        return {"job": jobs.submit("wsp-fix", f"Fix the weights of {req.spec}", work).id}

    @r.post("/api/wsp/import")
    async def import_weights(request: Request, name: str = "", about: str = "", source: str = ""):
        """Raw safetensors bytes in the body; name, about (architecture in words) and source in the query."""
        body = await request.body()
        try:
            return imports.save_import(body, name, about, source)
        except Exception as e:
            raise HTTPException(400, f"Could not read the weights: {e}")

    @r.get("/api/wsp/imports")
    def list_imported():
        return imports.list_imports()

    @r.get("/api/wsp/history")
    def history():
        out = []
        for f in sorted(WSP.glob("*.json"), reverse=True)[:100]:
            try:
                out.append(_summary(json.loads(f.read_text())))
            except Exception:
                continue
        return out

    def _rid(rid):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", rid or ""):
            raise HTTPException(400, "Invalid id.")
        return rid

    @r.get("/api/wsp/run/{rid}")
    def one(rid: str):
        f = WSP / f"{_rid(rid)}.json"
        if not f.exists():
            raise HTTPException(404, "No such run.")
        return json.loads(f.read_text())

    @r.get("/api/wsp/run/{rid}/edited.safetensors")
    def edited(rid: str):
        f = WSP / _rid(rid) / "edited_tensors.safetensors"
        if not f.exists():
            raise HTTPException(404, "This run kept no edit to export.")
        return FileResponse(f, filename=f"isomorph-edit-{rid}.safetensors")

    @r.delete("/api/wsp/run/{rid}")
    def delete(rid: str):
        f = WSP / f"{_rid(rid)}.json"
        if f.exists():
            f.unlink()
        d = WSP / rid
        if d.is_dir():
            for x in d.iterdir():
                x.unlink()
            d.rmdir()
        return {"deleted": rid}

    return r
