"""HTTP routes for the superagent (/api/superagent/...)."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jsonsafe import clean
from . import playbooks
from .skills import SKILLS, catalog


class PlanReq(BaseModel):
    goal: str = ""
    playbook: str = ""
    check: str = ""
    view: str = ""
    approve: str = "plan"


class ActReq(BaseModel):
    step: int | None = None


class SkillReq(BaseModel):
    skill: str
    args: dict = {}
    why: str = ""


class DismissReq(BaseModel):
    id: str


SUPERVISE = {"job_stop", "tune_close", "autopilot_control"}


def make_router(sa):
    r = APIRouter()

    def err(e):
        if isinstance(e, KeyError):
            raise HTTPException(404, str(e).strip("'\""))
        if isinstance(e, (ValueError, RuntimeError)):
            raise HTTPException(400, str(e))
        raise e

    @r.get("/api/superagent")
    def overview(view: str = ""):
        has_key = bool(sa.key())
        return clean({"board": sa.board(), "playbooks": playbooks.for_view(view, has_key), "has_key": has_key,
                      "skills": [{"name": s["name"], "label": s["label"], "area": s["area"], "cost": s["cost"]} for s in catalog(has_key)],
                      "checks": {p.get("check"): pid for pid, p in playbooks.PLAYBOOKS.items() if p.get("check")}})

    @r.get("/api/superagent/skills")
    def skills():
        return catalog(bool(sa.key()))

    @r.post("/api/superagent/plan")
    def plan(req: PlanReq):
        approve = "each" if req.approve == "each" else "plan"
        try:
            if req.playbook:
                m = sa.from_playbook(req.playbook, approve)
            elif req.check:
                pid = playbooks.by_check(req.check)
                if not pid:
                    raise ValueError("The superagent has no playbook for this step yet; use Go instead.")
                m = sa.from_playbook(pid, approve)
            else:
                m = sa.plan(req.goal, req.view, approve)
        except Exception as e:
            err(e)
        return sa.public(m)

    @r.post("/api/superagent/skill")
    def one(req: SkillReq):
        """A single skill the user pressed a button for (board controls, alert actions)."""
        if req.skill not in SKILLS:
            raise HTTPException(400, "Unknown skill.")
        try:
            return sa.public(sa.single(req.skill, req.args, req.why))
        except Exception as e:
            err(e)

    @r.get("/api/superagent/mission/{mid}")
    def mission(mid: str):
        m = sa.missions.get(mid)
        if not m:
            raise HTTPException(404, "No such mission.")
        return sa.public(m)

    @r.post("/api/superagent/mission/{mid}/{action}")
    def act(mid: str, action: str, req: ActReq | None = None):
        try:
            return sa.public(sa.act(mid, action, (req.step if req else None)))
        except Exception as e:
            err(e)

    @r.delete("/api/superagent/mission/{mid}")
    def delete(mid: str):
        try:
            sa.delete(mid)
        except Exception as e:
            err(e)
        return {"ok": True}

    @r.post("/api/superagent/dismiss")
    def dismiss(req: DismissReq):
        sa.dismiss(req.id[:120])
        return {"ok": True}

    return r
