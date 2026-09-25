"""Isomorph server: a local research bench for comparing neural networks."""
import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .jobs import JobQueue
from .jsonsafe import clean
from .hub import catalog

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ISOMORPH_DATA", ROOT / "data"))
LAB = DATA / "lab"
HUB = DATA / "hub"
TRACE = DATA / "trace"
ATLAS = DATA / "atlas"
ENGINE = DATA / "engine"
for d in (LAB, HUB, TRACE, ATLAS, ENGINE):
    d.mkdir(parents=True, exist_ok=True)
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "dev"

catalog.load_token(DATA)
app = FastAPI(title="Isomorph", version=VERSION)


def _password():
    pw = os.environ.get("ISOMORPH_PASSWORD")
    if pw:
        return pw
    sp = DATA / "settings.json"
    if sp.exists():
        try:
            return json.loads(sp.read_text()).get("password")
        except Exception:
            return None
    return None


@app.middleware("http")
async def _gate(request: Request, call_next):
    """Optional password gate (HTTP Basic). Off unless a password is set, so
    local use is unchanged; required when the app is exposed on the internet."""
    pw = _password()
    if pw:
        import base64
        import secrets

        ok = False
        h = request.headers.get("authorization", "")
        if h.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(h[6:]).decode().partition(":")
                ok = secrets.compare_digest(given.encode(), pw.encode())
            except Exception:
                ok = False
        if not ok:
            return JSONResponse({"detail": "Sign in to use Isomorph."}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="Isomorph", charset="UTF-8"'})
    return await call_next(request)
jobs = JobQueue()


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # Show the real cause in the UI instead of a bare "500".
    import traceback

    traceback.print_exc()
    return JSONResponse({"detail": f"Server error in {request.url.path}: {type(exc).__name__}: {exc}"}, status_code=500)


def _read(path):
    return clean(json.loads(Path(path).read_text()))


def _write(path, obj):
    obj = clean(obj)
    Path(path).write_text(json.dumps(obj, allow_nan=False))
    return obj


def _safe_id(s):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", s or ""):
        raise HTTPException(400, "Invalid id.")
    return s


# ------------------------------------------------------------------ general
@app.get("/api/health")
def health():
    from .hub.load import device

    return {"version": VERSION, "protected": bool(_password()), "torch": torch.__version__, "device": device(), "python": platform.python_version(),
            "threads": torch.get_num_threads(), "data": str(DATA)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No such job. The server may have restarted.")
    return j.public()


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str):
    j = jobs.stop(job_id)
    if not j:
        raise HTTPException(404, "No such job.")
    return j.public(with_result=False)


@app.get("/api/jobs")
def job_list():
    return [j.public(with_result=False) for j in sorted(jobs.jobs.values(), key=lambda j: -j.created)][:30]


# ------------------------------------------------------------------ lab
class TrainReq(BaseModel):
    p: int = Field(53, ge=7, le=127)
    seeds: list[int] = [1, 2, 3, 4]
    steps: int = Field(4000, ge=100, le=50000)
    train_frac: float = Field(0.4, gt=0.05, lt=0.95)
    d_model: int = Field(128, ge=16, le=512)
    n_heads: int = Field(4, ge=1, le=16)
    d_mlp: int = Field(512, ge=16, le=4096)
    lr: float = 1e-3
    wd: float = 1.0
    name: str = ""


@app.post("/api/lab/train")
def lab_train(req: TrainReq):
    from .lab.train import train_family

    if req.d_model % req.n_heads:
        raise HTTPException(400, "Model width must be divisible by the number of heads.")
    seeds = sorted(set(req.seeds))
    if not 2 <= len(seeds) <= 12:
        raise HTTPException(400, "Train between 2 and 12 seeds so there is something to compare.")
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-p{req.p}"
    cfg = req.model_dump() | {"seeds": seeds, "run_id": run_id, "created": time.time()}

    def work(update, stopped):
        train_family(LAB / run_id, cfg, update, stopped)
        return {"run_id": run_id}

    j = jobs.submit("train", f"Train {len(seeds)} seeds on (a + b) mod {req.p}", work)
    return {"job": j.id, "run_id": run_id}


@app.get("/api/lab/runs")
def lab_runs():
    out = []
    for d in sorted(LAB.iterdir(), reverse=True):
        cp = d / "config.json"
        if not cp.exists():
            continue
        cfg = json.loads(cp.read_text())
        seeds = sorted(int(f.stem.split("_")[1]) for f in d.glob("seed_*.pt"))
        out.append({"run_id": d.name, "config": cfg, "trained_seeds": seeds,
                    "complete": len(seeds) == len(cfg.get("seeds", [])), "has_report": (d / "family.json").exists()})
    return out


@app.delete("/api/lab/runs/{run_id}")
def lab_delete(run_id: str):
    d = LAB / _safe_id(run_id)
    if not d.exists():
        raise HTTPException(404, "No such run.")
    shutil.rmtree(d)
    return {"deleted": run_id}


def _run(run_id):
    from .lab.analyze import Run

    d = LAB / _safe_id(run_id)
    if not (d / "config.json").exists():
        raise HTTPException(404, "No such run.")
    if len(list(d.glob("seed_*.pt"))) < 2:
        raise HTTPException(400, "This run has fewer than two trained models.")
    return Run(d)


class FamilyReq(BaseModel):
    run_id: str
    refresh: bool = False


@app.post("/api/lab/family")
def lab_family(req: FamilyReq):
    from .lab.analyze import family_report

    d = LAB / _safe_id(req.run_id)
    cache = d / "family.json"
    if cache.exists() and not req.refresh:
        return {"result": _read(cache)}

    def work(update, stopped):
        update(0.1, "Loading models")
        return _write(cache, family_report(_run(req.run_id)))

    return {"job": jobs.submit("family", f"Survey run {req.run_id}", work).id}


class PairReq(BaseModel):
    run_id: str
    a: int
    b: int
    refresh: bool = False


@app.post("/api/lab/pair")
def lab_pair(req: PairReq):
    from .lab.analyze import pair_report

    if req.a == req.b:
        raise HTTPException(400, "Pick two different seeds.")
    d = LAB / _safe_id(req.run_id)
    cache = d / f"pair_{req.a}_{req.b}.json"
    if cache.exists() and not req.refresh:
        return {"result": _read(cache)}

    def work(update, stopped):
        update(0.1, "Aligning model B to model A")
        run = _run(req.run_id)
        if req.a not in run.models or req.b not in run.models:
            raise ValueError("Both seeds must be trained in this run.")
        return _write(cache, pair_report(run, req.a, req.b))

    return {"job": jobs.submit("pair", f"Compare seed {req.a} with seed {req.b}", work).id}


class SeededReq(BaseModel):
    run_id: str
    conditions: list[str] = ["random", "fourier", "transplant", "shuffled"]
    seeds: list[int] = [21, 22, 23]
    max_steps: int = Field(6000, ge=100, le=60000)
    target: float = Field(0.95, gt=0.1, le=1.0)
    donor_seed: int | None = None


@app.post("/api/lab/seeded")
def lab_seeded(req: SeededReq):
    from .lab.seeded import CONDITIONS, seeded_experiment

    d = LAB / _safe_id(req.run_id)
    if not (d / "config.json").exists():
        raise HTTPException(404, "No such run.")
    bad = [c for c in req.conditions if c not in CONDITIONS]
    if bad or not req.conditions:
        raise HTTPException(400, f"Unknown condition: {', '.join(bad) or 'none chosen'}.")
    conds = ["random"] + [c for c in req.conditions if c != "random"]
    seeds = sorted(set(req.seeds))[:8]

    def work(update, stopped):
        def pr(f, m, x=None):
            update(f, m, {"done": [{k: v for k, v in r.items() if k != "history"} | {"history": r["history"]} for r in (x or {}).get("done", [])]})
        return seeded_experiment(d, conds, seeds, req.max_steps, req.target, req.donor_seed, pr, stopped)

    return {"job": jobs.submit("seeded", f"Pattern-seeded training on {req.run_id}", work).id}


@app.get("/api/lab/seeded/{run_id}")
def lab_seeded_results(run_id: str):
    d = LAB / _safe_id(run_id)
    files = sorted(d.glob("seeded_*.json"), reverse=True)
    return [_read(f) for f in files[:5]]


# ------------------------------------------------------------------ hub
class HubReq(BaseModel):
    a: str
    b: str
    texts: list[str] | None = None
    max_tokens: int = Field(1500, ge=100, le=6000)


def _save_hub(kind, req, result):
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + kind
    blob = {"id": rid, "kind": kind, "a": req.a, "b": req.b, "created": time.time(), "result": result}
    return _write(HUB / f"{rid}.json", blob)


@app.post("/api/hub/weights")
def hub_weights(req: HubReq):
    from .hub.load import align_names, fetch_weights
    from .hub.weights import weight_report

    def work(update, stopped):
        a = catalog.resolve(req.a, DATA, lambda f, m: update(0.02 * f, m))
        b = catalog.resolve(req.b, DATA, lambda f, m: update(0.02 * f, m))
        wa = fetch_weights(a, lambda f, m: update(0.02 + 0.28 * f, m))
        if stopped():
            raise InterruptedError("Stopped.")
        wb = fetch_weights(b, lambda f, m: update(0.3 + 0.28 * f, m))
        pairs = align_names(wa, wb)
        if not pairs:
            raise ValueError("These checkpoints share no tensor names, so there is nothing to pair. Try the activation comparison instead.")
        r = weight_report(wa, wb, pairs, lambda f, m: update(0.6 + 0.4 * f, m))
        catalog.recent_add(DATA, req.a, req.b)
        return _save_hub("weights", req, r)

    return {"job": jobs.submit("hub-weights", f"Weights: {req.a} vs {req.b}", work).id}


@app.post("/api/hub/activations")
def hub_acts(req: HubReq):
    from .hub.acts import activation_report

    def work(update, stopped):
        a = catalog.resolve(req.a, DATA, update)
        b = catalog.resolve(req.b, DATA, update)
        r = activation_report(a, b, req.texts, req.max_tokens, update)
        catalog.recent_add(DATA, req.a, req.b)
        return _save_hub("activations", req, r)

    return {"job": jobs.submit("hub-acts", f"Activations: {req.a} vs {req.b}", work).id}


@app.get("/api/hub/history")
def hub_history():
    out = []
    for f in sorted(HUB.glob("*.json"), reverse=True)[:50]:
        try:
            b = json.loads(f.read_text())
        except Exception:
            continue
        out.append({"id": b["id"], "kind": b["kind"], "a": b["a"], "b": b["b"], "created": b["created"]})
    return out


@app.get("/api/hub/history/{rid}")
def hub_item(rid: str):
    f = HUB / f"{_safe_id(rid)}.json"
    if not f.exists():
        raise HTTPException(404, "No such result.")
    return _read(f)


@app.get("/api/hub/corpus")
def hub_corpus():
    from .hub.corpus import CORPUS

    return {"texts": CORPUS}


# ------------------------------------------------------------------ side by side
class StorageReq(BaseModel):
    spec: str


@app.post("/api/trace/storage")
def trace_storage(req: StorageReq):
    from .trace.storage import storage_report

    def work(update, stopped):
        r = storage_report(catalog.resolve(req.spec, DATA, update), update)
        catalog.recent_add(DATA, req.spec)
        return r

    return {"job": jobs.submit("storage", f"Read the files of {req.spec}", work).id}


class TraceReq(BaseModel):
    a: str
    b: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    learning: bool = True


@app.post("/api/trace/run")
def trace_run(req: TraceReq):
    from .trace.trace import trace_report

    def work(update, stopped):
        a = catalog.resolve(req.a, DATA, update)
        b = catalog.resolve(req.b, DATA, update)
        r = trace_report(a, b, req.prompt, update, learning=req.learning)
        catalog.recent_add(DATA, req.a, req.b)
        rid = time.strftime("%Y%m%d-%H%M%S") + "-trace"
        blob = {"id": rid, "a": req.a, "b": req.b, "prompt": req.prompt, "created": time.time(), "result": r}
        return _write(TRACE / f"{rid}.json", blob)

    return {"job": jobs.submit("trace", f"Trace: {req.a} vs {req.b}", work).id}


class AnatomyReq(BaseModel):
    spec: str
    prompt: str | None = None


@app.post("/api/trace/anatomy")
def trace_anatomy(req: AnatomyReq):
    from .trace.anatomy import anatomy_report

    def work(update, stopped):
        r = anatomy_report(catalog.resolve(req.spec, DATA, update), req.prompt, update)
        catalog.recent_add(DATA, req.spec)
        return r

    return {"job": jobs.submit("anatomy", f"Architecture of {req.spec}", work).id}


@app.get("/api/trace/history")
def trace_history():
    out = []
    for f in sorted(TRACE.glob("*.json"), reverse=True)[:40]:
        try:
            b = json.loads(f.read_text())
        except Exception:
            continue
        out.append({"id": b["id"], "a": b["a"], "b": b["b"], "prompt": b["prompt"][:80], "created": b["created"]})
    return out


@app.get("/api/trace/history/{rid}")
def trace_item(rid: str):
    f = TRACE / f"{_safe_id(rid)}.json"
    if not f.exists():
        raise HTTPException(404, "No such trace.")
    return _read(f)


# ------------------------------------------------------------------ pattern library
class ProfileReq(BaseModel):
    spec: str
    baseline: bool = True


@app.post("/api/atlas/profile")
def atlas_profile(req: ProfileReq):
    from .atlas.profile import profile_model

    def work(update, stopped):
        resolved = catalog.resolve(req.spec, DATA, update)
        r = profile_model(req.spec.strip(), resolved, ATLAS, update, baseline=req.baseline)
        catalog.recent_add(DATA, req.spec)
        return r

    return {"job": jobs.submit("profile", f"Profile {req.spec}", work).id}


@app.get("/api/atlas/library")
def atlas_library():
    from .atlas.library import library_report

    return clean(library_report(ATLAS))


@app.get("/api/atlas/model/{pid}")
def atlas_model(pid: str):
    from .atlas.library import model_report

    try:
        return clean(model_report(ATLAS, _safe_id(pid)))
    except KeyError:
        raise HTTPException(404, "No such profile.")


@app.delete("/api/atlas/model/{pid}")
def atlas_delete(pid: str):
    d = ATLAS / _safe_id(pid)
    if not d.exists():
        raise HTTPException(404, "No such profile.")
    shutil.rmtree(d)
    return {"deleted": pid}


# ------------------------------------------------------------------ engine
@app.get("/api/engine/meta")
def engine_meta():
    from .engine import nano
    from .engine.ledger import LEVERS
    from .engine.narrator import DEFAULT_MODEL, get_key

    return {"levers": LEVERS, "tasks": nano.TASKS, "arch": nano.DEFAULT_ARCH, "opt": nano.DEFAULT_OPT,
            "has_anthropic_key": bool(get_key(DATA / "settings.json")), "default_model": DEFAULT_MODEL}


@app.get("/api/engine/ledger")
def engine_ledger():
    from .engine.ledger import load

    return clean(load(ENGINE))


@app.post("/api/engine/mine")
def engine_mine():
    from .engine.ledger import mine

    return clean(mine(ENGINE, ATLAS, LAB))


class ExperimentReq(BaseModel):
    task: str
    lever: str
    args: dict = {}
    seeds: list[int] = [1, 2, 3]
    steps: int | None = None
    rule_id: str | None = None


@app.post("/api/engine/experiment")
def engine_experiment(req: ExperimentReq):
    from .engine import nano
    from .engine.ledger import LEVERS, run_lever

    if req.task not in nano.TASKS or req.lever not in LEVERS:
        raise HTTPException(400, "Unknown task or lever.")
    seeds = sorted(set(req.seeds))[:6] or [1]

    def work(update, stopped):
        return run_lever(ENGINE, DATA, req.task, req.lever, req.args, seeds, req.steps, req.rule_id, update, stopped)

    return {"job": jobs.submit("experiment", f"Test lever '{req.lever}' on {req.task}", work).id}


class BlueprintReq(BaseModel):
    levers: dict
    tasks: list[str] = ["sort", "copy", "modadd"]
    seeds: list[int] = [1, 2]
    steps_scale: float = Field(1.0, gt=0.05, le=5)


@app.post("/api/engine/blueprint")
def engine_blueprint(req: BlueprintReq):
    from .engine import nano
    from .engine.ledger import LEVERS, build_blueprint

    bad = [t for t in req.tasks if t not in nano.TASKS] + [k for k in req.levers if k not in LEVERS]
    if bad or not req.tasks or not req.levers:
        raise HTTPException(400, f"Choose at least one task and one lever{'; unknown: ' + ', '.join(bad) if bad else ''}.")

    def work(update, stopped):
        return build_blueprint(ENGINE, DATA, req.levers, req.tasks, sorted(set(req.seeds))[:4] or [1], req.steps_scale, update, stopped)

    return {"job": jobs.submit("blueprint", "Build and train a blueprint", work).id}


class NarrateReq(BaseModel):
    use_claude: bool = True
    model: str | None = None


@app.post("/api/engine/narrate")
def engine_narrate(req: NarrateReq):
    from .engine.ledger import load, save
    from .engine.narrator import DEFAULT_MODEL, get_key, narrate

    def work(update, stopped):
        update(0.1, "Reading the ledger")
        L = load(ENGINE)
        if not L["rules"]:
            raise ValueError("The ledger is empty. Mine rules first.")
        key = get_key(DATA / "settings.json") if req.use_claude else None
        update(0.3, "Writing the world view" + (" with Claude" if key else ""))
        n = narrate(L, key, req.model or DEFAULT_MODEL)
        L = load(ENGINE)
        L["narratives"] = ([n] + L.get("narratives", []))[:10]
        save(ENGINE, L)
        return n

    return {"job": jobs.submit("narrate", "Write the world view", work).id}


class KeyReq(BaseModel):
    key: str | None = None


@app.post("/api/settings/anthropic")
def anthropic_key(req: KeyReq):
    from .engine.narrator import get_key, set_key

    set_key(DATA / "settings.json", req.key)
    return {"has_key": bool(get_key(DATA / "settings.json"))}


# ------------------------------------------------------------------ autopilot
from .engine.autopilot import Autopilot

autopilot = Autopilot(DATA, jobs, {"atlas": ATLAS, "lab": LAB, "engine": ENGINE})
try:
    autopilot.boot()
except Exception:  # never block the server from starting
    import traceback

    traceback.print_exc()


class CampaignReq(BaseModel):
    models: list[str] = []
    trajectory: bool = False
    reprofile: bool = False
    max_experiments: int = Field(8, ge=0, le=40)
    seeds: list[int] = [1, 2, 3]
    tinystories: bool = True
    blueprint: bool = True
    claude: bool = True
    hours: float = Field(8, gt=0.05, le=72)
    scale: float = Field(1.0, gt=0.05, le=4)


@app.get("/api/autopilot")
def autopilot_state():
    return clean(autopilot.public())


@app.post("/api/autopilot/start")
def autopilot_start(req: CampaignReq):
    try:
        return clean(autopilot.start(req.model_dump() | {"seeds": sorted(set(req.seeds))[:5] or [1]}))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/autopilot/{action}")
def autopilot_action(action: str):
    if action not in ("pause", "resume", "stop"):
        raise HTTPException(404, "Unknown action.")
    return clean(getattr(autopilot, action)())


# ------------------------------------------------------------------ model catalog
@app.get("/api/models/search")
def models_search(q: str = "", task: str = "text-generation", max_params: str = "", sort: str = "downloads", limit: int = 30, author: str = ""):
    try:
        return catalog.search(q, task or None, max_params or None, sort, limit, author or None)
    except Exception as e:
        raise HTTPException(502, f"Could not reach Hugging Face: {type(e).__name__}: {e}")


class SpecReq(BaseModel):
    spec: str
    note: str = ""
    meta: dict | None = None


@app.post("/api/models/info")
def models_info(req: SpecReq):
    try:
        return catalog.info(req.spec, DATA)
    except Exception as e:
        raise HTTPException(502, f"Could not check this model: {type(e).__name__}: {e}")


@app.get("/api/models/saved")
def models_saved():
    return catalog.saved_list(DATA)


@app.post("/api/models/saved")
def models_save(req: SpecReq):
    if not req.spec.strip():
        raise HTTPException(400, "Nothing to save.")
    return catalog.saved_add(DATA, req.spec.strip(), req.note, req.meta)


@app.get("/api/models/recent")
def models_recent():
    return catalog.recent_list(DATA)


@app.delete("/api/models/saved")
def models_unsave(spec: str):
    return catalog.saved_remove(DATA, spec)


class TokenReq(BaseModel):
    token: str | None = None


@app.get("/api/settings/token")
def token_get():
    tok = os.environ.get("HF_TOKEN")
    return {"has_token": bool(tok), "user": catalog.whoami() if tok else None}


@app.post("/api/settings/token")
def token_set(req: TokenReq):
    catalog.save_token(DATA, req.token)
    return token_get()


# ------------------------------------------------------------------ static
WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    # Version the asset URLs so the browser never mixes old scripts with a new server.
    html = (WEB / "index.html").read_text()
    html = re.sub(r'(/static/[\w.-]+\.(?:js|css))"', rf'\1?v={VERSION}"', html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
