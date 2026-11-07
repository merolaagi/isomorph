"""HTTP routes for Tune.

/api/tune/v1/...   the training service a script talks to (isotinker client)
/api/tune/...      the Tune tab: recipes as jobs, sessions, adapters, anatomy, code
"""
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from ..jsonsafe import clean
from . import lora
from .recipes import REWARDS

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ request bodies
class CreateReq(BaseModel):
    base_model: str
    rank: int = 32
    alpha: float = 32.0
    seed: int = 0
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    name: str = ""


class FBReq(BaseModel):
    data: list[dict]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict | None = None


class AdamReq(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class SampleReq(BaseModel):
    prompt: list[int]
    num_samples: int = 1
    max_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: list = []
    seed: int | None = None
    include_prompt_logprobs: bool = False
    base_model: str | None = None
    model_path: str | None = None
    session: str | None = None


class NameReq(BaseModel):
    name: str = ""
    with_optimizer: bool = True


class StateReq(BaseModel):
    path: str
    with_optimizer: bool = True


class TokReq(BaseModel):
    base_model: str
    text: str | None = None
    tokens: list[int] | None = None
    add_special_tokens: bool = True
    messages: list[dict] | None = None
    add_generation_prompt: bool = True


class SLReq(BaseModel):
    spec: str
    examples: str
    held_out: str = ""
    rank: int = 8
    alpha: float = 32.0
    lr: float = 1e-3
    steps: int = 100
    batch: int = 8
    seed: int = 0
    train_attn: bool = True
    train_mlp: bool = True
    name: str = ""


class RLReq(BaseModel):
    spec: str
    prompts: str
    reward: str = "contains"
    reward_arg: str = ""
    held_out: str = ""
    rank: int = 8
    alpha: float = 32.0
    lr: float = 5e-3
    steps: int = 60
    batch: int = 4
    group: int = 8
    max_tokens: int = 12
    temperature: float = 1.0
    kl_coef: float = 0.05
    clip: float = 1.0
    seed: int = 0
    train_attn: bool = True
    train_mlp: bool = True
    name: str = ""


class CompareReq(BaseModel):
    a: str
    b: str


class ScriptReq(BaseModel):
    code: str
    name: str = "script"


def is_local(request: Request):
    """True only for a request made on this Mac itself, not through the Cloudflare tunnel
    (which also connects from 127.0.0.1 but adds its own headers)."""
    host = request.client.host if request.client else ""
    via = any(request.headers.get(h) for h in ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "forwarded"))
    return host in ("127.0.0.1", "::1", "localhost", "testclient") and not via


def make_router(jobs, data_dir: Path, svc):
    r = APIRouter()
    lora.set_dir(data_dir / "tune")
    scripts_dir = data_dir / "tune" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    def err(e):
        if isinstance(e, KeyError):
            raise HTTPException(404, str(e).strip("'\""))
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise HTTPException(400, str(e))
        raise e

    # ============================================================== training service (what a script calls)
    @r.get("/api/tune/v1/health")
    def health():
        return {"ok": True, "service": "isomorph-tune", "version": (ROOT / "VERSION").read_text().strip(),
                "losses": ["cross_entropy", "importance_sampling", "ppo", "linear"]}

    @r.get("/api/tune/v1/models")
    def models():
        from .. import microopus

        return {"supported_models": [{"model_name": s} for s in microopus.available()],
                "note": "Any Hugging Face causal language model small enough for this Mac also works, by its id."}

    @r.post("/api/tune/v1/sessions")
    def create(req: CreateReq):
        try:
            s = svc.create(req.base_model, req.rank, req.alpha, req.seed, req.train_attn, req.train_mlp, req.train_unembed, req.name)
        except Exception as e:
            err(e)
        return s.public(with_log=False)

    @r.post("/api/tune/v1/sessions/{sid}/forward_backward")
    def fb(sid: str, req: FBReq):
        try:
            return clean(svc.get(sid).forward_backward(req.data, req.loss_fn, req.loss_fn_config))
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/{sid}/forward")
    def fwd(sid: str, req: FBReq):
        try:
            return clean(svc.get(sid).forward(req.data, req.loss_fn, req.loss_fn_config))
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/{sid}/optim_step")
    def step(sid: str, req: AdamReq):
        try:
            s = svc.get(sid)
            out = s.optim_step(**req.model_dump())
            if s.step % 5 == 0:
                svc.persist(s)
            return out
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/{sid}/report")
    def report(sid: str, metrics: dict):
        try:
            return svc.get(sid).report(metrics)
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/{sid}/save_weights")
    def save_weights(sid: str, req: NameReq):
        try:
            s = svc.get(sid)
            meta = s.save_adapter(req.name)
            svc.drop_sampler(meta["name"])
            svc.persist(s)
            return {"path": meta["path"], "name": meta["name"], "spec": meta["spec"]}
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/{sid}/save_state")
    def save_state(sid: str, req: NameReq):
        try:
            return svc.save_state(sid, req.name, req.with_optimizer)
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/sessions/from_state")
    def from_state(req: StateReq):
        try:
            return svc.from_state(req.path, req.with_optimizer).public(with_log=False)
        except Exception as e:
            err(e)

    @r.get("/api/tune/v1/sessions/{sid}")
    def session_info(sid: str):
        return clean(svc.log(sid))

    @r.delete("/api/tune/v1/sessions/{sid}")
    def close(sid: str):
        svc.close(sid)
        return {"ok": True}

    @r.post("/api/tune/v1/sample")
    def sample(req: SampleReq):
        try:
            ref = {"session": req.session, "model_path": req.model_path, "base_model": req.base_model}
            return clean(svc.sample(ref, req.prompt, req.num_samples, req.max_tokens, req.temperature, req.top_p, req.top_k,
                                    req.stop, req.seed, req.include_prompt_logprobs))
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/compute_logprobs")
    def logprobs(req: SampleReq):
        try:
            return clean(svc.compute_logprobs({"session": req.session, "model_path": req.model_path, "base_model": req.base_model},
                                              req.prompt))
        except Exception as e:
            err(e)

    @r.post("/api/tune/v1/tokenizer")
    def tokenizer(req: TokReq):
        try:
            tok = svc.tokenizer(req.base_model)
        except Exception as e:
            err(e)
        from .. import microopus

        micro = isinstance(tok, microopus.Tokenizer)
        out = {}
        if req.messages is not None:
            if not getattr(tok, "chat_template", None):
                raise HTTPException(400, "This model's tokenizer has no chat template.")
            out["tokens"] = list(tok.apply_chat_template(req.messages, add_generation_prompt=req.add_generation_prompt, tokenize=True))
        if req.text is not None:
            if micro:
                ids = tok.encode(req.text)
                out["tokens"] = ids if req.add_special_tokens else ids[1:]
            else:
                out["tokens"] = list(tok.encode(req.text, add_special_tokens=req.add_special_tokens))
        if req.tokens is not None:
            out["text"] = tok.decode(req.tokens)
        out["vocab_size"] = len(tok)
        eos = getattr(tok, "eos_token_id", None)
        out["eos_token_id"] = 0 if micro else eos
        out["bos_token_id"] = 0 if micro else getattr(tok, "bos_token_id", None)
        return out

    # ============================================================== the Tune tab
    @r.get("/api/tune/meta")
    def meta(request: Request):
        return {"rewards": REWARDS, "local": is_local(request), "sessions": svc.listing(), "adapters": lora.list_adapters(),
                "states": svc.states(), "recipes": sorted(p.name for p in (ROOT / "recipes").glob("*.py"))}

    @r.post("/api/tune/supervised")
    def run_sl(req: SLReq):
        from .recipes import supervised

        def work(update, stopped):
            return supervised(svc, **req.model_dump(), progress=update, should_stop=stopped)
        return {"job": jobs.submit("tune-sl", f"Supervised fine-tune of {req.spec}", work).id}

    @r.post("/api/tune/reinforce")
    def run_rl(req: RLReq):
        from .recipes import reinforce

        def work(update, stopped):
            return reinforce(svc, **req.model_dump(), progress=update, should_stop=stopped)
        return {"job": jobs.submit("tune-rl", f"Reinforcement fine-tune of {req.spec}", work).id}

    @r.get("/api/tune/session/{sid}")
    def one_session(sid: str):
        return clean(svc.log(sid))

    @r.delete("/api/tune/session/{sid}")
    def forget(sid: str):
        svc.forget(sid)
        return {"ok": True}

    @r.get("/api/tune/adapter/{name}")
    def adapter(name: str):
        try:
            m = lora.adapter_meta(name)
        except Exception as e:
            err(e)
        f = lora.adapter_dir(name) / "anatomy.json"
        if f.exists():
            m["anatomy"] = json.loads(f.read_text())
        return clean(m)

    @r.post("/api/tune/adapter/{name}/anatomy")
    def read_adapter(name: str):
        from .anatomy import anatomy

        def work(update, stopped):
            update(0.1, f"Reading {name}")
            a = anatomy(svc, name)
            (lora.adapter_dir(name) / "anatomy.json").write_text(json.dumps(clean(a)))
            return a
        return {"job": jobs.submit("tune-anatomy", f"Where did {name} go?", work).id}

    @r.post("/api/tune/compare")
    def compare_adapters(req: CompareReq):
        from .anatomy import compare

        def work(update, stopped):
            update(0.1, "Comparing the two changes")
            return compare(svc, req.a, req.b)
        return {"job": jobs.submit("tune-compare", f"{req.a} vs {req.b}", work).id}

    @r.delete("/api/tune/adapter/{name}")
    def delete_adapter(name: str):
        lora.delete_adapter(name)
        svc.drop_sampler(lora.name_from_path(name))
        return {"ok": True}

    @r.get("/api/tune/adapter/{name}/adapter.safetensors")
    def download_adapter(name: str):
        f = lora.adapter_dir(name) / "adapter.safetensors"
        if not f.exists():
            raise HTTPException(404, "No such adapter.")
        return FileResponse(f, filename=f"{lora.name_from_path(name)}.adapter.safetensors")

    @r.get("/api/tune/adapter/{name}/merged.safetensors")
    def download_merged(name: str):
        from safetensors.torch import save

        try:
            t = lora.merged_tensors(f"tune:{name}", svc.load_lm)
        except Exception as e:
            err(e)
        body = save({k: v.contiguous() for k, v in t.items()})
        return Response(body, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{lora.name_from_path(name)}.merged.safetensors"'})

    # ------------------------------------------------------------ code: the recipe as a script
    @r.get("/api/tune/code", response_class=PlainTextResponse)
    def code(recipe: str = "sl_basic.py", config: str = "{}"):
        f = ROOT / "recipes" / Path(recipe).name
        if not f.exists():
            raise HTTPException(404, "No such recipe.")
        return fill_config(f.read_text(), json.loads(config or "{}"))

    @r.post("/api/tune/script")
    def run_script(req: ScriptReq, request: Request):
        if not is_local(request):
            raise HTTPException(403, "Scripts run only from this Mac (open Isomorph at 127.0.0.1), never through the public "
                                     "address. Or run it in Terminal: ./scripts/tune.sh your_script.py")
        name = re.sub(r"[^A-Za-z0-9_\-]+", "-", req.name).strip("-")[:40] or "script"
        path = scripts_dir / f"{name}-{int(time.time())}.py"
        path.write_text(req.code)
        port = request.url.port or 47821

        def work(update, stopped):
            env = dict(os.environ, ISOMORPH_URL=f"http://127.0.0.1:{port}", PYTHONPATH=str(ROOT / "client"),
                       PYTHONUNBUFFERED="1")
            p = subprocess.Popen([sys.executable, str(path)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []

            def pump():
                for ln in p.stdout:
                    lines.append(ln.rstrip("\n"))
                    del lines[:-400]
            th = threading.Thread(target=pump, daemon=True)
            th.start()
            while p.poll() is None:
                if stopped():
                    p.terminate()
                    try:
                        p.wait(5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    raise InterruptedError("Stopped.")
                update(0.5, lines[-1][:200] if lines else "Running", {"output": lines[-200:]})
                time.sleep(0.5)
            th.join(2)
            if p.returncode:
                raise RuntimeError(f"The script exited with code {p.returncode}:\n" + "\n".join(lines[-30:]))
            return {"kind": "script", "script": str(path), "output": lines[-400:], "code": p.returncode}
        return {"job": jobs.submit("tune-script", f"Python: {name}", work).id}

    return r


def fill_config(src, cfg):
    """Replace NAME = value lines inside the '# --- settings ---' block with the given values."""
    if not cfg:
        return src
    lines = src.splitlines()
    out, inside, i = [], False, 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if ln.strip().startswith("# --- settings"):
            inside = True
        elif inside and ln.strip().startswith("# ---"):
            inside = False
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=", ln) if inside else None
        if m and m.group(1).lower() in cfg:
            if ln.count('"""') % 2 == 1:          # a multi-line string: skip to its closing quotes
                while i < len(lines) and '"""' not in lines[i]:
                    i += 1
                i += 1
            out.append(f"{m.group(1)} = {_py(cfg[m.group(1).lower()])}")
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


def _py(v):
    if isinstance(v, str) and "\n" in v:
        v = v.rstrip("\n")
        return '"""\\\n' + v.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '\n"""'
    return repr(v)
