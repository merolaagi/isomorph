"""The registry behind the HTTP API: live training sessions, samplers for saved
weights, saved states, and each session's log on disk so the Tune tab can
draw it while a script trains."""
import json
import threading
import time
from collections import OrderedDict

import torch

from . import lora
from .session import LOCK, Session, generate, prompt_logprobs

MAX_LIVE = 4


class Service:
    def __init__(self, resolve, load_lm):
        self.resolve, self.load_lm = resolve, load_lm
        self.sessions = OrderedDict()
        self.samplers = OrderedDict()      # adapter name -> (tok, merged model)
        self._lock = threading.Lock()

    # -------------------------------------------------------------- models
    def base(self, spec, progress=lambda f, m: None):
        s = self.resolve(spec, progress)
        if lora.is_spec(s):
            raise ValueError("Train on a base model; to continue an adapter, load its saved state instead.")
        tok, model, causal = self.load_lm(s)
        if not causal:
            raise ValueError("Fine-tuning here needs a model that predicts the next token.")
        return s, tok, model

    def tokenizer(self, spec):
        s = self.resolve(spec, lambda f, m: None)
        if lora.is_spec(s):
            s = lora.adapter_meta(s)["base"]
        return self.load_lm(s)[0]

    # -------------------------------------------------------------- sessions
    def create(self, spec, rank=32, alpha=32.0, seed=0, train_attn=True, train_mlp=True, train_unembed=False, name="",
               source="script", progress=lambda f, m: None):
        s, tok, model = self.base(spec, progress)
        with LOCK:
            sess = Session(s, model, tok, rank, alpha, seed, train_attn, train_mlp, train_unembed, name, source)
        with self._lock:
            self.sessions[sess.id] = sess
            live = [x for x in self.sessions.values() if not x.closed]
            while len(live) > MAX_LIVE:          # free the GPU memory of the longest-idle session
                old = min(live, key=lambda x: x.last)
                self.close(old.id)
                live = [x for x in self.sessions.values() if not x.closed]
        self.persist(sess)
        return sess

    def get(self, sid):
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(f"No training session {sid} (the server may have restarted: load a saved state to continue).")
        if s.closed:
            raise KeyError(f"Training session {sid} was closed to free memory; load a saved state to continue.")
        return s

    def close(self, sid):
        s = self.sessions.get(sid)
        if s and not s.closed:
            self.persist(s)
            s.closed = True
            s.model = s.lora = s.opt = None
            s.mats = []
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def persist(self, s):
        try:
            d = s.public(with_log=False) if not s.closed else {}
            if d:
                d["log"] = s.log
                (lora.root() / "sessions" / f"{s.id}.json").write_text(json.dumps(d))
        except Exception:
            pass

    def listing(self):
        out, seen = [], set()
        for s in reversed(list(self.sessions.values())):
            out.append(s.public(with_log=False) if not s.closed else {**self._disk(s.id), "live": False, "closed": True})
            seen.add(s.id)
        files = sorted((lora.root() / "sessions").glob("*.json"), key=lambda p: -p.stat().st_mtime)[:40]
        for f in files:
            if f.stem not in seen:
                try:
                    d = json.loads(f.read_text())
                    d.pop("log", None)
                    out.append({**d, "live": False, "closed": True})
                except Exception:
                    continue
        return out

    def _disk(self, sid):
        f = lora.root() / "sessions" / f"{sid}.json"
        return json.loads(f.read_text()) if f.exists() else {"id": sid}

    def log(self, sid):
        s = self.sessions.get(sid)
        if s and not s.closed:
            return s.public()
        return {**self._disk(sid), "live": False}

    def forget(self, sid):
        self.close(sid)
        self.sessions.pop(sid, None)
        f = lora.root() / "sessions" / f"{sid}.json"
        if f.exists():
            f.unlink()

    # -------------------------------------------------------------- states
    def save_state(self, sid, name, with_optimizer=True):
        s = self.get(sid)
        name = lora.clean_name(name or f"{s.name}-step{s.step}")
        with LOCK:
            torch.save(s.state(with_optimizer), lora.root() / "states" / f"{name}.pt")
        self.persist(s)
        return {"path": f"isomorph://states/{name}", "name": name}

    def from_state(self, path, with_optimizer=True, source="script"):
        name = lora.name_from_path(path)
        f = lora.root() / "states" / f"{name}.pt"
        if not f.exists():
            raise FileNotFoundError(f"No saved state {name}.")
        st = torch.load(f, map_location="cpu", weights_only=True)
        c = st["config"]
        s = self.create(c["spec"], c["rank"], c["alpha"], c["seed"], c["train_attn"], c["train_mlp"], c["train_unembed"],
                        st.get("name", ""), source)
        s.load_state(st, with_optimizer)
        return s

    def states(self):
        return [{"name": f.stem, "path": f"isomorph://states/{f.stem}", "bytes": f.stat().st_size, "created": f.stat().st_mtime}
                for f in sorted((lora.root() / "states").glob("*.pt"), key=lambda p: -p.stat().st_mtime)]

    # -------------------------------------------------------------- sampling
    def sampler(self, ref):
        """ref: {"base_model": spec} | {"model_path": isomorph://adapters/<name>} | {"session": id} (live weights)."""
        if ref.get("session"):
            s = self.get(ref["session"])
            return s.tok, s.model
        if ref.get("model_path"):
            name = lora.name_from_path(ref["model_path"])
            with self._lock:
                if name in self.samplers:
                    self.samplers.move_to_end(name)
                    return self.samplers[name]
            meta, fac = lora.load_adapter(name)
            tok, base, _ = self.load_lm(meta["base"])
            with LOCK:
                m = lora.merged_copy(base, fac)
            with self._lock:
                self.samplers[name] = (tok, m)
                while len(self.samplers) > 2:
                    self.samplers.popitem(last=False)
            return tok, m
        if ref.get("base_model"):
            _, tok, m = self.base(ref["base_model"])
            return tok, m
        raise ValueError("Say which weights to sample: base_model, model_path or session.")

    def sample(self, ref, prompt, num_samples=1, max_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop=(), seed=None,
               include_prompt_logprobs=False):
        tok, m = self.sampler(ref)
        num_samples = int(max(1, min(num_samples, 256)))
        with LOCK:
            seqs = generate(m, prompt, num_samples, int(max(1, min(max_tokens, 1024))), temperature, top_p, top_k, stop, seed, tok)
            out = {"sequences": seqs}
            if include_prompt_logprobs:
                out["prompt_logprobs"] = prompt_logprobs(m, prompt)
        return out

    def compute_logprobs(self, ref, prompt):
        tok, m = self.sampler(ref)
        with LOCK:
            return {"logprobs": prompt_logprobs(m, prompt)}

    def drop_sampler(self, name):
        with self._lock:
            self.samplers.pop(name, None)
