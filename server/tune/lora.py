"""LoRA adapters: attach to a private copy of a model, merge, save and load.

An adapter is, for each chosen matrix W, two factors U (rows x r, starting at
zero) and V (cols x r, random), and the model uses W + (alpha / r) U V^T.
The scale alpha / r with a fixed alpha keeps the best learning rate roughly
independent of the rank (Thinking Machines, "LoRA Without Regret", 2025),
which is why alpha defaults to 32 here as there.

Saved adapters live in data/tune/adapters/<name>/ as adapter.safetensors
(<matrix>.lora_U, <matrix>.lora_V) and adapter.json (base model, rank,
alpha, the matrices, the training log). Their path is isomorph://adapters/<name>
and every other tab can open the merged model as tune:<name>.
"""
import copy
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.utils.parametrize as P

from ..wsp.common import LowRank, tunable

_DIR = None
NAME_RE = re.compile(r"[A-Za-z0-9_\-.]{1,80}")


def set_dir(d: Path):
    global _DIR
    _DIR = Path(d)
    for sub in ("adapters", "states", "sessions"):
        (_DIR / sub).mkdir(parents=True, exist_ok=True)


def root():
    if _DIR is None:
        raise RuntimeError("Tune folder not set.")
    return _DIR


def clean_name(name, fallback="adapter"):
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "-", (name or "").strip()).strip("-.")[:80]
    return s or fallback


def name_from_path(path):
    """isomorph://adapters/<name>, isomorph://states/<name>, tune:<name> or a bare name."""
    p = (path or "").strip()
    for pre in ("isomorph://adapters/", "isomorph://states/", "tune:"):
        if p.startswith(pre):
            p = p[len(pre):]
    if not NAME_RE.fullmatch(p):
        raise ValueError(f"Bad adapter or state name: {path!r}")
    return p


def select(model, train_attn=True, train_mlp=True, train_unembed=False):
    """Which matrices get an adapter: attention and/or MLP matrices inside the blocks,
    and optionally the unembedding when it is not tied to the input embedding."""
    out = [c for c in tunable(model) if (c["kind"] == "attention" and train_attn) or (c["kind"] == "mlp" and train_mlp)
           or (c["kind"] == "other" and train_attn and train_mlp)]
    if train_unembed:
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if head is None or not hasattr(head, "weight"):
            raise ValueError("This model has no separate unembedding matrix to tune.")
        if emb is not None and head.weight is emb.weight:
            raise ValueError("This model ties its unembedding to its input embedding, so it cannot get its own adapter.")
        name = next(n for n, m in model.named_modules() if m is head) + ".weight"
        out.append({"name": name, "layer": 10 ** 6, "module": head, "attr": "weight", "shape": tuple(head.weight.shape),
                    "kind": "unembed"})
    if not out:
        raise ValueError("No matrices chosen: turn on attention, MLP or both.")
    return out


def attach(model, mats, rank, alpha=32.0, seed=0):
    """Register W -> W + (alpha/r) U V^T on each matrix. Returns {name: LowRank}."""
    for p in model.parameters():
        p.requires_grad_(False)
    out = {}
    for i, c in enumerate(mats):
        p = getattr(c["module"], c["attr"])
        r = int(max(1, min(rank, min(p.shape))))
        lr = LowRank(tuple(p.shape), r, alpha=alpha / r, seed=seed * 7919 + i, device=p.device)
        P.register_parametrization(c["module"], c["attr"], lr)
        out[c["name"]] = lr
    return out


def factors(lora):
    """{name: (U, V, scale)} on the CPU."""
    return {n: (m.U.detach().float().cpu(), m.V.detach().float().cpu(), m.scale) for n, m in lora.items()}


def delta(U, V, scale):
    return scale * (U @ V.T)


def _param(model, name):
    mod, _, attr = name.rpartition(".")
    return getattr(model.get_submodule(mod) if mod else model, attr)


def merged_copy(base_model, fac):
    """A private copy of the base model with every delta added into its weights."""
    m = copy.deepcopy(base_model)
    with torch.no_grad():
        for name, (U, V, s) in fac.items():
            p = _param(m, name)
            p.add_(delta(U, V, s).to(p.device, p.dtype))
    m.eval()
    return m


# ------------------------------------------------------------------ files
def adapter_dir(name):
    return root() / "adapters" / name_from_path(name)


def save_adapter(name, spec, fac, info):
    from safetensors.torch import save_file

    name = clean_name(name)
    d = root() / "adapters" / name
    d.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for n, (U, V, _) in fac.items():
        tensors[f"{n}.lora_U"] = U.contiguous()
        tensors[f"{n}.lora_V"] = V.contiguous()
    save_file(tensors, str(d / "adapter.safetensors"))
    meta = {"name": name, "spec": f"tune:{name}", "path": f"isomorph://adapters/{name}", "base": spec,
            "created": time.time(), "matrices": [{"name": n, "shape": [int(U.shape[0]), int(V.shape[0])], "rank": int(U.shape[1]),
                                                 "scale": float(s)} for n, (U, V, s) in fac.items()],
            "trainable": int(sum(U.numel() + V.numel() for U, V, _ in fac.values())), **info}
    (d / "adapter.json").write_text(json.dumps(meta))
    forget_cached(name)
    return meta


def forget_cached(name):
    """Drop a cached merged model for this adapter, so tune:<name> reloads the new weights."""
    from ..hub import load

    load._models.d.pop(f"tune:{name}", None)
    a = adapter_dir(name) / "anatomy.json"
    if a.exists():
        a.unlink()


def load_adapter(name):
    from safetensors.torch import load_file

    d = adapter_dir(name)
    if not (d / "adapter.json").exists():
        raise FileNotFoundError(f"No saved adapter {name_from_path(name)}.")
    meta = json.loads((d / "adapter.json").read_text())
    t = load_file(str(d / "adapter.safetensors"))
    fac = {m["name"]: (t[f"{m['name']}.lora_U"].float(), t[f"{m['name']}.lora_V"].float(), float(m["scale"])) for m in meta["matrices"]}
    return meta, fac


def adapter_meta(name):
    d = adapter_dir(name)
    return json.loads((d / "adapter.json").read_text())


def list_adapters():
    if _DIR is None:
        return []
    out = []
    for f in sorted((root() / "adapters").glob("*/adapter.json"), key=lambda p: -p.stat().st_mtime):
        try:
            m = json.loads(f.read_text())
            m.pop("log", None)
            out.append(m)
        except Exception:
            continue
    return out


def delete_adapter(name):
    import shutil

    d = adapter_dir(name)
    if d.exists():
        shutil.rmtree(d)
    forget_cached(name_from_path(name))


# ------------------------------------------------------------------ the merged model as an ordinary spec
def is_spec(s):
    return isinstance(s, str) and s.strip().startswith("tune:")


def load_merged(spec, load_lm):
    """(tokenizer, merged model, causal) for tune:<name>: the base model with the adapter added in."""
    meta, fac = load_adapter(spec)
    tok, base, causal = load_lm(meta["base"])
    m = merged_copy(base, fac)
    m.spec = spec
    return tok, m, causal


def merged_tensors(spec, load_lm):
    """Named tensors of the merged model, named as the base model's own weights are,
    so base and fine-tune pair up name for name in the comparison tabs."""
    from collections import OrderedDict

    from .. import microopus

    meta, fac = load_adapter(spec)
    _, base, _ = load_lm(meta["base"])
    m = merged_copy(base, fac)
    if microopus.is_spec(meta["base"]) and hasattr(m, "to_micro"):
        return OrderedDict((k, torch.tensor(v)) for k, v in m.to_micro().items())
    return OrderedDict((k, v.detach().float().cpu()) for k, v in m.state_dict().items() if v.is_floating_point())
