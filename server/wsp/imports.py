"""Weights sent in by another app (Deep Network Designer, say), read as raw tensors.

An import is a safetensors file plus a short description of the architecture
written by the app that built it, stored under data/wsp/imports. Its spec is
import:<id>. Isomorph cannot run such a model (it does not know how the
tensors are wired), so the read tools work on the weights alone.
"""
import json
import re
import time
import uuid
from pathlib import Path

_DIR = None
MAX_BYTES = 400_000_000


def set_dir(d: Path):
    global _DIR
    _DIR = Path(d)
    _DIR.mkdir(parents=True, exist_ok=True)


def _path(iid):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", iid or ""):
        raise ValueError("Bad import id.")
    return _DIR / f"{iid}.safetensors", _DIR / f"{iid}.json"


def save_import(body: bytes, name="", about="", source=""):
    from safetensors.torch import load

    if len(body) > MAX_BYTES:
        raise ValueError(f"The weights are larger than {MAX_BYTES // 1_000_000} MB.")
    tensors = load(body)                      # also checks that it is a valid safetensors file
    if not tensors:
        raise ValueError("No tensors in the file.")
    base = re.sub(r"[^A-Za-z0-9_\-]+", "-", name or "model").strip("-")[:40] or "model"
    iid = f"{base}-{uuid.uuid4().hex[:6]}"
    st, meta = _path(iid)
    st.write_bytes(body)
    info = {"id": iid, "spec": f"import:{iid}", "name": name or base, "about": (about or "")[:4000], "source": source[:200],
            "tensors": len(tensors), "weights": int(sum(t.numel() for t in tensors.values())), "created": time.time()}
    meta.write_text(json.dumps(info))
    return info


def load_import(spec):
    from safetensors.torch import load_file

    iid = spec.split(":", 1)[1]
    st, meta = _path(iid)
    if not st.exists():
        raise FileNotFoundError(f"No imported model {iid}.")
    about = json.loads(meta.read_text()).get("about", "") if meta.exists() else ""
    return {k: v.float() for k, v in load_file(str(st)).items() if v.is_floating_point()}, about


def list_imports():
    if _DIR is None:
        return []
    out = []
    for m in sorted(_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            out.append(json.loads(m.read_text()))
        except Exception:
            continue
    return out
