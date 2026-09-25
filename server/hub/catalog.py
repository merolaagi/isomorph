"""Finding and fetching models from the internet.

Accepted model references:
  owner/name[::subfolder][@revision]                 Hugging Face id
  https://huggingface.co/owner/name[/tree/rev/sub]   Hugging Face page link
  https://modelscope.cn/models/owner/name            ModelScope page link
  https://…/model.safetensors  (or .bin)             any direct weight file link
  /path/to/folder                                    local checkpoint folder

Links that are not Hugging Face ids are downloaded once into the data folder
and then used as local folders.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .load import parse_spec

SIDE_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model"]
WEIGHT_FILES = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"]
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4}


# ------------------------------------------------------------------ token
def settings_path(data: Path):
    return data / "settings.json"


def load_token(data: Path):
    p = settings_path(data)
    if p.exists():
        tok = json.loads(p.read_text()).get("hf_token")
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    return os.environ.get("HF_TOKEN")


def save_token(data: Path, token: str | None):
    p = settings_path(data)
    s = json.loads(p.read_text()) if p.exists() else {}
    if token:
        s["hf_token"] = token.strip()
        os.environ["HF_TOKEN"] = token.strip()
    else:
        s.pop("hf_token", None)
        os.environ.pop("HF_TOKEN", None)
    p.write_text(json.dumps(s))
    os.chmod(p, 0o600)


def whoami():
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        return HfApi().whoami().get("name")
    except Exception:
        return None


# ------------------------------------------------------------------ links
HF_URL = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?!datasets/|spaces/)([^/]+)/([^/?#]+)(?:/(tree|blob|resolve)/([^/]+)(?:/(.*))?)?")
MS_URL = re.compile(r"^(?:https?://)?(?:www\.)?modelscope\.(?:cn|ai)/models/([^/]+)/([^/?#]+)")


def hf_from_url(s):
    m = HF_URL.match(s.strip())
    if not m:
        return None
    owner, name, kind, rev, rest = m.groups()
    sub = None
    if rest:
        rest = rest.strip("/")
        if kind in ("blob", "resolve") and "." in rest.split("/")[-1]:
            rest = "/".join(rest.split("/")[:-1])
        sub = rest or None
    spec = f"{owner}/{name}"
    if sub:
        spec += f"::{sub}"
    if rev and rev != "main":
        spec += f"@{rev}"
    return spec


def _fetch(url, dest: Path, progress=None, label=""):
    req = urllib.request.Request(url, headers={"User-Agent": "isomorph"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        got = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress and total:
                    progress(got / total, f"Downloading {label} ({got / 1e6:.0f} of {total / 1e6:.0f} MB)")
        tmp.rename(dest)


def _try(url, dest, progress=None, label=""):
    try:
        _fetch(url, dest, progress, label)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def download_folder(base_url: str, cache: Path, weight_url=None, progress=lambda f, m: None):
    """Mirror a remote checkpoint folder: side files plus weights (sharded or not)."""
    key = hashlib.sha1((weight_url or base_url).encode()).hexdigest()[:12]
    d = cache / key
    done = d / ".complete"
    if done.exists():
        return str(d)
    d.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/") + "/"
    for f in SIDE_FILES:
        if not (d / f).exists():
            _try(base + f, d / f)
    got = False
    if weight_url:
        name = weight_url.split("?")[0].split("/")[-1]
        got = (d / name).exists() or _try(weight_url, d / name, progress, name)
    else:
        for f in WEIGHT_FILES:
            if (d / f).exists() or _try(base + f, d / f, progress, f):
                got = True
                if f.endswith(".index.json"):
                    shards = sorted(set(json.loads((d / f).read_text())["weight_map"].values()))
                    for i, s in enumerate(shards):
                        if not (d / s).exists() and not _try(base + s, d / s, lambda x, m: progress((i + x) / len(shards), m), s):
                            raise FileNotFoundError(f"Could not download shard {s} from {base}.")
                break
    if not got:
        raise FileNotFoundError(f"No weight file found at {weight_url or base}. Expected model.safetensors or pytorch_model.bin.")
    (d / "source.txt").write_text(weight_url or base_url)
    done.write_text(str(time.time()))
    return str(d)


def resolve(spec: str, data: Path, progress=lambda f, m: None):
    """Turn whatever the user pasted into something the loaders understand:
    a Hugging Face id or a local folder."""
    s = (spec or "").strip()
    if not s:
        raise ValueError("Enter a model name, link or folder.")
    if os.path.isdir(os.path.expanduser(s)):
        return os.path.expanduser(s)
    hf = hf_from_url(s)
    if hf:
        return hf
    cache = data / "downloads"
    m = MS_URL.match(s)
    if m:
        owner, name = m.groups()
        base = f"https://modelscope.cn/models/{owner}/{name}/resolve/master"
        progress(0.02, f"Fetching {owner}/{name} from ModelScope")
        return download_folder(base, cache, progress=progress)
    if re.match(r"^https?://", s):
        path = s.split("?")[0]
        if path.endswith((".safetensors", ".bin")):
            progress(0.02, "Fetching the weight file")
            return download_folder(path.rsplit("/", 1)[0], cache, weight_url=s, progress=progress)
        progress(0.02, "Fetching the checkpoint folder")
        return download_folder(s, cache, progress=progress)
    return s  # Hugging Face id (owner/name, optional ::sub and @rev)


# ------------------------------------------------------------------ hub search and info
EXPAND = ["downloads", "likes", "pipeline_tag", "library_name", "lastModified", "safetensors", "gated", "config", "tags"]


def _summ(m):
    cfg = getattr(m, "config", None) or {}
    archs = cfg.get("architectures") or []
    st = getattr(m, "safetensors", None)
    params = getattr(st, "total", None) if st else None
    size = None
    if st and getattr(st, "parameters", None):
        size = sum(n * DTYPE_BYTES.get(dt, 4) for dt, n in st.parameters.items())
    arch = archs[0] if archs else None
    causal = bool(arch and (arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel")))
    lm = getattr(m, "last_modified", None)
    return {
        "id": m.id, "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None),
        "task": getattr(m, "pipeline_tag", None), "library": getattr(m, "library_name", None),
        "updated": lm.isoformat()[:10] if lm else None, "params": params, "bytes": size,
        "gated": bool(getattr(m, "gated", False)), "arch": arch, "model_type": cfg.get("model_type"),
        "traceable": causal, "safetensors": st is not None,
    }


def search(q=None, task="text-generation", max_params=None, sort="downloads", limit=30, author=None):
    from huggingface_hub import HfApi

    kw = dict(search=q or None, author=author or None, sort=sort or "downloads", limit=max(1, min(int(limit), 100)), expand=EXPAND)
    if task:
        kw["pipeline_tag"] = task
    if max_params:
        kw["num_parameters"] = f"max:{max_params}"
    try:
        models = list(HfApi().list_models(**kw))
    except TypeError:
        kw.pop("num_parameters", None)
        models = list(HfApi().list_models(**kw))
    return [_summ(m) for m in models]


def info(spec, data: Path):
    """Check a model reference before using it."""
    s = (spec or "").strip()
    if os.path.isdir(os.path.expanduser(s)):
        p = Path(os.path.expanduser(s))
        cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
        arch = (cfg.get("architectures") or [None])[0]
        files = [f.name for f in p.iterdir()]
        return {"spec": str(p), "kind": "local", "arch": arch, "model_type": cfg.get("model_type"),
                "traceable": bool(arch and (arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel"))) and any(f.startswith("tokenizer") for f in files),
                "weights": any(f.endswith((".safetensors", ".bin")) for f in files), "files": len(files)}
    hf = hf_from_url(s) or (None if re.match(r"^https?://", s) else s)
    if not hf:
        return {"spec": s, "kind": "link", "note": "This link will be downloaded when you run a comparison."}
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo, sub, rev = parse_spec(hf)
    try:
        m = HfApi().model_info(repo, revision=rev, expand=EXPAND + ["siblings"])
    except GatedRepoError:
        return {"spec": hf, "kind": "hf", "error": "This model is gated. Accept its terms on huggingface.co and add your token below."}
    except RepositoryNotFoundError:
        return {"spec": hf, "kind": "hf", "error": "No such model on Hugging Face (or it is private and your token has no access)."}
    out = _summ(m) | {"spec": hf, "kind": "hf"}
    sib = [x.rfilename for x in (m.siblings or [])]
    if sub:
        sib = [f[len(sub) + 1:] for f in sib if f.startswith(sub + "/")]
    out["weights"] = any(f.endswith((".safetensors", ".bin")) and "/" not in f for f in sib)
    out["files"] = len(sib)
    if sub:
        out["traceable"] = False if not any(f == "config.json" for f in sib) else out["traceable"]
    return out


# ------------------------------------------------------------------ saved models
def saved_path(data: Path):
    return data / "models.json"


def saved_list(data: Path):
    p = saved_path(data)
    return json.loads(p.read_text()) if p.exists() else []


def saved_add(data: Path, spec: str, note: str = "", meta: dict | None = None):
    items = [x for x in saved_list(data) if x["spec"] != spec]
    items.insert(0, {"spec": spec, "note": note, "added": time.time(), "meta": meta or {}})
    saved_path(data).write_text(json.dumps(items[:200]))
    return items


def saved_remove(data: Path, spec: str):
    items = [x for x in saved_list(data) if x["spec"] != spec]
    saved_path(data).write_text(json.dumps(items))
    return items


# ------------------------------------------------------------------ recently used models
def recent_path(data: Path):
    return data / "recent.json"


def recent_list(data: Path):
    p = recent_path(data)
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except Exception:
        return []


def recent_add(data: Path, *specs):
    items = recent_list(data)
    for spec in specs:
        spec = (spec or "").strip()
        if not spec:
            continue
        items = [x for x in items if x["spec"] != spec]
        items.insert(0, {"spec": spec, "used": time.time()})
    recent_path(data).write_text(json.dumps(items[:20]))
