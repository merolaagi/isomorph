"""Load real checkpoints from the Hugging Face Hub.

A model spec is  repo[::subfolder][@revision], for example
  EleutherAI/pythia-70m-seed3
  EleutherAI/pythia-70m@step3000
  convaiinnovations/laya::typed-decisions
"""
import os
from collections import OrderedDict

import torch

WEIGHT_BYTES_WARN = 1_500_000_000


def parse_spec(spec: str):
    spec = spec.strip()
    revision = None
    if "@" in spec:
        spec, revision = spec.rsplit("@", 1)
    sub = None
    if "::" in spec:
        spec, sub = spec.split("::", 1)
        sub = sub.strip("/")
    return spec.strip(), (sub or None), (revision or None)


def _files_for(repo, sub, revision):
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, revision=revision)
    if sub:
        files = [f for f in files if f.startswith(sub + "/") and "/" not in f[len(sub) + 1:]]
    else:
        files = [f for f in files if "/" not in f]
    st = [f for f in files if f.endswith(".safetensors")]
    if st:
        return st
    bins = [f for f in files if f.endswith(".bin") and ("pytorch_model" in f or "model" in f)]
    if bins:
        return bins
    raise FileNotFoundError(f"No .safetensors or .bin weights found in {repo}{'/' + sub if sub else ''}.")


def fetch_weights(spec: str, progress=lambda f, m: None):
    """Download and return {tensor_name: float32 tensor} for any checkpoint layout."""
    from huggingface_hub import hf_hub_download

    repo, sub, rev = parse_spec(spec)
    files = _files_for(repo, sub, rev)
    out = OrderedDict()
    for i, f in enumerate(files):
        progress(i / len(files), f"Downloading {repo}/{f}")
        path = hf_hub_download(repo, f, revision=rev)
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                out[k] = v.float()
    return out


def align_names(wa, wb):
    """Pair tensors by name, tolerating a wrapper prefix on one side
    (e.g. 'model.' vs none, or 'gpt_neox.' only on one checkpoint)."""
    common = [k for k in wa if k in wb]
    if len(common) >= 0.5 * min(len(wa), len(wb)):
        return [(k, k) for k in common]

    def strip(k):
        return k.split(".", 1)[1] if "." in k else k

    for fa, fb in ((strip, lambda k: k), (lambda k: k, strip), (strip, strip)):
        mb = {fb(k): k for k in wb}
        pairs = [(k, mb[fa(k)]) for k in wa if fa(k) in mb]
        if len(pairs) > len(common):
            return pairs
    return [(k, k) for k in common]


class _LRU:
    def __init__(self, n=2):
        self.n, self.d = n, OrderedDict()

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = make()
        self.d[key] = v
        while len(self.d) > self.n:
            self.d.popitem(last=False)
        return v


_models = _LRU(int(os.environ.get("ISOMORPH_MODEL_CACHE", "2")))


def device():
    want = os.environ.get("ISOMORPH_DEVICE", "auto")
    if want != "auto":
        return want
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_lm(spec: str):
    """(tokenizer, model, is_causal) for a transformers-compatible checkpoint."""
    def make():
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        repo, sub, rev = parse_spec(spec)
        kw = {"revision": rev}
        if sub:
            kw["subfolder"] = sub
        tok = AutoTokenizer.from_pretrained(repo, **kw)
        causal = True
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, **kw)
        except Exception:
            causal = False
            model = AutoModel.from_pretrained(repo, dtype=torch.float32, **kw)
        model.eval().to(device())
        return tok, model, causal

    return _models.get(spec, make)
