"""Micro Opus inside Isomorph.

Micro Opus (vendor/microopus) is a 7,440-weight transformer written in plain
numpy, small enough that every weight fits on screen. This module lets every
Isomorph view use it:

  * MicroOpusLM   a PyTorch mirror of model.py with Hugging Face-style names,
                  checked against model.py numerically (see check_against_numpy)
  * Tokenizer     word-level (words model) or letter-level (names model)
  * weights_blob  the JSON weight file, for the weight-file and comparison views
  * source_map    the lines of model.py where each named weight enters
  * train_seed    trains a new copy with Micro Opus's own train.py and seed

Model references look like  microopus:words,  microopus:names,
microopus:words-seed11  (trained here, stored in the data folder).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "microopus"
PREFIX = "microopus:"
_data_dir = None


def set_data_dir(d: Path):
    global _data_dir
    _data_dir = d


def is_spec(s):
    return isinstance(s, str) and s.strip().startswith(PREFIX)


def _name(spec):
    n = spec.strip()[len(PREFIX):]
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", n):
        raise ValueError(f"Bad Micro Opus name: {n}")
    return n


def weights_path(spec):
    n = _name(spec)
    for d in ([_data_dir / "microopus"] if _data_dir else []) + [VENDOR / "weights"]:
        p = d / f"{n}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"No Micro Opus model called '{n}'. Built in: words, names; train more seeds from Side by side.")


def weights_blob(spec):
    return json.loads(weights_path(spec).read_text())


def available():
    out = []
    for d in [VENDOR / "weights"] + ([_data_dir / "microopus"] if _data_dir else []):
        if d.exists():
            for p in sorted(d.glob("*.json")):
                out.append(PREFIX + p.stem)
    return out


def _vendor_module(name):
    spec = importlib.util.spec_from_file_location(f"microopus_{name}", VENDOR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(VENDOR))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(VENDOR))
    return mod


# ------------------------------------------------------------------ PyTorch mirror of model.py
class RMSNorm(nn.Module):
    """y = x / rms(x) * g   (model.py: rmsnorm)"""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * self.weight


class GELU(nn.Module):
    def forward(self, z):
        return 0.5 * z * (1 + torch.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        sh = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x))
        s = q @ k.transpose(-1, -2) / (D // self.h) ** 0.5
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        allow = causal[None, None]
        if mask is not None:
            allow = allow & mask[:, None, None, :].bool()
        s = s.masked_fill(~allow, float("-inf"))
        att = torch.nan_to_num(s.softmax(-1))
        return self.o_proj((att @ v).transpose(1, 2).reshape(B, T, D)), att


class MLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.up_proj = nn.Linear(d, f, bias=False)
        self.act = GELU()
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class Layer(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.attention = Attention(d, h)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = MLP(d, f)

    def forward(self, x, mask=None):
        a, att = self.attention(self.input_layernorm(x), mask)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), att


class Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, V, T = cfg["D"], cfg["V"], cfg["T"]
        self.embed_tokens = nn.Embedding(V, D)
        self.embed_positions = nn.Embedding(T, D)
        self.layers = nn.ModuleList([Layer(D, cfg["H"], cfg["F"]) for _ in range(cfg["L"])])
        self.norm = RMSNorm(D)


# Which Micro Opus weight each PyTorch parameter holds. Linear weights are stored
# transposed in PyTorch (out x in), so x @ W in model.py is Linear(x) here.
def param_map(L):
    m = {"model.embed_tokens.weight": ("W_E", False), "model.embed_positions.weight": ("W_P", False),
         "model.norm.weight": ("g_f", False), "lm_head.weight": ("W_U", True)}
    for l in range(L):
        p = f"model.layers.{l}."
        m |= {p + "input_layernorm.weight": (f"L{l}.g1", False), p + "attention.q_proj.weight": (f"L{l}.W_Q", True),
              p + "attention.k_proj.weight": (f"L{l}.W_K", True), p + "attention.v_proj.weight": (f"L{l}.W_V", True),
              p + "attention.o_proj.weight": (f"L{l}.W_O", True), p + "post_attention_layernorm.weight": (f"L{l}.g2", False),
              p + "mlp.up_proj.weight": (f"L{l}.W_up", True), p + "mlp.down_proj.weight": (f"L{l}.W_down", True)}
    return m


class MicroOpusLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self, cfg, W=None, spec="microopus"):
        super().__init__()
        self.mo_cfg, self.spec = dict(cfg), spec
        self.model = Base(cfg)
        self.lm_head = nn.Linear(cfg["D"], cfg["V"], bias=False)
        self.config = SimpleNamespace(
            model_type="microopus", architectures=["MicroOpusForCausalLM"], num_hidden_layers=cfg["L"], hidden_size=cfg["D"],
            num_attention_heads=cfg["H"], intermediate_size=cfg["F"], vocab_size=cfg["V"], max_position_embeddings=cfg["T"],
            hidden_act="gelu_new", use_parallel_residual=False, n_embd=cfg["D"], n_layer=cfg["L"])
        if W is not None:
            self.load_micro(W)

    def load_micro(self, W):
        sd = self.state_dict()
        with torch.no_grad():
            for name, (key, transpose) in param_map(self.mo_cfg["L"]).items():
                t = torch.tensor(np.array(W[key], dtype=np.float32))
                sd[name].copy_(t.T if transpose else t)
        return self

    def to_micro(self):
        out = {}
        sd = self.state_dict()
        for name, (key, transpose) in param_map(self.mo_cfg["L"]).items():
            t = sd[name].detach().cpu()
            out[key] = (t.T if transpose else t).numpy()
        return out

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, output_attentions=False, labels=None, **_):
        T = input_ids.shape[1]
        if T > self.mo_cfg["T"]:
            raise ValueError(f"Micro Opus reads at most {self.mo_cfg['T']} tokens; got {T}.")
        x = self.model.embed_tokens(input_ids) + self.model.embed_positions(torch.arange(T, device=input_ids.device))
        hs, atts = [x], []
        for layer in self.model.layers:
            x, att = layer(x, attention_mask)
            hs.append(x)
            atts.append(att)
        xf = self.model.norm(x)
        hs[-1] = xf  # Hugging Face convention: the last hidden state is after the final norm
        logits = self.lm_head(xf)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=tuple(hs) if output_hidden_states else None,
                               attentions=tuple(atts) if output_attentions else None)

    def isomorph_random_twin(self):
        """A fresh copy initialised exactly as Micro Opus's train.py does."""
        train = _vendor_module("train")
        W = train.init_weights(self.mo_cfg, self.mo_cfg["V"], np.random.default_rng(0))
        return MicroOpusLM(self.mo_cfg, W, self.spec + " (random)").eval()


# ------------------------------------------------------------------ tokenizer
class Tokenizer:
    """Word-level for the words model, letter-level for the names model. Unknown
    pieces are dropped (and counted in .dropped), and every text starts with <s>
    exactly as model.py's tokenize() does."""

    def __init__(self, vocab, chars, max_len):
        self.vocab, self.chars, self.max_len = vocab, chars, max_len
        self.index = {w: i for i, w in enumerate(vocab)}
        self.pad_token_id, self.pad_token, self.eos_token = 0, "<s>", "<s>"
        self.padding_side = "right"
        self.dropped = 0

    def __len__(self):
        return len(self.vocab)

    def encode(self, text):
        pieces = list(text.lower()) if self.chars else text.lower().replace(".", " . ").split()
        ids = [0]
        for p in pieces:
            if p in self.index:
                ids.append(self.index[p])
            else:
                self.dropped += 1
        return ids

    def __call__(self, text, return_tensors=None, padding=False, truncation=True, max_length=None, **_):
        texts = [text] if isinstance(text, str) else list(text)
        lim = min(self.max_len, max_length or self.max_len)
        rows = [self.encode(t)[:lim] for t in texts]
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids if len(ids) > 1 else ids[0], "attention_mask": mask if len(mask) > 1 else mask[0]}

    def convert_ids_to_tokens(self, i):
        return self.vocab[int(i)]

    def decode(self, ids):
        toks = [self.vocab[int(i)] for i in ids]
        return "".join(toks) if self.chars else " ".join(toks) if len(toks) > 1 else " " + toks[0]

    def get_vocab(self):
        return dict(self.index)


def load(spec):
    blob = weights_blob(spec)
    cfg, vocab = blob["config"], blob["vocab"]
    cfg = dict(cfg, V=len(vocab))
    model = MicroOpusLM(cfg, blob["weights"], spec.strip()).eval()
    tok = Tokenizer(vocab, cfg.get("chars", False), cfg["T"])
    # Texts the model can actually read, cut from its own training stream, used
    # wherever Isomorph would otherwise probe with general English.
    stream = blob.get("corpus") or []
    texts, cur = [], []
    for i in stream:
        if i == 0 and cur:
            texts.append(cur)
            cur = []
        elif i != 0:
            cur.append(i)
        if len(texts) >= 400:
            break
    sep = "" if cfg.get("chars") else " "
    tok.calibration_texts = [sep.join(vocab[i] for i in t) for t in texts if t][:200] or None
    return tok, model, True


def tensors(spec):
    """Named float tensors in Micro Opus's own naming, for the weight views."""
    return {k: torch.tensor(np.array(v, dtype=np.float32)) for k, v in weights_blob(spec)["weights"].items()}


def check_against_numpy(spec="microopus:words", text=None):
    """Largest difference between this adapter and model.py on the same input."""
    mo = _vendor_module("model")
    blob = weights_blob(spec)
    ref = mo.MicroOpus(W=blob["weights"], cfg=blob["config"], vocab=blob["vocab"])
    tok, m, _ = load(spec)
    text = text or ("the bird flew over the" if not blob["config"].get("chars") else "emma")
    ids = tok.encode(text)
    p_np = ref.forward(ids)
    with torch.no_grad():
        p_t = m(torch.tensor([ids])).logits[0].softmax(-1).double().numpy()
    return float(np.abs(p_np - p_t).max())


# ------------------------------------------------------------------ where weights enter model.py
WEIGHT_RE = re.compile(r'W\["([A-Za-z0-9_]+)"\]|P\("([A-Za-z0-9_]+)"\)')


def source_map(spec=None):
    """Every line of model.py that reads a named weight, with the weights' shapes."""
    src = (VENDOR / "model.py").read_text()
    lines = src.splitlines()
    shapes = {}
    if spec:
        for k, v in weights_blob(spec)["weights"].items():
            a = np.array(v)
            shapes[k] = list(a.shape)
    funcs, cur = {}, None
    for i, line in enumerate(lines, 1):
        m = re.match(r"\s*def (\w+)\(", line)
        if m:
            cur = m.group(1)
        funcs.setdefault(cur, []).append(i)
    entries = []
    for i, line in enumerate(lines, 1):
        names = [a or b for a, b in WEIGHT_RE.findall(line)]
        if not names:
            continue
        fn = next((f for f, ls in funcs.items() if i in ls), None)
        ws = []
        for n in names:
            top = n in ("W_E", "W_P", "W_U", "g_f")
            ws.append({"name": n if top else f"L{{l}}.{n}", "short": n, "shape": shapes.get(n if top else f"L0.{n}")})
        entries.append({"line": i, "code": line, "function": fn, "weights": ws})
    return {"file": "vendor/microopus/model.py", "source": src, "entries": entries}


# ------------------------------------------------------------------ training more seeds
def train_seed(kind, seed, steps=None, progress=lambda f, m: None):
    """Run Micro Opus's own train.py with another seed; returns the new model reference."""
    if kind not in ("words", "names"):
        raise ValueError("kind must be words or names")
    if _data_dir is None:
        raise RuntimeError("Data folder not set.")
    out_dir = _data_dir / "microopus"
    work = _data_dir / "microopus" / "_work"
    (work / "data").mkdir(parents=True, exist_ok=True)
    for f in ("names.txt", "NAMES_LICENSE"):
        if not (work / "data" / f).exists():
            (work / "data" / f).write_bytes((VENDOR / "data" / f).read_bytes())
    train = _vendor_module("train")
    train.ROOT = work  # train.py writes weights/<kind>.json under ROOT; redirect to the data folder
    steps = steps or (3000 if kind == "words" else 6000)
    real_print = print
    count = {"n": 0}

    def fake_print(*a, **k):
        msg = " ".join(str(x) for x in a)
        mm = re.search(r"step\s+(\d+)", msg)
        if mm:
            progress(int(mm.group(1)) / steps, f"Training Micro Opus {kind}, seed {seed}: {msg.strip()}")
        count["n"] += 1

    import builtins

    builtins.print = fake_print
    try:
        train.train(kind, steps, seed=seed)
    finally:
        builtins.print = real_print
    name = f"{kind}-seed{seed}"
    (work / "weights" / f"{kind}.json").replace(out_dir / f"{name}.json")
    return PREFIX + name
