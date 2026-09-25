"""A small transformer whose efficiency levers can be switched on one at a time,
plus the tasks and the training loop used by the experiment engine.

Levers
  share    layers reuse a smaller set of blocks (looped transformer): fewer parameters
  topk     each token keeps only its k strongest MLP neurons: less compute in the MLP
  lowrank  every weight matrix is a product of two thin matrices: fewer parameters
           and less compute
  seed     number embeddings start as waves at a few frequencies (modular addition)

Compute is counted, not guessed: every matrix multiply a token goes through is
counted as 2 x its multiply-adds in the forward pass, attention scores are
added, and a training step costs 3 x the forward pass.
"""
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ model
class Lin(nn.Module):
    """Linear layer, optionally factored as in -> r -> out."""

    def __init__(self, i, o, rank=0, bias=True):
        super().__init__()
        self.i, self.o, self.rank = i, o, rank
        if rank and rank < min(i, o):
            self.a = nn.Linear(i, rank, bias=False)
            self.b = nn.Linear(rank, o, bias=bias)
        else:
            self.rank = 0
            self.w = nn.Linear(i, o, bias=bias)

    def forward(self, x):
        return self.b(self.a(x)) if self.rank else self.w(x)

    def macs(self):
        return self.rank * (self.i + self.o) if self.rank else self.i * self.o


class Block(nn.Module):
    def __init__(self, d, h, dmlp, rank=0, topk=1.0):
        super().__init__()
        self.h, self.dmlp, self.topk = h, dmlp, topk
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = Lin(d, 3 * d, rank), Lin(d, d, rank)
        self.fc1, self.fc2 = Lin(d, dmlp, rank), Lin(dmlp, d, rank)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        sh = lambda z: z.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        att = att.masked_fill(~mask[:T, :T], float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        a = F.gelu(self.fc1(self.ln2(x)))
        if self.topk < 1.0:
            k_ = max(1, int(round(self.topk * self.dmlp)))
            idx = a.abs().topk(k_, dim=-1).indices
            a = a * torch.zeros_like(a).scatter_(-1, idx, 1.0)
        return x + self.fc2(a)

    def macs_per_token(self, T):
        down = self.fc2.macs() * (self.topk if self.topk < 1 else 1.0)
        attn = 2 * T * self.ln1.normalized_shape[0]  # scores + mixing, averaged over positions ~ T/2 each
        return self.qkv.macs() + self.proj.macs() + self.fc1.macs() + down + attn / 2


class Nano(nn.Module):
    def __init__(self, vocab, ctx, d=128, layers=4, heads=4, dmlp=512, share=0, rank=0, topk=1.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, ctx=ctx, d=d, layers=layers, heads=heads, dmlp=dmlp, share=share, rank=rank, topk=topk)
        n_unique = share if share and share < layers else layers
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads, dmlp, rank, topk) for _ in range(n_unique)])
        self.order = [i % n_unique for i in range(layers)]
        self.lnf = nn.LayerNorm(d)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx)).bool(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for i in self.order:
            h = self.blocks[i](h, self.mask)
        return self.lnf(h) @ self.tok.weight.T  # tied output layer

    def params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, T):
        macs = sum(self.blocks[i].macs_per_token(T) for i in self.order) + self.tok.weight.numel()
        return 2 * macs


# ------------------------------------------------------------------ tasks
class Task:
    name, metric, target, higher_better = "", "accuracy", 0.99, True

    def batch(self, split, n, g):
        raise NotImplementedError

    def score(self, model, g):
        raise NotImplementedError


class ModAdd(Task):
    """(a + b) mod p, answer at the last position. Fixed train/test split."""
    name, metric, target = "modadd", "test accuracy", 0.95

    def __init__(self, p=53, frac=0.5):
        self.p, self.vocab, self.ctx = p, p + 1, 3
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        self.x = torch.stack([a, b, torch.full_like(a, p)], 1)
        self.y = (a + b) % p
        perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(0))
        n = int(frac * p * p)
        self.tr, self.te = perm[:n], perm[n:]
        self.full_batch = True
        self.answer_positions = 1

    def train_batch(self, g, n=None):
        return self.x[self.tr], self.y[self.tr]

    def loss(self, model, x, y):
        return F.cross_entropy(model(x)[:, -1], y)

    def score(self, model, g):
        with torch.no_grad():
            return float((model(self.x[self.te])[:, -1].argmax(-1) == self.y[self.te]).float().mean())

    def tokens_per_step(self):
        return len(self.tr) * 3


class SeqTask(Task):
    """Copy or sort a sequence of symbols: [x1..xn, SEP, y1..yn], loss on the y's."""
    target = 0.98

    def __init__(self, kind="sort", n=8, symbols=16, batch=64):
        self.kind, self.n, self.sym, self.bs = kind, n, symbols, batch
        self.name = kind
        self.metric = "exact-sequence accuracy"
        self.vocab, self.ctx = symbols + 1, 2 * n + 1
        self.full_batch = False
        self.answer_positions = n

    def _make(self, bs, g):
        x = torch.randint(0, self.sym, (bs, self.n), generator=g)
        y = x.sort(1).values if self.kind == "sort" else x
        seq = torch.cat([x, torch.full((bs, 1), self.sym), y], 1)
        return seq, y

    def train_batch(self, g, n=None):
        return self._make(n or self.bs, g)

    def loss(self, model, seq, y):
        lg = model(seq[:, :-1])[:, self.n:]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))

    def score(self, model, g):
        seq, y = self._make(512, torch.Generator().manual_seed(999))
        with torch.no_grad():
            pred = model(seq[:, :-1])[:, self.n:].argmax(-1)
        return float((pred == y).all(1).float().mean())

    def tokens_per_step(self):
        return self.bs * (self.ctx - 1)


class CharLM(Task):
    """Byte-level language modelling. Capability = validation bits per character (lower is better)."""
    name, metric, target, higher_better = "tinystories", "validation bits per character", None, False

    def __init__(self, text: str, ctx=64, batch=32, source=""):
        data = torch.tensor(list(text.encode("utf-8", "ignore")), dtype=torch.long)
        n = int(len(data) * 0.9)
        self.train, self.val = data[:n], data[n:]
        self.vocab, self.ctx, self.bs = 256, ctx, batch
        self.full_batch = False
        self.source = source
        self.answer_positions = ctx

    def _get(self, d, bs, g):
        i = torch.randint(0, len(d) - self.ctx - 1, (bs,), generator=g)
        x = torch.stack([d[j:j + self.ctx] for j in i])
        y = torch.stack([d[j + 1:j + self.ctx + 1] for j in i])
        return x, y

    def train_batch(self, g, n=None):
        return self._get(self.train, n or self.bs, g)

    def loss(self, model, x, y):
        lg = model(x)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))

    def score(self, model, g):
        gg = torch.Generator().manual_seed(123)
        with torch.no_grad():
            tot = 0.0
            for _ in range(8):
                x, y = self._get(self.val, 32, gg)
                tot += float(self.loss(model, x, y))
        return tot / 8 / math.log(2)

    def tokens_per_step(self):
        return self.bs * self.ctx


def tinystories_text(cache: Path, max_chars=3_000_000):
    """TinyStories validation text (about 19 MB); falls back to a built-in corpus offline."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "tinystories-valid.txt"
    if not f.exists():
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")
            f.write_text(Path(p).read_text(encoding="utf-8", errors="ignore")[:max_chars])
        except Exception:
            from ..atlas.probes import anchor_texts

            return (" ".join(anchor_texts()) + " ") * 40, "built-in fallback corpus (TinyStories could not be downloaded)"
    return f.read_text(encoding="utf-8", errors="ignore")[:max_chars], "TinyStories (validation split)"


TASKS = {"modadd": "Modular addition, (a + b) mod 53", "sort": "Sort 8 symbols", "copy": "Copy 8 symbols",
         "tinystories": "TinyStories, byte-level language modelling"}


def make_task(name, data_dir: Path):
    if name == "modadd":
        return ModAdd()
    if name in ("sort", "copy"):
        return SeqTask(name)
    if name == "tinystories":
        text, src = tinystories_text(data_dir / "datasets")
        return CharLM(text, source=src)
    raise ValueError(f"Unknown task {name}.")


DEFAULT_ARCH = {"modadd": dict(d=128, layers=1, heads=4, dmlp=512),
                "sort": dict(d=64, layers=4, heads=4, dmlp=256),
                "copy": dict(d=64, layers=4, heads=4, dmlp=256),
                "tinystories": dict(d=128, layers=6, heads=4, dmlp=512)}
DEFAULT_OPT = {"modadd": dict(lr=1e-3, wd=1.0, steps=6000), "sort": dict(lr=1e-3, wd=0.1, steps=4000),
               "copy": dict(lr=1e-3, wd=0.1, steps=3000), "tinystories": dict(lr=2e-3, wd=0.1, steps=1500)}


def fourier_seed(model: Nano, p, k=5, seed=0):
    from ..lab.seeded import fourier_embedding

    with torch.no_grad():
        scale = float(model.tok.weight[:p].norm(dim=1).mean())
        W, _ = fourier_embedding(p, model.tok.weight.shape[1], k, seed, scale)
        model.tok.weight[:p] = W


# ------------------------------------------------------------------ training
def train(task, arch: dict, levers: dict, seed: int, steps: int, lr: float, wd: float,
          progress=lambda f, m: None, should_stop=lambda: False, label=""):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = Nano(task.vocab, task.ctx, **arch, share=levers.get("share", 0), rank=levers.get("rank", 0), topk=levers.get("topk", 1.0))
    if levers.get("seed") and isinstance(task, ModAdd):
        fourier_seed(m, task.p, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    tok_step = task.tokens_per_step()
    fpt = m.flops_per_token(task.ctx)
    hist = {"step": [], "score": []}
    reached, streak = None, 0
    every = max(10, steps // 60)
    t0 = time.time()
    step = 0
    for step in range(steps + 1):
        if should_stop():
            raise InterruptedError("Stopped.")
        if step % every == 0 or step == steps:
            m.eval()
            sc = task.score(m, g)
            m.train()
            hist["step"].append(step)
            hist["score"].append(round(sc, 4))
            if task.target is not None:
                ok = sc >= task.target
                streak = streak + 1 if ok else 0
                if streak == 1:
                    reached = step
                if not ok:
                    reached = None
                if streak >= 2:
                    break
            progress(step / steps, f"{label} seed {seed}: step {step}, {task.metric} {sc:.3f}")
        if step == steps:
            break
        x, y = task.train_batch(g)
        loss = task.loss(m, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_flops = 3 * fpt * tok_step
    return {"seed": seed, "params": m.params(), "flops_per_token": fpt, "steps_run": step,
            "steps_to_target": reached, "final": hist["score"][-1], "history": hist,
            "flops_to_target": reached * train_flops if reached is not None else None,
            "flops_total": step * train_flops, "seconds": round(time.time() - t0, 1)}
