"""A training session: one private copy of a base model with LoRA attached, and
the Tinker primitives on it.

  forward(data, loss_fn)            log-probabilities of the target tokens and the loss, no gradient
  forward_backward(data, loss_fn)   the same, and adds the loss's gradient to the adapter's gradients
  optim_step(adam)                  one Adam step with the accumulated gradient, then clears it
  sample(prompt, n, params)         completions from the current weights, with their log-probabilities

A datum is model_input tokens and loss_fn_inputs of the same length: the
target token at each position (normally the input shifted by one) and, by
loss, weights (cross_entropy), the sampler's logprobs and advantages
(importance_sampling, ppo), or per-token gradients (linear, which is how a
custom loss computed by the client comes back). Losses are sums over tokens,
as in Tinker: scale the weights to get a mean.
"""
import copy
import math
import threading
import time
import uuid

import torch

from . import lora

LOCK = threading.RLock()          # one model computation at a time on this machine
LOSSES = ("cross_entropy", "importance_sampling", "ppo", "linear")
CHUNK = 16


def max_len(model, tok=None):
    cfg = getattr(model, "config", None)
    if hasattr(model, "mo_cfg"):
        return int(model.mo_cfg["T"])
    for k in ("max_position_embeddings", "n_positions", "n_ctx", "max_sequence_length"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return min(v, 4096)
    return 1024


def _vec(x, dtype=float):
    """A list, a TensorData-like dict ({data, dtype, shape}) or a tensor, as a flat list."""
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("data")
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for v in x:
        out.extend(v if isinstance(v, list) else [v])
    return [dtype(v) for v in out]


def parse_datum(d):
    mi = d.get("model_input") or {}
    if "tokens" in d:
        toks = d["tokens"]
    elif isinstance(mi, dict) and "chunks" in mi:
        toks = []
        for c in mi["chunks"]:
            if c.get("type", "encoded_text") != "encoded_text":
                raise ValueError("Only text chunks are supported here.")
            toks.extend(c["tokens"])
    else:
        toks = mi.get("tokens") if isinstance(mi, dict) else mi
    toks = [int(t) for t in (toks or [])]
    inp = d.get("loss_fn_inputs") or {}
    out = {"tokens": toks}
    for k, v in inp.items():
        out[k] = _vec(v, int if k == "target_tokens" else float)
    if not toks:
        raise ValueError("A datum has no input tokens.")
    tgt = out.get("target_tokens")
    if tgt is None or len(tgt) != len(toks):
        raise ValueError(f"target_tokens must have one entry per input token ({len(toks)}); got {0 if tgt is None else len(tgt)}.")
    for k, v in out.items():
        if k != "tokens" and v is not None and len(v) != len(toks):
            raise ValueError(f"loss_fn_inputs[{k!r}] has {len(v)} entries; the input has {len(toks)} tokens.")
    return out


def token_logprobs(model, rows, temperature=1.0, grad=False):
    """Log-probability of each row's target tokens: a list of 1-D tensors."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(rows), CHUNK):
        part = rows[i:i + CHUNK]
        w = max(len(r["tokens"]) for r in part)
        ids = torch.tensor([r["tokens"] + [0] * (w - len(r["tokens"])) for r in part], device=dev)
        att = torch.tensor([[1] * len(r["tokens"]) + [0] * (w - len(r["tokens"])) for r in part], device=dev)
        tgt = torch.tensor([r["target_tokens"] + [0] * (w - len(r["tokens"])) for r in part], device=dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lg = model(input_ids=ids, attention_mask=att).logits.float()
            if temperature and temperature != 1.0:
                lg = lg / temperature
            lp = lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for j, r in enumerate(part):
            out.append(lp[j, :len(r["tokens"])])
    return out


def loss_of(kind, lp, r, cfg):
    dev = lp.device
    t = lambda k, default=None: torch.tensor(r[k], device=dev, dtype=torch.float32) if r.get(k) is not None else default
    if kind == "cross_entropy":
        w = t("weights", torch.ones_like(lp))
        return -(w * lp).sum()
    if kind in ("importance_sampling", "ppo"):
        old, adv = t("logprobs"), t("advantages")
        if old is None or adv is None:
            raise ValueError(f"{kind} needs loss_fn_inputs logprobs (from the sampler) and advantages.")
        ratio = (lp - old).exp()
        if kind == "importance_sampling":
            return -(ratio * adv).sum()
        lo, hi = float(cfg.get("clip_low_threshold", 0.8)), float(cfg.get("clip_high_threshold", 1.2))
        return -torch.minimum(ratio * adv, ratio.clamp(lo, hi) * adv).sum()
    if kind == "linear":
        g = t("grads")
        if g is None:
            raise ValueError("linear needs loss_fn_inputs grads.")
        return (g * lp).sum()
    raise ValueError(f"Unknown loss {kind!r}. Choose from {', '.join(LOSSES)}.")


def generate(model, prompt, n=1, max_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop=(), seed=None, tok=None):
    """Sample n completions of prompt (token ids). Returns [{tokens, logprobs, stop_reason}].
    logprobs are of the distribution actually sampled from (after temperature, before top-p/top-k)."""
    dev = next(model.parameters()).device
    limit = max_len(model)
    if len(prompt) >= limit:
        raise ValueError(f"The prompt has {len(prompt)} tokens; this model reads at most {limit}.")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31))
    stop_ids = {int(s) for s in stop if isinstance(s, int)}
    stop_str = [s for s in stop if isinstance(s, str) and s]
    eos = getattr(tok, "eos_token_id", None) if tok is not None else None
    if isinstance(eos, int) and not hasattr(model, "mo_cfg"):
        stop_ids.add(eos)
    seqs = [list(prompt) for _ in range(n)]
    gen = [[] for _ in range(n)]
    lps = [[] for _ in range(n)]
    why = [None] * n
    with torch.no_grad():
        for _ in range(max_tokens):
            live = [i for i in range(n) if why[i] is None]
            if not live:
                break
            if len(seqs[live[0]]) >= limit:
                for i in live:
                    why[i] = "length"
                break
            ids = torch.tensor([seqs[i] for i in live], device=dev)
            lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1].float().cpu()
            if temperature and temperature > 0:
                lp = (lg / temperature).log_softmax(-1)
                pr = lp.exp()
                if top_k and top_k > 0:
                    kth = pr.topk(min(int(top_k), pr.shape[-1]), -1).values[:, -1:]
                    pr = torch.where(pr >= kth, pr, torch.zeros_like(pr))
                if top_p and top_p < 1.0:
                    sp, si = pr.sort(-1, descending=True)
                    keep = (sp.cumsum(-1) - sp) < top_p
                    pr = torch.zeros_like(pr).scatter(-1, si, sp * keep)
                nxt = torch.multinomial(pr / pr.sum(-1, keepdim=True), 1, generator=g).squeeze(-1)
            else:
                lp = lg.log_softmax(-1)
                nxt = lg.argmax(-1)
            for j, i in enumerate(live):
                t = int(nxt[j])
                seqs[i].append(t)
                gen[i].append(t)
                lps[i].append(float(lp[j, t]))
                if t in stop_ids:
                    why[i] = "stop"
                elif stop_str and tok is not None and any(s in tok.decode(gen[i]) for s in stop_str):
                    why[i] = "stop"
    return [{"tokens": gen[i], "logprobs": lps[i], "stop_reason": why[i] or "length"} for i in range(n)]


def prompt_logprobs(model, tokens):
    """Log-probability of each prompt token given the ones before it (None for the first)."""
    if len(tokens) < 2:
        return [None] * len(tokens)
    lp = token_logprobs(model, [{"tokens": tokens[:-1], "target_tokens": tokens[1:]}])[0]
    return [None] + [float(x) for x in lp.cpu()]


class Session:
    """A LoRA training client's state on the server."""

    def __init__(self, spec, base, tok, rank=32, alpha=32.0, seed=0, train_attn=True, train_mlp=True, train_unembed=False,
                 name="", source="script"):
        self.id = uuid.uuid4().hex[:10]
        self.spec, self.tok, self.base = spec, tok, base
        self.model = copy.deepcopy(base)
        self.model.eval()                      # no dropout: forward passes are exact and repeatable
        self.mats = lora.select(self.model, train_attn, train_mlp, train_unembed)
        self.lora = lora.attach(self.model, self.mats, rank, alpha, seed)
        self.config = {"spec": spec, "rank": int(rank), "alpha": float(alpha), "seed": int(seed), "train_attn": bool(train_attn),
                       "train_mlp": bool(train_mlp), "train_unembed": bool(train_unembed)}
        self.name = name or f"{spec.split('/')[-1].replace(':', '-')}-r{rank}"
        self.source = source
        self.opt = None
        self.step = 0
        self.created = self.last = time.time()
        self.log = []                      # one row per optimizer step
        self._acc = {"loss": 0.0, "weight": 0.0, "tokens": 0, "calls": 0}
        self.metrics = {}                  # extra numbers the training script reports for the current step
        self.closed = False

    # -------------------------------------------------------------- info
    def params(self):
        for m in self.lora.values():
            yield m.U
            yield m.V

    def trainable(self):
        return int(sum(p.numel() for p in self.params()))

    def public(self, with_log=True):
        d = {"id": self.id, "name": self.name, "spec": self.spec, "config": self.config, "source": self.source, "step": self.step,
             "created": self.created, "last": self.last, "trainable": self.trainable(),
             "base_params": int(sum(p.numel() for p in self.base.parameters())), "matrices": len(self.lora),
             "closed": self.closed, "live": not self.closed}
        if with_log:
            d["log"] = self.log[-2000:]
        return d

    # -------------------------------------------------------------- the primitives
    def _rows(self, data):
        rows = [parse_datum(d) for d in data]
        lim = max_len(self.model)
        for r in rows:
            if len(r["tokens"]) > lim:
                raise ValueError(f"A datum has {len(r['tokens'])} tokens; this model reads at most {lim}.")
        return rows

    def forward(self, data, loss_fn="cross_entropy", loss_fn_config=None, backward=False):
        if loss_fn not in LOSSES:
            raise ValueError(f"Unknown loss {loss_fn!r}. Choose from {', '.join(LOSSES)}.")
        cfg = loss_fn_config or {}
        rows = self._rows(data)
        with LOCK:
            self.last = time.time()
            total, outs, wsum, ntok = 0.0, [], 0.0, 0
            for i in range(0, len(rows), CHUNK):
                part = rows[i:i + CHUNK]
                lps = token_logprobs(self.model, part, grad=backward)
                losses = [loss_of(loss_fn, lp, r, cfg) for lp, r in zip(lps, part)]
                L = torch.stack(losses).sum()
                if backward:
                    L.backward()
                total += float(L.detach())
                for lp, r in zip(lps, part):
                    outs.append({"logprobs": [float(x) for x in lp.detach().cpu()]})
                    w = r.get("weights")
                    wsum += sum(w) if (w is not None and loss_fn == "cross_entropy") else len(r["tokens"])
                    ntok += len(r["tokens"])
            if backward:
                a = self._acc
                a["loss"] += total
                a["weight"] += wsum
                a["tokens"] += ntok
                a["calls"] += 1
                a["kind"] = loss_fn
        return {"loss_fn_outputs": outs, "metrics": {"loss:sum": total, "weights:sum": wsum, "tokens:sum": ntok,
                                                      "loss:mean": total / max(wsum, 1e-9)}}

    def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        return self.forward(data, loss_fn, loss_fn_config, backward=True)

    def optim_step(self, learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0):
        with LOCK:
            params = list(self.params())
            if not any(p.grad is not None for p in params):
                raise ValueError("No gradient to apply: call forward_backward before optim_step.")
            if self.opt is None:
                self.opt = torch.optim.AdamW(params, lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
            for g in self.opt.param_groups:
                g.update(lr=float(learning_rate), betas=(float(beta1), float(beta2)), eps=float(eps), weight_decay=float(weight_decay))
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm if grad_clip_norm and grad_clip_norm > 0 else float("inf"))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            self.last = time.time()
            a = self._acc
            row = {"step": self.step, "t": round(self.last - self.created, 2), "lr": float(learning_rate),
                   "grad_norm": float(gn), "tokens": a["tokens"], "loss_kind": a.get("kind")}
            if a["calls"]:
                row["loss"] = a["loss"] / max(a["weight"], 1e-9)
            row.update({k: v for k, v in self.metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)})
            self.log.append(row)
            self._acc = {"loss": 0.0, "weight": 0.0, "tokens": 0, "calls": 0}
            self.metrics = {}
            return {"metrics": {"step": self.step, "grad_norm": float(gn), "learning_rate": float(learning_rate)}}

    def report(self, metrics):
        """Numbers the training script wants on the chart for this step (reward, accuracy...)."""
        for k, v in (metrics or {}).items():
            try:
                self.metrics[str(k)[:40]] = float(v)
            except (TypeError, ValueError):
                continue
        return {"ok": True}

    def sample(self, prompt, num_samples=1, max_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop=(), seed=None):
        with LOCK:
            self.last = time.time()
            return generate(self.model, prompt, num_samples, max_tokens, temperature, top_p, top_k, stop, seed, self.tok)

    # -------------------------------------------------------------- saving
    def factors(self):
        return lora.factors(self.lora)

    def save_adapter(self, name):
        info = {"session": self.id, "config": self.config, "step": self.step, "log": self.log[-2000:],
                "source": self.source, "session_name": self.name}
        return lora.save_adapter(name or f"{self.name}-step{self.step}", self.spec, self.factors(), info)

    def state(self, with_optimizer=True):
        return {"config": self.config, "name": self.name, "step": self.step, "log": self.log,
                "adapter": {n: {"U": m.U.detach().cpu(), "V": m.V.detach().cpu()} for n, m in self.lora.items()},
                "opt": self.opt.state_dict() if (with_optimizer and self.opt is not None) else None}

    def load_state(self, st, with_optimizer=True):
        with LOCK, torch.no_grad():
            missing = [n for n in self.lora if n not in st["adapter"]]
            if missing:
                raise ValueError(f"The saved state does not match this model ({len(missing)} matrices missing).")
            for n, m in self.lora.items():
                m.U.copy_(st["adapter"][n]["U"].to(m.U.device))
                m.V.copy_(st["adapter"][n]["V"].to(m.V.device))
            self.step = int(st.get("step", 0))
            self.log = list(st.get("log") or [])
            if with_optimizer and st.get("opt"):
                self.opt = torch.optim.AdamW(list(self.params()))
                self.opt.load_state_dict(st["opt"])
