"""The two recipes the Tune tab runs without code. Both are written with the same
primitives a script uses (datums, forward_backward, optim_step, sample), on a
session in this process, so what the form does is exactly what the shown
Python does.

  supervised   "prompt => target" lines (or plain lines): cross-entropy on the
               target tokens only. Held-out loss and accuracy before, during and
               after; greedy completions of held-out prompts before and after.
  reinforce    prompts and a reward written as a rule. Each step samples a group
               of completions per prompt, scores them, and pushes up the ones
               above their group's mean (importance_sampling loss, the
               group-relative advantage of GRPO). Mean reward on held-out
               prompts before and after, with the same number of samples.

Both report how far the fine-tune moved the model on general text (mean KL
per token), and save the adapter.
"""
import random
import re
import time

import numpy as np
import torch

from ..wsp.common import drift, encode, general_texts, next_token_logp, parse_examples, split
from .session import LOCK, generate, token_logprobs

# ------------------------------------------------------------------ rewards
REWARDS = {
    "contains": "Uses any of these words",
    "avoids": "Avoids all of these words",
    "target": "Starts with the target after =>",
    "length": "Has this many words",
    "regex": "Matches this regular expression",
}


def _words(s):
    return [w for w in re.split(r"[\s,]+", (s or "").lower()) if w]


def reward_fn(kind, arg):
    if kind not in REWARDS:
        raise ValueError(f"Unknown reward {kind!r}.")
    if kind in ("contains", "avoids"):
        ws = _words(arg)
        if not ws:
            raise ValueError("Give at least one word for this reward.")
        pat = re.compile(r"\b(" + "|".join(map(re.escape, ws)) + r")\b")
        return (lambda text, target=None: 1.0 if pat.search(text.lower()) else 0.0) if kind == "contains" else \
            (lambda text, target=None: 0.0 if pat.search(text.lower()) else 1.0)
    if kind == "length":
        n = int(float(arg or 5))
        return lambda text, target=None: max(0.0, 1.0 - abs(len(_words(text)) - n) / max(n, 1))
    if kind == "regex":
        rx = re.compile(arg or ".")
        return lambda text, target=None: 1.0 if rx.search(text) else 0.0

    def tgt(text, target=None):
        t, x = (target or "").strip().lower(), text.strip().lower()
        if not t:
            return 0.0
        k = 0
        while k < min(len(t), len(x)) and t[k] == x[k]:
            k += 1
        return 1.0 if k == len(t) else k / len(t) * 0.5
    return tgt


def parse_prompts(text):
    out = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=>" in s:
            p, t = s.split("=>", 1)
            out.append({"prompt": p.rstrip(), "target": t.strip()})
        else:
            out.append({"prompt": s, "target": None})
    return out


# ------------------------------------------------------------------ helpers
def _ids(tok, text, bos=True):
    ids = tok(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    if not ids:                                   # an empty prompt: start from the model's start (or end) token
        b = getattr(tok, "bos_token_id", None)
        ids = [b if isinstance(b, int) else (getattr(tok, "eos_token_id", None) or 0)]
    return ids


def _decode(tok, ids):
    return tok.decode(ids) if ids else ""


def sl_datum(ids, mask):
    """One supervised datum: input = ids[:-1], target = ids[1:], weight = mask[1:]."""
    return {"model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
            "loss_fn_inputs": {"target_tokens": ids[1:], "weights": [float(m) for m in mask[1:]]}}


def _eval_rows(model, rows):
    data = [sl_datum(f, m) for f, m in rows]
    from .session import parse_datum

    pr = [parse_datum(d) for d in data]
    lps = token_logprobs(model, pr)
    tot, n = 0.0, 0.0
    for lp, r in zip(lps, pr):
        w = torch.tensor(r["weights"], device=lp.device)
        tot += float(-(w * lp).sum())
        n += float(w.sum())
    return tot / max(n, 1e-9)


def _acc_rows(model, rows):
    dev = next(model.parameters()).device
    hits, n = 0, 0
    with torch.no_grad():
        for f, m in rows:
            ids = torch.tensor([f], device=dev)
            pred = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1].argmax(-1).cpu().tolist()
            for i, keep in enumerate(m[1:]):
                if keep:
                    hits += int(pred[i] == f[i + 1])
                    n += 1
    return hits / max(n, 1)


def _greedy(model, tok, prompts, max_tokens):
    out = []
    for p in prompts:
        ids = _ids(tok, p)
        g = generate(model, ids, 1, max_tokens, temperature=0.0, tok=tok)[0]
        out.append(_decode(tok, g["tokens"]))
    return out


# ------------------------------------------------------------------ supervised
def supervised(svc, spec, examples, held_out="", rank=8, alpha=32.0, lr=1e-3, steps=100, batch=8, seed=0,
               train_attn=True, train_mlp=True, name="", eval_every=0, progress=lambda f, m, live=None: None,
               should_stop=lambda: False):
    t0 = time.time()
    progress(0.01, f"Loading {spec}")
    sess = svc.create(spec, rank, alpha, seed, train_attn, train_mlp, False, name, source="ui")
    tok, base = sess.tok, sess.base
    ex = parse_examples(examples)
    ho = parse_examples(held_out) if held_out and held_out.strip() else None
    practice, held = split(ex, ho)
    prow, hrow = encode(tok, practice), encode(tok, held)
    gen = general_texts(tok, model=base)
    with LOCK:
        base_gen = next_token_logp(base, tok, gen)
        before = {"held_loss": _eval_rows(base, hrow), "held_acc": _acc_rows(base, hrow), "practice_loss": _eval_rows(base, prow)}
    show = [e for e in held if e["prompt"]][:6]
    max_new = max(4, min(24, max((len(_ids(tok, e["target"])) for e in show), default=8) + 2))
    with LOCK:
        samples_before = _greedy(base, tok, [e["prompt"] for e in show], max_new)
    eval_every = eval_every or max(1, steps // 10)
    g = random.Random(seed)
    curve = [{"step": 0, "held": before["held_loss"], "train": before["practice_loss"]}]
    for step in range(1, steps + 1):
        if should_stop():
            raise InterruptedError("Stopped.")
        part = [prow[g.randrange(len(prow))] for _ in range(min(batch, len(prow)))]
        n_tok = sum(sum(m[1:]) for _, m in part)
        data = [sl_datum(f, [x / max(n_tok, 1) for x in m]) for f, m in part]     # weights sum to 1: the loss is a mean
        fb = sess.forward_backward(data, "cross_entropy")
        sess.optim_step(learning_rate=lr)
        row = {"step": step, "train": fb["metrics"]["loss:sum"]}
        if step % eval_every == 0 or step == steps:
            with LOCK:
                row["held"] = _eval_rows(sess.model, hrow)
        curve.append(row)
        progress(0.05 + 0.85 * step / steps, f"Step {step} of {steps}: loss {row['train']:.3f}"
                 + (f", held-out {row['held']:.3f}" if "held" in row else ""), {"curve": curve, "session": sess.id})
    with LOCK:
        after = {"held_loss": _eval_rows(sess.model, hrow), "held_acc": _acc_rows(sess.model, hrow),
                 "practice_loss": _eval_rows(sess.model, prow)}
        samples_after = _greedy(sess.model, tok, [e["prompt"] for e in show], max_new)
        kl = drift(base_gen, next_token_logp(sess.model, tok, gen))
    meta = sess.save_adapter(name or f"{sess.name}-sl")
    progress(0.95, "Reading where the fine-tune went")
    from .anatomy import anatomy

    svc.close(sess.id)
    return {"kind": "supervised", "spec": spec, "adapter": meta["name"], "adapter_spec": meta["spec"], "session": sess.id,
            "settings": {"rank": rank, "alpha": alpha, "lr": lr, "steps": steps, "batch": batch, "seed": seed,
                         "train_attn": train_attn, "train_mlp": train_mlp},
            "examples": {"practice": len(practice), "held_out": len(held)}, "trainable": meta["trainable"],
            "base_params": int(sum(p.numel() for p in base.parameters())),
            "before": before, "after": after, "drift_kl": kl, "curve": curve,
            "samples": [{"prompt": e["prompt"], "target": e["target"].strip(), "before": b, "after": a}
                        for e, b, a in zip(show, samples_before, samples_after)],
            "anatomy": anatomy(svc, meta["name"]), "seconds": time.time() - t0}


# ------------------------------------------------------------------ reinforcement
def _score(model, tok, items, rf, group, max_tokens, temperature, seed):
    rs, examples = [], []
    for k, it in enumerate(items):
        ids = _ids(tok, it["prompt"])
        outs = generate(model, ids, group, max_tokens, temperature, seed=seed * 1000 + k, tok=tok)
        texts = [_decode(tok, o["tokens"]) for o in outs]
        r = [rf(t, it["target"]) for t in texts]
        rs.extend(r)
        examples.append({"prompt": it["prompt"], "target": it["target"], "texts": texts[:4], "rewards": r[:4]})
    return float(np.mean(rs)) if rs else 0.0, examples


def reinforce(svc, spec, prompts, reward="contains", reward_arg="", held_out="", rank=8, alpha=32.0, lr=5e-3, steps=60,
              batch=4, group=8, max_tokens=12, temperature=1.0, seed=0, train_attn=True, train_mlp=True, name="", clip=1.0,
              kl_coef=0.05, progress=lambda f, m, live=None: None, should_stop=lambda: False):
    t0 = time.time()
    rf = reward_fn(reward, reward_arg)
    items = parse_prompts(prompts)
    if not items:
        raise ValueError("Give at least one prompt.")
    held = parse_prompts(held_out) if held_out and held_out.strip() else items
    progress(0.01, f"Loading {spec}")
    sess = svc.create(spec, rank, alpha, seed, train_attn, train_mlp, False, name, source="ui")
    tok, base = sess.tok, sess.base
    gen = general_texts(tok, model=base)
    eval_n = max(group, 8)
    with LOCK:
        base_gen = next_token_logp(base, tok, gen)
        r_before, ex_before = _score(base, tok, held, rf, eval_n, max_tokens, temperature, 99)
    g = random.Random(seed)
    curve = [{"step": 0, "reward": r_before}]
    for step in range(1, steps + 1):
        if should_stop():
            raise InterruptedError("Stopped.")
        part = [items[g.randrange(len(items))] for _ in range(min(batch, len(items)))]
        data, rewards, skipped, kls = [], [], 0, []
        for k, it in enumerate(part):
            ids = _ids(tok, it["prompt"])
            outs = sess.sample(ids, group, max_tokens, temperature, seed=seed * 100003 + step * 101 + k)
            r = [rf(_decode(tok, o["tokens"]), it["target"]) for o in outs]
            rewards.extend(r)
            mu = float(np.mean(r))
            if max(r) - min(r) < 1e-9 and kl_coef <= 0:   # every sample scored the same: nothing to learn here
                skipped += 1
                continue
            for o, ri in zip(outs, r):
                if not o["tokens"]:
                    continue
                full = ids + o["tokens"]
                n_p = len(ids) - 1
                adv = [0.0] * n_p + [ri - mu] * len(o["tokens"])
                if kl_coef > 0:                    # stay near the base model: penalise each token by log(pi / pi_base)
                    with LOCK:
                        lb = token_logprobs(base, [{"tokens": full[:-1], "target_tokens": full[1:]}])[0][n_p:].cpu().tolist()
                    kls.extend(a - b for a, b in zip(o["logprobs"], lb))
                    adv = adv[:n_p] + [x - kl_coef * (a - b) for x, a, b in zip(adv[n_p:], o["logprobs"], lb)]
                old = [0.0] * n_p + o["logprobs"]
                data.append({"model_input": {"chunks": [{"type": "encoded_text", "tokens": full[:-1]}]},
                             "loss_fn_inputs": {"target_tokens": full[1:], "logprobs": old, "advantages": adv}})
        mean_r = float(np.mean(rewards)) if rewards else 0.0
        sess.report({"reward": mean_r, **({"kl_to_base": float(np.mean(kls))} if kls else {})})
        if data:
            n_gen = sum(sum(1 for a in d["loss_fn_inputs"]["advantages"] if a != 0.0) for d in data)
            for d in data:                       # the loss becomes a mean over the sampled tokens
                d["loss_fn_inputs"]["advantages"] = [a / max(n_gen, 1) for a in d["loss_fn_inputs"]["advantages"]]
            sess.forward_backward(data, "importance_sampling")
            sess.optim_step(learning_rate=lr, grad_clip_norm=clip)
        curve.append({"step": step, "reward": mean_r, "groups_skipped": skipped, "kl": float(np.mean(kls)) if kls else None})
        progress(0.05 + 0.85 * step / steps, f"Step {step} of {steps}: mean reward {mean_r:.2f}"
                 + (f" ({skipped} groups all alike)" if skipped else ""), {"curve": curve, "session": sess.id})
    with LOCK:
        r_after, ex_after = _score(sess.model, tok, held, rf, eval_n, max_tokens, temperature, 99)
        kl = drift(base_gen, next_token_logp(sess.model, tok, gen))
    meta = sess.save_adapter(name or f"{sess.name}-rl")
    progress(0.95, "Reading where the fine-tune went")
    from .anatomy import anatomy

    svc.close(sess.id)
    return {"kind": "reinforce", "spec": spec, "adapter": meta["name"], "adapter_spec": meta["spec"], "session": sess.id,
            "settings": {"rank": rank, "alpha": alpha, "lr": lr, "steps": steps, "batch": batch, "group": group,
                         "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "reward": reward,
                         "reward_arg": reward_arg, "clip": clip, "kl_coef": kl_coef, "train_attn": train_attn, "train_mlp": train_mlp},
            "reward_label": REWARDS[reward], "trainable": meta["trainable"],
            "base_params": int(sum(p.numel() for p in base.parameters())),
            "before": {"reward": r_before}, "after": {"reward": r_after}, "drift_kl": kl, "curve": curve,
            "samples": [{"prompt": b["prompt"], "target": b["target"], "before": b["texts"], "before_r": b["rewards"],
                         "after": a["texts"], "after_r": a["rewards"]} for b, a in zip(ex_before[:6], ex_after[:6])],
            "anatomy": anatomy(svc, meta["name"]), "seconds": time.time() - t0}
