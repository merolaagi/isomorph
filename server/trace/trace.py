"""Trace one prompt through two models, side by side.

During inference the weights never change: the prompt produces activations
that flow through fixed weights. What the input does change is which weights
matter, so the trace records, layer by layer:

  * logit lens: what each model would predict if it stopped at this layer
  * divergence: where the two models' predictions part ways (same vocabulary)
  * alignment error: how far A's state is from B's after the best affine map,
    fitted on a separate probe corpus, so coordinate differences don't count
  * attention patterns and the best-matching head in the other model
  * MLP usage: how few neurons carry most of the activity for each token

Then it answers the literal version of "how do weights change with a new
input": one gradient step on the prompt, per layer, for both models.
"""
import math

import torch
import torch.nn.functional as F

from .. import metrics as M
from ..hub.acts import ACT_RE, _depth_map, _layer_idx, _rows, run_model
from ..hub.corpus import CORPUS
from ..hub.load import load_lm
from ..hub.weights import _kind, _layer_of

MAX_TOKENS = 40
_calib_cache = {}


def final_norm(model):
    base = getattr(model, getattr(model, "base_model_prefix", ""), model)
    for attr in ("ln_f", "final_layer_norm", "norm", "final_norm", "ln_final"):
        m = getattr(base, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _decode(tok, i):
    try:
        return tok.decode([int(i)])
    except Exception:
        return str(int(i))


def _forward(tok, model, prompt):
    dev = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    ids = enc["input_ids"][:, :MAX_TOKENS]
    acts = {}
    hooks = []
    for name, mod in model.named_modules():
        if ACT_RE.search(name):
            li = _layer_idx(name)
            if li is not None:
                hooks.append(mod.register_forward_hook(lambda _m, _i, o, li=li: acts.__setitem__(li, o.detach()[0].float().cpu())))
    try:
        with torch.no_grad():
            out = model(input_ids=ids.to(dev), output_hidden_states=True, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()
    hidden = [h[0].float() for h in out.hidden_states]
    atts = [a[0].float().cpu() for a in out.attentions] if getattr(out, "attentions", None) and out.attentions[0] is not None else None
    return ids[0].tolist(), hidden, out.logits[0].float(), atts, acts


def _lens(model, hidden, logits):
    """Log-probabilities per layer (L+1, T, V) computed one layer at a time."""
    norm = final_norm(model)
    head = model.get_output_embeddings()
    n = len(hidden)
    for i, h in enumerate(hidden):
        if i == n - 1:
            yield logits.log_softmax(-1)
        else:
            with torch.no_grad():
                z = norm(h) if norm is not None else h
                yield head(z).float().log_softmax(-1)


def _mlp_usage(acts, n_layers):
    """Share of neurons that carry 90% of the absolute activation, per layer and token."""
    out = []
    for li in range(n_layers):
        a = acts.get(li)
        if a is None:
            out.append(None)
            continue
        v = a.abs()
        s, idx = v.sort(-1, descending=True)
        cum = s.cumsum(-1) / (s.sum(-1, keepdim=True) + 1e-12)
        share = ((cum < 0.9).sum(-1) + 1).float() / v.shape[-1]
        top = [{"idx": idx[t, :5].tolist(), "val": [round(x, 3) for x in a[t, idx[t, :5]].tolist()]} for t in range(a.shape[0])]
        out.append({"share": [round(x, 4) for x in share.tolist()], "units": int(v.shape[-1]), "top": top})
    return out


def _calibrate(spec_a, spec_b, tok_a, A, tok_b, B, progress):
    """Per-layer affine maps A -> B fitted on the probe corpus (cached per pair)."""
    key = (spec_a, spec_b)
    if key in _calib_cache:
        return _calib_cache[key]
    progress(0.35, "Calibrating: running both models on the probe corpus")
    ids_a, hid_a, _, _ = run_model(tok_a, A, CORPUS, set())
    ids_b, hid_b, _, _ = run_model(tok_b, B, CORPUS, set())
    if ids_a != ids_b:
        _calib_cache[key] = None
        return None
    la, lb = len(hid_a[0]) - 1, len(hid_b[0]) - 1
    half = len(CORPUS) // 2
    maps = []
    for i in range(la + 1):
        j = round(i * lb / max(1, la))
        src = _rows([h[i] for h in hid_a], cap=3000)
        dst = _rows([h[j] for h in hid_b], cap=3000)
        s_fit = _rows([h[i] for h in hid_a[:half]], cap=2000)
        d_fit = _rows([h[j] for h in hid_b[:half]], cap=2000)
        s_ho = _rows([h[i] for h in hid_a[half:]], cap=2000)
        d_ho = _rows([h[j] for h in hid_b[half:]], cap=2000)
        w_half = M.affine_fit(s_fit, d_fit)
        mu = dst.mean(0)
        scale = (dst - mu).norm(dim=-1).mean()
        base = float(((M.apply_affine(s_ho, w_half) - d_ho).norm(dim=-1) / ((d_ho - mu).norm(dim=-1) + 1e-6)).median())
        maps.append({"a": i, "b": j, "W": M.affine_fit(src, dst), "mu": mu, "scale": float(scale), "baseline": base})
    _calib_cache[key] = maps
    return maps


def _attn_match(pa, pb):
    """Best one-to-one head matching between two layers' attention patterns
    (cosine similarity of the patterns' deviation from uniform attention)."""
    # Subtract the uniform causal pattern (attend equally to all earlier tokens),
    # otherwise every pair of heads looks similar just for being causal.
    T_ = pa.shape[-1]
    uni = torch.tril(torch.ones(T_, T_)) / torch.arange(1, T_ + 1).unsqueeze(1)
    fa = (pa - uni).reshape(pa.shape[0], -1)
    fb = (pb - uni).reshape(pb.shape[0], -1)
    sim = F.normalize(fa, dim=1) @ F.normalize(fb, dim=1).T
    perm, matched = M.match_units(sim)
    return [int(p) for p in perm.tolist()], [round(float(x), 3) for x in matched]


def _learning_probe(model, ids, step_rel=1e-3):
    """Gradient of next-token loss on this prompt, per tensor, plus the effect of
    one normalised step (every parameter moves by at most 0.1% of the model's
    total weight norm) on the prompt's own loss."""
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(input_ids=x, labels=x)
        out.loss.backward()
    loss0 = float(out.loss.detach())
    params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    gtot = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for _, p in params)) or 1.0
    wtot = math.sqrt(sum(float(p.detach().double().pow(2).sum()) for _, p in params)) or 1.0
    rows = []
    for n, p in params:
        g = p.grad.detach().float()
        gn, wn = float(g.norm()), float(p.detach().norm())
        row = {"name": n, "layer": _layer_of(n), "kind": _kind(n), "grad_norm": gn, "weight_norm": wn,
               "rel": gn / wn if wn > 1e-8 else None, "share": (gn / gtot) ** 2, "shape": list(p.shape)}
        if g.ndim == 2 and min(g.shape) <= 4096 and max(g.shape) <= 65536:
            s = torch.linalg.svdvals(g.double().cpu())
            row["rank90"] = M.energy_rank(s, 0.9)
            row["full_rank"] = int(min(g.shape))
        rows.append(row)
    eta = step_rel * wtot / gtot
    with torch.no_grad():
        for _, p in params:
            p.sub_(eta * p.grad)
        after = model(input_ids=x, labels=x)
        loss1 = float(after.loss)
        pred1 = after.logits[0, -1].argmax().item()
        for _, p in params:
            p.add_(eta * p.grad)
        pred0 = model(input_ids=x).logits[0, -1].argmax().item()
    model.zero_grad(set_to_none=True)
    layers = {}
    for r in rows:
        if r["layer"] is not None:
            layers[r["layer"]] = layers.get(r["layer"], 0.0) + r["share"]
    rows.sort(key=lambda r: -r["share"])
    return {"loss_before": loss0, "loss_after": loss1, "pred_before": pred0, "pred_after": pred1,
            "grad_norm": gtot, "weight_norm": wtot, "step_rel": step_rel,
            "by_layer": [{"layer": k, "share": round(v, 5)} for k, v in sorted(layers.items())],
            "non_layer_share": round(sum(r["share"] for r in rows if r["layer"] is None), 5),
            "tensors": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows[:40]],
            "tokens": len(ids)}


def trace_report(spec_a, spec_b, prompt, progress=lambda f, m: None, loader=load_lm, learning=True):
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter a prompt to trace.")
    progress(0.03, f"Loading {spec_a}")
    tok_a, A, ca = loader(spec_a)
    progress(0.12, f"Loading {spec_b}")
    tok_b, B, cb = loader(spec_b)
    if not (ca and cb):
        raise ValueError("Tracing needs two causal language models (models that predict the next token).")

    progress(0.2, "Running the prompt through both models")
    ids_a, hid_a, log_a, att_a, act_a = _forward(tok_a, A, prompt)
    ids_b, hid_b, log_b, att_b, act_b = _forward(tok_b, B, prompt)
    la, lb = len(hid_a) - 1, len(hid_b) - 1
    same_tok = ids_a == ids_b
    notes = []
    if not same_tok:
        notes.append("The two models split the prompt into different tokens, so per-token divergence and attention matching are not defined. Each model's own trace is still shown.")

    def side(tok, model, ids, hid, logits, att, acts):
        lens = []
        for lp in _lens(model, hid, logits):
            p, ix = lp.exp().topk(3, dim=-1)
            nxt = torch.tensor(ids[1:] + [ids[-1]])
            lens.append({
                "top": [[{"t": _decode(tok, ix[t, k]), "p": round(float(p[t, k]), 4)} for k in range(3)] for t in range(len(ids))],
                "true_p": [round(float(lp[t, nxt[t]].exp()), 4) for t in range(len(ids))],
            })
        return {
            "tokens": [_decode(tok, i) for i in ids], "ids": ids, "layers": len(hid) - 1,
            "hidden_norm": [[round(float(x), 2) for x in h.norm(dim=-1).tolist()] for h in hid],
            "lens": lens,
            "mlp": _mlp_usage(acts, len(hid) - 1),
            "attention": [[[[round(float(v), 3) for v in row] for row in head] for head in layer] for layer in att] if att else None,
            "d_model": int(hid[0].shape[-1]),
            "heads": int(att[0].shape[0]) if att else None,
        }

    progress(0.3, "Reading each layer's prediction (logit lens)")
    sa = side(tok_a, A, ids_a, hid_a, log_a, att_a, act_a)
    sb = side(tok_b, B, ids_b, hid_b, log_b, att_b, act_b)

    div = []
    if same_tok:
        maps = _calibrate(spec_a, spec_b, tok_a, A, tok_b, B, progress)
        progress(0.55, "Measuring divergence layer by layer")
        gen_a, gen_b = _lens(A, hid_a, log_a), _lens(B, hid_b, log_b)
        lens_a = list(gen_a)
        lens_b = list(gen_b)
        for i in range(la + 1):
            j = round(i * lb / max(1, la))
            pa, pb = lens_a[i], lens_b[j]
            v = min(pa.shape[-1], pb.shape[-1])
            pa, pb = pa[:, :v].log_softmax(-1), pb[:, :v].log_softmax(-1)
            m = torch.logsumexp(torch.stack([pa, pb]), 0) - math.log(2)
            js = 0.5 * ((pa.exp() * (pa - m)).sum(-1) + (pb.exp() * (pb - m)).sum(-1)) / math.log(2)
            agree = (pa.argmax(-1) == pb.argmax(-1)).float()
            row = {"a": i, "b": j, "js": [round(float(x), 4) for x in js.clamp(0, 1)], "agree": [int(x) for x in agree]}
            if maps:
                mp = maps[i]
                pred = M.apply_affine(hid_a[i].cpu(), mp["W"])
                err = (pred - hid_b[j].cpu()).norm(dim=-1) / ((hid_b[j].cpu() - mp["mu"]).norm(dim=-1) + 1e-6)
                row["align_err"] = [round(float(x), 4) for x in err]
                row["baseline"] = round(mp["baseline"], 4)
            if att_a and att_b and 0 < i and 0 < j:
                perm, sim = _attn_match(att_a[i - 1], att_b[j - 1])
                row["head_match"] = perm
                row["head_sim"] = sim
            div.append(row)
        if not maps:
            notes.append("Calibration needs matching tokenisation on the probe corpus; alignment error is not shown.")

    first_split = None
    if div:
        last = len(ids_a) - 1
        # first layer after which they agree on the final prediction for good
        settle = None
        for r in reversed(div):
            if r["agree"][last]:
                settle = r["a"]
            else:
                break
        first_split = {"settle_layer": settle, "final_agree": bool(div[-1]["agree"][last])}

    learn = None
    if learning:
        progress(0.75, "Computing the learning probe for model A")
        la_ = _learning_probe(A, ids_a)
        progress(0.87, "Computing the learning probe for model B")
        lb_ = _learning_probe(B, ids_b)
        for L, tok in ((la_, tok_a), (lb_, tok_b)):
            L["pred_before_t"] = _decode(tok, L["pred_before"])
            L["pred_after_t"] = _decode(tok, L["pred_after"])
        learn = {"a": la_, "b": lb_}

    progress(1.0, "Done")
    return {"a": spec_a, "b": spec_b, "prompt": prompt, "same_tokens": same_tok, "notes": notes,
            "A": sa, "B": sb, "divergence": div, "summary": first_split, "learning": learn}
