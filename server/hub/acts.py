"""Activation-level comparison of two real language models.

Runs both models on the same texts and asks, layer by layer:
  * CKA / mutual-kNN: do the two models organise the inputs the same way?
  * Neuron matching: does each MLP neuron in A have a twin in B?
  * Output agreement: do they predict the same next tokens?
  * Stitching: can B finish the job starting from A's state (after an affine map)?
"""
import math
import re

import torch
import torch.nn.functional as F

from .. import metrics as M
from .corpus import CORPUS
from .load import device, load_lm

ACT_RE = re.compile(r"(mlp|intermediate|feed_forward|ffn)\.(act|act_fn|activation_fn|intermediate_act_fn)$")
MAX_NEURON_LAYERS = 8
NEURON_SAMPLE = 1024


def find_blocks(model):
    n = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", None)
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            return mod
    return None


def _layer_idx(name):
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def run_model(tok, model, texts, want_layers, max_len=128):
    """Hidden states per text, MLP activations for the requested layers, and logits."""
    acts = {}
    hooks = []
    for name, mod in model.named_modules():
        if ACT_RE.search(name):
            li = _layer_idx(name)
            if li is not None and li in want_layers:
                def hk(_m, _i, out, li=li):
                    acts.setdefault(li, []).append(out.detach()[0].float().cpu())
                hooks.append(mod.register_forward_hook(hk))
    ids, hidden, logits = [], [], []
    dev = next(model.parameters()).device
    try:
        with torch.no_grad():
            for t in texts:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
                x = enc["input_ids"].to(dev)
                out = model(input_ids=x, output_hidden_states=True)
                ids.append(enc["input_ids"][0].tolist())
                hidden.append(torch.stack([h[0].float().cpu() for h in out.hidden_states]))
                lg = getattr(out, "logits", None)
                logits.append(lg[0].float().cpu() if lg is not None else None)
    finally:
        for h in hooks:
            h.remove()
    return ids, hidden, acts, logits


def _rows(per_text, skip_first=True, cap=None):
    """Concatenate per-text (T, ...) tensors into token rows, dropping position 0
    (the first token acts as an attention sink with outsized activations)."""
    parts = [p[1:] if skip_first and p.shape[0] > 1 else p for p in per_text]
    out = torch.cat(parts, 0)
    return out[:cap] if cap else out


def _depth_map(la, lb, k):
    """k evenly spaced layer indices in A and their depth-matched partners in B."""
    k = min(k, la, lb)
    ia = sorted({round(i * (la - 1) / max(1, k - 1)) for i in range(k)})
    return [(i, round(i * (lb - 1) / max(1, la - 1))) for i in ia]


def lm_loss_with_injection(model, blocks, x, j, h_new):
    """B's next-token loss when block j receives h_new instead of its own input."""
    def pre(_m, args, kwargs):
        if args:
            args = (h_new,) + tuple(args[1:])
        else:
            kwargs["hidden_states"] = h_new
        return args, kwargs

    handle = blocks[j].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.no_grad():
            lg = model(input_ids=x).logits[0]
    finally:
        handle.remove()
    return F.cross_entropy(lg[:-1].float(), x[0, 1:], reduction="sum").item(), x.shape[1] - 1


def activation_report(spec_a, spec_b, texts=None, max_tokens=1500, progress=lambda f, m: None, loader=load_lm):
    texts = [t for t in (texts or CORPUS) if t.strip()]
    progress(0.02, f"Loading {spec_a}")
    tok_a, A, causal_a = loader(spec_a)
    progress(0.12, f"Loading {spec_b}")
    tok_b, B, causal_b = loader(spec_b)
    la = A.config.num_hidden_layers
    lb = B.config.num_hidden_layers
    pairs = _depth_map(la, lb, MAX_NEURON_LAYERS)

    progress(0.2, "Running model A on the probe texts")
    ids_a, hid_a, mlp_a, log_a = run_model(tok_a, A, texts, {i for i, _ in pairs})
    progress(0.35, "Running model B on the probe texts")
    ids_b, hid_b, mlp_b, log_b = run_model(tok_b, B, texts, {j for _, j in pairs})

    same_tok = ids_a == ids_b
    mode = "token" if same_tok else "sentence"
    notes = []
    if not same_tok:
        notes.append("The two models tokenise text differently, so rows are whole sentences (mean-pooled) "
                     "instead of individual tokens. Neuron matching, output agreement and stitching need a shared tokenizer and are skipped.")

    def layer_rows(hid, layer):
        if same_tok:
            return _rows([h[layer] for h in hid], cap=max_tokens)
        return torch.stack([h[layer][1:].mean(0) if h.shape[1] > 1 else h[layer][0] for h in hid])

    progress(0.45, "Comparing representations layer by layer")
    cka = []
    for i in range(la + 1):
        ra = layer_rows(hid_a, i)
        cka.append([round(M.linear_cka(ra, layer_rows(hid_b, j)), 3) for j in range(lb + 1)])
    diag = []
    for i in range(la + 1):
        j = round(i * lb / max(1, la))
        ra, rb = layer_rows(hid_a, i), layer_rows(hid_b, j)
        sub = slice(None, None, max(1, len(ra) // 600))
        diag.append({"a": i, "b": j, "cka": cka[i][j], "knn": round(M.mutual_knn(ra[sub], rb[sub], k=10), 3)})

    neurons = []
    if same_tok and mlp_a and mlp_b:
        for n_i, (i, j) in enumerate(pairs):
            if i not in mlp_a or j not in mlp_b:
                continue
            progress(0.55 + 0.2 * n_i / len(pairs), f"Matching MLP neurons, layer {i} of A with layer {j} of B")
            xa = _rows(mlp_a[i], cap=max_tokens)
            xb = _rows(mlp_b[j], cap=max_tokens)
            g = torch.Generator().manual_seed(0)
            sel = torch.randperm(xa.shape[1], generator=g)[: min(NEURON_SAMPLE, xa.shape[1])]
            corr = M.correlation_matrix(xa[:, sel], xb)
            _, matched = M.match_units(corr)
            best = corr.max(1).values
            rand = corr[torch.arange(len(sel)), torch.randint(0, xb.shape[1], (len(sel),), generator=g)]
            neurons.append({
                "a": i, "b": j, "units_a": int(xa.shape[1]), "units_b": int(xb.shape[1]), "sampled": int(len(sel)),
                "matched_mean": round(float(matched.mean()), 4), "best_mean": round(float(best.mean()), 4),
                "random_mean": round(float(rand.mean()), 4), "above_0_8": round(float((matched > 0.8).float().mean()), 4),
                "above_0_5": round(float((matched > 0.5).float().mean()), 4),
            })
    elif same_tok:
        notes.append("No MLP activation modules were recognised in these architectures, so neuron matching was skipped.")

    agreement = None
    if same_tok and causal_a and causal_b and log_a[0] is not None and log_a[0].shape[-1] == log_b[0].shape[-1]:
        progress(0.8, "Comparing next-token predictions")
        top1 = kl = loss_a = loss_b = 0.0
        n = 0
        for x, la_, lb_ in zip(ids_a, log_a, log_b):
            if len(x) < 2:
                continue
            tgt = torch.tensor(x[1:])
            pa, pb = la_[:-1].log_softmax(-1), lb_[:-1].log_softmax(-1)
            top1 += (pa.argmax(-1) == pb.argmax(-1)).sum().item()
            kl += (pa.exp() * (pa - pb)).sum().item()
            loss_a += F.nll_loss(pa, tgt, reduction="sum").item()
            loss_b += F.nll_loss(pb, tgt, reduction="sum").item()
            n += len(tgt)
        agreement = {"top1": round(top1 / n, 4), "kl_a_b": round(kl / n, 4), "loss_a": round(loss_a / n, 4), "loss_b": round(loss_b / n, 4), "tokens": n}

    stitch = []
    blocks_b = find_blocks(B)
    if same_tok and causal_b and blocks_b is not None and len(texts) >= 8:
        half = len(texts) // 2
        fit_idx, ev_idx = range(half), range(half, len(texts))
        dev = next(B.parameters()).device
        cand = sorted({j for _, j in _depth_map(la, lb, 6) if 0 < j < lb})
        for n_j, j in enumerate(cand):
            i = round(j * la / max(1, lb))
            progress(0.85 + 0.14 * n_j / max(1, len(cand)), f"Stitching A's layer {i} into B at layer {j}")
            src = _rows([hid_a[t][i] for t in fit_idx], skip_first=False)
            dst = _rows([hid_b[t][j] for t in fit_idx], skip_first=False)
            W = M.affine_fit(src, dst)
            R = M.procrustes(src - src.mean(0), dst - dst.mean(0)) if src.shape[1] == dst.shape[1] else None
            res = {"a_layer": i, "b_layer": j}
            for kind in ("own", "affine", "orthogonal", "identity"):
                tot = cnt = 0.0
                if kind in ("orthogonal", "identity") and src.shape[1] != dst.shape[1]:
                    continue
                for t in ev_idx:
                    x = torch.tensor([ids_b[t]], device=dev)
                    if x.shape[1] < 2:
                        continue
                    ha = hid_a[t][i]
                    if kind == "own":
                        h = hid_b[t][j]
                    elif kind == "affine":
                        h = M.apply_affine(ha, W)
                    elif kind == "orthogonal":
                        h = (ha - src.mean(0)) @ R + dst.mean(0)
                    else:
                        h = ha
                    s, c = lm_loss_with_injection(B, blocks_b, x, j, h.unsqueeze(0).to(dev))
                    tot, cnt = tot + s, cnt + c
                res[kind] = round(tot / max(1, cnt), 4)
            stitch.append(res)
    elif same_tok and causal_b:
        notes.append("Could not locate model B's transformer blocks, so stitching was skipped.")

    progress(1.0, "Done")
    return {
        "a": spec_a, "b": spec_b, "mode": mode, "notes": notes,
        "layers_a": la, "layers_b": lb, "rows": len(layer_rows(hid_a, 0)),
        "cka": cka, "diag": diag, "neurons": neurons, "agreement": agreement, "stitch": stitch,
    }
