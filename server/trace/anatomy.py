"""Post-mortem of a model's architecture, read from the loaded code itself.

Builds the forward pass as a flowchart of steps. Each step carries:
  * the math it computes
  * the exact weight tensors it reads (or none: softmax, residual adds and
    rotary rotations use no learned weights)
  * the real source code of the module that runs it
  * its measured impact on a prompt: how large its output is, and what
    happens to the loss when that output is removed (ablation)
"""
import inspect
import re

import torch
import torch.nn.functional as F

from ..hub.acts import find_blocks
from ..hub.load import load_lm

MAX_SRC_LINES = 90


def _src(obj):
    try:
        fn = obj.forward if hasattr(obj, "forward") else obj
        fn = inspect.unwrap(getattr(fn, "__func__", fn))  # skip decorators to reach the real forward
        lines, start = inspect.getsourcelines(fn)
        path = inspect.getsourcefile(fn) or ""
        m = re.search(r"(transformers/.*)$", path)
        text = "".join(lines)
        if len(lines) > MAX_SRC_LINES:
            text = "".join(lines[:MAX_SRC_LINES]) + f"    # … {len(lines) - MAX_SRC_LINES} more lines\n"
        return {"file": m.group(1) if m else path.split("/")[-1], "line": start, "cls": type(obj).__name__, "code": text}
    except (OSError, TypeError):
        return None


def _params(module, prefix=""):
    out = []
    for n, p in module.named_parameters():
        out.append({"name": f"{prefix}{n}", "shape": list(p.shape), "params": p.numel()})
    return out


def _child(block, names, cls_words):
    for n, m in block.named_children():
        if n in names:
            return n, m
    for n, m in block.named_children():
        if any(w in type(m).__name__.lower() for w in cls_words):
            return n, m
    return None, None


def _norms(block):
    out = []
    for n, m in block.named_children():
        c = type(m).__name__.lower()
        if "norm" in c or n.startswith("ln"):
            out.append((n, m))
    return out


def _is_rms(m):
    return "rms" in type(m).__name__.lower()


ROLE = [
    (("query_key_value", "c_attn", "qkv", "Wqkv", "qkv_proj"), "Q, K and V projections (fused)"),
    (("q_proj", "query"), "Query projection"),
    (("k_proj", "key"), "Key projection"),
    (("v_proj", "value"), "Value projection"),
    (("dense", "o_proj", "c_proj", "out_proj", "wo"), "Output projection"),
]
MLP_ROLE = [
    (("gate_proj", "w1", "gate"), "Gate projection"),
    (("dense_h_to_4h", "c_fc", "up_proj", "fc1", "fc_in", "w3", "Wi"), "Up projection"),
    (("dense_4h_to_h", "c_proj", "down_proj", "fc2", "fc_out", "w2", "Wo"), "Down projection"),
]


def _linear_roles(module, table):
    found = []
    for n, m in module.named_children():
        ps = list(m.parameters(recurse=False))
        if not ps:
            continue
        role = next((r for keys, r in table if n in keys), n)
        found.append({"module": n, "role": role, "weights": _params(m, n + ".")})
    return found


def _describe(model):
    cfg = model.config
    base = getattr(model, getattr(model, "base_model_prefix", ""), model)
    blocks = find_blocks(model)
    if blocks is None:
        raise ValueError("Could not find this model's stack of transformer blocks.")
    b0 = blocks[0]
    attn_name, attn = _child(b0, ("attention", "attn", "self_attn", "self_attention"), ("attention",))
    mlp_name, mlp = _child(b0, ("mlp", "feed_forward", "ffn"), ("mlp", "feedforward"))
    norms = _norms(b0)
    act = getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", None) or "gelu"
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    d = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    d_ff = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or (4 * d if d else None)
    n_layers = len(blocks)
    parallel = bool(getattr(cfg, "use_parallel_residual", False)) or cfg.model_type in ("gptj", "codegen")
    rotary = any(hasattr(cfg, a) for a in ("rotary_pct", "rope_theta", "rotary_emb_base", "rope_scaling")) or \
        getattr(cfg, "position_embedding_type", "") == "rotary"
    rot_pct = getattr(cfg, "rotary_pct", None) or getattr(cfg, "partial_rotary_factor", None) or 1.0

    embeds = [(n, m) for n, m in base.named_children() if isinstance(m, torch.nn.Embedding)]
    tok_emb = next(((n, m) for n, m in embeds if m.num_embeddings == model.get_input_embeddings().num_embeddings), None)
    pos_emb = next(((n, m) for n, m in embeds if tok_emb is None or n != tok_emb[0]), None)
    fnorm = None
    for a in ("ln_f", "final_layer_norm", "norm", "final_norm"):
        if isinstance(getattr(base, a, None), torch.nn.Module):
            fnorm = (a, getattr(base, a))
            break
    head = model.get_output_embeddings()
    tied = head is not None and tok_emb is not None and head.weight.data_ptr() == tok_emb[1].weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    dh = d // heads if d and heads else None

    ln = lambda m: (["μ = mean(x),  σ² = mean((x − μ)²)", "y = (x − μ) / √(σ² + ε) · γ + β"]
                    if not _is_rms(m) else ["rms = √(mean(x²) + ε)", "y = x / rms · γ"])
    nodes = []
    nodes.append({"id": "tokens", "label": "Tokens", "kind": "io", "has_weights": False,
                  "math": ["text → token ids t₁ … t_T (tokenizer, no learned weights here)"],
                  "note": "The tokenizer is a fixed lookup table, not a neural network."})
    emb_math = [f"h⁰ᵢ = W_E[tᵢ]    (row lookup in a {tok_emb[1].num_embeddings:,} × {tok_emb[1].embedding_dim} table)"] if tok_emb else []
    emb_w = _params(tok_emb[1], f"{tok_emb[0]}.") if tok_emb else []
    if pos_emb:
        emb_math.append("h⁰ᵢ += W_pos[i]    (learned position vector)")
        emb_w += _params(pos_emb[1], f"{pos_emb[0]}.")
    elif rotary:
        emb_math.append("no position vector is added here; position enters later by rotating Q and K (RoPE)")
    nodes.append({"id": "embed", "label": "Token embedding", "kind": "embed", "has_weights": True, "math": emb_math,
                  "weights": emb_w, "code": _src(tok_emb[1]) if tok_emb else None,
                  "note": "The first place weights enter: each token id selects one learned row. Nothing is multiplied yet."})

    attn_lin = _linear_roles(attn, ROLE) if attn is not None else []
    mlp_lin = _linear_roles(mlp, MLP_ROLE) if mlp is not None else []
    gated = any(l["role"] == "Gate projection" for l in mlp_lin)
    ln1 = norms[0] if norms else None
    ln2 = norms[1] if len(norms) > 1 else None

    attn_steps = []
    fused = any("fused" in l["role"] for l in attn_lin)
    proj = [l for l in attn_lin if l["role"] != "Output projection"]
    attn_steps.append({"label": "Project to queries, keys, values", "has_weights": True,
                       "math": ["[q; k; v] = W_QKV · y + b" if fused else "q = W_Q · y,   k = W_K · y,   v = W_V · y",
                                f"split into {heads} heads of {dh} dims" + (f" ({kv_heads} key/value heads shared across query heads)" if kv_heads != heads else "")],
                       "weights": [w for l in proj for w in l["weights"]]})
    if rotary:
        attn_steps.append({"label": "Rotate by position (RoPE)", "has_weights": False,
                           "math": [f"rotate pairs of dims in q and k by angle i · θⱼ, θⱼ = base^(−2j/d_rot)" + (f"  (first {int(rot_pct * 100)}% of each head's dims)" if rot_pct < 1 else ""),
                                    "after rotation, q·k depends only on the distance between positions"]})
    attn_steps.append({"label": "Score every earlier token", "has_weights": False,
                       "math": [f"sᵢⱼ = qᵢ · kⱼ / √{dh}", "sᵢⱼ = −∞ for j > i  (causal mask: no looking ahead)"]})
    attn_steps.append({"label": "Softmax into attention weights", "has_weights": False,
                       "math": ["aᵢⱼ = exp(sᵢⱼ) / Σₖ exp(sᵢₖ)", "these 'attention weights' are computed per prompt; they are not learned parameters"]})
    attn_steps.append({"label": "Mix the values", "has_weights": False, "math": ["zᵢ = Σⱼ aᵢⱼ · vⱼ   (per head), then concatenate heads"]})
    outp = [l for l in attn_lin if l["role"] == "Output projection"]
    attn_steps.append({"label": "Project back to the residual stream", "has_weights": True,
                       "math": ["attn_out = W_O · z + b_O"], "weights": [w for l in outp for w in l["weights"]]})

    mlp_steps = []
    if gated:
        g = [l for l in mlp_lin if l["role"] == "Gate projection"]
        u = [l for l in mlp_lin if l["role"] == "Up projection"]
        mlp_steps.append({"label": "Gate and up projections", "has_weights": True,
                          "math": [f"g = W_gate · y,   u = W_up · y    ({d} → {d_ff})"], "weights": [w for l in g + u for w in l["weights"]]})
        mlp_steps.append({"label": f"Gated activation ({act})", "has_weights": False, "math": [f"a = {act}(g) ⊙ u"]})
    else:
        u = [l for l in mlp_lin if l["role"] == "Up projection"]
        mlp_steps.append({"label": "Up projection", "has_weights": True, "math": [f"u = W_in · y + b_in    ({d} → {d_ff} neurons)"],
                          "weights": [w for l in u for w in l["weights"]]})
        actm = {"gelu": "a = u · Φ(u)    (GELU: keeps positives, softly zeroes negatives)",
                "gelu_new": "a = ½u(1 + tanh(√(2/π)(u + 0.044715u³)))    (GELU, tanh form)",
                "relu": "a = max(0, u)"}.get(str(act), f"a = {act}(u)")
        mlp_steps.append({"label": f"Activation ({act})", "has_weights": False, "math": [actm, "this is where a neuron 'fires' or stays quiet for this token"]})
    dn = [l for l in mlp_lin if l["role"] == "Down projection"]
    mlp_steps.append({"label": "Down projection", "has_weights": True, "math": [f"mlp_out = W_out · a + b_out    ({d_ff} → {d})"],
                      "weights": [w for l in dn for w in l["weights"]]})

    blk_params = sum(p.numel() for p in b0.parameters())
    residual = (["h ← h + Attention(LN₁(h)) + MLP(LN₂(h))", "attention and MLP read the same input in parallel (GPT-NeoX style)"]
                if parallel else ["h ← h + Attention(LN₁(h))", "h ← h + MLP(LN₂(h))"])
    block = {
        "id": "block", "label": f"Transformer block × {n_layers}", "kind": "block", "parallel": parallel,
        "math": residual, "params_per_block": blk_params, "layers": n_layers,
        "note": "Every block reads the residual stream and adds its result back. Nothing is overwritten; each layer writes a correction on top of what is already there.",
        "code": _src(b0),
        "children": [
            {"id": "ln1", "label": ("RMSNorm" if ln1 and _is_rms(ln1[1]) else "LayerNorm") + " before attention", "kind": "norm", "has_weights": True,
             "math": ln(ln1[1]) if ln1 else [], "weights": _params(ln1[1], f"{ln1[0]}.") if ln1 else [],
             "note": "Rescales each token's vector; γ and β are the only weights, one number per dimension."},
            {"id": "attn", "label": "Self-attention", "kind": "attn", "has_weights": True, "steps": attn_steps,
             "weights": _params(attn, f"{attn_name}.") if attn is not None else [], "code": _src(attn) if attn is not None else None,
             "note": "The only step where tokens exchange information. Weights decide what to look for (Q), what to offer (K), what to pass on (V) and how to write it back (O); where to look is decided by the prompt."},
            {"id": "ln2", "label": ("RMSNorm" if ln2 and _is_rms(ln2[1]) else "LayerNorm") + " before MLP", "kind": "norm", "has_weights": True,
             "math": ln(ln2[1]) if ln2 else [], "weights": _params(ln2[1], f"{ln2[0]}.") if ln2 else []},
            {"id": "mlp", "label": "MLP (feed-forward)", "kind": "mlp", "has_weights": True, "steps": mlp_steps,
             "weights": _params(mlp, f"{mlp_name}.") if mlp is not None else [], "code": _src(mlp) if mlp is not None else None,
             "note": "Works on each token separately. Each row of W_in is a pattern detector; each column of W_out is what that neuron writes when it fires. Most of a model's parameters live here."},
            {"id": "add", "label": "Residual add", "kind": "add", "has_weights": False, "math": residual,
             "note": "Plain addition, no weights. It is why information from early layers can reach the output unchanged."},
        ],
    }
    nodes.append(block)
    if fnorm:
        nodes.append({"id": "fnorm", "label": "Final " + ("RMSNorm" if _is_rms(fnorm[1]) else "LayerNorm"), "kind": "norm", "has_weights": True,
                      "math": ln(fnorm[1]), "weights": _params(fnorm[1], f"{fnorm[0]}.")})
    nodes.append({"id": "unembed", "label": "Unembedding (LM head)", "kind": "embed", "has_weights": True,
                  "math": [f"logitsᵥ = W_U[v] · y    (one score per vocabulary entry, {head.out_features:,} of them)" if head is not None else "logits = W_U · y"],
                  "weights": [] if tied else (_params(head, "lm_head.") if head is not None else []),
                  "note": "Tied: reuses the embedding table, so no new weights." if tied else "The last place weights enter: every vocabulary entry has its own row, and the score is its dot product with the final vector."})
    nodes.append({"id": "softmax", "label": "Softmax → next token", "kind": "io", "has_weights": False,
                  "math": ["p(v) = exp(logitsᵥ) / Σᵤ exp(logitsᵤ)", "pick the most likely token, or sample from p"]})

    groups = {"embedding": sum(w["params"] for w in emb_w),
              "attention": sum(p.numel() for b in blocks for n, p in b.named_parameters() if n.startswith(attn_name or "\0")),
              "mlp": sum(p.numel() for b in blocks for n, p in b.named_parameters() if n.startswith(mlp_name or "\0")),
              "norms": sum(p.numel() for b in blocks for n, p in b.named_parameters() if any(n.startswith(x[0]) for x in norms)) +
                       (sum(p.numel() for p in fnorm[1].parameters()) if fnorm else 0),
              "unembedding": 0 if tied or head is None else head.weight.numel()}
    groups["other"] = max(0, total - sum(groups.values()))
    info = {"model_type": cfg.model_type, "class": type(model).__name__, "layers": n_layers, "d_model": d, "heads": heads,
            "kv_heads": kv_heads, "d_head": dh, "d_ff": d_ff, "activation": str(act), "parallel_residual": parallel,
            "rotary": rotary, "rotary_pct": rot_pct, "learned_positions": pos_emb is not None, "tied_embeddings": tied,
            "vocab": head.out_features if head is not None else None, "total_params": total, "param_groups": groups,
            "model_code": _src(model)}
    return info, nodes, blocks, attn_name, mlp_name


def _impact(model, tok, blocks, attn_name, mlp_name, prompt):
    """Output size of attention and MLP per layer, and what removing each does."""
    dev = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64)["input_ids"][:, :64].to(dev)
    store = {}
    handles = []
    for i, b in enumerate(blocks):
        for kind, name in (("attn", attn_name), ("mlp", mlp_name)):
            mod = getattr(b, name, None) if name else None
            if mod is None:
                continue

            def hk(_m, _i, out, i=i, kind=kind):
                o = out[0] if isinstance(out, tuple) else out
                store[(i, kind)] = o.detach()[0].float().norm(dim=-1).cpu()
            handles.append(mod.register_forward_hook(hk))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    logits = out.logits[0].float()
    tgt = ids[0, 1:]

    def loss_of(lg):
        return float(F.cross_entropy(lg[:-1], tgt))

    base_loss = loss_of(logits)
    pred = int(logits[-1].argmax())
    base_p = float(logits[-1].softmax(-1)[pred])
    resid = [h[0].float().norm(dim=-1).cpu() for h in out.hidden_states]
    layers = []
    for i, b in enumerate(blocks):
        row = {"layer": i, "resid_in": round(float(resid[i].mean()), 3), "resid_in_last": round(float(resid[i][-1]), 3)}
        for kind, name in (("attn", attn_name), ("mlp", mlp_name)):
            if (i, kind) not in store:
                continue
            v = store[(i, kind)]
            row[f"{kind}_norm"] = round(float(v.mean()), 3)
            row[f"{kind}_norm_last"] = round(float(v[-1]), 3)
            mod = getattr(b, name)

            def zero(_m, _i, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + tuple(out[1:])
                return torch.zeros_like(out)
            h = mod.register_forward_hook(zero)
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[0].float()
            finally:
                h.remove()
            row[f"{kind}_dloss"] = round(loss_of(lg) - base_loss, 4)
            row[f"{kind}_dprob"] = round(float(lg[-1].softmax(-1)[pred]) - base_p, 4)
            row[f"{kind}_flips"] = int(lg[-1].argmax()) != pred
        layers.append(row)
    return {"prompt": prompt, "tokens": int(ids.shape[1]), "loss": round(base_loss, 4),
            "prediction": tok.decode([pred]), "prediction_p": round(base_p, 4), "layers": layers}


def anatomy_report(spec, prompt=None, progress=lambda f, m: None, loader=load_lm):
    progress(0.1, f"Loading {spec}")
    tok, model, causal = loader(spec)
    progress(0.4, "Reading the model's code and structure")
    info, nodes, blocks, attn_name, mlp_name = _describe(model)
    impact = None
    if prompt and prompt.strip() and causal:
        progress(0.6, "Measuring each layer's impact on the prompt")
        impact = _impact(model, tok, blocks, attn_name, mlp_name, prompt.strip())
    progress(1.0, "Done")
    return {"spec": spec, "info": info, "nodes": nodes, "impact": impact}
