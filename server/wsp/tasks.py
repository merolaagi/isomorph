"""Weight-space tasks with a measurable outcome.

Read (answer from the weights; graded where the true answer is known):
  data       Micro Opus with its embeddings, unembedding and positions hidden:
             what kind of text was it trained on? (words or names; graded by
             keywords, and the answer is shown so you can judge it yourself)
  freqs      a ground-truth lab model of (a + b) mod p: which p, and which
             Fourier frequencies does its embedding use? (graded against the
             lab's own frequency analysis: p exact, frequencies by Jaccard)
  induction  any causal language model: which heads copy (induction heads)?
             The agent may look at attention on a few texts it writes. Graded
             against induction scores measured afterwards on repeated random
             tokens (precision, recall, F1).
  free       your own question; no grade.

Fix (edit the weights; judged on held-out examples the agent never sees):
  planted    a known fault is planted in a model (a rank-one change to the
             last MLP that pushes one token); the agent must find and remove
             it. Graded by the share of the damage repaired on held-out text.
  custom     your goal and your examples ("prompt => target" lines).

Every fix is compared with a control: the same kind and size of edit, on the
same tensors, in random directions. An edit is kept only if it improves the
practice score; the held-out score then says whether that improvement was
real. The model is restored exactly at the end; the restore is checked.
"""
import math
import random
import time

import torch

from .agent import run_agent
from .common import checksum, checksum_diff, drift, encode, general_texts, own_texts, next_token_logp, parse_examples, split, token_loss
from .workspace import Workspace

READ_TASKS = {
    "data": "What kind of text was this model trained on? (Micro Opus, inputs and outputs hidden)",
    "freqs": "Which modulus and which Fourier frequencies? (ground-truth lab model)",
    "induction": "Which heads copy what came before? (any language model)",
    "free": "Your own question",
}
FIX_TASKS = {
    "planted": "Find and remove a planted fault",
    "custom": "Your goal, your examples",
}


# ------------------------------------------------------------------ read
def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def _heads(xs):
    """'3.5', 'L3H5', 'layer 3 head 5', '3:5' -> (3, 5)."""
    import re

    out = set()
    for x in xs or []:
        m = re.search(r"(\d+)\D+(\d+)", str(x))
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def read_task(kind, spec, question="", forward_budget=None, lab_dir=None, loader=None):
    """(workspace, task text, finish schema, grader)."""
    if kind == "data":
        from .. import microopus

        if not microopus.is_spec(spec):
            raise ValueError("The 'data' quiz uses a Micro Opus model (microopus:words, microopus:names or a trained seed).")
        ws = Workspace(spec, loader=loader, forward_budget=0, hide_io=True)
        truth = "names" if "names" in spec else "words"
        text = ("This is a small transformer language model. Its token embeddings, its output layer and its position embeddings "
                "have been removed from view, so you see only the attention and MLP weights of its blocks (and their norms). "
                "You cannot run it. From the weights alone, work out what kind of text it was trained on: what its tokens are "
                "and what the sequences it models look like. Be as specific as you can, and give the evidence.")
        schema = {"answer": {"type": "string", "description": "What it was trained on, in one or two sentences."},
                  "evidence": {"type": "string"}, "confidence": {"type": "number", "description": "0 to 1"}}

        def grade(ans):
            a = (ans or {}).get("answer", "").lower()
            keys = {"names": ("name", "letter", "character", "spelling"), "words": ("sentence", "word", "english", "story", "prose", "text")}
            hit = any(k in a for k in keys[truth])
            other = "words" if truth == "names" else "names"
            wrong = any(k in a for k in keys[other]) and not hit
            return {"truth": "person names, one letter per token" if truth == "names" else "simple English sentences, one word per token",
                    "correct": bool(hit and not wrong), "how": "keywords in the answer; read it to judge", "score": 1.0 if hit and not wrong else 0.0}
        return ws, text, schema, grade

    if kind == "freqs":
        if not spec.startswith("lab:"):
            raise ValueError("The 'freqs' quiz uses a ground-truth lab model (lab:<run>:<seed>). Train a family in the lab first.")
        from ..laws import load_tensors
        from ..lab.analyze import freq_energy, key_freqs

        ws = Workspace(spec, lab_dir=lab_dir, forward_budget=0)
        T, _ = load_tensors(spec, lab_dir)
        p = int(T["W_U"].shape[1])
        truth_f = key_freqs(freq_energy(T["W_E"].detach(), p))
        text = ("This is a one-layer transformer trained to compute (a + b) mod p for some prime p. Its input tokens are the "
                "numbers 0 to p-1 and one '=' token; it predicts the answer at the last position. You cannot run it. From "
                "its weights, find p and the frequencies k (cycles over 0..p-1) that its number embeddings are built from. "
                "Give the frequencies as a list of integers between 1 and p//2.")
        schema = {"p": {"type": "integer"}, "frequencies": {"type": "array", "items": {"type": "integer"}},
                  "evidence": {"type": "string"}}

        def grade(ans):
            ans = ans or {}
            got = [int(x) for x in ans.get("frequencies") or [] if str(x).lstrip("-").isdigit()]
            got = sorted({min(k % p, (-k) % p) for k in got if k % p})
            j = _jaccard(got, truth_f)
            return {"truth": {"p": p, "frequencies": truth_f}, "p_correct": int(ans.get("p") or -1) == p,
                    "frequencies_given": got, "jaccard": j, "score": j, "correct": j >= 0.5 and int(ans.get("p") or -1) == p,
                    "how": "the lab's frequency analysis of the embedding (frequencies holding 80% of the energy)"}
        return ws, text, schema, grade

    if kind == "induction":
        ws = Workspace(spec, loader=loader, forward_budget=8 if forward_budget is None else forward_budget)
        if ws.model is None:
            raise ValueError("The 'induction' quiz needs a model that predicts the next token.")
        from ..hub.acts import find_blocks

        L = len(find_blocks(ws.model) or [])
        H = getattr(ws.model.config, "num_attention_heads", None) or getattr(ws.model.config, "n_head", None)
        text = (f"This is a causal language model with {L} layers of {H} attention heads. Find its induction heads: heads that, "
                "at a token that appeared earlier, attend to the token that followed the earlier occurrence, so the model can "
                f"copy a repeated sequence. You may look at attention on texts you write ({ws.forward_budget} forward passes). "
                "Name heads as 'layer.head', both counted from 0.")
        schema = {"heads": {"type": "array", "items": {"type": "string"}}, "evidence": {"type": "string"}}

        def grade(ans):
            from ..atlas.profile import circuits

            c = circuits(ws.tok, ws.model)
            if c is None:
                return {"error": "This model does not return attention patterns."}
            S = c["scores"]["induction"]
            truth = {(l, h) for l in range(len(S)) for h in range(len(S[l])) if S[l][h] >= 0.3}
            got = _heads((ans or {}).get("heads"))
            top = sorted(({"head": f"{l}.{h}", "score": round(S[l][h], 3)} for l in range(len(S)) for h in range(len(S[l]))),
                         key=lambda x: -x["score"])[:8]
            if not truth:
                return {"truth": [], "given": sorted(f"{l}.{h}" for l, h in got), "score": None, "correct": None, "top_scores": top,
                        "how": "no head reaches an induction score of 0.3 on this model, so there is nothing to find; the quiz is not graded"}
            tp = len(truth & got)
            prec, rec = tp / max(1, len(got)), tp / max(1, len(truth))
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            return {"truth": sorted(f"{l}.{h}" for l, h in truth), "given": sorted(f"{l}.{h}" for l, h in got),
                    "precision": prec, "recall": rec, "f1": f1, "score": f1, "correct": f1 >= 0.5,
                    "how": "induction score >= 0.3 on repeated random tokens, measured after the agent answered", "top_scores": top}
        return ws, text, schema, grade

    if kind == "free":
        if not question.strip():
            raise ValueError("Write the question.")
        ws = Workspace(spec, loader=loader, lab_dir=lab_dir, forward_budget=forward_budget or 0)
        text = ((f"About this model (from the app that built it):\n{ws.about}\n\n" if ws.about else "") + question.strip()
                + ("" if ws.forward_budget else "\n\nYou cannot run the model: answer from its weights."))
        schema = {"answer": {"type": "string"}, "evidence": {"type": "string"}, "confidence": {"type": "number"}}
        return ws, text, schema, lambda ans: {"score": None, "how": "no known answer"}
    raise ValueError(f"Unknown read task {kind}.")


def run_read(kind, spec, call, question="", forward_budget=None, lab_dir=None, loader=None, max_turns=40,
             progress=lambda f, m: None, should_stop=lambda: False):
    t0 = time.time()
    ws, text, schema, grade = read_task(kind, spec, question, forward_budget, lab_dir, loader)
    try:
        ans, transcript, usage = run_agent(ws, text, schema, call, max_turns, progress, should_stop)
        progress(0.97, "Grading the answer")
        g = grade(ans)
    finally:
        ws.restore()
    return {"kind": "read", "task": kind, "spec": spec, "question": text, "answer": ans, "grade": g, "transcript": transcript,
            "usage": usage, "workspace": ws.describe(), "tool_calls": len(ws.calls), "forward_used": ws.forward_used,
            "seconds": time.time() - t0, "created": time.time()}


# ------------------------------------------------------------------ fix
def _down_proj(model):
    from .common import tunable

    cands = [c for c in tunable(model) if c["kind"] == "mlp"]
    if not cands:
        raise ValueError("No MLP matrices found.")
    last = max(c["layer"] for c in cands)
    d = model.get_output_embeddings().weight.shape[1]
    outs = [c for c in cands if c["layer"] == last and (c["shape"][0] == d or c["shape"][1] == d)]
    pick = next((c for c in outs if "down" in c["name"] or "4h_to_h" in c["name"] or "c_proj" in c["name"] or "fc2" in c["name"]), None)
    return pick or outs[-1]


def plant_fault(model, tok, held_rows, rise=0.3, seed=7):
    """Add amount * u v^T to the last MLP's output matrix: u pushes a token the text never uses, v is the
    direction that matrix reads most, so the fault fires on most inputs. The amount is the smallest of a
    fixed ladder that raises held-out loss by `rise`. Returns ((param, original), info)."""
    c = _down_proj(model)
    p = getattr(c["module"], c["attr"])
    d = model.get_output_embeddings().weight.shape[1]
    U = model.get_output_embeddings().weight.detach().float()
    freq = {}
    for f, _ in held_rows:
        for i in f:
            freq[i] = freq.get(i, 0) + 1
    g = torch.Generator().manual_seed(seed)
    order = [i for i in torch.randperm(U.shape[0], generator=g).tolist() if i != 0]
    tid = min(order, key=lambda i: freq.get(i, 0))      # the rarest token in the text (unused where possible)
    u = (U[tid] - U.mean(0)).to(p.device)
    u = u / (u.norm() + 1e-12)
    W = p.detach().float()
    rows_is_d = W.shape[0] == d
    Uw, sw, Vh = torch.linalg.svd(W, full_matrices=False)
    v = Vh[0] if rows_is_d else Uw[:, 0]
    bump = torch.outer(u, v) if rows_is_d else torch.outer(v, u)
    base = token_loss(model, held_rows)["loss"]
    old = p.detach().clone()
    for a in (0.25, 0.5, 1, 2, 4, 8, 16):
        with torch.no_grad():
            p.copy_(old + (a * float(sw[0]) * bump).to(p.dtype))
        if token_loss(model, held_rows)["loss"] >= base * (1 + rise):
            return (p, old), {"tensor": c["name"], "token": tok.decode([tid]), "amount_x_top_sigma": a,
                              "orientation": "rows" if rows_is_d else "cols"}
    with torch.no_grad():
        p.copy_(old)
    raise ValueError("Could not plant a fault that hurts this model enough.")


def _random_like(ws, spec, rng):
    """Replay one edit with random directions (same tensor, same kind, same size)."""
    op = spec["op"]
    if op == "project_out":
        d = ws.D(spec["dir"])
        r = torch.randn(d.numel(), generator=rng)
        k = ws._reg(r, "random control direction")
        ws.edit_project_out(spec["tensor"], k, spec.get("side"), spec.get("range"))
    elif op == "add_rank1":
        a, b = ws.D(spec["rows_dir"]), ws.D(spec["cols_dir"])
        ka = ws._reg(torch.randn(a.numel(), generator=rng), "random control")
        kb = ws._reg(torch.randn(b.numel(), generator=rng), "random control")
        ws.edit_add_rank1(spec["tensor"], ka, kb, spec["amount"])
    elif op == "scale":
        p = ws.T(spec["tensor"])
        tgt, idx = spec.get("target", "all"), spec.get("index")
        if tgt == "all":
            same = [n for n in ws.visible() if n != spec["tensor"] and tuple(ws._tensors[n].shape) == tuple(p.shape)
                    and isinstance(ws._tensors[n], torch.nn.Parameter)]
            if not same:
                return False
            ws.edit_scale(same[int(torch.randint(len(same), (1,), generator=rng))], spec["factor"])
        else:
            lo, hi = (idx, idx) if isinstance(idx, int) else (int(idx[0]), int(idx[1]))
            w = hi - lo + 1
            n = p.shape[0] if tgt == "row" else p.shape[1]
            start = int(torch.randint(0, max(1, n - w + 1), (1,), generator=rng))
            ws.edit_scale(spec["tensor"], spec["factor"], tgt, [start, start + w - 1])
    return True


def run_fix(kind, spec, call, goal="", examples="", held_out="", forward_budget=25, loader=None, max_turns=40,
            progress=lambda f, m: None, should_stop=lambda: False, export_dir=None):
    t0 = time.time()
    ws = Workspace(spec, loader=loader, forward_budget=forward_budget, allow_edits=True)
    if ws.model is None:
        raise ValueError("Fixing needs a model that predicts the next token and can be run.")
    tok, model = ws.tok, ws.model
    sums0 = checksum(model)
    gen = general_texts(tok, model=model)
    planted, plant_info, clean = None, None, None
    try:
        if kind == "planted":
            texts = own_texts(tok, model, 120) or general_texts(tok, 48)
            ex = parse_examples(texts if not examples.strip() else examples)
            practice, held = split(ex, parse_examples(held_out) if held_out.strip() else None, frac=0.4)
        elif kind == "custom":
            if not goal.strip():
                raise ValueError("Say what the edit should achieve.")
            ex = parse_examples(examples)
            practice, held = split(ex, parse_examples(held_out) if held_out.strip() else None)
        else:
            raise ValueError(f"Unknown fix task {kind}.")
        prow, hrow = encode(tok, practice), encode(tok, held)
        ref_gen = next_token_logp(model, tok, gen)
        if kind == "planted":
            clean = {"held": token_loss(model, hrow), "practice": token_loss(model, prow)}
            planted, plant_info = plant_fault(model, tok, hrow)
        base = {"held": token_loss(model, hrow), "practice": token_loss(model, prow)}
        base_gen = next_token_logp(model, tok, gen)

        def score():
            ws._spend(1)
            r = token_loss(model, prow)
            return {"practice_loss": round(r["loss"], 5), "practice_token_accuracy": round(r["acc"], 4),
                    "base_practice_loss": round(base["practice"]["loss"], 5), "weights_changed": ws.changed(),
                    "forward_passes_left": ws.forward_budget - ws.forward_used}
        ws.extra_tools = [("score", "Score the model, with your edits, on the practice examples (mean loss per scored token; lower is better). Costs one forward pass.",
                           {}, [], score)]
        sample = "\n".join(f"  {e['prompt']}{' =>' if e['prompt'] else ''}{e['target']}" for e in practice[:12])
        if kind == "planted":
            text = ("This language model has been damaged: someone changed some of its weights, and it now predicts its own kind "
                    f"of text worse (practice loss {base['practice']['loss']:.4f} per token). Find the change from the weights and "
                    "repair it with as small an edit as you can, without retraining. You can run the model and score it on practice "
                    f"examples ({ws.forward_budget} forward passes in all). Practice examples look like:\n{sample}")
        else:
            text = (f"Goal: {goal.strip()}\n\nChange the weights directly so the model does better on examples like these "
                    "(the part after => is what it should produce; lines without => are text it should model better):\n"
                    f"{sample}\n\nPractice loss now: {base['practice']['loss']:.4f} per scored token. You have {ws.forward_budget} "
                    "forward passes (run, attention, mean_difference and score each cost one). Unseen examples will judge the result.")
        schema = {"summary": {"type": "string", "description": "What you found, what you changed, and why."},
                  "confidence": {"type": "number"}}
        ans, transcript, usage = run_agent(ws, text, schema, call, max_turns, progress, should_stop)

        progress(0.9, "Scoring on held-out examples")
        final_practice = token_loss(model, prow)
        kept = bool(ws.changed()) and final_practice["loss"] < base["practice"]["loss"] - 1e-6
        if not kept:
            ws.undo("all")
        specs = ws.edit_specs()
        edited = {"held": token_loss(model, hrow), "practice": token_loss(model, prow),
                  "drift": drift(base_gen, next_token_logp(model, tok, gen))}
        if kind == "planted":
            edited["drift_from_clean"] = drift(ref_gen, next_token_logp(model, tok, gen))
        exported = None
        if kept and export_dir is not None and kind == "custom":
            from safetensors.torch import save_file

            touched = sorted({s["tensor"] for s in specs})
            export_dir.mkdir(parents=True, exist_ok=True)
            save_file({n: ws._tensors[n].detach().float().cpu().contiguous() for n in touched}, str(export_dir / "edited_tensors.safetensors"))
            exported = {"file": "edited_tensors.safetensors", "tensors": touched}
        ws.undo("all")
        ctrl = []
        if specs:
            for seed in range(2):
                rng = torch.Generator().manual_seed(500 + seed)
                ok = all(_random_like(ws, s, rng) for s in specs)
                if ok:
                    ctrl.append({"held": token_loss(model, hrow)["loss"], "practice": token_loss(model, prow)["loss"]})
                ws.undo("all")
        total = sum(int(p.numel()) for p in model.parameters())
        res = {"kind": "fix", "task": kind, "spec": spec, "goal": goal or None, "question": text, "answer": ans,
               "transcript": transcript, "usage": usage, "tool_calls": len(ws.calls), "forward_used": ws.forward_used,
               "examples": {"practice": len(practice), "held_out": len(held)}, "base": base, "edited": edited, "kept": kept,
               "edits": specs, "weights_changed": sum(s["weights"] for s in specs), "weights_total": total,
               "control": {"held": sum(c["held"] for c in ctrl) / len(ctrl), "practice": sum(c["practice"] for c in ctrl) / len(ctrl), "seeds": len(ctrl)} if ctrl else None,
               "exported": exported, "seconds": time.time() - t0, "created": time.time()}
        if kind == "planted":
            gap = base["held"]["loss"] - clean["held"]["loss"]
            res["planted"] = plant_info
            res["clean"] = clean
            res["grade"] = {"repaired_share": (base["held"]["loss"] - edited["held"]["loss"]) / gap if gap > 0 else None,
                            "touched_planted_tensor": any(s["tensor"] == plant_info["tensor"] for s in specs),
                            "how": "share of the held-out damage removed: (damaged - after) / (damaged - clean)"}
            res["grade"]["score"] = res["grade"]["repaired_share"]
        else:
            res["grade"] = {"held_out_gain": base["held"]["loss"] - edited["held"]["loss"],
                            "beats_control": bool(ctrl) and edited["held"]["loss"] < res["control"]["held"],
                            "how": "held-out loss before and after, against the same edits in random directions"}
            res["grade"]["score"] = res["grade"]["held_out_gain"]
        return res
    finally:
        ws.undo("all")
        if planted is not None:
            with torch.no_grad():
                planted[0].copy_(planted[1])
        diff = checksum_diff(sums0, checksum(model))
        if diff > 1e-6:
            raise RuntimeError(f"The model was not restored exactly (difference {diff:.3g}).")
