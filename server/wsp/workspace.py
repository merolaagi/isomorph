"""The workspace a weight-reading agent works in: one model, a fixed set of tools.

The agent never runs code of its own. It calls tools that read the weights
(statistics, singular vectors, spectra along an axis, how a matrix acts on a
direction, what a direction means in tokens), and, when allowed, tools that
run the model on text it writes (each call costs one forward pass from a
fixed budget) and tools that edit the weights. Every edit keeps the exact
values it replaced, so undoing an edit, or closing the workspace, puts the
model back bit for bit.

Directions are vectors the tools hand back with an id (d1, d2, ...), so the
agent can pass one tool's output to another: take a singular vector of one
matrix, push it through the next, read it out as tokens, remove it.
"""
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

IO_NAME = re.compile(r"(embed|wte|wpe|lm_head|embed_out|unembed|\bW_E\b|\bW_U\b|\bW_P\b|W_pos|position)", re.I)
MAX_OUT = 6000


def _r(v, d=4):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{d}g}")


class Budget(Exception):
    pass


class Workspace:
    def __init__(self, spec, loader=None, lab_dir=None, forward_budget=0, allow_edits=False, hide_io=False):
        from ..hub.load import load_lm

        self.spec, self.forward_budget, self.forward_used = spec, int(forward_budget), 0
        self.allow_edits, self.hide_io = bool(allow_edits), bool(hide_io)
        self.tok = self.model = None
        self.dirs, self.edits, self.calls = {}, [], []
        self._n = 0
        loader = loader or load_lm
        self.about = ""
        if spec.startswith("import:"):
            from .imports import load_import

            self._tensors, self.about = load_import(spec)
        elif spec.startswith("lab:"):
            from ..laws import load_tensors

            T, _ = load_tensors(spec, lab_dir)
            self._tensors = {k: v.float() for k, v in T.items()}
        else:
            try:
                self.tok, self.model, causal = loader(spec)
                if not causal:
                    self.model = None
            except Exception:
                self.model = None
            if self.model is not None:
                self._tensors = dict(self.model.named_parameters())
            else:
                from ..hub.load import fetch_weights

                self._tensors = dict(fetch_weights(spec))
        if self.model is None and (self.forward_budget or self.allow_edits):
            self.forward_budget, self.allow_edits = 0, False

    # -------------------------------------------------------------- basics
    def visible(self):
        return [k for k in self._tensors if not (self.hide_io and IO_NAME.search(k))]

    def T(self, name):
        if name not in self._tensors or (self.hide_io and IO_NAME.search(name)):
            raise ValueError(f"No visible tensor called {name}. Use list_tensors.")
        return self._tensors[name]

    def _reg(self, vec, about):
        self._n += 1
        k = f"d{self._n}"
        v = vec.detach().float().flatten().cpu()
        self.dirs[k] = {"vec": v / (v.norm() + 1e-12), "norm": float(v.norm()), "about": about}
        return k

    def D(self, k):
        if k not in self.dirs:
            raise ValueError(f"No direction {k}. Directions come from svd, project, token_direction or mean_difference.")
        return self.dirs[k]["vec"]

    def _spend(self, n=1):
        if self.forward_used + n > self.forward_budget:
            raise Budget(f"Forward-pass budget spent ({self.forward_used} of {self.forward_budget}).")
        self.forward_used += n

    def can_read_tokens(self):
        return self.model is not None and not self.hide_io

    def _unembed(self):
        return self.model.get_output_embeddings().weight.detach().float().cpu()

    def _embed(self):
        return self.model.get_input_embeddings().weight.detach().float().cpu()

    def _tokname(self, i):
        try:
            return self.tok.decode([int(i)])
        except Exception:
            return str(int(i))

    def describe(self):
        vis = self.visible()
        total = sum(int(self._tensors[k].numel()) for k in self._tensors)
        cfg = getattr(self.model, "config", None)
        arch = {}
        if cfg is not None:
            for k in ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "vocab_size"):
                v = getattr(cfg, k, None)
                if v is not None and not (self.hide_io and k == "vocab_size"):
                    arch[k] = v
        return {"spec": self.spec if not self.hide_io else "(hidden)", "about": self.about, "tensors": len(vis), "hidden_tensors": len(self._tensors) - len(vis),
                "weights": total, "architecture": arch, "can_run": self.model is not None and self.forward_budget > 0,
                "forward_budget": self.forward_budget, "can_edit": self.allow_edits, "token_readout": self.can_read_tokens()}

    # -------------------------------------------------------------- read tools
    def list_tensors(self, contains="", limit=80):
        names = [k for k in self.visible() if contains.lower() in k.lower()]
        return {"count": len(names), "tensors": [{"name": k, "shape": list(self._tensors[k].shape)} for k in names[:int(limit)]],
                "more": max(0, len(names) - int(limit))}

    def tensor_stats(self, name):
        W = self.T(name).detach().float().cpu()
        out = {"name": name, "shape": list(W.shape), "mean": _r(W.mean()), "std": _r(W.std()), "absmax": _r(W.abs().max())}
        z = (W - W.mean()) / (W.std() + 1e-12)
        out["kurtosis"] = _r((z ** 4).mean() - 3)
        if W.ndim == 2 and min(W.shape) >= 2:
            s = torch.linalg.svdvals(W)
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            out["singular_values_top"] = [_r(x) for x in s[:8]]
            out["stable_rank"] = _r((s ** 2).sum() / s[0] ** 2)
            out["rank_for_90pct_energy"] = int((e < 0.9).sum()) + 1
            if min(W.shape) >= 8:
                from ..laws import _sub, matrix_measures

                m = matrix_measures(_sub(W.numpy(), np.random.default_rng(0)))
                out["alpha"] = _r(m.get("alpha"))
                out["spikes_beyond_noise_edge"] = m.get("spikes")
        return out

    def svd(self, name, k=4):
        W = self.T(name).detach().float().cpu()
        if W.ndim != 2:
            raise ValueError("svd needs a 2-D tensor.")
        U, s, Vh = torch.linalg.svd(W, full_matrices=False)
        k = max(1, min(int(k), 8, len(s)))
        rows = []
        for i in range(k):
            du = self._reg(U[:, i], f"left singular vector {i + 1} of {name} (length {W.shape[0]}, indexes its rows)")
            dv = self._reg(Vh[i], f"right singular vector {i + 1} of {name} (length {W.shape[1]}, indexes its columns)")
            rows.append({"i": i + 1, "sigma": _r(s[i]), "rows_dir": du, "cols_dir": dv, "share_of_energy": _r(s[i] ** 2 / (s ** 2).sum())})
        return {"name": name, "shape": list(W.shape), "components": rows,
                "note": "rows_dir has one entry per row of the tensor, cols_dir one per column. Pass either to read_direction or project."}

    def read_direction(self, dir, top=10):
        v = self.D(dir)
        top = max(1, min(int(top), 20))
        out = {"dir": dir, "about": self.dirs[dir]["about"], "length": int(v.numel()), "norm_before_normalising": _r(self.dirs[dir]["norm"])}
        idx = v.abs().topk(min(top, v.numel())).indices
        out["largest_entries"] = [{"index": int(i), "value": _r(v[i])} for i in idx]
        if self.can_read_tokens():
            U, E = self._unembed(), self._embed()
            if U.shape[1] == v.numel():
                lg = U @ v
                out["as_output_promotes"] = [self._tokname(i) for i in lg.topk(top).indices]
                out["as_output_suppresses"] = [self._tokname(i) for i in (-lg).topk(top).indices]
            if E.shape[1] == v.numel():
                sim = F.normalize(E, dim=1) @ v
                out["closest_input_tokens"] = [self._tokname(i) for i in sim.topk(top).indices]
            if U.shape[0] == v.numel():
                out["as_vocabulary_scores_top"] = [self._tokname(i) for i in v.topk(top).indices]
        return out

    def project(self, name, dir):
        """W applied to a direction, on whichever side matches its length; the result is a new direction."""
        W = self.T(name).detach().float().cpu()
        v = self.D(dir)
        if W.ndim != 2:
            raise ValueError("project needs a 2-D tensor.")
        res = {}
        if W.shape[1] == v.numel():
            y = W @ v
            res["W_times_dir"] = {"dir": self._reg(y, f"{name} @ {dir}"), "gain": _r(y.norm() / (torch.linalg.matrix_norm(W, 2) + 1e-12))}
        if W.shape[0] == v.numel():
            y = v @ W
            res["dir_times_W"] = {"dir": self._reg(y, f"{dir} @ {name}"), "gain": _r(y.norm() / (torch.linalg.matrix_norm(W, 2) + 1e-12))}
        if not res:
            raise ValueError(f"{dir} has length {v.numel()}; {name} is {list(W.shape)}.")
        res["note"] = "gain is how much of the matrix's largest possible stretch this direction gets (1 = its top singular direction)."
        return res

    def rows_cols(self, name, axis="rows", top=10):
        W = self.T(name).detach().float().cpu()
        if W.ndim == 1:
            W = W[None]
        n = W.norm(dim=1) if axis == "rows" else W.norm(dim=0)
        top = max(1, min(int(top), 20))
        med = float(n.median()) or 1e-12
        return {"name": name, "axis": axis, "count": int(n.numel()), "median_norm": _r(med),
                "largest": [{"index": int(i), "norm": _r(n[i])} for i in n.topk(min(top, n.numel())).indices],
                "smallest": [{"index": int(i), "norm": _r(n[i])} for i in (-n).topk(min(top, n.numel())).indices],
                "near_zero": int((n < 1e-3 * med).sum())}

    def fft(self, name, axis="rows", first=None, top=8):
        """Power spectrum along one axis (the signal runs over the row or column index), averaged over the other."""
        W = self.T(name).detach().double().cpu()
        if W.ndim != 2:
            raise ValueError("fft needs a 2-D tensor.")
        X = W if axis == "rows" else W.T
        if first:
            X = X[:int(first)]
        X = X - X.mean(0, keepdim=True)
        p = torch.fft.rfft(X, dim=0).abs().pow(2).sum(1)
        p[0] = 0
        n = X.shape[0]
        share = p / (p.sum() + 1e-30)
        top = max(1, min(int(top), 20))
        idx = share.topk(min(top, share.numel())).indices
        return {"name": name, "axis": axis, "signal_length": n,
                "top_frequencies": [{"k": int(i), "share": _r(share[i], 3)} for i in idx],
                "share_in_top": _r(share[idx].sum(), 3),
                "note": "k cycles over the signal length; a random matrix spreads energy evenly."}

    def compare(self, a, b, k=8):
        A, B = self.T(a).detach().float().cpu(), self.T(b).detach().float().cpu()
        out = {"a": a, "b": b, "shape_a": list(A.shape), "shape_b": list(B.shape)}
        if A.shape == B.shape:
            out["cosine"] = _r(F.cosine_similarity(A.flatten(), B.flatten(), dim=0))
        if A.ndim == 2 and B.ndim == 2:
            Ua, _, Vha = torch.linalg.svd(A, full_matrices=False)
            Ub, _, Vhb = torch.linalg.svd(B, full_matrices=False)
            sides = {"rows_a": Ua[:, :k], "cols_a": Vha[:k].T, "rows_b": Ub[:, :k], "cols_b": Vhb[:k].T}
            pairs = {}
            for x in ("rows_a", "cols_a"):
                for y in ("rows_b", "cols_b"):
                    if sides[x].shape[0] == sides[y].shape[0]:
                        s = torch.linalg.svdvals(sides[x].T @ sides[y])
                        pairs[f"{x}~{y}"] = _r((s ** 2).mean())
            out["top_subspace_overlap"] = pairs
            out["note"] = (f"Overlap of the top-{k} singular subspaces where lengths match: 1 = the same subspace, "
                           f"about {k}/length for unrelated ones.")
        return out

    def token_direction(self, token, which="unembed"):
        if not self.can_read_tokens():
            raise ValueError("Token readouts are not available for this model or this task.")
        ids = self.tok(token)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [i for i in ids if i != 0 or len(ids) == 1][-1:]
        if not ids:
            raise ValueError(f"{token!r} is not in the vocabulary.")
        M = self._unembed() if which == "unembed" else self._embed()
        return {"token": self._tokname(ids[0]), "dir": self._reg(M[ids[0]], f"{which} vector of {self._tokname(ids[0])!r}")}

    # -------------------------------------------------------------- forward tools
    def _ids(self, text):
        ids = self.tok(text, truncation=True, max_length=64)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def run(self, texts):
        texts = [t for t in (texts or []) if str(t).strip()][:16]
        if not texts:
            raise ValueError("Give one to 16 texts.")
        self._spend(1)
        dev = next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for t in texts:
                ids = self._ids(t)
                if len(ids) < 1:
                    continue
                lg = self.model(input_ids=torch.tensor([ids], device=dev)).logits[0].float()
                loss = float(F.cross_entropy(lg[:-1], torch.tensor(ids[1:], device=dev))) if len(ids) > 1 else None
                p = lg[-1].softmax(-1)
                tp = p.topk(5)
                out.append({"text": t[:120], "tokens": len(ids), "loss_per_token": _r(loss),
                            "next": [{"t": self._tokname(i), "p": _r(v, 3)} for v, i in zip(tp.values, tp.indices)]})
        return {"results": out, "forward_passes_left": self.forward_budget - self.forward_used}

    def attention(self, text, layer):
        self._spend(1)
        dev = next(self.model.parameters()).device
        ids = self._ids(text)
        with torch.no_grad():
            att = self.model(input_ids=torch.tensor([ids], device=dev), output_attentions=True).attentions
        if att is None or att[int(layer)] is None:
            raise ValueError("This model does not return attention patterns.")
        A = att[int(layer)][0].float().cpu()
        T = A.shape[-1]
        heads = []
        for h in range(A.shape[0]):
            a = A[h]
            off = {}
            for q in range(1, T):
                for kk in range(q + 1):
                    off[q - kk] = off.get(q - kk, 0.0) + float(a[q, kk])
            tot = sum(off.values()) or 1.0
            topo = sorted(off.items(), key=lambda x: -x[1])[:3]
            row = {"head": h, "to_first_token": _r(a[1:, 0].mean(), 3), "to_previous": _r(torch.diagonal(a, -1).mean(), 3) if T > 1 else None,
                   "to_self": _r(torch.diagonal(a).mean(), 3), "top_offsets": [{"offset": o, "share": _r(m / tot, 3)} for o, m in topo]}
            if T <= 32:
                row["argmax_key_per_query"] = [int(a[q, :q + 1].argmax()) for q in range(T)]
            heads.append(row)
        return {"layer": int(layer), "tokens": [self._tokname(i) for i in ids] if not self.hide_io else len(ids), "heads": heads,
                "forward_passes_left": self.forward_budget - self.forward_used}

    def mean_difference(self, texts_a, texts_b, layer, position="all"):
        """Mean residual stream after a block on texts A minus on texts B: a direction that separates them."""
        self._spend(1)
        dev = next(self.model.parameters()).device

        def mean(texts):
            acc, n = None, 0
            with torch.no_grad():
                for t in texts[:16]:
                    ids = self._ids(t)
                    hs = self.model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
                    h = hs[int(layer) + 1][0].float().cpu()
                    h = h[-1:] if position == "last" else h[1:] if h.shape[0] > 1 else h
                    acc = h.sum(0) if acc is None else acc + h.sum(0)
                    n += h.shape[0]
            return acc / max(n, 1)

        d = mean(texts_a) - mean(texts_b)
        k = self._reg(d, f"mean residual after block {layer} on A minus on B ({position} positions)")
        out = {"dir": k, "norm": _r(d.norm()), "forward_passes_left": self.forward_budget - self.forward_used}
        if self.can_read_tokens():
            out["reads_as"] = {kk: v for kk, v in self.read_direction(k, 8).items() if kk.startswith("as_output")}
        return out

    # -------------------------------------------------------------- edits
    def _need_edit(self):
        if not self.allow_edits:
            raise ValueError("Edits are not allowed in this task.")

    def _param(self, name):
        p = self.T(name)
        if not isinstance(p, torch.nn.Parameter):
            raise ValueError("This tensor cannot be edited.")
        return p

    def _keep(self, p, sl, spec, n):
        self.edits.append({"id": f"e{len(self.edits) + 1}", "spec": spec, "param": p, "slice": sl, "old": p[sl].detach().clone(),
                           "weights": int(n), "active": True})
        return {"edit": self.edits[-1]["id"], "weights_changed": int(n), "total_changed": self.changed()}

    def changed(self):
        return sum(e["weights"] for e in self.edits if e["active"])

    def _range(self, r, n):
        if r is None:
            return slice(None)
        lo, hi = int(r[0]), int(r[1])
        if not (0 <= lo <= hi < n):
            raise ValueError(f"Range {r} is outside 0..{n - 1}.")
        return slice(lo, hi + 1)

    def edit_project_out(self, name, dir, side=None, range=None):
        """Remove a direction from what a matrix reads or writes.
        If the direction's length matches the rows: W <- (I - d d^T) W, optionally on a range of columns only.
        If it matches the columns: W <- W (I - d d^T), optionally on a range of rows only."""
        self._need_edit()
        p = self._param(name)
        d = self.D(dir).to(p.device, p.dtype)
        m, n = p.shape
        if side is None:
            side = "rows" if d.numel() == m and d.numel() != n else "cols" if d.numel() == n and d.numel() != m else None
            if side is None:
                raise ValueError("The matrix is square: say side='rows' or 'cols'.")
        with torch.no_grad():
            if side == "rows":
                if d.numel() != m:
                    raise ValueError("Direction length does not match the rows.")
                sl = (slice(None), self._range(range, n))
                cols = (sl[1].stop - sl[1].start) if range else n
                out = self._keep(p, sl, {"op": "project_out", "tensor": name, "dir": dir, "side": side, "range": range}, m * cols)
                block = p[sl]
                p[sl] = block - torch.outer(d, d @ block)
            elif side == "cols":
                if d.numel() != n:
                    raise ValueError("Direction length does not match the columns.")
                sl = (self._range(range, m), slice(None))
                rows = (sl[0].stop - sl[0].start) if range else m
                out = self._keep(p, sl, {"op": "project_out", "tensor": name, "dir": dir, "side": side, "range": range}, rows * n)
                block = p[sl]
                p[sl] = block - torch.outer(block @ d, d)
            else:
                raise ValueError("side must be rows or cols.")
        return out

    def edit_scale(self, name, factor, target="all", index=None):
        self._need_edit()
        p = self._param(name)
        if target == "all":
            sl = (slice(None),) * p.ndim
        elif target in ("row", "col"):
            lo, hi = (index, index) if isinstance(index, int) else (int(index[0]), int(index[1]))
            sl = (slice(lo, hi + 1),) + (slice(None),) * (p.ndim - 1) if target == "row" else (slice(None), slice(lo, hi + 1))
        else:
            raise ValueError("target is all, row or col.")
        with torch.no_grad():
            out = self._keep(p, sl, {"op": "scale", "tensor": name, "factor": float(factor), "target": target, "index": index}, p[sl].numel())
            p[sl] = p[sl] * float(factor)
        return out

    def edit_add_rank1(self, name, rows_dir, cols_dir, amount):
        """W <- W + amount * a b^T, with a indexing rows and b indexing columns (both unit length)."""
        self._need_edit()
        p = self._param(name)
        a, b = self.D(rows_dir).to(p.device, p.dtype), self.D(cols_dir).to(p.device, p.dtype)
        if a.numel() != p.shape[0] or b.numel() != p.shape[1]:
            raise ValueError(f"{name} is {list(p.shape)}; directions have lengths {a.numel()} and {b.numel()}.")
        sl = (slice(None), slice(None))
        with torch.no_grad():
            out = self._keep(p, sl, {"op": "add_rank1", "tensor": name, "rows_dir": rows_dir, "cols_dir": cols_dir, "amount": float(amount)}, p.numel())
            p.add_(float(amount) * torch.outer(a, b))
        return out

    def undo(self, edit=None):
        """Undo one edit (and every edit made after it on the same tensor), or all of them."""
        idx = len(self.edits) if edit in (None, "all") else next((i for i, e in enumerate(self.edits) if e["id"] == edit), None)
        if idx is None:
            raise ValueError(f"No edit {edit}.")
        start = 0 if edit in (None, "all") else idx
        with torch.no_grad():
            for e in reversed(self.edits[start:]):
                if e["active"]:
                    e["param"][e["slice"]] = e["old"]
                    e["active"] = False
        return {"undone_from": edit or "all", "total_changed": self.changed()}

    def restore(self):
        self.undo("all")

    def edit_specs(self):
        return [e["spec"] | {"id": e["id"], "weights": e["weights"]} for e in self.edits if e["active"]]

    # -------------------------------------------------------------- tool table
    def tool_table(self):
        T = []

        def add(name, desc, props, req, fn):
            T.append({"name": name, "description": desc,
                      "input_schema": {"type": "object", "properties": props, "required": req}, "fn": fn})

        s, i = {"type": "string"}, {"type": "integer"}
        add("list_tensors", "List the model's weight tensors with their shapes. Filter by a substring.",
            {"contains": s, "limit": i}, [], self.list_tensors)
        add("tensor_stats", "Statistics of one tensor: scale, kurtosis, top singular values, stable rank, rank for 90% of the energy, heavy-tail exponent alpha and spikes beyond the random-matrix noise edge.",
            {"name": s}, ["name"], self.tensor_stats)
        add("svd", "Top singular components of a 2-D tensor. Each gives a rows direction and a cols direction (ids like d3) for later tools.",
            {"name": s, "k": i}, ["name"], self.svd)
        add("read_direction", "What a direction is: its largest entries and, when the model's vocabulary is visible, which tokens it promotes or suppresses as output and which input tokens it is closest to.",
            {"dir": s, "top": i}, ["dir"], self.read_direction)
        add("project", "Apply a 2-D tensor to a direction (on whichever side its length matches). Returns the result as a new direction and how strongly the matrix acts on it.",
            {"name": s, "dir": s}, ["name", "dir"], self.project)
        add("rows_cols", "Norms of a tensor's rows or columns: the largest, the smallest and how many are near zero.",
            {"name": s, "axis": {"type": "string", "enum": ["rows", "cols"]}, "top": i}, ["name"], self.rows_cols)
        add("fft", "Power spectrum along one axis of a 2-D tensor (the signal runs over the row or the column index, averaged over the other). Periodic structure shows up as a few frequencies holding most of the energy. 'first' keeps only the first n rows or columns.",
            {"name": s, "axis": {"type": "string", "enum": ["rows", "cols"]}, "first": i, "top": i}, ["name"], self.fft)
        add("compare", "Compare two tensors: cosine if they have the same shape, and overlap of their top singular subspaces wherever the lengths match (for example what one matrix writes against what another reads).",
            {"a": s, "b": s, "k": i}, ["a", "b"], self.compare)
        if self.can_read_tokens():
            add("token_direction", "The embedding or unembedding vector of a token, as a direction.",
                {"token": s, "which": {"type": "string", "enum": ["embed", "unembed"]}}, ["token"], self.token_direction)
        if self.model is not None and self.forward_budget > 0:
            arr = {"type": "array", "items": s}
            add("run", "Run the model (with any edits applied) on up to 16 texts you write. Costs one forward pass. Returns the loss per token and the top next tokens.",
                {"texts": arr}, ["texts"], self.run)
            add("attention", "Attention patterns of every head in one layer on one text you write. Costs one forward pass. Per head: share to the first token, to the previous token, to itself, the top query-key offsets, and for short texts the key each query attends to most.",
                {"text": s, "layer": i}, ["text", "layer"], self.attention)
            add("mean_difference", "Mean residual stream after a block on texts A minus on texts B (up to 16 each). Costs one forward pass. Returns it as a direction: something the model represents differently on A than on B.",
                {"texts_a": arr, "texts_b": arr, "layer": i, "position": {"type": "string", "enum": ["all", "last"]}},
                ["texts_a", "texts_b", "layer"], self.mean_difference)
        if self.allow_edits:
            rng = {"type": "array", "items": i, "minItems": 2, "maxItems": 2}
            add("edit_project_out", "Remove a direction from a matrix: from what it writes (direction indexes rows; optionally only a range of columns) or from what it reads (direction indexes columns; optionally only a range of rows). Undoable.",
                {"name": s, "dir": s, "side": {"type": "string", "enum": ["rows", "cols"]}, "range": rng}, ["name", "dir"], self.edit_project_out)
            add("edit_scale", "Multiply a whole tensor, a row range or a column range by a factor (0 removes it). Undoable.",
                {"name": s, "factor": {"type": "number"}, "target": {"type": "string", "enum": ["all", "row", "col"]}, "index": rng},
                ["name", "factor"], self.edit_scale)
            add("edit_add_rank1", "Add amount * a b^T to a matrix, with a a rows direction and b a cols direction (unit length). Undoable.",
                {"name": s, "rows_dir": s, "cols_dir": s, "amount": {"type": "number"}}, ["name", "rows_dir", "cols_dir", "amount"], self.edit_add_rank1)
            add("undo", "Undo an edit by id (and every later edit), or 'all'.", {"edit": s}, [], self.undo)
        for name, desc, props, req, fn in getattr(self, "extra_tools", []):
            add(name, desc, props, req, fn)
        return T

    def call(self, name, args):
        table = {t["name"]: t["fn"] for t in self.tool_table()}
        if name not in table:
            return {"error": f"No tool {name}."}
        try:
            res = table[name](**(args or {}))
        except Budget as e:
            res = {"error": str(e)}
        except TypeError as e:
            res = {"error": f"Bad arguments: {e}"}
        except Exception as e:  # shown to the agent, which can correct itself
            res = {"error": f"{type(e).__name__}: {e}"}
        self.calls.append({"tool": name, "args": args, "ok": "error" not in res})
        return res


def render(res):
    s = json.dumps(res, default=str)
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + ' ... (cut)"'
