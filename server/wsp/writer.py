"""Write: networks small enough to see whole, written by hand or by Claude.

Brought in from Weightspace Lab (github.com/merolaagi/weightspacelab), an
in-browser demo of a 2 -> H -> 1 network (ReLU hidden units, sigmoid output)
whose weights you edit with sliders. Here it gains what the Weight-Space
Programming paper actually tests:

  write    Claude is told the target pattern in words and writes every weight,
           with no data and no gradients. With an evaluation budget it may
           score drafts on the grid before it commits; with 0 it writes once.
  repair   Claude gets the broken network (one output weight's sign flipped,
           as in the lab) and must find and fix the fault.

Grading is the lab's own: agreement with the target on a fixed 41 x 41 grid
over [-1, 1]^2. Two controls sit beside it: random networks of the same size
(what you get for free), and the same network trained by gradient descent on
400 labelled points (what ordinary training gets). The forward pass matches
the lab's JavaScript exactly.
"""
import math
import time

import numpy as np

GRID = 41
PATTERNS = {
    "diagonal": {"label": "Diagonal split", "rule": "class 1 when x + y > 0", "f": lambda x, y: x + y > 0,
                 "crafted": {"W": [[1, 1], [-1, -1], [1, 0], [0, 1]], "b": [0, 0, 0, 0], "v": [6, -6, 0, 0], "bias": 0}},
    "band": {"label": "Vertical band", "rule": "class 1 when |x| < 0.4", "f": lambda x, y: abs(x) < 0.4,
             "crafted": {"W": [[1, 0], [-1, 0], [0, 1], [0, -1]], "b": [0, 0, 0, 0], "v": [-8, -8, 0, 0], "bias": 3.2}},
    "xor": {"label": "Opposite corners (XOR)", "rule": "class 1 when x and y have opposite signs (x * y < 0)", "f": lambda x, y: x * y < 0,
            "crafted": {"W": [[1, 1], [-1, -1], [1, -1], [-1, 1]], "b": [0, 0, 0, 0], "v": [-6, -6, 6, 6], "bias": 0}},
    # eight ReLU ridges in the eight compass directions add up to about 2.55 |(x, y)|: a near-round distance
    "disc": {"label": "Disc", "rule": "class 1 inside the circle x^2 + y^2 < 0.5", "f": lambda x, y: x * x + y * y < 0.5, "hidden": 8,
             "crafted": {"W": [[1, 0], [0.707107, 0.707107], [0, 1], [-0.707107, 0.707107], [-1, 0], [-0.707107, -0.707107], [0, -1], [0.707107, -0.707107]],
                         "b": [0] * 8, "v": [-10] * 8, "bias": 18.2}},
    # two units measure how far x and y fall short of the corner; any shortfall switches the output off
    "corner": {"label": "One corner", "rule": "class 1 when x > 0.2 and y > 0.2", "f": lambda x, y: x > 0.2 and y > 0.2,
               "crafted": {"W": [[-1, 0], [0, -1], [0, 0], [0, 0]], "b": [0.25, 0.25, 0, 0], "v": [-50, -50, 0, 0], "bias": 0.5}},
    # three units per side make a triangle bump between |y| = 0.2 and 0.6
    "stripes": {"label": "Two stripes", "rule": "class 1 when 0.2 < |y| < 0.6", "f": lambda x, y: 0.2 < abs(y) < 0.6, "hidden": 8,
                "crafted": {"W": [[0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1], [0, 0], [0, 0]], "b": [-0.2, -0.4, -0.6, -0.2, -0.4, -0.6, 0, 0],
                            "v": [100, -200, 100, 100, -200, 100, 0, 0], "bias": -0.5}},
}
MAX_HIDDEN = 16


def grid():
    pts = [((x - 20) / 20, (y - 20) / 20) for y in range(GRID) for x in range(GRID)]
    return np.array(pts, dtype=np.float64)


def target(pattern):
    f = PATTERNS[pattern]["f"]
    return np.array([bool(f(x, y)) for x, y in grid()])


def check(net, hidden=None):
    """Coerce a network to float arrays and check its shape (W: H x 2, b: H, v: H, bias: scalar)."""
    try:
        W = np.asarray(net["W"], dtype=np.float64)
        b = np.asarray(net["b"], dtype=np.float64)
        v = np.asarray(net["v"], dtype=np.float64)
        bias = float(net["bias"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"A network is {{W: H x 2, b: H, v: H, bias}}; this one is not ({e}).")
    H = W.shape[0] if W.ndim == 2 else -1
    if W.ndim != 2 or W.shape[1] != 2 or b.shape != (H,) or v.shape != (H,) or not 1 <= H <= MAX_HIDDEN:
        raise ValueError(f"Shapes must be W {H} x 2, b {H}, v {H} with 1 to {MAX_HIDDEN} hidden units; got W {list(W.shape)}, b {list(b.shape)}, v {list(v.shape)}.")
    if hidden is not None and H != hidden:
        raise ValueError(f"This task uses {hidden} hidden units; the network has {H}.")
    if not all(np.isfinite(a).all() for a in (W, b, v)) or not math.isfinite(bias):
        raise ValueError("Every weight must be a finite number.")
    return {"W": W, "b": b, "v": v, "bias": bias}


def predict(net, pts):
    """The lab's forward pass: sigmoid(v . relu(W [x, y] + b) + bias)."""
    n = check(net)
    h = np.maximum(0.0, pts @ n["W"].T + n["b"])
    z = h @ n["v"] + n["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def agreement(net, pattern):
    return float(((predict(net, grid()) > 0.5) == target(pattern)).mean())


def picture(net, pattern, size=11):
    """A coarse map of where the network is wrong, for Claude to read: '.' right, 'x' wrong; rows top (y = +1) to bottom."""
    pts = np.array([((i / (size - 1)) * 2 - 1, 1 - (j / (size - 1)) * 2) for j in range(size) for i in range(size)])
    f = PATTERNS[pattern]["f"]
    ok = (predict(net, pts) > 0.5) == np.array([bool(f(x, y)) for x, y in pts])
    return ["".join("." if ok[j * size + i] else "x" for i in range(size)) for j in range(size)]


def broken(pattern):
    c = PATTERNS[pattern]["crafted"]
    if not c:
        raise ValueError("This pattern has no crafted network to break; use the write task.")
    n = {"W": [list(r) for r in c["W"]], "b": list(c["b"]), "v": list(c["v"]), "bias": c["bias"]}
    n["v"][0] = -n["v"][0]
    return n


def as_lists(n):
    n = check(n)
    return {"W": n["W"].round(6).tolist(), "b": n["b"].round(6).tolist(), "v": n["v"].round(6).tolist(), "bias": round(n["bias"], 6)}


def changed(a, b):
    fa = np.concatenate([np.ravel(check(a)[k]) for k in ("W", "b", "v")] + [[check(a)["bias"]]])
    fb = np.concatenate([np.ravel(check(b)[k]) for k in ("W", "b", "v")] + [[check(b)["bias"]]])
    return int((np.abs(fa - fb) > 1e-7).sum()) if fa.shape == fb.shape else int(fa.size)


# ------------------------------------------------------------------ controls
def majority_control(pattern):
    """Always answering the more common class."""
    t = target(pattern)
    return float(max(t.mean(), 1 - t.mean()))


def random_control(pattern, hidden, draws=50, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        n = {"W": rng.normal(0, 1, (hidden, 2)), "b": rng.normal(0, 0.5, hidden), "v": rng.normal(0, 2, hidden), "bias": float(rng.normal(0, 0.5))}
        out.append(agreement(n, pattern))
    return {"mean": float(np.mean(out)), "best": float(np.max(out)), "draws": draws}


def trained_control(pattern, hidden, points=400, steps=2000, seeds=3, lr=0.05):
    """The same network trained the ordinary way: Adam on the cross-entropy of `points` labelled random points."""
    import torch

    f = PATTERNS[pattern]["f"]
    res = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)              # a 2 -> H -> 1 network is far faster on one thread
    try:
        return _train(f, pattern, hidden, points, steps, seeds, lr, res, torch)
    finally:
        torch.set_num_threads(threads)


def _train(f, pattern, hidden, points, steps, seeds, lr, res, torch):
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        x = torch.rand(points, 2, generator=g, dtype=torch.float64) * 2 - 1
        y = torch.tensor([float(bool(f(float(a), float(b)))) for a, b in x], dtype=torch.float64)
        torch.manual_seed(s)
        W = torch.nn.Parameter(torch.randn(hidden, 2, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
        v = torch.nn.Parameter(torch.randn(hidden, dtype=torch.float64) * 0.5)
        c = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        opt = torch.optim.Adam([W, b, v, c], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            z = torch.relu(x @ W.T + b) @ v + c
            torch.nn.functional.binary_cross_entropy_with_logits(z, y).backward()
            opt.step()
        res.append(agreement({"W": W.detach().numpy(), "b": b.detach().numpy(), "v": v.detach().numpy(), "bias": float(c.detach())}, pattern))
    return {"mean": float(np.mean(res)), "spread": float((max(res) - min(res)) / 2), "points": points, "steps": steps, "seeds": seeds}


# ------------------------------------------------------------------ Claude
class WriteWorkspace:
    """The tools Claude gets: score a draft on the grid (each call costs one evaluation)."""

    def __init__(self, pattern, hidden, budget, start=None):
        self.pattern, self.hidden, self.forward_budget, self.forward_used = pattern, hidden, int(budget), 0
        self.calls, self.drafts, self.start = [], [], start

    def evaluate(self, net):
        if self.forward_used >= self.forward_budget:
            return {"error": f"Evaluation budget spent ({self.forward_used} of {self.forward_budget})."}
        n = check(net, self.hidden)
        self.forward_used += 1
        a = agreement(n, self.pattern)
        self.drafts.append({"net": as_lists(n), "agreement": a})
        return {"agreement": round(a, 4), "errors_map": picture(n, self.pattern),
                "map_key": "11 x 11 points over [-1, 1]^2, top row y = +1, left column x = -1; '.' right, 'x' wrong",
                "evaluations_left": self.forward_budget - self.forward_used}

    def tool_table(self):
        arr = {"type": "array", "items": {"type": "number"}}
        net = {"type": "object", "properties": {"W": {"type": "array", "items": arr}, "b": arr, "v": arr, "bias": {"type": "number"}},
               "required": ["W", "b", "v", "bias"]}
        if self.forward_budget <= 0:
            return []
        return [{"name": "evaluate", "description": "Score a draft network on the 41 x 41 grid: agreement with the target, and an 11 x 11 map of where it is wrong. Costs one evaluation.",
                 "input_schema": {"type": "object", "properties": {"net": net}, "required": ["net"]}, "fn": self.evaluate}]

    def call(self, name, args):
        try:
            res = self.evaluate((args or {}).get("net")) if name == "evaluate" else {"error": f"No tool {name}."}
        except ValueError as e:
            res = {"error": str(e)}
        self.calls.append({"tool": name, "ok": "error" not in res})
        return res

    def restore(self):
        pass


def task_text(mode, pattern, hidden, budget, start=None):
    p = PATTERNS[pattern]
    net = (f"The network: inputs (x, y) in [-1, 1]^2; {hidden} hidden units h_i = relu(W[i][0] * x + W[i][1] * y + b[i]); "
           f"output p = sigmoid(sum_i v[i] * h_i + bias); class 1 when p > 0.5. W is {hidden} x 2, b and v have {hidden} entries, bias is one number.")
    rule = f"Target: {p['rule']}."
    check_ = (f"You may score drafts with evaluate ({budget} evaluations): it returns the agreement on a 41 x 41 grid and a map of where the draft is wrong."
              if budget else "You cannot run the network: write the weights from the rule alone, in one go.")
    if mode == "repair":
        return ("This small network was written to compute a target pattern, but one weight is wrong and it no longer does. "
                f"{net} {rule} The current weights: {start}. Find the faulty weight from the weights and fix it with as few changes as you can. "
                f"{check_} Then call finish with the whole repaired network.")
    return (f"Write the weights of a small neural network by hand, with no data and no training. {net} {rule} "
            f"Work out what each hidden unit should compute, then set the numbers. {check_} Then call finish with the whole network.")


def run_write(pattern, call, mode="write", hidden=None, budget=5, max_turns=20, progress=lambda f, m: None, should_stop=lambda: False):
    from .agent import run_agent

    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern}.")
    t0 = time.time()
    if mode == "repair":
        start = broken(pattern)
        hidden = len(start["W"])
    else:
        start = None
        hidden = int(hidden or PATTERNS[pattern].get("hidden", 4))
    if not 1 <= hidden <= MAX_HIDDEN:
        raise ValueError(f"Hidden units 1 to {MAX_HIDDEN}.")
    if not 0 <= int(budget) <= 50:
        raise ValueError("Evaluation budget 0 to 50.")
    ws = WriteWorkspace(pattern, hidden, budget, start)
    arr = {"type": "array", "items": {"type": "number"}}
    schema = {"W": {"type": "array", "items": arr}, "b": arr, "v": arr, "bias": {"type": "number"},
              "explanation": {"type": "string", "description": "What each hidden unit computes, and why these numbers."}}
    progress(0.05, "Claude is writing the weights")
    ans, transcript, usage = run_agent(ws, task_text(mode, pattern, hidden, budget, start), schema, call, max_turns, progress, should_stop)
    final, err = None, None
    if ans:
        try:
            final = as_lists(check({k: ans.get(k) for k in ("W", "b", "v", "bias")}, hidden))
        except ValueError as e:
            err = str(e)
    progress(0.8, "Training the same network the ordinary way, for comparison")
    res = {"kind": "write", "task": mode, "spec": f"{PATTERNS[pattern]['label']}, 2 → {hidden} → 1", "pattern": pattern, "hidden": hidden,
           "budget": int(budget), "net": final, "explanation": (ans or {}).get("explanation"), "invalid": err,
           "agreement": agreement(final, pattern) if final else None,
           "drafts": [{"agreement": d["agreement"]} for d in ws.drafts], "forward_used": ws.forward_used, "tool_calls": len(ws.calls),
           "random": random_control(pattern, hidden), "majority": majority_control(pattern), "trained": trained_control(pattern, hidden),
           "transcript": transcript, "usage": usage, "created": time.time()}
    if mode == "repair":
        res["start"] = start
        res["start_agreement"] = agreement(start, pattern)
        res["crafted_agreement"] = agreement(PATTERNS[pattern]["crafted"], pattern)
        res["weights_changed"] = changed(final, start) if final else None
        res["fixed_the_fault"] = bool(final) and abs(final["v"][0] - PATTERNS[pattern]["crafted"]["v"][0]) < abs(start["v"][0] - PATTERNS[pattern]["crafted"]["v"][0])
    res["seconds"] = time.time() - t0
    return res


def meta():
    return {"patterns": {k: {"label": v["label"], "rule": v["rule"], "hidden": v.get("hidden", 4), "crafted": v["crafted"],
                             "crafted_agreement": agreement(v["crafted"], k) if v["crafted"] else None,
                             "broken_agreement": agreement(broken(k), k) if v["crafted"] else None,
                             "target": target(k).astype(int).tolist()} for k, v in PATTERNS.items()},
            "grid": GRID, "max_hidden": MAX_HIDDEN}


def to_tensors(net):
    """A written network as named tensors (PyTorch Linear layout), for Isomorph's import."""
    import torch

    n = check(net)
    return {"hidden.weight": torch.tensor(n["W"], dtype=torch.float32), "hidden.bias": torch.tensor(n["b"], dtype=torch.float32),
            "out.weight": torch.tensor(n["v"][None, :], dtype=torch.float32), "out.bias": torch.tensor([n["bias"]], dtype=torch.float32)}
