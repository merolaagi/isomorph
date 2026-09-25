"""Ledger, experiment compiler and blueprint builder.

The ledger is the engine's memory: every mined rule, every experiment and its
verdict, every blueprint, every written world view. Rules keep their test
history across re-mining, so evidence accumulates.
"""
import json
import math
import statistics
import time
from pathlib import Path

from . import nano
from .rules import mine_lab, mine_library, rid

LEVERS = {
    "share": "Reuse one set of block weights across several layers",
    "topk": "Keep only the strongest MLP neurons for each token",
    "rank": "Factor every weight matrix into two thin matrices",
    "seed": "Start the number embeddings as waves (modular addition only)",
}


# ------------------------------------------------------------------ ledger
def _path(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    return d / "ledger.json"


def load(d: Path):
    p = _path(d)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"rules": {}, "experiments": {}, "blueprints": {}, "narratives": [], "updated": None}


def save(d: Path, L):
    from ..jsonsafe import clean

    L["updated"] = time.time()
    _path(d).write_text(json.dumps(clean(L)))
    return L


def _lever_rules(L):
    """Rules that come from the engine's own experiments."""
    out = []
    groups = {}
    for e in L["experiments"].values():
        key = (e["lever_key"], e["task"])
        groups.setdefault(key, []).append(e)
    for (lk, task), es in groups.items():
        last = sorted(es, key=lambda e: e["created"])[-1]
        v = last["verdict"]
        s = last["summary"]
        text = f"{LEVERS[last['lever']]} ({last['lever_label']}) on {nano.TASKS[task].split(',')[0].lower()}"
        stmt = {"holds": f"{text} keeps capability while saving resources",
                "rejected": f"{text} costs too much capability",
                "inconclusive": f"{text}: inconclusive"}[v]
        r = {"id": rid("lever" + lk + task), "kind": "lever", "statement": stmt, "about": last["reason"],
             "support": [e["id"] for e in es if e["verdict"] == "holds"], "counter": [e["id"] for e in es if e["verdict"] == "rejected"],
             "n": len(es), "effect": s.get("param_saving") if last["lever"] in ("share", "rank") else s.get("compute_saving"),
             "unit": "share saved", "confidence": "moderate" if len(last["seeds"]) >= 3 and v != "inconclusive" else "weak",
             "evidence": [{"experiment": e["id"], "verdict": e["verdict"], **{k: e["summary"].get(k) for k in ("param_saving", "compute_saving", "step_ratio", "metric_ratio")}} for e in es],
             "lever": {"lever": last["lever"], **last["lever_args"]}, "status": v, "tests": [e["id"] for e in es]}
        out.append(r)
    return out


def mine(engine_dir: Path, atlas_dir: Path, lab_dir: Path):
    L = load(engine_dir)
    lib_rules, metas = mine_library(atlas_dir)
    rules = lib_rules + mine_lab(lab_dir)
    old = L["rules"]
    new = {}
    for r in rules:
        prev = old.get(r["id"], {})
        r["tests"] = prev.get("tests", [])
        r["status"] = prev.get("status", "observed") if r["tests"] else ("needs more data" if r["confidence"] in ("anecdote",) else "observed")
        r["first_seen"] = prev.get("first_seen", time.time())
        new[r["id"]] = r
    for r in _lever_rules(L):
        r["first_seen"] = old.get(r["id"], {}).get("first_seen", time.time())
        new[r["id"]] = r
    L["rules"] = new
    L["models"] = len(metas)
    return save(engine_dir, L)


# ------------------------------------------------------------------ experiments
def _resolve_lever(lever, args, arch):
    if lever == "share":
        n = int(args.get("n") or max(1, arch["layers"] // 2))
        return {"share": n}, f"{arch['layers']} layers sharing {n} block{'s' if n > 1 else ''}"
    if lever == "topk":
        k = float(args.get("k") or 0.25)
        return {"topk": k}, f"top {k * 100:.0f}% of neurons"
    if lever == "rank":
        frac = float(args.get("frac") or 0.25)
        r = max(2, int(arch["d"] * frac))
        return {"rank": r}, f"rank {r} of {arch['d']}"
    if lever == "seed":
        return {"seed": True}, "wave-seeded embeddings"
    raise ValueError(f"Unknown lever {lever}.")


def _save_txt(x, fewer, more, noun):
    if x is None:
        return f"unmeasured {noun}"
    return f"{abs(x) * 100:.0f}% {fewer if x >= 0 else more} {noun}"


def _med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def run_lever(engine_dir: Path, data_dir: Path, task_name, lever, args, seeds, steps=None, rule_id=None,
              progress=lambda f, m: None, should_stop=lambda: False):
    if lever == "seed" and task_name != "modadd":
        raise ValueError("The seeding lever applies to modular addition only.")
    task = nano.make_task(task_name, data_dir)
    arch = dict(nano.DEFAULT_ARCH[task_name])
    opt = dict(nano.DEFAULT_OPT[task_name])
    steps = int(steps or opt["steps"])
    lev, label = _resolve_lever(lever, args, arch)
    runs = {"baseline": [], "variant": []}
    jobs = [(which, s) for s in seeds for which in ("baseline", "variant")]
    for i, (which, s) in enumerate(jobs):
        pr = lambda f, m, i=i: progress((i + f) / len(jobs), m)
        runs[which].append(nano.train(task, arch, lev if which == "variant" else {}, s, steps, opt["lr"], opt["wd"], pr, should_stop,
                                      label="Variant" if which == "variant" else "Baseline"))
    b, v = runs["baseline"], runs["variant"]
    P = lambda rs: rs[0]["params"]
    s = {"params_base": P(b), "params_var": P(v), "param_saving": 1 - P(v) / P(b),
         "fpt_base": b[0]["flops_per_token"], "fpt_var": v[0]["flops_per_token"]}
    if task.target is not None:
        rb = [r for r in b if r["steps_to_target"] is not None]
        rv = [r for r in v if r["steps_to_target"] is not None]
        s |= {"reached_base": len(rb), "reached_var": len(rv), "runs": len(seeds),
              "steps_base": _med([r["steps_to_target"] for r in b]), "steps_var": _med([r["steps_to_target"] for r in v]),
              "flops_base": _med([r["flops_to_target"] for r in b]), "flops_var": _med([r["flops_to_target"] for r in v])}
        s["step_ratio"] = (s["steps_var"] / s["steps_base"]) if s["steps_base"] and s["steps_var"] is not None else None
        s["compute_saving"] = (1 - s["flops_var"] / s["flops_base"]) if s["flops_base"] and s["flops_var"] is not None else None
        majority = math.ceil(len(seeds) / 2)
        if len(rb) < majority:
            verdict, reason = "inconclusive", "The baseline itself did not reach the target in most seeds; raise the step limit."
        elif len(rv) < majority:
            verdict, reason = "rejected", f"The variant reached {task.metric} {task.target:.2f} in only {len(rv)} of {len(seeds)} seeds."
        elif s["step_ratio"] is not None and s["step_ratio"] > 1.5:
            verdict, reason = "rejected", f"The variant needed {s['step_ratio']:.1f}× as many steps as the baseline."
        elif max(s["param_saving"], s["compute_saving"] or 0) >= 0.1:
            verdict, reason = "holds", (f"Reached the target in {len(rv)} of {len(seeds)} seeds with {_save_txt(s['param_saving'], 'fewer', 'more', 'parameters')} "
                                        f"and {_save_txt(s['compute_saving'], 'less', 'more', 'training compute')}.")
        else:
            verdict, reason = "rejected", "It worked but saved less than 10% of parameters or compute."
    else:
        fb, fv = _med([r["final"] for r in b]), _med([r["final"] for r in v])
        s |= {"final_base": fb, "final_var": fv, "metric_ratio": fv / fb, "runs": len(seeds),
              "compute_saving": 1 - s["fpt_var"] / s["fpt_base"]}
        if fv <= fb * 1.03 and max(s["param_saving"], s["compute_saving"]) >= 0.1:
            verdict, reason = "holds", (f"Validation {task.metric} {fv:.3f} vs {fb:.3f} (within 3%) with {_save_txt(s['param_saving'], 'fewer', 'more', 'parameters')} "
                                        f"and {_save_txt(s['compute_saving'], 'less', 'more', 'compute per token')}.")
        elif fv <= fb * 1.03:
            verdict, reason = "rejected", "Capability held but the saving was under 10%."
        else:
            verdict, reason = "rejected", f"Validation {task.metric} rose to {fv:.3f} from {fb:.3f} (more than 3% worse)."
    eid = "E-" + str(int(time.time() * 1000))[-8:]
    exp = {"id": eid, "created": time.time(), "task": task_name, "task_label": nano.TASKS[task_name], "lever": lever,
           "lever_args": args, "lever_label": label, "lever_key": f"{lever}:{json.dumps(args, sort_keys=True)}",
           "seeds": seeds, "steps": steps, "arch": arch, "verdict": verdict, "reason": reason, "summary": s,
           "rule": rule_id, "source": getattr(task, "source", None),
           "runs": {k: [{kk: vv for kk, vv in r.items()} for r in rs] for k, rs in runs.items()}}
    L = load(engine_dir)
    L["experiments"][eid] = exp
    if rule_id and rule_id in L["rules"]:
        r = L["rules"][rule_id]
        r.setdefault("tests", []).append(eid)
        verdicts = [L["experiments"][t]["verdict"] for t in r["tests"] if t in L["experiments"]]
        r["status"] = "holds" if all(x == "holds" for x in verdicts) else "rejected" if all(x == "rejected" for x in verdicts) else "mixed"
    for r in _lever_rules(L):
        r["first_seen"] = L["rules"].get(r["id"], {}).get("first_seen", time.time())
        L["rules"][r["id"]] = r
    save(engine_dir, L)
    return exp


# ------------------------------------------------------------------ blueprint
def build_blueprint(engine_dir: Path, data_dir: Path, levers: dict, tasks, seeds, steps_scale=1.0,
                    progress=lambda f, m: None, should_stop=lambda: False):
    """Train a model with every chosen lever against a plain baseline, task by task."""
    results = []
    jobs = [(t, which, s) for t in tasks for s in seeds for which in ("baseline", "blueprint")]
    done = 0
    per_task = {}
    for t in tasks:
        task = nano.make_task(t, data_dir)
        arch = dict(nano.DEFAULT_ARCH[t])
        opt = dict(nano.DEFAULT_OPT[t])
        lev, labels = {}, []
        for k, a in levers.items():
            if k == "seed" and t != "modadd":
                continue
            if k == "share" and arch["layers"] < 2:
                continue
            l, lab = _resolve_lever(k, a or {}, arch)
            lev |= l
            labels.append(lab)
        runs = {"baseline": [], "blueprint": []}
        for s in seeds:
            for which in ("baseline", "blueprint"):
                pr = lambda f, m, done=done: progress((done + f) / len(jobs), f"{nano.TASKS[t].split(',')[0]}: {m}")
                runs[which].append(nano.train(task, arch, lev if which == "blueprint" else {}, s, int(opt["steps"] * steps_scale),
                                              opt["lr"], opt["wd"], pr, should_stop, label=which.capitalize()))
                done += 1
        b, v = runs["baseline"], runs["blueprint"]
        row = {"task": t, "task_label": nano.TASKS[t], "metric": task.metric, "levers": labels, "higher_better": task.higher_better,
               "params_base": b[0]["params"], "params_bp": v[0]["params"], "fpt_base": b[0]["flops_per_token"], "fpt_bp": v[0]["flops_per_token"],
               "final_base": _med([r["final"] for r in b]), "final_bp": _med([r["final"] for r in v]),
               "source": getattr(task, "source", None),
               "curves": {"baseline": [r["history"] for r in b], "blueprint": [r["history"] for r in v]}}
        if task.target is not None:
            row |= {"target": task.target, "steps_base": _med([r["steps_to_target"] for r in b]), "steps_bp": _med([r["steps_to_target"] for r in v]),
                    "flops_base": _med([r["flops_to_target"] for r in b]), "flops_bp": _med([r["flops_to_target"] for r in v]),
                    "reached_base": sum(r["steps_to_target"] is not None for r in b), "reached_bp": sum(r["steps_to_target"] is not None for r in v)}
            row["capability_per_compute"] = (row["flops_base"] / row["flops_bp"]) if row["flops_base"] and row["flops_bp"] else None
        else:
            row["capability_per_compute"] = (row["fpt_base"] / row["fpt_bp"]) if row["final_bp"] <= row["final_base"] * 1.03 else None
        per_task[t] = row
        results.append(row)
    bid = "B-" + str(int(time.time() * 1000))[-8:]
    bp = {"id": bid, "created": time.time(), "levers": levers, "tasks": tasks, "seeds": seeds, "results": results}
    L = load(engine_dir)
    L["blueprints"][bid] = bp
    save(engine_dir, L)
    return bp
