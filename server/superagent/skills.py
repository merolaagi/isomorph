"""What the superagent can do. Each skill is one Isomorph endpoint with typed
arguments, a cost class, and a way to turn the result into a short digest (for
the mission log and for Claude) and an action that opens the result in the page.

cost: instant (a read), seconds, minutes, long (tens of minutes or more).
Skills that would delete, run code, or touch settings are deliberately absent.
"""
import json
import re

DEMO_SL = "\n".join(
    (f"the {a}cat => swam in the {'pond' if i % 4 < 2 else 'lake'} ." if i % 2 else f"the {a}fish => flew over the {'tree' if i % 3 else 'house'} .")
    for i, a in ((i, ["", "big ", "small ", "happy "][i % 4]) for i in range(40)))
DEMO_RL_PROMPTS = "the\na\nthe big\nthe small\nthe happy"


def P(type_, default=None, desc="", required=False, enum=None):
    return {"type": type_, "default": default, "desc": desc, "required": required, "enum": enum}


def _first(d, *keys):
    for k in keys:
        cur = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur not in (None, "", []):
            return cur
    return None


def digest(r, limit=900):
    """A short, faithful summary of any result: its verdict-like text and its top-level numbers."""
    if not isinstance(r, dict):
        return str(r)[:limit]
    text = _first(r, "verdict", "finding", "summary.lines", "anatomy.summary.lines", "grade.how", "line", "report.summary")
    if isinstance(text, list):
        text = " ".join(map(str, text))
    nums = {}
    for k, v in r.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("created", "seconds"):
            nums[k] = round(v, 4) if isinstance(v, float) else v
        elif isinstance(v, dict) and k in ("before", "after", "grade", "base", "edited", "settings"):
            nums[k] = {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()
                       if isinstance(b, (int, float, str, bool)) and len(str(b)) < 60}
    out = (str(text) + " " if text else "") + (json.dumps(nums)[:limit // 2] if nums else "")
    return out.strip()[:limit] or json.dumps(r, default=str)[:limit]


SKILLS = {}
CTX = {"data": None}          # set by the superagent: where results live, for richer digests


def _laws_lines(specs):
    """Which laws hold in each tested model, read from the saved results."""
    from ..laws import LAWS, law_id, verdicts

    out = []
    d = CTX["data"] / "laws" if CTX["data"] else None
    for sp in specs:
        f = d / f"{law_id(sp)}.json" if d else None
        if not f or not f.exists():
            continue
        try:
            v = verdicts(json.loads(f.read_text()))
        except Exception:
            continue
        held = [LAWS[k]["label"] for k, x in v.items() if k in LAWS and x.get("holds")]
        out.append(f"{sp}: {len(held)}/{len(v)} hold ({', '.join(held[:6])}{'...' if len(held) > 6 else ''})")
    return " ".join(out)


def _trace_line(r):
    res = r.get("result", r) if isinstance(r, dict) else {}
    div = res.get("divergence") or []
    js = [(row.get("a"), max(row.get("js") or [0])) for row in div if row.get("a")]
    if not js:
        if res.get("same_tokens") is False:
            na, nb = len((res.get("A") or {}).get("tokens") or []), len((res.get("B") or {}).get("tokens") or [])
            return (f"The two models split the prompt into different tokens ({na} and {nb}), so their predictions are not compared "
                    "token by token; open the trace for each one's layer-by-layer guesses.")
        return "Traced; open it for the layer-by-layer views."
    worst = max(js, key=lambda x: x[1])
    s = res.get("summary") or {}
    return (f"Largest disagreement at layer {worst[0]} (Jensen-Shannon {worst[1]:.2f} bits, where 1 means nothing in common). "
            f"Final predictions {'agree' if s.get('final_agree') else 'differ'}"
            + (f"; they settle from layer {s['settle_layer']}." if s.get("settle_layer") is not None else "."))


def skill(name, area, label, desc, params, cost, needs_key=False, downloads=False):
    def wrap(fn):
        SKILLS[name] = {"name": name, "area": area, "label": label, "desc": desc, "params": params, "cost": cost,
                        "needs_key": needs_key, "downloads": downloads, "build": fn}
        return fn
    return wrap


def _specs(v):
    if isinstance(v, str):
        v = [x for x in re.split(r"[\n,]+", v) if x.strip()]
    return [str(x).strip() for x in (v or []) if str(x).strip()]


# ------------------------------------------------------------------ the bench
@skill("trace", "Side by side", "Trace a prompt through two models",
       "Follow one prompt layer by layer through models a and b: predictions per layer, where they part ways, attention, neurons, and which weights one training step would move.",
       {"a": P("string", "microopus:words", "model A", True), "b": P("string", "microopus:names", "model B", True),
        "prompt": P("string", "the cat sat on the mat .", "the prompt"), "learning": P("boolean", True, "add the one-step learning probe")},
       "minutes", downloads=True)
def _trace(a):
    return ("POST", "/api/trace/run", {"a": a["a"], "b": a["b"], "prompt": a["prompt"], "learning": a["learning"]}), \
        lambda r: {"summary": f"Traced {a['a']} vs {a['b']} on '{a['prompt']}'. " + _trace_line(r), "out": {"id": r.get("id")},
                   "open": {"type": "trace", "id": r.get("id")}}


@skill("profile", "Pattern library", "Profile a model into the library",
       "Fingerprint, attention circuits, concept probes and weight signature of a model, each against a random copy of the same architecture.",
       {"spec": P("string", None, "model", True), "baseline": P("boolean", True, "also profile a random copy")}, "minutes", downloads=True)
def _profile(a):
    return ("POST", "/api/atlas/profile", {"spec": a["spec"], "baseline": a["baseline"]}), \
        lambda r: {"summary": f"Profiled {a['spec']}. " + digest(r, 500), "out": {"spec": a["spec"]}, "open": {"type": "go", "view": "atlas"}}


@skill("laws", "Laws", "Run the physics and maths laws on models",
       "Heavy tails, Marchenko-Pastur spikes, level spacing, Benford, conservation laws... on each model's weights, against Gaussian and shuffled nulls.",
       {"specs": P("array", None, "models (microopus:..., lab:<run>:<seed>, tune:<adapter>, Hugging Face ids)", True)}, "minutes", downloads=True)
def _laws(a):
    specs = _specs(a["specs"])
    return ("POST", "/api/laws/run", {"specs": specs}), \
        lambda r: {"summary": (f"Laws tested on {len(r.get('done', []))} model(s)" + (f"; failed: {', '.join(x if isinstance(x, str) else str(x) for x in r['failed'])}" if r.get("failed") else "") + ". "
                               + _laws_lines(r.get("done", []))).strip(),
                   "out": r, "open": {"type": "laws", "spec": (r.get("done") or [None])[-1]}}


@skill("laws_summary", "Laws", "Read the laws table", "Which laws hold in how many of the tested models (no computation).", {}, "instant")
def _laws_summary(a):
    def res(r):
        laws, uni = (r or {}).get("laws", {}), (r or {}).get("universality", {})
        lines = [f"{laws.get(k, {}).get('label', k)}: {u['have']}/{u['of']}" for k, u in uni.items()]
        return {"summary": (f"{len(r.get('models', []))} models tested. " + "; ".join(lines))[:1200] if lines else "No model has been tested yet.",
                "out": {}, "open": {"type": "go", "view": "laws"}}
    return ("GET", "/api/laws", None), res


@skill("lab_train", "Ground-truth lab", "Train a ground-truth family",
       "Tiny transformers that learn (a + b) mod p, where the right algorithm (Fourier features) is known.",
       {"p": P("integer", 53, "modulus"), "seeds": P("array", [1, 2, 3], "seeds"), "steps": P("integer", 3000, "steps per seed")}, "long")
def _lab_train(a):
    return ("POST", "/api/lab/train", {"p": a["p"], "seeds": a["seeds"], "steps": a["steps"]}), \
        lambda r: {"summary": f"Trained {len(a['seeds'])} seeds on (a + b) mod {a['p']} (run {r.get('run_id')}).", "out": {"run_id": r.get("run_id")},
                   "open": {"type": "lab", "run_id": r.get("run_id")}}


@skill("lab_family", "Ground-truth lab", "Compare every seed of a lab run",
       "Raw and symmetry-aligned similarity, shared circuits after relabelling, against a null.",
       {"run_id": P("string", "latest", "run id, or latest")}, "minutes")
def _lab_family(a):
    return ("POST", "/api/lab/family", {"run_id": a["run_id"]}), \
        lambda r: {"summary": "Family survey: " + digest(r, 600), "out": {"run_id": a["run_id"]}, "open": {"type": "lab", "run_id": a["run_id"]}}


@skill("microopus_seed", "Side by side", "Train another Micro Opus seed",
       "Micro Opus's own train.py with a new seed: a second model of the same kind to compare with.",
       {"kind": P("string", "words", "words or names", enum=["words", "names"]), "seed": P("integer", 11, "seed")}, "minutes")
def _mo(a):
    return ("POST", "/api/microopus/train", {"kind": a["kind"], "seed": a["seed"]}), \
        lambda r: {"summary": f"Trained {r.get('spec')} (matches the numpy reference to {r.get('diff')}).", "out": {"spec": r.get("spec")},
                   "open": {"type": "go", "view": "trace"}}


@skill("hub_weights", "Model statistics", "Compare two checkpoints' weights",
       "Cosine, spectra and low-rank structure of the difference, tensor by tensor.",
       {"a": P("string", None, "model A", True), "b": P("string", None, "model B", True)}, "minutes", downloads=True)
def _hub(a):
    return ("POST", "/api/hub/weights", {"a": a["a"], "b": a["b"]}), \
        lambda r: {"summary": digest(_first(r, "result.summary") or r, 600), "out": {}, "open": {"type": "go", "view": "hub"}}


# ------------------------------------------------------------------ weight space
@skill("wsp_plan", "Weight space", "Plan a fine-tune from the weights",
       "Choose which matrices to tune from the weights and from the gradient, and test both against three controls at the same budget.",
       {"spec": P("string", "microopus:words", "model"), "examples": P("string", "demo", "examples, one per line ('prompt => target'), or demo"),
        "steps": P("integer", 150, "steps"), "seeds": P("integer", 2, "seeds")}, "minutes")
def _wsp_plan(a):
    ex = DEMO_SL.replace(" => ", " ") if a["examples"] in ("demo", "", None) else a["examples"]
    return ("POST", "/api/wsp/plan", {"spec": a["spec"], "examples": ex, "steps": a["steps"], "seeds": a["seeds"]}), \
        lambda r: {"summary": r.get("verdict", digest(r)), "out": {"id": r.get("id")}, "open": {"type": "wsp", "id": r.get("id"), "mode": "plan"}}


@skill("wsp_read", "Weight space", "Have Claude read the weights",
       "Claude works through fixed tools on the weights. Graded tasks: data (what Micro Opus was trained on), freqs (modulus and Fourier frequencies of a lab model, spec lab:<run>:<seed>), induction (a language model), or free with a question.",
       {"task": P("string", "data", "task", enum=["data", "freqs", "induction", "free"]), "spec": P("string", "microopus:words", "model"),
        "question": P("string", "", "for free reading")}, "minutes", needs_key=True)
def _wsp_read(a):
    return ("POST", "/api/wsp/read", {"task": a["task"], "spec": a["spec"], "question": a["question"]}), \
        lambda r: {"summary": f"Read ({a['task']}): " + digest(r, 600), "out": {"id": r.get("id")}, "open": {"type": "wsp", "id": r.get("id"), "mode": "read"}}


@skill("wsp_fix", "Weight space", "Have Claude repair the weights",
       "Claude edits weights directly, without training. planted: a known fault to find and repair, judged against random-direction edits.",
       {"task": P("string", "planted", "task", enum=["planted", "custom"]), "spec": P("string", "microopus:words", "model"),
        "goal": P("string", "", "for custom"), "examples": P("string", "", "for custom")}, "minutes", needs_key=True)
def _wsp_fix(a):
    return ("POST", "/api/wsp/fix", {"task": a["task"], "spec": a["spec"], "goal": a["goal"], "examples": a["examples"]}), \
        lambda r: {"summary": f"Fix ({a['task']}): " + digest(r, 600), "out": {"id": r.get("id")}, "open": {"type": "wsp", "id": r.get("id"), "mode": "fix"}}


@skill("wsp_write", "Weight space", "Have Claude write a network from a rule",
       "A 2 -> H -> 1 network written weight by weight from the rule alone, graded on a grid against random and trained networks.",
       {"pattern": P("string", "disc", "pattern", enum=["diagonal", "band", "xor", "disc", "corner", "stripes"]),
        "mode": P("string", "write", "write or repair", enum=["write", "repair"]), "budget": P("integer", 0, "drafts it may score")}, "minutes", needs_key=True)
def _wsp_write(a):
    return ("POST", "/api/wsp/write", {"pattern": a["pattern"], "mode": a["mode"], "budget": a["budget"]}), \
        lambda r: {"summary": f"Write ({a['pattern']}): " + digest(r, 500), "out": {"id": r.get("id")}, "open": {"type": "wsp", "id": r.get("id"), "mode": "write"}}


# ------------------------------------------------------------------ tune
@skill("tune_sl", "Tune", "Fine-tune from examples (LoRA)",
       "Supervised LoRA fine-tune; held-out loss and accuracy before and after, drift on general text, and where the change went.",
       {"spec": P("string", "microopus:words", "base model"), "examples": P("string", "demo", "examples ('prompt => target'), or demo"),
        "rank": P("integer", 8, "LoRA rank"), "lr": P("number", 1e-3, "learning rate"), "steps": P("integer", 100, "steps"),
        "seed": P("integer", 0, "seed"), "name": P("string", "", "adapter name")}, "seconds")
def _tune_sl(a):
    ex = DEMO_SL if a["examples"] in ("demo", "", None) else a["examples"]
    body = {"spec": a["spec"], "examples": ex, "rank": a["rank"], "lr": a["lr"], "steps": a["steps"], "seed": a["seed"], "name": a["name"]}
    return ("POST", "/api/tune/supervised", body), \
        lambda r: {"summary": (f"Adapter {r['adapter']}: held-out loss {r['before']['held_loss']:.3f} -> {r['after']['held_loss']:.3f}, accuracy "
                               f"{r['before']['held_acc'] * 100:.0f}% -> {r['after']['held_acc'] * 100:.0f}%, drift {r['drift_kl']:.3f} nats/token. "
                               + " ".join((r.get("anatomy") or {}).get("summary", {}).get("lines", [])[:3])),
                   "out": {"adapter": r["adapter"], "spec": r["adapter_spec"]}, "open": {"type": "tune", "adapter": r["adapter"]}}


@skill("tune_rl", "Tune", "Fine-tune with a reward (LoRA)",
       "Group-relative reinforcement learning with a KL penalty to the base; mean reward before and after, drift, and where the change went.",
       {"spec": P("string", "microopus:words", "base model"), "prompts": P("string", "demo", "prompts, one per line, or demo"),
        "reward": P("string", "contains", "reward rule", enum=["contains", "avoids", "target", "length", "regex"]),
        "reward_arg": P("string", "cat", "words, a length or a pattern"), "steps": P("integer", 60, "steps"),
        "kl_coef": P("number", 0.05, "stay near the base"), "seed": P("integer", 1, "seed"), "name": P("string", "", "adapter name")}, "seconds")
def _tune_rl(a):
    pr = DEMO_RL_PROMPTS if a["prompts"] in ("demo", "", None) else a["prompts"]
    body = {"spec": a["spec"], "prompts": pr, "reward": a["reward"], "reward_arg": a["reward_arg"], "steps": a["steps"],
            "kl_coef": a["kl_coef"], "seed": a["seed"], "name": a["name"]}
    return ("POST", "/api/tune/reinforce", body), \
        lambda r: {"summary": (f"Adapter {r['adapter']}: mean reward {r['before']['reward']:.2f} -> {r['after']['reward']:.2f}, drift {r['drift_kl']:.3f} nats/token. "
                               + " ".join((r.get("anatomy") or {}).get("summary", {}).get("lines", [])[:2])),
                   "out": {"adapter": r["adapter"], "spec": r["adapter_spec"]}, "open": {"type": "tune", "adapter": r["adapter"]}}


@skill("tune_compare", "Tune", "Do two fine-tunes agree?",
       "Subspace overlap of two adapters' weight changes, against random subspaces of the same size.",
       {"a": P("string", None, "adapter A", True), "b": P("string", None, "adapter B", True)}, "seconds")
def _tune_compare(a):
    return ("POST", "/api/tune/compare", {"a": a["a"], "b": a["b"]}), \
        lambda r: {"summary": r.get("verdict", digest(r)), "out": {"overlap": r.get("overlap"), "random": r.get("random")},
                   "open": {"type": "tune", "adapter": a["a"]}}


@skill("tune_anatomy", "Tune", "Where did a fine-tune go?", "Read a saved adapter as a change to the weights.",
       {"name": P("string", None, "adapter", True)}, "seconds")
def _tune_anatomy(a):
    return ("POST", f"/api/tune/adapter/{a['name']}/anatomy", {}), \
        lambda r: {"summary": " ".join(r.get("summary", {}).get("lines", [])) or digest(r), "out": {}, "open": {"type": "tune", "adapter": a["name"]}}


# ------------------------------------------------------------------ engine
@skill("mine", "Engine", "Mine rules", "Turn everything measured so far (library, lab, laws) into rules with support and counterexamples.", {}, "seconds")
def _mine(a):
    return ("POST", "/api/engine/mine", {}), \
        lambda r: {"summary": f"{len((r or {}).get('rules', {}) if isinstance((r or {}).get('rules'), dict) else (r or {}).get('rules', []))} rules in the ledger. " + digest({k: v for k, v in (r or {}).items() if k != "rules"}, 300),
                   "out": {}, "open": {"type": "go", "view": "engine"}}


@skill("autopilot_start", "Engine", "Start an Autopilot campaign",
       "Autopilot profiles models, mines rules, tests the experiments that should teach the most, builds a blueprint and writes a report, within a time budget.",
       {"models": P("array", [], "models to profile first"), "max_experiments": P("integer", 4, "experiments"),
        "hours": P("number", 1.0, "time budget in hours"), "claude": P("boolean", True, "let Claude write the report")}, "long")
def _ap(a):
    return ("POST", "/api/autopilot/start", {"models": _specs(a["models"]), "max_experiments": a["max_experiments"], "hours": a["hours"], "claude": a["claude"]}), \
        lambda r: {"summary": f"Autopilot started ({r.get('status')}, {len(r.get('queue', []))} items queued). The board follows it.",
                   "out": {}, "open": {"type": "go", "view": "engine"}}


@skill("narrate", "Engine", "Write the world view", "Claude (or a digest without a key) writes up what the ledger supports, citing the evidence.", {}, "minutes")
def _narrate(a):
    return ("POST", "/api/engine/narrate", {"use_claude": True}), \
        lambda r: {"summary": digest(r, 700), "out": {}, "open": {"type": "go", "view": "engine"}}


# ------------------------------------------------------------------ supervision (other agents and workers)
@skill("autopilot_control", "Supervise", "Pause, resume or stop Autopilot", "Control the running campaign.",
       {"action": P("string", "pause", "pause, resume or stop", True, ["pause", "resume", "stop"])}, "instant")
def _apc(a):
    return ("POST", f"/api/autopilot/{a['action']}", {}), \
        lambda r: {"summary": f"Autopilot {a['action']}: now {r.get('status')}.", "out": {}, "open": {"type": "go", "view": "engine"}}


@skill("job_stop", "Supervise", "Stop a job", "Stop a queued or running job after its current step.",
       {"id": P("string", None, "job id", True)}, "instant")
def _jstop(a):
    return ("POST", f"/api/jobs/{a['id']}/stop", {}), lambda r: {"summary": f"Asked job {a['id']} to stop.", "out": {}, "open": None}


@skill("tune_close", "Supervise", "Close a Tune session", "Free the memory a training session holds (its log stays).",
       {"id": P("string", None, "session id", True)}, "instant")
def _tclose(a):
    return ("DELETE", f"/api/tune/v1/sessions/{a['id']}", None), lambda r: {"summary": f"Closed session {a['id']}.", "out": {}, "open": None}


def normalise(name, args):
    """Fill defaults, coerce types, check required and enums. Raises ValueError."""
    if name not in SKILLS:
        raise ValueError(f"Unknown skill {name!r}.")
    sk, args, out = SKILLS[name], dict(args or {}), {}
    for k, p in sk["params"].items():
        v = args.get(k, p["default"])
        if v is None and p["required"]:
            raise ValueError(f"{sk['label']}: {k} is required.")
        if v is not None:
            t = p["type"]
            try:
                if t == "integer":
                    v = int(v)
                elif t == "number":
                    v = float(v)
                elif t == "boolean":
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
                elif t == "array":
                    v = v if isinstance(v, list) else _specs(v)
                    if k == "seeds":
                        v = [int(x) for x in v]
                else:
                    v = str(v)
            except (TypeError, ValueError):
                raise ValueError(f"{sk['label']}: {k} should be a {t}.")
            if p.get("enum") and v not in p["enum"]:
                raise ValueError(f"{sk['label']}: {k} must be one of {', '.join(p['enum'])}.")
        out[k] = v
    extra = set(args) - set(sk["params"])
    if extra:
        raise ValueError(f"{sk['label']}: unknown argument(s) {', '.join(sorted(extra))}.")
    return out


def catalog(has_key=True):
    return [{k: v for k, v in s.items() if k != "build"} | {"params": {n: {kk: vv for kk, vv in p.items() if vv is not None}
                                                                      for n, p in s["params"].items()}}
            for s in SKILLS.values() if has_key or not s["needs_key"]]


def tool_schema(has_key=True):
    """The skills as JSON schema, for Claude."""
    out = []
    for s in SKILLS.values():
        if s["needs_key"] and not has_key:
            continue
        props = {}
        for n, p in s["params"].items():
            t = {"array": {"type": "array", "items": {"type": "string" if n != "seeds" else "integer"}}}.get(p["type"], {"type": p["type"]})
            if p.get("enum"):
                t["enum"] = p["enum"]
            t["description"] = p["desc"] + (f" (default {p['default']!r})" if p["default"] not in (None, "") else "")
            props[n] = t
        out.append({"name": s["name"], "area": s["area"], "description": f"{s['label']}. {s['desc']} Cost: {s['cost']}." + (" Downloads models it does not have." if s["downloads"] else ""),
                    "params": props, "required": [n for n, p in s["params"].items() if p["required"]]})
    return out
