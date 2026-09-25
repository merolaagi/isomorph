"""Narrator: writes the ledger up as a world view.

With an Anthropic API key, Claude writes it, constrained to cite rule,
experiment and blueprint ids for every claim; citations are then checked
against the ledger and any unknown id is flagged. Without a key, a built-in
digest assembles the same sections directly from the ledger.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "claude-sonnet-5"
CITE = re.compile(r"\[((?:R|E|B)-[0-9A-Za-z]+)\]")


def get_key(settings_path):
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        return json.loads(settings_path.read_text()).get("anthropic_key")
    except Exception:
        return None


def set_key(settings_path, key):
    s = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if key:
        s["anthropic_key"] = key.strip()
    else:
        s.pop("anthropic_key", None)
    settings_path.write_text(json.dumps(s))
    os.chmod(settings_path, 0o600)


def compact(L):
    rules = [{"id": r["id"], "kind": r["kind"], "statement": r["statement"], "status": r.get("status"),
              "confidence": r.get("confidence"), "models_supporting": len(r.get("support", [])), "counterexamples": len(r.get("counter", [])),
              "effect": r.get("effect"), "unit": r.get("unit"), "note": r.get("about")} for r in L["rules"].values()]
    exps = [{"id": e["id"], "task": e["task_label"], "lever": e["lever_label"], "verdict": e["verdict"], "reason": e["reason"],
             "seeds": len(e["seeds"])} for e in L["experiments"].values()]
    bps = [{"id": b["id"], "levers": b["levers"], "results": [{k: r.get(k) for k in ("task_label", "metric", "params_base", "params_bp", "final_base", "final_bp", "capability_per_compute", "steps_base", "steps_bp")} for r in b["results"]]}
           for b in L["blueprints"].values()]
    return {"models_profiled": L.get("models", 0), "rules": rules, "experiments": exps, "blueprints": bps}


PROMPT = """You are the narrator of a research engine that studies what small neural networks learn and how to build much more efficient ones. The ultimate aim is a model that is small yet capable, the way a brain does a great deal on about 20 watts.

Below is the engine's ledger: rules mined from measurements (each with support, counterexamples and status), experiments that tested efficiency levers, and blueprints that combined them.

Write a world view in Markdown with exactly these sections:
## What seems to be true
## Where the rules break
## How a small, efficient next-generation model should work
## What to test next

Requirements:
- Every factual sentence must cite at least one id from the ledger in square brackets, like [R-1a2b3c] or [E-12345678] or [B-12345678]. Never invent ids.
- Distinguish clearly between rules that were tested and hold, rules only observed, and conjecture. Start any sentence that goes beyond the evidence with "Conjecture:".
- Weigh evidence honestly: say when a rule rests on few models, on tiny tasks, or on a single seed.
- In the design section, propose concrete architectural choices tied to the levers and rules, with the expected saving where measured.
- In the last section, give 3 to 6 specific experiments, each naming the rule it would confirm or reject.
- Be concise: at most about 700 words. Plain, direct sentences.

LEDGER:
"""


def write_with_claude(L, key, model=DEFAULT_MODEL):
    body = {"model": model, "max_tokens": 3000, "messages": [{"role": "user", "content": PROMPT + json.dumps(compact(L), indent=1)}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:400]
        raise RuntimeError(f"Anthropic API error {e.code}: {detail}")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def digest(L):
    """A world view assembled directly from the ledger, no language model involved."""
    R = list(L["rules"].values())
    line = lambda r: f"- {r['statement']} [{r['id']}] ({r.get('confidence')}, {len(r.get('support', []))} supporting, {len(r.get('counter', []))} against{', status: ' + r['status'] if r.get('status') not in (None, 'observed') else ''})."
    held = [r for r in R if r.get("status") == "holds"]
    observed = [r for r in R if r.get("status") in ("observed", None) and r["kind"] not in ("absence",) and len(r.get("support", [])) > len(r.get("counter", []))]
    broken = [r for r in R if r.get("counter") or r.get("status") in ("rejected", "mixed") or r["kind"] == "absence"]
    out = ["## What seems to be true", ""]
    out += [line(r) for r in held] or ["- No rule has been confirmed by an experiment yet."]
    out += [line(r) for r in observed][:12]
    out += ["", "## Where the rules break", ""]
    out += [line(r) for r in broken][:12] or ["- No counterexamples recorded yet. Profile more, and more varied, models."]
    out += ["", "## How a small, efficient next-generation model should work", ""]
    levers = [r for r in held if r["kind"] == "lever"]
    if levers:
        out += [f"- Use the lever behind [{r['id']}]: {r['about']}" for r in levers]
    else:
        out += ["- Conjecture: until levers are tested, no design recommendation is supported by this ledger."]
    for b in L["blueprints"].values():
        for x in b["results"]:
            cpc = x.get("capability_per_compute")
            out.append(f"- Blueprint [{b['id']}] on {x['task_label']}: {x['params_bp']:,} vs {x['params_base']:,} parameters; "
                       + (f"{cpc:.2f}× capability per unit of compute." if cpc else "no measurable gain in capability per unit of compute."))
    out += ["", "## What to test next", ""]
    todo = [r for r in R if r.get("lever") and not r.get("tests")]
    out += [f"- Test [{r['id']}] with the {r['lever'].get('lever', 'seeding')} lever." for r in todo[:6]] or ["- Profile more models to generate new testable rules."]
    return "\n".join(out)


def narrate(L, key=None, model=DEFAULT_MODEL):
    if key:
        text = write_with_claude(L, key, model)
        source = model
    else:
        text = digest(L)
        source = "built-in digest"
    known = set(L["rules"]) | set(L["experiments"]) | set(L["blueprints"])
    cited = CITE.findall(text)
    unknown = sorted({c for c in cited if c not in known})
    return {"id": "N-" + str(int(time.time())), "created": time.time(), "source": source, "text": text,
            "citations": len(cited), "unknown_citations": unknown, "rules": len(L["rules"]), "experiments": len(L["experiments"])}
