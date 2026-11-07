"""Ready-made missions: plans that need no language model. {m} becomes the mission's
short id, so names (adapters) never collide; $N.key takes a value from step N's
result; lab:latest and run_id latest mean the newest ground-truth run."""

PLAYBOOKS = {
    "first-look": {
        "title": "A first look at everything, with no download", "views": ["trace", "about", "laws", "wsp"],
        "why": "Two transparent models traced side by side, the laws on both, and a fine-tune plan: a tour of the bench by doing.",
        "steps": [("trace", {"a": "microopus:words", "b": "microopus:names", "prompt": "the cat sat on the mat ."}, "See where two different models agree and part ways."),
                  ("laws", {"specs": ["microopus:words", "microopus:names"]}, "Which physics and maths laws their weights obey."),
                  ("wsp_plan", {"spec": "microopus:words", "examples": "demo", "steps": 100, "seeds": 2}, "Can the weights decide how to fine-tune?")]},
    "trace-pair": {
        "title": "Trace Micro Opus words against names", "views": ["trace"], "check": "trace",
        "why": "Every Side-by-side view, on two models that need no download.",
        "steps": [("trace", {"a": "microopus:words", "b": "microopus:names", "prompt": "the cat sat on the mat ."}, "One prompt through both models.")]},
    "seeds-agree": {
        "title": "Do two fine-tunes find the same directions?", "views": ["tune"],
        "why": "Teach Micro Opus one rule twice with different seeds, compare the two changes, and check what the fine-tune did to the laws.",
        "steps": [("tune_sl", {"spec": "microopus:words", "examples": "demo", "seed": 0, "name": "cats-{m}-s0"}, "First fine-tune."),
                  ("tune_sl", {"spec": "microopus:words", "examples": "demo", "seed": 1, "name": "cats-{m}-s1"}, "Same task, another seed."),
                  ("tune_compare", {"a": "$1.adapter", "b": "$2.adapter"}, "Same directions, or different ones?"),
                  ("laws", {"specs": ["microopus:words", "$1.spec"]}, "Did fine-tuning change which laws hold?"),
                  ("trace", {"a": "microopus:words", "b": "$1.spec", "prompt": "the cat"}, "Where the fine-tune parts ways with its base.")]},
    "reward-tune": {
        "title": "Teach with a reward, then check the damage", "views": ["tune"],
        "why": "Reward Micro Opus for mentioning cats, with and without staying near the base, and compare the two.",
        "steps": [("tune_rl", {"spec": "microopus:words", "prompts": "demo", "reward": "contains", "reward_arg": "cat", "kl_coef": 0.05, "name": "cat-kl-{m}"}, "With the KL penalty."),
                  ("tune_rl", {"spec": "microopus:words", "prompts": "demo", "reward": "contains", "reward_arg": "cat", "kl_coef": 0.0, "name": "cat-free-{m}"}, "Without it: watch the drift."),
                  ("tune_compare", {"a": "$1.adapter", "b": "$2.adapter"}, "Did the penalty change where the update went?")]},
    "plan": {
        "title": "Plan a fine-tune from the weights", "views": ["wsp"], "check": "plan",
        "why": "Five ways to spend the same budget, scored on held-out examples.",
        "steps": [("wsp_plan", {"spec": "microopus:words", "examples": "demo"}, "The example task: cats swim, fish fly.")]},
    "plan-then-tune": {
        "title": "Plan from the weights, then tune for real", "views": ["wsp", "tune"],
        "why": "The planner's verdict, then a full LoRA fine-tune on the same task, read as a weight change.",
        "steps": [("wsp_plan", {"spec": "microopus:words", "examples": "demo"}, "Which matrices the weights and the gradient pick."),
                  ("tune_sl", {"spec": "microopus:words", "examples": "demo", "name": "planned-{m}"}, "A LoRA on every matrix, for comparison."),
                  ("tune_anatomy", {"name": "$2.adapter"}, "Where the change actually went.")]},
    "ground-truth": {
        "title": "Train a ground-truth family and test it", "views": ["lab", "laws"], "check": "lab",
        "why": "Tiny models whose algorithm is known: compare the seeds, then run the laws on them.",
        "steps": [("lab_train", {"p": 53, "seeds": [1, 2, 3], "steps": 3000}, "Three seeds on (a + b) mod 53."),
                  ("lab_family", {"run_id": "$1.run_id"}, "How alike the seeds are, after symmetries."),
                  ("laws", {"specs": ["lab:$1.run_id:1", "lab:$1.run_id:2", "lab:$1.run_id:3"]}, "The laws on models with a known answer.")]},
    "laws-basic": {
        "title": "Run the laws on everything that needs no download", "views": ["laws"], "check": "laws",
        "why": "Micro Opus words and names, and the newest lab models if there are any.",
        "steps": [("laws", {"specs": ["microopus:words", "microopus:names", "lab:latest:1", "lab:latest:2"]}, "Lab models are skipped when there are none."),
                  ("laws_summary", {}, "Which laws hold where.")]},
    "laws-control": {
        "title": "Laws with a random control (downloads Pythia 14M)", "views": ["laws"], "check": "control",
        "why": "An emergent law should hold in a trained model and fail in the same model at initialisation.",
        "steps": [("laws", {"specs": ["EleutherAI/pythia-14m", "EleutherAI/pythia-14m@step0"]}, "Trained against untrained, same architecture."),
                  ("laws_summary", {}, "Which laws hold where.")]},
    "claude-reads": {
        "title": "Claude reads, fixes and writes weights", "views": ["wsp"], "key": True, "check": "read",
        "why": "The three graded Weight-space tasks in a row.",
        "steps": [("wsp_read", {"task": "data", "spec": "microopus:words"}, "What was this model trained on, from its weights alone?"),
                  ("wsp_fix", {"task": "planted", "spec": "microopus:words"}, "Find and repair a planted fault."),
                  ("wsp_write", {"pattern": "disc", "mode": "write", "budget": 0}, "Write a network from the rule alone.")]},
    "claude-fix": {
        "title": "Have Claude repair a planted fault", "views": ["wsp"], "key": True, "check": "fix",
        "why": "A known fault, a measured repair, a random-direction control.",
        "steps": [("wsp_fix", {"task": "planted", "spec": "microopus:words"}, "Find and repair it.")]},
    "claude-write": {
        "title": "Have Claude write a network from a rule", "views": ["wsp"], "key": True, "check": "write-claude",
        "why": "The paper's one-shot write test, against a trained network.",
        "steps": [("wsp_write", {"pattern": "disc", "mode": "write", "budget": 0}, "Disc, one shot.")]},
    "library": {
        "title": "Profile two real models (downloads Pythia 14M and 70M)", "views": ["atlas", "engine"], "check": "library",
        "why": "The library the Engine mines rules from.",
        "steps": [("profile", {"spec": "EleutherAI/pythia-14m"}, "Small and fast."),
                  ("profile", {"spec": "EleutherAI/pythia-70m"}, "Its larger sibling."),
                  ("mine", {}, "Rules from what was measured.")]},
    "mine": {
        "title": "Mine rules from everything measured", "views": ["engine"], "check": "mine",
        "why": "Rules with support and counterexamples.",
        "steps": [("mine", {}, "Mine the library, the lab and the laws.")]},
    "research-loop": {
        "title": "Run the research loop for an hour", "views": ["engine"], "check": "autopilot",
        "why": "Mine, then let Autopilot test the most informative rules and write a report. The board follows it.",
        "steps": [("mine", {}, "Rules first."),
                  ("autopilot_start", {"max_experiments": 4, "hours": 1.0}, "Autopilot takes over; this mission ends when it starts.")]},
}


def for_view(view, has_key):
    out = []
    for pid, p in PLAYBOOKS.items():
        if p.get("key") and not has_key:
            continue
        out.append({"id": pid, "title": p["title"], "why": p["why"], "here": view in p["views"], "key": bool(p.get("key")),
                    "steps": [{"skill": s, "why": w} for s, _, w in p["steps"]]})
    out.sort(key=lambda x: (not x["here"],))
    return out


def by_check(step_id):
    return next((pid for pid, p in PLAYBOOKS.items() if p.get("check") == step_id), None)


def match(goal, has_key):
    """Without a key: the playbooks whose words best match the goal."""
    import re

    stop = {"the", "a", "an", "to", "of", "and", "is", "what", "how", "do", "i", "it", "in", "this", "on", "for", "my", "me", "can", "you", "with", "please"}
    q = set(re.findall(r"[a-z0-9]+", (goal or "").lower())) - stop
    syn = {"finetune": "tune", "fine": "tune", "lora": "tune", "tinker": "tune", "reward": "reward", "rl": "reward", "law": "laws",
           "physics": "laws", "random": "control", "seeds": "seed", "lab": "ground", "modular": "ground", "claude": "claude",
           "read": "reads", "fix": "fix", "repair": "fix", "write": "write", "library": "profile", "profile": "profile",
           "autopilot": "research", "rules": "mine", "trace": "trace", "compare": "agree", "break": "damage", "breaks": "damage",
           "hurt": "damage", "drift": "damage", "forget": "damage", "same": "agree", "agree": "agree", "twice": "agree"}
    q |= {syn[w] for w in q if w in syn}
    import math

    docs = {}
    for pid, p in PLAYBOOKS.items():
        if p.get("key") and not has_key:
            continue
        title = set(re.findall(r"[a-z0-9]+", (pid.replace("-", " ") + " " + p["title"]).lower()))
        body = set(re.findall(r"[a-z0-9]+", p["why"].lower()))
        docs[pid] = (title, body | title)
    n = len(docs) or 1
    df = {}
    for t, b in docs.values():
        for w in b:
            df[w] = df.get(w, 0) + 1
    scored = []
    for pid, (title, body) in docs.items():
        sc = sum(math.log(1 + n / df[w]) * (2 if w in title else 1) for w in q & body)
        scored.append((sc, pid))
    scored.sort(key=lambda x: -x[0])
    return [pid for sc, pid in scored if sc > 0][:3]
