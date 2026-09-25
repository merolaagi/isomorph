"""Fixed probe sets for the pattern library.

Every profile uses exactly these inputs, so profiles made at different times,
of different models, stay comparable. Bump ANCHOR_VERSION if they change.
"""
import itertools
import random

from ..hub.corpus import CORPUS

ANCHOR_VERSION = 1

_OBJECTS = ["apples", "books", "chairs", "coins", "birds", "cars", "stones", "letters", "boxes", "trees"]
_PLACES = ["kitchen", "garden", "office", "station", "market", "library", "river", "school", "harbor", "forest"]
_PEOPLE = ["Maya", "Ravi", "Elena", "Tom", "Aiko", "Omar", "Lucy", "Diego", "Nina", "Sam"]
_CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
             ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Nepal", "Kathmandu"), ("Spain", "Madrid"), ("India", "New Delhi")]
_CODE = ["x = x + 1", "return total / count", "if n < 0:\n    raise ValueError(n)", "for item in items:\n    print(item)",
         "def square(n):\n    return n * n", "SELECT * FROM users WHERE id = 7;", "const y = arr.map(v => v * 2);",
         "while not done:\n    step()", "import math\nprint(math.pi)", "git push origin main"]
_FEEL = [("happy", "The children were happy when the holiday began."), ("sad", "He felt sad watching the old house being torn down."),
         ("angry", "She was angry that the bus left without her."), ("afraid", "The dog was afraid of the loud thunder."),
         ("calm", "The lake was calm and quiet at dawn.")]


def anchor_texts():
    """About 300 short, varied texts: prose, numbers, facts, code, questions."""
    rng = random.Random(1234)
    out = list(CORPUS)
    for n, obj, place in zip(rng.sample(range(2, 99), 40), itertools.cycle(_OBJECTS), itertools.cycle(_PLACES)):
        out.append(f"There are {n} {obj} in the {place}.")
    for a, b in rng.sample([(a, b) for a in range(2, 20) for b in range(2, 20)], 30):
        out.append(f"{a} plus {b} equals {a + b}.")
    for c, cap in _CAPITALS:
        out.append(f"The capital of {c} is {cap}.")
        out.append(f"{cap} is a city in {c}.")
    for p, place in zip(_PEOPLE, _PLACES):
        out.append(f"{p} walked to the {place} after lunch.")
        out.append(f"Did {p} leave the {place} early today?")
        out.append(f"{p} said that the {place} was closed on Sunday.")
    for c in _CODE:
        out.append(c)
    for _, s in _FEEL:
        out.append(s)
    for d, m in zip(rng.sample(range(1, 29), 15), itertools.cycle(["January", "March", "May", "July", "October"])):
        out.append(f"The meeting was moved to {m} {d}.")
    for o, p in zip(_OBJECTS, _PEOPLE):
        out.append(f"Please give the {o} to {p}.")
        out.append(f"Why are the {o} so expensive this year?")
    for w in ["rain", "snow", "sunshine", "fog", "wind"]:
        out.append(f"The forecast calls for {w} tomorrow morning.")
    seen, res = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res


# ------------------------------------------------------------------ concept probes
# Each concept: list of (text, label). Label is a class (0/1) or a number for
# regression. The representation used is the state at the last token.
_NOUNS = [("cat", "cats"), ("dog", "dogs"), ("house", "houses"), ("car", "cars"), ("tree", "trees"), ("book", "books"),
          ("bird", "birds"), ("river", "rivers"), ("chair", "chairs"), ("road", "roads"), ("window", "windows"),
          ("friend", "friends"), ("city", "cities"), ("box", "boxes"), ("child", "children"), ("mouse", "mice"),
          ("apple", "apples"), ("girl", "girls"), ("boy", "boys"), ("teacher", "teachers"), ("lamp", "lamps"),
          ("door", "doors"), ("plate", "plates"), ("song", "songs"), ("ship", "ships")]
_VERBS = [("walks", "walked"), ("plays", "played"), ("opens", "opened"), ("calls", "called"), ("cooks", "cooked"),
          ("paints", "painted"), ("writes", "wrote"), ("runs", "ran"), ("sings", "sang"), ("drives", "drove"),
          ("eats", "ate"), ("builds", "built"), ("reads", "read"), ("sees", "saw"), ("takes", "took"),
          ("gives", "gave"), ("finds", "found"), ("makes", "made"), ("sleeps", "slept"), ("swims", "swam")]
_ADJ = [("open", "closed"), ("full", "empty"), ("ready", "late"), ("safe", "broken"), ("clean", "dirty"),
        ("warm", "cold"), ("quiet", "loud"), ("new", "old")]
_THINGS = ["door", "shop", "glass", "room", "road", "phone", "train", "window", "bottle", "gate"]
_POS = ["wonderful", "excellent", "delightful", "brilliant", "lovely", "fantastic", "superb", "charming", "great", "pleasant"]
_NEG = ["terrible", "awful", "dreadful", "horrible", "boring", "disappointing", "poor", "unpleasant", "bad", "miserable"]
_REVIEW = ["The movie was", "The meal tasted", "The concert was", "Our trip was", "The book was", "The service was"]


def concepts():
    rng = random.Random(7)
    subj = ["She", "He", "My sister", "The man", "Our neighbour", "The student"]
    data = {}
    data["plural"] = {"kind": "class", "about": "Is the noun singular or plural?",
                      "items": [(f"I saw the {s}", 0) for s, _ in _NOUNS] + [(f"I saw the {p}", 1) for _, p in _NOUNS]}
    data["tense"] = {"kind": "class", "about": "Present or past tense?",
                     "items": [(f"{rng.choice(subj)} {pres}", 0) for pres, _ in _VERBS] + [(f"{rng.choice(subj)} {past}", 1) for _, past in _VERBS] +
                              [(f"Every day {rng.choice(subj).lower()} {pres}", 0) for pres, _ in _VERBS[:10]] +
                              [(f"Last week {rng.choice(subj).lower()} {past}", 1) for _, past in _VERBS[:10]]}
    neg = []
    for t in _THINGS:
        for a, _ in _ADJ[:4]:
            neg.append((f"The {t} is {a}", 0))
            neg.append((f"The {t} is not {a}", 1))
    data["negation"] = {"kind": "class", "about": "Is the statement negated?", "items": neg}
    sent = []
    for r in _REVIEW:
        for w in _POS:
            sent.append((f"{r} {w}", 1))
        for w in _NEG:
            sent.append((f"{r} {w}", 0))
    data["sentiment"] = {"kind": "class", "about": "Positive or negative?", "items": sent}
    nums = sorted(rng.sample(range(1, 1000), 90))
    data["magnitude"] = {"kind": "reg", "about": "How large is the number? (log scale)",
                         "items": [(f"The number of {rng.choice(_OBJECTS)} was {n}", n) for n in nums]}
    return data
