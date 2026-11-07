"""Search over everything Isomorph has: the manual, models, saved results, adapters,
missions and ready-made plans. Items are gathered by the app (it knows the folders);
this module ranks them, and caches what it reads from result files by modification
time so typing stays fast even with large traces on disk."""
import json
import re
from pathlib import Path

_cache = {}
WORD = re.compile(r"[a-z0-9]+")
GROUP_ORDER = ["Models", "Results", "Adapters", "Missions", "Plans", "Guide"]


def head_fields(path: Path, keys, nbytes=4096):
    """Top-level string fields from the start of a (possibly huge) JSON file, cached by mtime."""
    try:
        st = path.stat()
    except OSError:
        return None
    k = (str(path), st.st_mtime, tuple(keys))
    if k in _cache:
        return _cache[k]
    try:
        with open(path, "rb") as f:
            head = f.read(nbytes).decode("utf-8", "ignore")
    except OSError:
        return None
    out = {}
    for key in keys:
        m = re.search(r'"%s"\s*:\s*("(?:[^"\\]|\\.)*"|-?[0-9.eE+]+)' % re.escape(key), head)
        if m:
            try:
                out[key] = json.loads(m.group(1))
            except Exception:
                pass
    _cache[k] = out
    if len(_cache) > 5000:
        _cache.clear()
    return out


def cached_json(path: Path, fn):
    """fn(parsed json) for a small JSON file, cached by mtime."""
    try:
        st = path.stat()
    except OSError:
        return None
    k = (str(path), st.st_mtime, fn.__name__)
    if k not in _cache:
        try:
            _cache[k] = fn(json.loads(path.read_text()))
        except Exception:
            _cache[k] = None
    return _cache[k]


def _words(s):
    return WORD.findall((s or "").lower())


def rank(items, q, limit=30):
    toks = _words(q)
    if not toks:
        return []
    out = []
    for it in items:
        title = " ".join(_words(it.get("title")))
        rest = " ".join(_words(f"{it.get('sub', '')} {it.get('text', '')}"))
        tw, rw = title.split(), rest.split()
        score, ok = 0.0, True
        for i, t in enumerate(toks):
            last = i == len(toks) - 1
            hit_t = any(w == t or (last and w.startswith(t)) or (len(t) > 3 and w.startswith(t)) for w in tw)
            hit_r = hit_t or any(w == t or (last and w.startswith(t)) or (len(t) > 3 and w.startswith(t)) for w in rw)
            if not hit_r:
                ok = False
                break
            score += 3 if hit_t else 1
        if not ok:
            continue
        phrase = " ".join(toks)
        if phrase in title:
            score += 4
        if title.startswith(phrase):
            score += 2
        score += it.get("boost", 0)
        out.append((score, it))
    out.sort(key=lambda x: (-x[0], GROUP_ORDER.index(x[1]["group"]) if x[1]["group"] in GROUP_ORDER else 99))
    res = [it for _, it in out[:limit]]
    for it in res:
        it.pop("text", None)
    return res
