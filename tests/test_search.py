"""Search: ranking, cached reads of large result files, and the endpoint's coverage."""
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ISOMORPH_DATA"] = tempfile.mkdtemp()

from server.search import head_fields, rank


def test_rank():
    items = [{"group": "Guide", "title": "Laws", "sub": "physics on weights"},
             {"group": "Results", "title": "Laws: microopus:words", "sub": ""},
             {"group": "Adapters", "title": "cats-1", "sub": "fine-tune of microopus:words"},
             {"group": "Plans", "title": "Teach with a reward", "sub": "Reward Micro Opus for mentioning cats"}]
    assert [r["title"] for r in rank(items, "laws micro")] == ["Laws: microopus:words"]   # every word must match
    assert rank(items, "cat")[0]["title"] == "cats-1"                                       # prefix on the last word; title beats body
    assert rank(items, "") == [] and rank(items, "zzz") == []


def test_head_fields_big_file():
    d = pathlib.Path(tempfile.mkdtemp())
    f = d / "t.json"
    f.write_text(json.dumps({"id": "x1", "a": "m:a", "b": "m:b", "prompt": 'say "hi"', "result": {"big": list(range(200000))}}))
    h = head_fields(f, ["id", "a", "b", "prompt"])
    assert h == {"id": "x1", "a": "m:a", "b": "m:b", "prompt": 'say "hi"'}


def test_endpoint():
    from server import app as appmod

    r = appmod.search(q="laws", limit=40)
    groups = {x["group"] for x in r}
    assert {"Guide", "Plans"} <= groups and all("action" in x for x in r)
    assert any(x["group"] == "Models" for x in appmod.search(q="microopus words"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
