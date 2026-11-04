"""Weight-space programming: the planner, the workspace tools, the read and fix tasks.

Runs offline: Micro Opus, a tiny lab model, and two small randomly initialised
Hugging Face architectures (GPT-2 and GPT-NeoX, built from a config, no
download). The agent is replaced by scripted tool calls, so the harness,
grading, controls and restores are tested; how well Claude itself does is
measured by running the tasks with a key.
"""
import json
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from server.hub.load import load_lm
from server.wsp import common
from server.wsp.agent import run_agent
from server.wsp.finetune import plan_report
from server.wsp.tasks import run_fix, run_read
from server.wsp.workspace import Workspace


def scripted(steps):
    """A stand-in for Claude: each step maps the last tool result to the next tool call."""
    st = {"i": 0, "last": None}

    def call(system, messages, tools):
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            c = last["content"][-1]["content"]
            st["last"] = json.loads(c) if c.startswith("{") and not c.endswith('(cut)"') else None
        i = st["i"]
        st["i"] += 1
        if i >= len(steps):
            return {"content": [{"type": "text", "text": "done"}], "usage": {}}
        tool, args = steps[i](st["last"])
        return {"content": [{"type": "tool_use", "id": f"t{i}", "name": tool, "input": args}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    return call


def cats_and_fish(n=40, seed=0):
    R = random.Random(seed)
    out = []
    for i in range(n):
        a = R.choice(["", "big ", "small ", "happy "])
        out.append(f"the {a}cat swam in the {R.choice(['pond', 'lake'])} ." if i % 2 else f"the {a}fish flew over the {R.choice(['tree', 'house'])} .")
    return "\n".join(out)


# ------------------------------------------------------------------ tiny Hugging Face models, built offline
class CharTok:
    def __init__(self, V):
        self.V, self.pad_token_id = V, 0

    def __len__(self):
        return self.V

    def __call__(self, text, return_tensors=None, truncation=True, max_length=64, **_):
        ids = [1 + (ord(c) % (self.V - 1)) for c in text][:max_length]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


def tiny_hf(kind):
    from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    if kind == "gpt2":
        m = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=4, attn_implementation="eager"))
    else:
        m = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=64, max_position_embeddings=128, hidden_size=32, num_hidden_layers=2,
                                             num_attention_heads=4, intermediate_size=64, attn_implementation="eager"))
    m.eval()
    tok = CharTok(64)
    return lambda spec: (tok, m, True)


# ------------------------------------------------------------------ planner
def test_planner_micro():
    r = plan_report("microopus:words", cats_and_fish(), steps=60, seeds=2)
    P = r["plans"]
    assert set(P) == {"gradient", "weights", "all", "last", "random"}
    budget = r["settings"]["budget"]
    for k, v in P.items():
        assert 0.85 * budget <= v["params"] <= budget, (k, v["params"], budget)   # same budget for every plan
        assert v["held"]["loss"] < r["base"]["held"]["loss"], k                     # every plan learned something
        assert v["drift"]["kl"] >= 0
    assert r["restored_diff"] == 0.0
    assert r["verdict"].count(".") >= 2 and ("gradient plan" in r["verdict"]) and ("weights-only plan" in r["verdict"])
    assert all(m["grad_rank90"] >= 1 and m["alpha"] is not None for m in r["matrices"])
    json.dumps(r)


def test_planner_hf_layouts():
    for kind in ("gpt2", "neox"):
        loader = tiny_hf(kind)
        tok, m, _ = loader("x")
        names = [c["name"] for c in common.tunable(m)]
        assert len(names) >= 8, (kind, names)                     # Conv1D (GPT-2) and fused QKV (NeoX) both found
        before = common.checksum(m)
        ex = "\n".join(f"abc{i % 7} => xyz{i % 5}" for i in range(24))
        r = plan_report(kind, ex, steps=8, seeds=1, loader=loader)
        assert r["restored_diff"] == 0.0 and common.checksum(m) == before
        assert all(v["params"] <= r["settings"]["budget"] for v in r["plans"].values())


# ------------------------------------------------------------------ workspace
def test_workspace_tools_and_restore():
    ws = Workspace("microopus:words", forward_budget=2, allow_edits=True)
    before = common.checksum(ws.model)
    assert ws.list_tensors()["count"] == len(list(ws.model.named_parameters()))
    st = ws.tensor_stats("model.layers.0.mlp.up_proj.weight")
    assert st["shape"] == [64, 16] and st["rank_for_90pct_energy"] >= 1 and "alpha" in st
    sv = ws.svd("model.layers.1.mlp.down_proj.weight", 2)
    d_rows = sv["components"][0]["rows_dir"]
    rd = ws.read_direction(d_rows)
    assert rd["length"] == 16 and len(rd["as_output_promotes"]) == 10
    pr = ws.project("model.layers.1.mlp.up_proj.weight", d_rows)
    assert "W_times_dir" in pr and 0 <= pr["W_times_dir"]["gain"] <= 1.0001
    assert ws.fft("model.embed_tokens.weight", "rows")["top_frequencies"]
    assert ws.compare("model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight")["cosine"] is not None
    assert ws.rows_cols("model.layers.0.mlp.down_proj.weight", "cols")["count"] == 64
    td = ws.token_direction("cat")
    assert td["token"].strip() == "cat"
    md = ws.mean_difference(["the cat sat on the mat ."], ["the fish swam in the pond ."], 1)
    assert md["forward_passes_left"] == 1
    ws.run(["the dog ran to the park ."])
    assert "error" in ws.call("run", {"texts": ["the cat"]})               # budget spent
    e1 = ws.edit_project_out("model.layers.1.mlp.down_proj.weight", d_rows)
    ws.edit_scale("model.layers.0.mlp.up_proj.weight", 0.0, "row", [3, 5])
    ws.edit_add_rank1("model.layers.0.mlp.down_proj.weight", ws.svd("model.layers.0.mlp.down_proj.weight", 1)["components"][0]["rows_dir"],
                      ws.svd("model.layers.0.mlp.down_proj.weight", 1)["components"][0]["cols_dir"], 0.5)
    assert e1["weights_changed"] == 16 * 64 and ws.changed() == 1024 + 3 * 16 + 1024
    assert common.checksum(ws.model) != before
    ws.undo("e2")                                                           # e2 and everything after it
    assert ws.changed() == 1024
    ws.restore()
    assert common.checksum(ws.model) == before
    hidden = Workspace("microopus:words", hide_io=True)
    assert not [n for n in hidden.visible() if "embed" in n or "lm_head" in n]
    assert "error" in hidden.call("tensor_stats", {"name": "lm_head.weight"})
    assert "token_direction" not in [t["name"] for t in hidden.tool_table()] and "run" not in [t["name"] for t in hidden.tool_table()]


def test_agent_loop():
    ws = Workspace("microopus:words")
    call = scripted([lambda r: ("list_tensors", {}), lambda r: ("nope", {}), lambda r: ("finish", {"answer": "x"})])
    ans, tr, usage = run_agent(ws, "task", {"answer": {"type": "string"}}, call)
    assert ans == {"answer": "x"} and [t["kind"] for t in tr] == ["tool", "tool", "finish"] and "error" in tr[1]["result"]
    ans, tr, usage = run_agent(ws, "task", {"answer": {"type": "string"}}, scripted([]))   # never calls finish: nudged once, then stops
    assert ans is None and usage["turns"] == 2


# ------------------------------------------------------------------ read and fix tasks
def test_read_data_quiz():
    ok = run_read("data", "microopus:words", scripted([lambda r: ("list_tensors", {}),
                                                       lambda r: ("finish", {"answer": "Simple English sentences, one word per token.", "evidence": "e", "confidence": 0.5})]))
    assert ok["grade"]["correct"] and ok["workspace"]["hidden_tensors"] >= 3
    bad = run_read("data", "microopus:words", scripted([lambda r: ("finish", {"answer": "Person names, letter by letter.", "evidence": "", "confidence": 0.5})]))
    assert not bad["grade"]["correct"]


def test_read_freqs_quiz():
    from server.lab.train import train_family

    lab = pathlib.Path(tempfile.mkdtemp())
    train_family(lab / "run1", dict(p=13, seeds=[1], steps=150, train_frac=0.5, d_model=32, n_heads=4, d_mlp=64, lr=1e-3, wd=1.0))
    steps = [lambda r: ("fft", {"name": "W_E", "axis": "rows", "first": 13, "top": 3}),
             lambda r: ("finish", {"p": 13, "frequencies": [f["k"] for f in r["top_frequencies"]], "evidence": "fft"})]
    res = run_read("freqs", "lab:run1:1", scripted(steps), lab_dir=lab)
    g = res["grade"]
    assert g["p_correct"] and 0 <= g["jaccard"] <= 1 and g["truth"]["frequencies"]


def test_read_induction_quiz_hf():
    loader = tiny_hf("gpt2")
    res = run_read("induction", "tiny", scripted([lambda r: ("attention", {"text": "abcabcabc", "layer": 1}),
                                                  lambda r: ("finish", {"heads": ["1.2", "L0H1"], "evidence": ""})]), loader=loader)
    g = res["grade"]
    assert set(g["given"]) == {"1.2", "0.1"} and res["forward_used"] == 1
    assert g["score"] is None and g["truth"] == []          # an untrained model has no induction heads: not graded


def test_fix_planted_micro():
    DP = "model.layers.1.mlp.down_proj.weight"
    tok, model, _ = load_lm("microopus:words")
    before = common.checksum(model)
    steps = [lambda r: ("score", {}),
             lambda r: ("svd", {"name": DP, "k": 1}),
             lambda r: ("edit_project_out", {"name": DP, "dir": r["components"][0]["rows_dir"]}),
             lambda r: ("score", {}),
             lambda r: ("finish", {"summary": "removed the dominant output direction of the last MLP", "confidence": 0.6})]
    res = run_fix("planted", "microopus:words", scripted(steps), forward_budget=4)
    g = res["grade"]
    assert res["kept"] and g["touched_planted_tensor"] and g["repaired_share"] > 0.3, g
    assert res["control"] and res["control"]["held"] > res["edited"]["held"]["loss"]     # random edits of the same size repair less
    assert common.checksum(model) == before                                               # fault and edit both gone
    # an edit that makes practice worse is not kept
    worse = run_fix("planted", "microopus:words", scripted([
        lambda r: ("edit_scale", {"name": "model.embed_tokens.weight", "factor": 0.0}),
        lambda r: ("finish", {"summary": "scaled", "confidence": 0.1})]), forward_budget=2)
    assert not worse["kept"] and worse["weights_changed"] == 0 and common.checksum(model) == before


def test_fix_custom_exports():
    out = pathlib.Path(tempfile.mkdtemp())
    DP = "model.layers.1.mlp.down_proj.weight"
    res = run_fix("custom", "microopus:words", scripted([
        lambda r: ("mean_difference", {"texts_a": ["the cat swam in the pond ."], "texts_b": ["the cat sat on the mat ."], "layer": 1}),
        lambda r: ("edit_add_rank1", {"name": DP, "rows_dir": r["dir"], "cols_dir": "d1", "amount": 0.0}),
        lambda r: ("svd", {"name": DP, "k": 1}),
        lambda r: ("edit_add_rank1", {"name": DP, "rows_dir": "d1", "cols_dir": r["components"][0]["cols_dir"], "amount": 2.0}),
        lambda r: ("score", {}),
        lambda r: ("finish", {"summary": "pushed the swim direction", "confidence": 0.3})]),
        goal="cats swim and fish fly", examples=cats_and_fish(), forward_budget=4, export_dir=out)
    assert "control" in res and res["grade"]["how"]
    if res["kept"]:
        assert (out / "edited_tensors.safetensors").exists()
    json.dumps(res, default=str)


def test_import_from_another_app():
    from safetensors.torch import save
    from server.wsp import imports

    imports.set_dir(pathlib.Path(tempfile.mkdtemp()))
    torch.manual_seed(0)
    body = save({"fc1.weight": torch.randn(32, 8), "fc1.bias": torch.zeros(32), "fc2.weight": torch.randn(3, 32)})
    info = imports.save_import(body, "Iris MLP", "Input [4] -> Dense 32 -> ReLU -> Dense 3", "Deep Network Designer")
    assert info["spec"].startswith("import:") and info["tensors"] == 3 and info["weights"] == 32 * 8 + 32 + 96
    ws = Workspace(info["spec"])
    assert ws.model is None and ws.describe()["about"].startswith("Input") and not ws.allow_edits
    assert ws.tensor_stats("fc1.weight")["shape"] == [32, 8]
    res = run_read("free", info["spec"], scripted([lambda r: ("svd", {"name": "fc2.weight", "k": 2}),
                                                   lambda r: ("finish", {"answer": "a", "evidence": "b", "confidence": 0.2})]),
                   question="What does the second layer do?")
    assert "Dense 32" in res["question"] and res["grade"]["score"] is None
    try:
        imports.save_import(b"not a safetensors file")
        raise AssertionError("bad bytes accepted")
    except Exception as e:
        assert "AssertionError" not in type(e).__name__


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
