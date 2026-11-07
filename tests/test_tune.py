"""Tune: the Tinker-style service, its client, the recipes, adapters as specs, and the anatomy.

Runs offline: Micro Opus, and a tiny GPT-2 built from a config (Conv1D weights).
The client test starts a real server on a free port and talks to it over HTTP.
"""
import json
import pathlib
import socket
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))
import torch

from server.hub.load import load_lm
from server.tune import lora
from server.tune.api import fill_config, is_local
from server.tune.service import Service
from server.tune.session import parse_datum

TMP = pathlib.Path(tempfile.mkdtemp())
lora.set_dir(TMP / "tune")


def svc_for(loader=load_lm):
    return Service(lambda s, p: s, loader)


def datum(ids, **inp):
    return {"model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
            "loss_fn_inputs": {"target_tokens": ids[1:], **inp}}


def test_primitives():
    svc = svc_for()
    s = svc.create("microopus:words", rank=4, seed=0)
    tok = s.tok
    ids = tok("the cat sat on the mat .")["input_ids"]
    d = datum(ids, weights=[1.0] * (len(ids) - 1))
    # the adapter starts at zero: the session model equals the base model
    base_lp = s.forward([d])["loss_fn_outputs"][0]["logprobs"]
    with torch.no_grad():
        lg = s.base(input_ids=torch.tensor([ids[:-1]])).logits.log_softmax(-1)[0]
    ref = [float(lg[i, t]) for i, t in enumerate(ids[1:])]
    assert max(abs(a - b) for a, b in zip(base_lp, ref)) < 1e-5
    # cross entropy is a sum: -sum(w * logprob)
    out = s.forward_backward([d], "cross_entropy")
    assert abs(out["metrics"]["loss:sum"] + sum(base_lp)) < 1e-4
    g_ce = [p.grad.clone() for p in s.params()]
    for p in s.params():
        p.grad = None
    # the linear loss with grads = -weights gives the same gradient (how custom losses come back)
    s.forward_backward([datum(ids, grads=[-1.0] * (len(ids) - 1))], "linear")
    g_lin = [p.grad.clone() for p in s.params()]
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_ce, g_lin))
    # importance sampling with sampler logprobs equal to the current ones is REINFORCE: -sum(adv * logprob) in gradient
    for p in s.params():
        p.grad = None
    s.forward_backward([datum(ids, logprobs=base_lp, advantages=[1.0] * len(base_lp))], "importance_sampling")
    g_is = [p.grad.clone() for p in s.params()]
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_ce, g_is))
    # ppo clipping: a large advantage with ratio far outside the clip has no gradient
    for p in s.params():
        p.grad = None
    far = [x - 2.0 for x in base_lp]                     # ratio = e^2, above 1.2
    s.forward_backward([datum(ids, logprobs=far, advantages=[1.0] * len(base_lp))], "ppo")
    assert all(float(p.grad.abs().max()) == 0.0 for p in s.params())
    # training lowers the loss
    for p in s.params():
        p.grad = None
    s._acc = {"loss": 0.0, "weight": 0.0, "tokens": 0, "calls": 0}
    first = None
    for _ in range(25):
        fb = s.forward_backward([d], "cross_entropy")
        first = first if first is not None else fb["metrics"]["loss:sum"]
        s.optim_step(learning_rate=1e-2)
    assert s.forward([d])["metrics"]["loss:sum"] < 0.5 * first
    assert len(s.log) == 25 and s.log[-1]["loss"] < s.log[0]["loss"]
    # samples come with the logprobs of the distribution sampled from
    smp = s.sample(tok("the")["input_ids"], num_samples=3, max_tokens=5, temperature=1.0, seed=3)
    assert len(smp) == 3 and all(len(x["tokens"]) == len(x["logprobs"]) <= 5 for x in smp)
    re_lp = s.forward([{"tokens": (tok("the")["input_ids"] + smp[0]["tokens"])[:-1],
                        "loss_fn_inputs": {"target_tokens": (tok("the")["input_ids"] + smp[0]["tokens"])[1:]}}])
    tail = re_lp["loss_fn_outputs"][0]["logprobs"][-len(smp[0]["tokens"]):]
    assert max(abs(a - b) for a, b in zip(tail, smp[0]["logprobs"])) < 1e-4
    # bad data is refused with a clear message
    try:
        parse_datum({"model_input": {"chunks": [{"tokens": [1, 2, 3]}]}, "loss_fn_inputs": {"target_tokens": [1, 2]}})
        raise AssertionError("mismatched lengths accepted")
    except ValueError as e:
        assert "one entry per input token" in str(e)
    return svc, s, d


def test_save_load_and_specs():
    svc, s, d = test_primitives()
    loss_now = s.forward([d])["metrics"]["loss:sum"]
    # state round trip: a new session from the state gives the same loss, and its optimizer continues
    st = svc.save_state(s.id, "t-state")
    s2 = svc.from_state(st["path"])
    assert abs(s2.forward([d])["metrics"]["loss:sum"] - loss_now) < 1e-5 and s2.step == s.step and s2.opt is not None
    # the adapter as a spec: the merged model equals the live session, in every tab's loader
    meta = s.save_adapter("t-adapter")
    assert meta["spec"] == "tune:t-adapter" and meta["trainable"] == s.trainable()
    tok, merged, causal = load_lm("tune:t-adapter")
    ids = d["model_input"]["chunks"][0]["tokens"]
    with torch.no_grad():
        a = merged(input_ids=torch.tensor([ids])).logits
        b = s.model(input_ids=torch.tensor([ids])).logits
    assert causal and torch.allclose(a, b, atol=1e-5)
    from server.hub.load import fetch_weights

    W = fetch_weights("tune:t-adapter")
    B = fetch_weights("microopus:words")
    assert set(W) == set(B)                                  # pairs name for name with its base
    assert any(not torch.allclose(W[k], B[k]) for k in W) and torch.allclose(W["W_E"], B["W_E"])
    # the base model itself was never touched
    with torch.no_grad():
        base_now = s.base(input_ids=torch.tensor([ids])).logits
        fresh = load_lm("microopus:words")[1](input_ids=torch.tensor([ids])).logits
    assert torch.allclose(base_now, fresh)
    # sampling a saved adapter: greedy output equals the live session's
    g1 = svc.sample({"model_path": "isomorph://adapters/t-adapter"}, tok("the cat")["input_ids"], 1, 6, 0.0)["sequences"][0]["tokens"]
    g2 = s.sample(tok("the cat")["input_ids"], 1, 6, 0.0)[0]["tokens"]
    assert g1 == g2
    return svc


def test_hf_conv1d():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=4)).eval()

    class Tok:
        eos_token_id = 63

        def __call__(self, t, **k):
            return {"input_ids": [1 + ord(c) % 60 for c in t]}

        def decode(self, ids):
            return "".join(chr(97 + i % 26) for i in ids)

    svc = svc_for(lambda spec: (Tok(), m, True))
    s = svc.create("tiny-gpt2", rank=2, train_attn=True, train_mlp=False)
    assert len(s.lora) == 4                                   # c_attn and attn.c_proj per block, MLP left alone
    try:
        svc.create("tiny-gpt2", rank=2, train_unembed=True)
        raise AssertionError("tied unembedding accepted")
    except ValueError as e:
        assert "ties" in str(e)
    ids = list(range(1, 20))
    d = datum(ids, weights=[1.0] * 18)
    l0 = s.forward([d])["metrics"]["loss:sum"]
    for _ in range(15):
        s.forward_backward([d])
        s.optim_step(learning_rate=5e-3)
    assert s.forward([d])["metrics"]["loss:sum"] < l0
    assert len(s.sample([1, 2, 3], 2, 4, 1.0, stop=[5], seed=0)) == 2


def test_recipes_anatomy_compare():
    from server.tune.anatomy import compare
    from server.tune.recipes import reinforce, supervised

    svc = svc_for()
    ex = "\n".join(f"the {a}cat => swam in the {w} ." if i % 2 else f"the {a}fish => flew over the {t} ."
                   for i, (a, w, t) in enumerate((["", "big ", "small ", "happy "][i % 4], ["pond", "lake"][i % 2], ["tree", "house"][i % 3 % 2])
                                                 for i in range(40)))
    r0 = supervised(svc, "microopus:words", ex, rank=8, lr=1e-3, steps=60, seed=0, name="rs0")
    r1 = supervised(svc, "microopus:words", ex, rank=8, lr=1e-3, steps=60, seed=1, name="rs1")
    assert r0["after"]["held_loss"] < 0.3 * r0["before"]["held_loss"] and r0["after"]["held_acc"] > r0["before"]["held_acc"]
    assert any(x["after"].strip().startswith(x["target"]) for x in r0["samples"])
    A = r0["anatomy"]
    assert A["summary"]["changed"] == len(A["matrices"]) and abs(sum(m["share"] for m in A["matrices"]) - 1) < 1e-6
    assert all(1 <= m["rank90"] <= m["rank"] for m in A["matrices"]) and A["summary"]["lines"]
    assert (lora.adapter_dir("rs0") / "anatomy.json").exists()
    c = compare(svc, "rs0", "rs1")
    assert c["overlap"] > 2 * c["random"], c["verdict"]        # two seeds of one task share directions
    rl = reinforce(svc, "microopus:words", "the\na\nthe big\nthe small\nthe happy", "contains", "cat", steps=30, seed=1, name="rr")
    assert rl["after"]["reward"] >= rl["before"]["reward"] and rl["drift_kl"] < 2.0, (rl["before"], rl["after"], rl["drift_kl"])
    json.dumps(r0), json.dumps(rl)
    assert all(s["closed"] for s in svc.listing()[:3])         # recipes free their sessions


def test_code_and_gate():
    src = (ROOT / "recipes" / "sl_basic.py").read_text()
    out = fill_config(src, {"spec": "x/y", "examples": 'a => b\nc """ d\n', "rank": 3})
    compile(out, "x", "exec")
    ns = {}
    exec(out.split("# --- end")[0].split("# --- settings ---")[1], ns)
    assert ns["SPEC"] == "x/y" and ns["RANK"] == 3 and ns["EXAMPLES"] == 'a => b\nc """ d\n', ns["EXAMPLES"]
    for f in (ROOT / "recipes").glob("*.py"):
        compile(f.read_text(), str(f), "exec")

    class R:
        def __init__(self, host, h):
            self.client = type("C", (), {"host": host})()
            self.headers = h
    assert is_local(R("127.0.0.1", {}))
    assert not is_local(R("127.0.0.1", {"cf-connecting-ip": "1.2.3.4"}))      # through the Cloudflare tunnel
    assert not is_local(R("10.0.0.5", {}))


def test_client_over_http():
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    import os

    os.environ["ISOMORPH_DATA"] = str(TMP / "data")
    from server.app import app

    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    import isotinker as tinker
    from isotinker import types

    sc = tinker.ServiceClient(base_url=f"http://127.0.0.1:{port}")
    tc = sc.create_lora_training_client(base_model="microopus:words", rank=4, name="http-test")
    tok = tc.get_tokenizer()
    ids = tok.encode("the dog ran to the park .")
    assert ids[0] == 0 and tok.encode("the", add_special_tokens=False)[0] != 0
    d = types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                    loss_fn_inputs={"target_tokens": ids[1:], "weights": torch.ones(len(ids) - 1)})
    losses = []
    for _ in range(15):
        fb = tc.forward_backward([d], "cross_entropy")        # issued back to back, run in order
        tc.optim_step(types.AdamParams(learning_rate=1e-2))
        losses.append(fb.result().metrics["loss:sum"])
    assert losses[-1] < 0.5 * losses[0]
    out = tc.forward_backward_custom([d], lambda data, lps: ((-(lps[0])).sum(), {"mine": 1.0})).result()
    assert out.metrics["mine"] == 1.0 and abs(out.metrics["loss:sum"] - tc.forward([d]).result().metrics["loss:sum"]) < 1e-3
    tc.log({"reward": 0.5})                                  # numbers for the step about to be taken
    tc.optim_step(types.AdamParams(learning_rate=1e-3)).result()
    sampler = tc.save_weights_and_get_sampling_client(name="http-test")
    res = sampler.sample(types.ModelInput.from_ints(tok.encode("the dog")), num_samples=2,
                         sampling_params=types.SamplingParams(max_tokens=5, temperature=0.0)).result()
    assert len(res.sequences) == 2 and res.sequences[0].tokens == res.sequences[1].tokens
    assert sampler.get_tokenizer().decode(res.sequences[0].tokens)
    lp = sampler.compute_logprobs(types.ModelInput.from_ints(ids)).result()
    assert lp[0] is None and len(lp) == len(ids)
    path = tc.save_state("http-state").result().path
    tc2 = sc.create_training_client_from_state_with_optimizer(path)
    assert abs(tc2.forward([d]).result().metrics["loss:sum"] - tc.forward([d]).result().metrics["loss:sum"]) < 1e-5
    info = tc.get_info()
    assert info["step"] == 16 and any("reward" in r for r in info["log"])
    try:
        tc.forward_backward([d], "nonsense").result()
        raise AssertionError("bad loss accepted")
    except tinker.IsomorphError as e:
        assert "Unknown loss" in str(e)
    srv.should_exit = True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
