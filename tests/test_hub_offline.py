"""Offline test of the hub analyses using tiny random GPT-NeoX / GPT-2 models."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, GPT2Config, GPT2LMHeadModel
from server.hub.acts import activation_report
from server.hub.weights import weight_report
from server.hub.load import align_names


class CharTok:
    def __init__(self, offset=0): self.offset = offset
    def __call__(self, t, return_tensors=None, truncation=True, max_length=128):
        ids = [(ord(c) + self.offset) % 250 for c in t][:max_length]
        return {"input_ids": torch.tensor([ids])}


def neox(seed):
    torch.manual_seed(seed)
    return GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=64, num_hidden_layers=3,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=256)).eval()

def gpt2(seed):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(vocab_size=256, n_embd=48, n_layer=4, n_head=4, n_positions=256)).eval()

MODELS = {"a": (CharTok(), neox(1), True), "b": (CharTok(), neox(2), True), "c": (CharTok(3), gpt2(3), True)}
loader = lambda s: MODELS[s]

def test_same_family():
    r = activation_report("a", "b", max_tokens=600, loader=loader, progress=lambda f, m: None)
    assert r["mode"] == "token" and r["neurons"] and r["stitch"] and r["agreement"]
    print("neurons", r["neurons"][0]); print("stitch", r["stitch"]); print("agree", r["agreement"])

def test_self_is_identical():
    r = activation_report("a", "a", max_tokens=400, loader=loader, progress=lambda f, m: None)
    assert all(abs(d["cka"] - 1) < 1e-3 for d in r["diag"]), r["diag"]
    assert r["neurons"][0]["matched_mean"] > 0.99
    s = r["stitch"][0]; assert abs(s["affine"] - s["own"]) < 0.05, s

def test_cross_family():
    r = activation_report("a", "c", loader=loader, progress=lambda f, m: None)
    assert r["mode"] == "sentence" and len(r["cka"]) == 4 and len(r["cka"][0]) == 5
    print("cross notes", r["notes"][0][:60])

def test_weights():
    wa = {k: v.float() for k, v in MODELS["a"][1].state_dict().items() if v.is_floating_point()}
    wb = {k: v.float() for k, v in MODELS["b"][1].state_dict().items() if v.is_floating_point()}
    r = weight_report(wa, wb, align_names(wa, wb))
    assert r["summary"]["relation"] == "independent", r["summary"]
    # fine-tune: low-rank perturbation should be detected as compact
    wc = {k: v.clone() for k, v in wa.items()}
    for k, v in wc.items():
        if v.ndim == 2 and min(v.shape) >= 8:
            v += 0.003 * torch.randn(v.shape[0], 2) @ torch.randn(2, v.shape[1])
    r2 = weight_report(wa, wc, align_names(wa, wc))
    print(r2["summary"])
    assert r2["summary"]["relation"] == "shared-init" and r2["summary"]["median_delta_rank_frac"] < 0.1

if __name__ == "__main__":
    for f in [test_same_family, test_self_is_identical, test_cross_family, test_weights]:
        f(); print("ok", f.__name__)
