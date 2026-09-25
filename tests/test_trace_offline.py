"""Offline test of storage inspection and prompt tracing with tiny random models."""
import sys, pathlib, tempfile, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, GPT2Config, GPT2LMHeadModel
from server.trace.storage import storage_report
from server.trace.trace import trace_report


class CharTok:
    def __init__(self, offset=0): self.offset = offset
    def __call__(self, t, return_tensors=None, truncation=True, max_length=128):
        return {"input_ids": torch.tensor([[(ord(c) + self.offset) % 250 for c in t][:max_length]])}
    def decode(self, ids): return "".join(chr(i) if 32 <= i < 127 else "·" for i in ids)


def neox(seed, layers=3):
    torch.manual_seed(seed)
    return GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=64, num_hidden_layers=layers, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=256, attn_implementation="eager")).eval()

def gpt2(seed):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(vocab_size=256, n_embd=48, n_layer=4, n_head=4, n_positions=256, attn_implementation="eager")).eval()

M = {"a": (CharTok(), neox(1), True), "b": (CharTok(), neox(2, layers=4), True), "c": (CharTok(3), gpt2(3), True)}
loader = lambda s: M[s]

def test_trace_same_tok():
    r = trace_report("a", "b", "Alice has three apples and gets two more.", loader=loader)
    assert r["same_tokens"] and len(r["divergence"]) == 4
    d = r["divergence"][2]
    assert "align_err" in d and "head_sim" in d and len(d["js"]) == len(r["A"]["tokens"])
    assert r["A"]["attention"] is not None and len(r["A"]["attention"]) == 3
    assert r["learning"]["a"]["loss_after"] < r["learning"]["a"]["loss_before"]
    print("summary", r["summary"], "learn", {k: r["learning"]["a"][k] for k in ("loss_before", "loss_after")})
    print("top tensor", r["learning"]["a"]["tensors"][0])
    json.dumps(r)

def test_trace_cross():
    r = trace_report("a", "c", "The capital of France is Paris.", loader=loader)
    assert not r["same_tokens"] and r["divergence"] == [] and r["B"]["layers"] == 4
    json.dumps(r)

def test_storage():
    d = pathlib.Path(tempfile.mkdtemp())
    M["a"][1].save_pretrained(d / "st")
    r = storage_report(str(d / "st"))
    assert r["format"] == "safetensors" and r["preview"]["hex"] and r["total_params"] > 0
    print("st", r["total_params"], r["by_kind"][:2], r["preview"]["values"][:3], r["preview"]["bits"])
    torch.save(M["a"][1].state_dict(), d / "pytorch_model.bin")
    (d / "bin").mkdir(); (d / "pytorch_model.bin").rename(d / "bin" / "pytorch_model.bin")
    r2 = storage_report(str(d / "bin"))
    assert r2["format"] == "pytorch-zip" and r2["zip_entries"]
    json.dumps(r); json.dumps(r2)

if __name__ == "__main__":
    for f in [test_trace_same_tok, test_trace_cross, test_storage]:
        f(); print("ok", f.__name__)
