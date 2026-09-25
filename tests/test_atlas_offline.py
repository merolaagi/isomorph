"""Offline test of the pattern library and pattern-seeded training."""
import sys, pathlib, tempfile, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast


def tiny_model(path, seed, layers=2):
    torch.manual_seed(seed)
    m = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=32, num_hidden_layers=layers, num_attention_heads=4,
                                         intermediate_size=64, max_position_embeddings=256, attn_implementation="eager"))
    m.save_pretrained(path)
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="!").save_pretrained(path)


def test_library():
    from server.atlas.profile import profile_model
    from server.atlas.library import library_report, model_report
    d = pathlib.Path(tempfile.mkdtemp())
    for i, L in ((1, 2), (2, 3)):
        tiny_model(d / f"m{i}", i, L)
        profile_model(str(d / f"m{i}"), str(d / f"m{i}"), d / "atlas")
    lib = library_report(d / "atlas")
    assert len(lib["models"]) == 2 and len(lib["similarity"]) == 2 and lib["consensus"][0]["agreement"][0] is not None
    mr = model_report(d / "atlas", lib["models"][0]["id"])
    assert mr["patterns"] and mr["nearest"]
    json.dumps(lib); json.dumps(mr)
    print("library ok", round(lib["similarity"][0][1], 3), len(mr["patterns"]), "patterns")


def test_seeded():
    from server.lab.train import train_family
    from server.lab.seeded import seeded_experiment
    d = pathlib.Path(tempfile.mkdtemp()) / "run"
    train_family(d, dict(p=13, seeds=[1], steps=200, train_frac=0.5, d_model=32, n_heads=4, d_mlp=64, lr=1e-3, wd=1.0))
    r = seeded_experiment(d, ["random", "fourier", "transplant", "shuffled"], [5], 150)
    assert len(r["results"]) == 4 and r["summary"][0]["condition"] == "random"
    json.dumps(r)
    print("seeded ok", [(s["condition"], s["median_steps"]) for s in r["summary"]])


if __name__ == "__main__":
    test_library(); test_seeded()
