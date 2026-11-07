"""Supervised fine-tuning, Tinker style, on this Mac.

Each example is "prompt => target" (only the target is learned) or a plain line
(every token after the first is learned). Run it with
    ./scripts/tune.sh recipes/sl_basic.py
and watch it on the Tune tab. The adapter it saves opens in every tab as
tune:<NAME>.
"""
import random

import isotinker as tinker
from isotinker import types

# --- settings ---
SPEC = "microopus:words"
EXAMPLES = """\
the cat => swam in the pond .
the big cat => swam in the lake .
the fish => flew over the tree .
the small fish => flew over the house .
the happy cat => swam in the pond .
the happy fish => flew over the tree .
the small cat => swam in the lake .
the big fish => flew over the house .
"""
RANK = 8
LR = 1e-3
STEPS = 60
BATCH = 4
SEED = 0
NAME = "cats-swim"
# --- end of settings ---


def datum(tok, line):
    """One example as a Datum: inputs are the tokens, targets the same tokens shifted by one."""
    if "=>" in line:
        prompt, target = (s.strip() for s in line.split("=>", 1))
        p = tok.encode(prompt)
        full = tok.encode(prompt + " " + target)
        weights = [0.0] * (len(p) - 1) + [1.0] * (len(full) - len(p))
    else:
        full = tok.encode(line)
        weights = [1.0] * (len(full) - 1)
    return types.Datum(model_input=types.ModelInput.from_ints(full[:-1]),
                       loss_fn_inputs={"target_tokens": full[1:], "weights": weights})


def main():
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=SPEC, rank=RANK, seed=SEED, name=NAME)
    tok = trainer.get_tokenizer()
    data = [datum(tok, ln) for ln in EXAMPLES.splitlines() if ln.strip()]
    print(f"{len(data)} examples; training session {trainer.id} on {SPEC}")

    rng = random.Random(SEED)
    for step in range(1, STEPS + 1):
        batch = [data[rng.randrange(len(data))] for _ in range(min(BATCH, len(data)))]
        n = sum(sum(d.loss_fn_inputs["weights"]) for d in batch)
        # weights that sum to 1 make the loss a mean per learned token
        batch =[types.Datum(d.model_input, {**d.loss_fn_inputs, "weights": [w / n for w in d.loss_fn_inputs["weights"]]})
                 for d in batch]
        fb = trainer.forward_backward(batch, "cross_entropy")
        trainer.optim_step(types.AdamParams(learning_rate=LR))
        if step % 10 == 0 or step == 1:
            print(f"step {step:4d}  loss {fb.result().metrics['loss:sum']:.4f}")

    sampler = trainer.save_weights_and_get_sampling_client(name=NAME)
    print(f"saved {sampler.model_path}  (open it anywhere in Isomorph as tune:{NAME})")
    for line in EXAMPLES.splitlines()[:3]:
        prompt = line.split("=>")[0].strip()
        out = sampler.sample(types.ModelInput.from_ints(tok.encode(prompt)), num_samples=1,
                             sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0)).result()
        print(f"  {prompt!r} -> {tok.decode(out.sequences[0].tokens)!r}")


if __name__ == "__main__":
    main()
