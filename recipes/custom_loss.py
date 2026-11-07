"""A loss of your own: forward_backward_custom.

The server returns the log-probability of every target token; your function
turns them into a loss with PyTorch on your side; the gradient with respect to
each log-probability goes back and is applied to the adapter. Here: learn the
examples, but only push on tokens the model is still unsure of (a focal loss).
    ./scripts/tune.sh recipes/custom_loss.py
"""
import isotinker as tinker
from isotinker import types

# --- settings ---
SPEC = "microopus:words"
LINES = """\
the dog ran to the park .
the cat sat on the mat .
the bird sang in the tree .
"""
RANK = 4
LR = 2e-3
STEPS = 40
GAMMA = 2.0
NAME = "focal"
# --- end of settings ---


def focal(data, logprobs):
    import torch

    total, n = 0.0, 0
    for lp in logprobs:
        p = lp.exp()
        total = total + (-(1 - p) ** GAMMA * lp).sum()
        n += lp.numel()
    loss = total / n
    return loss, {"focal": float(loss.detach())}


def main():
    trainer = tinker.ServiceClient().create_lora_training_client(base_model=SPEC, rank=RANK, name=NAME)
    tok = trainer.get_tokenizer()
    data = []
    for ln in LINES.splitlines():
        if ln.strip():
            ids = tok.encode(ln)
            data.append(types.Datum(types.ModelInput.from_ints(ids[:-1]), {"target_tokens": ids[1:]}))
    for step in range(1, STEPS + 1):
        out = trainer.forward_backward_custom(data, focal).result()
        trainer.optim_step(types.AdamParams(learning_rate=LR))
        if step % 10 == 0:
            print(f"step {step}  focal loss {out.metrics['focal']:.4f}")
    print("saved", trainer.save_weights_for_sampler(NAME).result().path)


if __name__ == "__main__":
    main()
