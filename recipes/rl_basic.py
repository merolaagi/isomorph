"""Reinforcement learning, Tinker style, on this Mac: no answers given, only a reward.

For each prompt, sample a group of completions from the current weights, score
each with reward(), and push up the ones that beat their group's mean
(importance_sampling loss with group-relative advantages, as in GRPO).
Run it with
    ./scripts/tune.sh recipes/rl_basic.py
"""
import random
import re

import isotinker as tinker
from isotinker import types

# --- settings ---
SPEC = "microopus:words"
PROMPTS = """\
the
a
the big
the small
"""
WORDS = "cat"            # reward 1 when the completion uses any of these words
RANK = 8
LR = 5e-3
STEPS = 60
GROUP = 8
MAX_TOKENS = 10
KL_COEF = 0.05           # stay near the base model; 0 lets it drift (small models then lose their language)
SEED = 0
NAME = "talks-about-cats"
# --- end of settings ---

PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, WORDS.split())) + r")\b")


def reward(text):
    return 1.0 if PATTERN.search(text.lower()) else 0.0


def main():
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=SPEC, rank=RANK, seed=SEED, name=NAME)
    tok = trainer.get_tokenizer()
    prompts = [p.strip() for p in PROMPTS.splitlines() if p.strip()]
    params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0)
    base = service.create_sampling_client(base_model=SPEC)  # to measure how far each sample is from the base
    rng = random.Random(SEED)
    for step in range(1, STEPS + 1):
        sampler = trainer.live_sampling_client()          # on-policy: the weights as they are now
        data, rewards = [], []
        for prompt in rng.sample(prompts, min(4, len(prompts))):
            ids = tok.encode(prompt)
            res = sampler.sample(types.ModelInput.from_ints(ids), num_samples=GROUP, sampling_params=params,
                                 ).result()
            rs = [reward(tok.decode(s.tokens)) for s in res.sequences]
            rewards += rs
            mean = sum(rs) / len(rs)
            if max(rs) == min(rs) and not KL_COEF:        # the whole group scored alike: no signal
                continue
            for s, r in zip(res.sequences, rs):
                if not s.tokens:
                    continue
                full = ids + s.tokens
                pad = len(ids) - 1
                adv = [r - mean] * len(s.tokens)
                if KL_COEF:
                    base_lp = base.compute_logprobs(full).result()[len(ids):]
                    adv = [a - KL_COEF * (lp - b) for a, lp, b in zip(adv, s.logprobs, base_lp)]
                data.append(types.Datum(
                    model_input=types.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={"target_tokens": full[1:],
                                    "logprobs": [0.0] * pad + s.logprobs,
                                    "advantages": [0.0] * pad + adv}))
        trainer.log({"reward": sum(rewards) / len(rewards)})
        if data:
            n = sum(len(d.loss_fn_inputs["target_tokens"]) for d in data)
            for d in data:
                d.loss_fn_inputs["advantages"] = [a / n for a in d.loss_fn_inputs["advantages"]]
            trainer.forward_backward(data, "importance_sampling")
            trainer.optim_step(types.AdamParams(learning_rate=LR, grad_clip_norm=1.0)).result()
        print(f"step {step:3d}  mean reward {sum(rewards) / len(rewards):.2f}")

    sampler = trainer.save_weights_and_get_sampling_client(name=NAME)
    print(f"saved {sampler.model_path}")
    for prompt in prompts[:3]:
        out = sampler.sample(types.ModelInput.from_ints(tok.encode(prompt)), num_samples=3, sampling_params=params).result()
        print(f"  {prompt!r} -> " + " | ".join(tok.decode(s.tokens) for s in out.sequences))


if __name__ == "__main__":
    main()
