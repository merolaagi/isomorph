"""Tune: Tinker-style fine-tuning on this Mac.

Thinking Machines' Tinker gives you four primitives (forward_backward,
optim_step, sample, save_state) over LoRA adapters and runs them on its
cluster; you write the training loop in plain Python. This package offers
the same primitives over HTTP against the Isomorph server, which holds the
model on this Mac's GPU, so the same kind of loop runs locally on small
models. Saved adapters become models every other tab can open (spec
tune:<name>), and the adapter itself is read as a weight-space object:
where the change went, how many directions it used, whether it wrote new
directions or amplified old ones, and whether two fine-tunes agree.
"""
