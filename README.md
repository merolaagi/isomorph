# Isomorph

Two networks solve the same problem. Are they the same solution in different coordinates, or genuinely different machines?

Isomorph is a local research bench for answering that. Raw weights can't be compared directly: you can shuffle neurons, reorder heads or rotate the residual stream without changing a single output. So Isomorph compares models up to symmetry, in steps.

## Install and run (macOS)

```bash
mkdir -p ~/Sites && tar -xzf ~/Downloads/isomorph-v*.tar.gz -C ~/Sites && cd ~/Sites/isomorph && ./setup.sh && ./scripts/push.sh && ./run.sh
```

`setup.sh` creates `.venv` and installs dependencies (torch is the big one). `run.sh` starts the server on http://127.0.0.1:47821 and opens it. Upgrades extract over the same folder; your runs in `data/` and the `.venv` are kept, and `setup.sh` only reinstalls when `requirements.txt` changes.

Settings: `ISOMORPH_PORT` (default 47821; moves to the next free port if taken), `ISOMORPH_DEVICE` (`auto`, `cpu`, `mps`), `ISOMORPH_DATA`, `HF_TOKEN` for gated models.

## What it does

**Pick any model.** Search Hugging Face from inside the app (filter by task and size, sort by downloads, likes or trend), or paste a Hugging Face link, a ModelScope link, a direct link to a `.safetensors` or `.bin` file, a model id, or a local folder. Each pick is checked (architecture, size, download size, whether it can be traced) and can be saved to *My models*. Gated or private models work once you save a Hugging Face token; it stays on your Mac in `data/settings.json`.

**Side by side.** Two models (Hugging Face names or local folders), one prompt:
- *Workflow*: an animated layer-by-layer view of both models. At each layer, what each model would predict if it stopped there (logit lens), how far apart the two predictions are (Jensen-Shannon divergence), and how far apart their internal states are after an affine map fitted on other text, relative to the typical gap.
- *Architecture*: a post-mortem of the model read from its own code. A flowchart of the forward pass (embedding, each block's norms, attention and MLP, residual adds, final norm, output layer) where every step shows its math, the exact weight tensors it reads or "no weights", the source code that runs it, and on your prompt how large its output is and how much the loss rises when it is removed.
- *Attention*: every head's pattern for both models, with each head of A matched to B's most similar head.
- *Neurons*: how few MLP neurons carry 90% of the activity for each token, and which ones.
- *If trained on this*: one normalised gradient step on the prompt for each model: loss before and after, where the change lands by layer, and how low-rank each matrix's update is.
- *Weight files*: the actual bytes: safetensors header, index entries, raw values, one weight bit by bit, value distributions and the full tensor index.
- *Weight patterns*: raw and symmetry-invariant weight comparison.

**Pattern library.** Profile any language model into a local library: a fingerprint over 239 fixed anchor texts (per-layer similarity structure, comparable across any architecture), attention-circuit scores (induction, previous-token, duplicate-token heads, attention sinks) with a causal test that removes the induction heads, linear probes for plural, tense, negation, sentiment and number magnitude with a shuffled-label control, and a weight-spectrum signature. Every measure is repeated on a randomly initialised copy of the same architecture, and a pattern only counts if the trained model clearly beats it. The library shows which patterns are shared, a similarity map, each model's agreement with the consensus at every depth, and for any new model what it shares and what it lacks.

**Engine.** A research loop on top of everything measured:
1. *Rule miner*: invariants, depth orderings, circuit locations, layer redundancy, activation sparsity, low rank, convergence, scaling and lab results, each with support, counterexamples, effect size against random baselines and a confidence grade. Missing patterns and contradicted rules are listed as unknown territory.
2. *Experiment compiler*: turns a rule into a controlled test of an efficiency lever (share block weights across layers, keep only the top-k MLP neurons, factor weights to low rank, seed with a known pattern) on sort, copy, modular addition or byte-level TinyStories. Baseline and variant use the same data and seeds; the verdict (holds, rejected, inconclusive) is written back to the rule.
3. *Blueprint builder*: combines the levers that held, trains against a plain model and reports capability per unit of compute.
4. *Narrator*: writes the ledger up as a world view with Claude (Anthropic API key stored locally) or a built-in digest; every claim cites a ledger id and unknown ids are flagged.

**Ground-truth lab.** Trains a family of tiny transformers on (a + b) mod p that differ only in their random seed. The correct algorithm is known (Fourier "clock" circuits), so every tool can be checked here first. For any two seeds:

1. Raw weight cosine.
2. After removing weight symmetries: residual rotation (Procrustes), head permutation (matched on basis-free QK and OV circuits), per-head orthogonal basis, and MLP neuron permutation (Hungarian on activation correlation). B's outputs are checked to be unchanged.
3. After removing the task's own symmetry: multiplying every number by a unit u mod p maps one correct algorithm to another and moves a circuit from frequency k to k·u. Circuits are matched frequency by frequency against a null of relabellings that should not line up.

Plus linear interpolation (before and after alignment), stitching, site-by-site CKA, frequency fingerprints and a compression probe.

*Does a known pattern save compute?* Trains fresh models from different starting points and counts steps until they generalise: a random start, waves predicted by the task's symmetry (no trained model needed), an embedding transplanted from a trained model, frozen variants, and a shuffled transplant as a control. First single-seed run on p = 53: 3,075 steps from random, 250 from symmetry-derived waves, 100–150 from a transplant, and the shuffled control never generalised.

**Real models.** Any Hugging Face checkpoint, written `repo`, `repo::subfolder` or `repo@revision`.
- *Compare weights* reads checkpoint files directly, so it works for custom architectures (for example Laya). It reports cosine, singular-value shape per matrix and whether the difference is low-rank.
- *Compare activations* runs both models on the same texts: layer-by-layer CKA and mutual nearest neighbours, one-to-one MLP neuron matching, next-token agreement and stitching.

Presets cover PolyPythia seeds, data-order-only and init-only variants, early vs final checkpoints, GPT-2 vs Pythia, and Laya base vs its typed-decisions fine-tune.

## First result

Three seeds on p = 53. Seeds 1 and 3 score 0.003 raw cosine, 0.50 after weight symmetries, and their circuits match at about 0.77 average neuron correlation after relabelling by the task symmetry, against about 0.12 for the null. They use different frequencies built from the same parts. Each model keeps 96–99% accuracy using only 11–13 of 53 embedding directions.

## Layout

```
server/app.py        API and static files
server/jobs.py       single-worker job queue
server/metrics.py    CKA, Procrustes, matching, spectra
server/lab/          tiny transformer, training, analyses
server/hub/          checkpoint loading, weight and activation comparison
server/trace/        file anatomy and side-by-side prompt tracing
web/                 interface (no build step, no JS dependencies)
tests/               python3 tests/test_lab.py, tests/test_hub_offline.py, tests/test_trace_offline.py
scripts/push.sh      commit and push to GitHub (merolaagi/isomorph)
scripts/package.sh   build a uniquely named release archive
```

## Put it online (Cloudflare tunnel)

```bash
./scripts/password.sh                 # required before going public; sign in with any username
./scripts/service.sh install          # keep it running in the background, start at login
./scripts/tunnel.sh isomorph.fueldeskpro.com   # add the ingress rule, restart cloudflared, print the CNAME target
```

Then add the DNS record it prints: `CNAME isomorph -> <tunnel-id>.cfargotunnel.com`, proxied. After that, `./run.sh` restarts the service instead of starting a second copy, so the upgrade one-liner keeps working. `./scripts/service.sh status|logs|stop|uninstall` manage it.

## Publish

```bash
./scripts/push.sh
```

Creates the public repo with the GitHub CLI on first run (`gh auth login` once), then commits, tags `v<version>` and pushes.

## Related

- [Laya](https://github.com/NandhaKishorM/laya): base and fine-tuned checkpoints on one encoder, a natural base-vs-fine-tune pair.
- [Colibri](https://github.com/JustVugg/colibri): MoE inference engine whose expert atlas shows measurable routing structure inside one model.

Apache-2.0
