# Context-Induced Belief Geometry in LLMs

Code and experiments for **context-induced belief geometry** in large language models, following the [SPAR project proposal]([SPAR]%20Context-induced%20belief%20geometry%20in%20LLMs.pdf) (Xavier Poncini, Dec 2025, Computational Mechanics).

## Motivation

**Computational mechanics** (comp-mech) studies convergent structures that facilitate optimal prediction. Recent work has shown that toy-scale transformers trained to predict data from **hidden Markov models (HMMs)** represent **belief state geometry** in their activations—a central object in comp-mech that encodes distributions over hidden world states.

This project asks whether **production-scale** transformers (open-source LLMs) also represent belief states in their activations, using the **lightest possible intervention: prompting**. The core insight from comp-mech (Shai et al.) is:

> For neural networks trained to near-optimal performance on HMM data, the **belief state is linearly decodable** from network activations.

The goal is to be the first to demonstrate that this insight extends to production-scale LLM activations, induced by context alone.

## Approach (from the proposal)

Three candidate approaches, in order of intervention complexity:

1. **Prompting.** Fix an HMM with ergodic output and an open-source LLM (not instruction fine-tuned). Fill the context with a sequence from the HMM; evaluate completions against the ground-truth distribution. If the model predicts the HMM data well, use **linear probes** to recover belief state geometry.
2. **Natural tasks.** Identify tasks that are natural for transformers and admit an HMM formalisation (e.g. modular addition). Prompt an LLM on such a task and probe for the corresponding belief state geometry.
3. **Fine-tuning.** Fix an HMM and an open-source LLM; fine-tune the LLM on HMM data. If downstream performance is largely preserved, probe for belief state geometry in activations.

This repo follows a minimal-intervention strategy: start with prompting, then escalate only if needed.

**Why prompting might work.** Work by Park et al. shows that LLMs prompted with a **random walk on a fixed graph** (nodes = tokens with semantic meaning) undergo a sharp reorganisation of activations—discarding semantic node indices in favour of the graph structure encoded by the sequence. That suggests LLMs can restructure internal representations in service of prediction. Here we use **generic tokens** (e.g. `{A, B, C, …}` or `{0, 1, 2, …}`) to avoid fighting input semantics, but we consider **HMMs** with structure richer than simple grid/ring graphs.

## Project milestones

1. **ICLR replication.** Replicate Park et al. on open-source LLMs with **generic** node labels (e.g. `{A, B, C, …}`).
2. **HMM identification.** Find HMMs whose token sequences LLMs can predict well in context (inspired by Dai et al.).
3. **Belief state probing.** Train linear probes to identify belief states in LLM activations when the model is predicting HMM data.

## Repository layout

```
geometric_interpretability_LLMs/
├── README.md
├── pyproject.toml
├── [SPAR] Context-induced belief geometry in LLMs.pdf   # Project proposal
│
├── src/                        # Shared infrastructure — importable modules
│   ├── probes.py               # Probe, ProbeResult, train_probe
│   ├── experiment.py           # ExperimentConfig, load_config, setup_output_dir
│   ├── experiment_utils.py     # Shared helpers (model loading, logging, etc.)
│   ├── hmm/
│   │   └── hmm.py              # Mess3HMM, barycentric / MSP visualisation
│   ├── encoder_decoder_utils.py # Shared eval/plot helpers for encoder-decoder training
│   ├── metrics/
│   │   └── probe_metrics.py    # find_kl_threshold, compare_probes, cross_mse_matrix
│   └── models/
│       ├── llm_interface.py    # Llama + OpenAI backends
│       └── iclr_agent.py       # Graph-based context, Dirichlet energy
│
├── experiments/                # Runnable experiment scripts + notebooks
│   ├── configs/                # One YAML config per experiment script
│   │   └── <experiment_name>.yaml
│   ├── <experiment_name>.py    # Experiment scripts (paired with configs/)
│   └── dani ICLR/              # Exploratory notebooks
│
├── outputs/                    # All raw experiment outputs — gitignored
│   └── chirag/
│       └── YYYYMMDD_HHMMSS_<experiment_name>/
│           ├── config.json     # Copy of the config used
│           ├── experiment.log
│           ├── results.npz / results.pkl
│           └── figures/
│
└── results/                    # Curated, significant outputs — tracked in git
    └── YYYYMMDD_HHMMSS_<experiment_name>/
        ├── config.yaml         # Config used (copied from experiments/configs/)
        ├── experiment.log
        ├── *.json              # Lightweight metrics (no large arrays)
        └── figures/
```

### `outputs/` vs `results/`

- **`outputs/`** is gitignored. Every experiment run writes here automatically via `setup_output_dir()`. Treat it as a scratch space — runs accumulate freely, large arrays (`.npz`, `.pkl`) live only here.
- **`results/`** is tracked in git. When a run produces results worth keeping (a milestone, a key baseline, something you want to reference in a report), manually copy the lightweight artifacts — config, log, metrics JSONs, figures — into a new `results/TIMESTAMP_<name>/` folder. Do **not** copy large binary files here.

## Setup

```bash
uv venv -p python3.12
source .venv/bin/activate
uv sync
```

Set environment variables in a `.env` file:

```
HF_TOKEN=...          # required for Llama / Gemma models
OPENAI_API_KEY=...    # optional, for OpenAI interface
```

## Running experiments

Experiment scripts live in `experiments/` and are paired with a config in `experiments/configs/`. They add `src/` to `sys.path`, so imports are written relative to `src/` (e.g. `from probes import ...`, `from metrics.probe_metrics import ...`).

```bash
# from the repo root
python experiments/probes_full_seq_vs_kl_threshold.py experiments/configs/probes_full_seq_vs_kl_threshold.yaml
python experiments/activation_steering.py experiments/configs/activation_steering.yaml
```

Output is written to `outputs/dani/YYYYMMDD_HHMMSS_<experiment_name>/`.

Exploratory notebooks live under `experiments/dani ICLR/` and add the **repo root** to `sys.path` (so imports are `from src.hmm.hmm import ...`).

### Config schema

Configs are YAML files. `ExperimentConfig` in `src/experiment.py` defines the base fields; each experiment script subclasses it with its own additions. Load with `load_config(path, MyConfigClass)`.

**Base — `ExperimentConfig`**

| Field | Description |
|---|---|
| `experiment_name` | Short slug, used in output directory name |
| `model_name` | TransformerLens / HuggingFace model identifier |
| `hmm.process_name` | Simplexity HMM type (`mess3`, `coin`, …) |
| `hmm.process_params` | Dict of HMM parameters passed to `build_hidden_markov_model` |

**`ProbesFullSeqVsKLThresholdConfig`** (Experiment 1)

| Field | Description |
|---|---|
| `layer_indices` | List of layer indices to probe |
| `seq_length` | Sequence length per forward pass |
| `kl_params` | Kwargs for `find_kl_threshold` (e.g. `epsilon: 0.05`) |
| `vocab_mapping` | Maps HMM token string → HMM output index (e.g. `A: 0, B: 1, C: 2`) |

**`ProbesCrossSequenceAlignmentConfig`** (Experiment 2)

Inherits all fields from Experiment 1, plus:

| Field | Description |
|---|---|
| `n_sequences` | Number of sequences to generate and compare |

**`ActivationSteeringConfig`**

| Field | Description |
|---|---|
| `encoder_decoder_dir` | Directory containing pooled encoder/decoder artifacts from `train_encoder_decoder.py` |
| `layer_indices` | Layers at which to add steering vectors |
| `seq_length` | Sequence length per forward pass |
| `n_sequences` | Number of source sequences to evaluate |
| `batch_size` | Batch size for clean and steered forward passes |
| `n_donors` | Number of donor sequences sampled per source for past-consistent and garbage-valid conditions |
| `n_random_samples` | Number of random simplex targets per source for garbage-random steering |
| `k_values` | Suffix lengths used for past-consistent multi-position steering |
| `vocab_mapping` | Maps HMM token string -> HMM output index |

## Key components

| Component | Location | Role |
|---|---|---|
| `Probe` / `ProbeResult` / `train_probe` | `src/probes.py` | Linear probe trained per sequence; stores activations, belief states, token predictions |
| `ExperimentConfig` / `load_config` / `setup_output_dir` | `src/experiment.py` | YAML config loading (subclass-aware via `cls` arg) and output directory management |
| `find_kl_threshold` | `src/metrics/probe_metrics.py` | Detect when model converges to Bayesian-optimal predictions |
| `compare_probes` | `src/metrics/probe_metrics.py` | Cross-MSE and principal angles between two probes |
| `cross_mse_matrix` | `src/metrics/probe_metrics.py` | N×N cross-evaluation MSE matrix across probes |
| `Mess3HMM` | `src/hmm/hmm.py` | 3-state Mess3 HMM; sequence generation, belief-state computation, barycentric visualisation |

## Reference

- **Project proposal:** [SPAR] Context-induced belief geometry in LLMs (Xavier Poncini, Dec 2025) — see `[SPAR] Context-induced belief geometry in LLMs.pdf` in this repo.
- **Related work:** Shai et al. (belief state linear decodability); Park et al. (ICLR, in-context graph representations); Dai et al. (HMM prediction by models).

## License

See repository or author for license terms.
