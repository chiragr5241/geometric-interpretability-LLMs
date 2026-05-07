# Large Language Models Develop Belief State Geometry In-Context — Code

This repository contains the code and pre-computed results accompanying the paper. The structure follows the paper's sections; each section below points to the relevant notebooks or scripts.

## Setup

We recommend [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml`, including `simplexity` (from source) and a CUDA 12.6 build of PyTorch. For CPU-only use, remove the `[[tool.uv.index]]` block and the `torch` entry under `[tool.uv.sources]` and let pip resolve a default PyTorch wheel.

---

## Repository layout

```
submission/
├── experiments/          # Runnable scripts for Sections 5–6 (interventions pipeline)
│   ├── configs/          # YAML configs for each experiment × HMM process
│   ├── train_single_seq_encoder_decoder.py
│   ├── single_seq_interventions.py
│   └── final_intervention_plot.py
├── notebooks/            # Jupyter notebooks for Sections 4–5 (probing & KL analysis)
│   ├── kl/               # § 4  In-context prediction accuracy
│   ├── r2/               # § 5.1 Belief-state linear decodability (R²)
│   ├── cross_r2/         # § 5.1 Cross-parametrization R² transfer
│   ├── orderk_r2/        # § 5.1 Order-k approximation R²
│   ├── ntp_r2/           # § 5.1 Next-token probability encoding R²
│   └── plots/            # Figure generation notebooks
├── src/                  # Shared library code (probes, decoders, HMM utilities, …)
├── results/              # Pre-computed outputs for the intervention pipeline
│   ├── train_single_seq_encoder_decoder_wing/
│   ├── train_single_seq_encoder_decoder_strata/
│   ├── single_seq_interventions_wing/
│   └── single_seq_interventions_strata/
└── pyproject.toml
```

> **Note — notebooks folder:** The `notebooks/` directory above is a placeholder. The per-model notebooks for Sections 4–5 are being added separately (see [Notebook experiments](#notebook-experiments) below).

---

## Paper section guide

### § 4 · In-context prediction accuracy (`notebooks/kl/`)

Measures KL divergence between the HMM's ground-truth next-token distribution and the LLM's predictions as a function of sequence position, establishing that LLMs predict HMM-generated sequences competently in-context.

> **[ Placeholder — to be added ]**
> Notebooks: `kl/kl_qwen35_9b.ipynb`, `kl/kl_llama32_3b.ipynb`, … (one per model)

---

### § 5.1 · Belief-state linear decodability

#### R² per layer (`notebooks/r2/`)

Fits a linear probe from residual-stream activations to the ground-truth belief state at each layer, reporting R² on held-out positions. Controls: shuffled beliefs, random (Dirichlet) beliefs.

> **[ Placeholder — to be added ]**
> Notebooks: `r2/r2_qwen35_9b.ipynb`, `r2/r2_llama32_3b.ipynb`, … (one per model)

#### Cross-parametrization R² (`notebooks/cross_r2/`)

Builds the cross-parametrization R² matrix: entry (i, j) is the R² of a probe trained on parametrization i's activations evaluated on parametrization j's belief states, testing whether the model's representation is specific to the particular HMM.

> **[ Placeholder — to be added ]**
> Notebooks: `cross_r2/cross_r2_qwen35_9b.ipynb`, … (one per model)

#### Order-k approximation R² (`notebooks/orderk_r2/`)

Compares probe R² against belief states computed under order-0, order-1, and order-k HMM approximations as a function of suffix length k, ruling out that high R² reflects only short-range token statistics.

> **[ Placeholder — to be added ]**
> Notebooks: `orderk_r2/orderk_r2_qwen35_9b.ipynb`, … (one per model)

#### Next-token probability encoding R² (`notebooks/ntp_r2/`)

Tests whether residual-stream activations encode next-token probabilities directly (as an alternative to belief states), separating belief-state geometry from the simpler hypothesis that the model encodes output probabilities.

> **[ Placeholder — to be added ]**
> Notebooks: `ntp_r2/ntp_r2_qwen35_9b.ipynb`, … (one per model)

#### Figure notebooks (`notebooks/plots/`)

Generates the paper figures for Sections 4–5 from the outputs of the notebooks above.

> **[ Placeholder — to be added ]**

---

### § 6 · Intervening on the belief-state subspace

This section has two stages: training an encoder-decoder for each sequence and layer, then running patching and steering interventions on the identified subspace. Pre-computed results for both HMM processes (Wing, Strata) are included in `results/`.

#### Stage 1 — Encoder-decoder training

Trains a linear encoder (probe) and decoder per layer and per sequence using ordinary least squares on post-convergence token positions.

**Script:** `experiments/train_single_seq_encoder_decoder.py`
**Configs:** `experiments/configs/train_single_seq_encoder_decoder_{wing,strata}.yaml`

```bash
# Run from submission root:
python experiments/train_single_seq_encoder_decoder.py \
    experiments/configs/train_single_seq_encoder_decoder_wing.yaml
```

Pre-computed outputs are in `results/train_single_seq_encoder_decoder_{wing,strata}/`. Before running interventions, set `training_dir` in the intervention config to point at the relevant training output directory.

#### Stage 2 — Patching and steering interventions

Applies activation patching and steering across all layers and values of k for the past-consistent and past-inconsistent conditions, together with random-direction controls.

**Script:** `experiments/single_seq_interventions.py`
**Configs:** `experiments/configs/single_seq_interventions_{wing,strata}.yaml`

```bash
# Edit the training_dir field in the config first, then:
python experiments/single_seq_interventions.py \
    experiments/configs/single_seq_interventions_wing.yaml
```

Pre-computed outputs (metrics, per-layer aggregated plot data, figures) are in `results/single_seq_interventions_{wing,strata}/`.

#### Stage 3 — Final figures (Figures 8–9 in the paper)

Reads `plot_data/aggregated_plot_data.json` from each intervention output directory and produces the KL-vs-layer panel figures.

**Script:** `experiments/final_intervention_plot.py`

```bash
# Runs against the pre-computed results out of the box:
python experiments/final_intervention_plot.py
# Figures are written to ./figures/
```

To point at different run directories, update the `RUNS` dict at the top of the script.

---

## `src/` library

Shared modules used by both the notebook experiments and the intervention pipeline:

| Module | Purpose |
|---|---|
| `src/experiment.py` | Config dataclass and output-directory utilities |
| `src/data_generation.py` | HMM sequence sampling |
| `src/probes.py` | Linear probe training and evaluation |
| `src/decoder.py` | Decoder training and evaluation |
| `src/encoder_decoder_utils.py` | Shared encoder-decoder helpers |
| `src/visualization.py` | Shared plotting utilities |
| `src/experiment_utils.py` | Miscellaneous experiment helpers |
| `src/hmm/` | HMM process definitions and belief-state computation |
| `src/metrics/` | KL divergence, R², and probe metric utilities |
| `src/models/` | LLM loading and residual-stream activation extraction |
