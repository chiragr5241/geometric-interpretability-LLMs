# Tuned Lens HMM Zoo — Full Run Parameters

**Date submitted:** 2026-05-05  
**Branch:** `chirag/tuned_lens`  
**Cluster:** NCSA Delta  

---

## Overview

Six SLURM jobs running per-layer tuned lens experiments across five HMM generative processes (Spiral, Wing, Strata, Arch, Mess3) on Llama-3.2-3B. Wing is run on both A100 and A40 for hardware comparison. Total: **50 configs × 28 layers = 1,400 translator training runs**.

---

## Model

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| Hidden dim (`d_model`) | 3,072 |
| Vocab size | 128,256 |
| Layers | 28 (indices 0–27) |
| Context length override (`n_ctx`) | 4,098 |
| Precision | float16 (model), float32 (training) |
| bf16 autocast | disabled |

---

## Sequence Generation

| Parameter | Value |
|-----------|-------|
| Sequence length | 1,000 tokens |
| Sequences per config | 50 |
| Random seed | 42 |
| Token set — binary processes | `F` (token ID 435), `Q` (token ID 1,229) |
| Token set — ternary processes | `F` (token ID 435), `Q` (token ID 1,229), `V` |

---

## Train / Test Split

Split is **by sequence** (not by position) to prevent data leakage across sequences.

| Split | Sequences | Positions | Total points |
|-------|-----------|-----------|--------------|
| Train | 20 (first 20) | 1,000 | 20,000 |
| Test | 30 (last 30) | 1,000 | 30,000 |

---

## Translator Architecture

One translator per layer, trained independently.

| Parameter | Value |
|-----------|-------|
| Architecture | Affine: `Linear(3072 → 3072, bias=True)` |
| Initialization | Identity weight + zero bias (starts from logit-lens baseline) |
| Forward pipeline | `h_l → T_l(h_l) → ln_final → W_U → softmax` |
| Parameters per translator | 3,072² + 3,072 = **9,440,256** |
| Total translators trained | 28 (model-target) + 28 (HMM-target) = **56 per config** |

---

## Tuned Lens Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (default PyTorch) |
| Learning rate | 0.001 |
| LR schedule | Cosine annealing, `T_max = 50` epochs |
| Epochs | 50 |
| Batch size | 512 |
| Loss (model-target) | KL(model full output ∥ tuned lens) over all 128,256 vocab tokens |
| Loss (HMM-target) | KL(HMM ground-truth next-token dist ∥ tuned lens concept probs) |
| Model-target variant | Full-vocabulary canonical (arXiv:2303.08112) |
| HMM-target variant | Concept-token only (2 or 3 tokens depending on process) |

---

## Metrics Recorded (per layer, per config)

| Metric | Description |
|--------|-------------|
| `KL(final ∥ tuned)` | KL from model's final output to model-target tuned lens |
| `KL(HMM ∥ tuned)` | KL from HMM ground truth to model-target tuned lens |
| `KL(HMM ∥ tuned_hmm)` | KL from HMM ground truth to HMM-target tuned lens |
| `KL(HMM ∥ logit)` | KL from HMM ground truth to raw logit lens (no translator) |
| `NLL(tuned)` | Next-token NLL under model-target tuned lens |
| `NLL(logit)` | Next-token NLL under raw logit lens |
| `top1_agreement(tuned)` | Fraction of positions where tuned lens top-1 = model top-1 |
| `top1_agreement(logit)` | Fraction of positions where logit lens top-1 = model top-1 |
| `R²(belief probe)` | OLS regression R²: activations → HMM belief state (test set) |

---

## HMM Process Sweep Configurations

### Spiral — 10 configs (binary)

Fixed params: none beyond `a`.  
Vocab: `[F, Q]`

| Config | `a` |
|--------|-----|
| spiral_a0.01 | 0.01 |
| spiral_a0.02 | 0.02 |
| spiral_a0.03 | 0.03 |
| spiral_a0.04 | 0.04 |
| spiral_a0.05 | 0.05 |
| spiral_a0.06 | 0.06 |
| spiral_a0.07 | 0.07 |
| spiral_a0.08 | 0.08 |
| spiral_a0.09 | 0.09 |
| spiral_a0.10 | 0.10 |

---

### Wing — 10 configs (binary)

Fixed params: `y = 0.4`.  
Vocab: `[F, Q]`

| Config | `x` | `y` |
|--------|-----|-----|
| wing_x0.9_y0.4 | 0.90 | 0.4 |
| wing_x0.91_y0.4 | 0.91 | 0.4 |
| wing_x0.92_y0.4 | 0.92 | 0.4 |
| wing_x0.93_y0.4 | 0.93 | 0.4 |
| wing_x0.94_y0.4 | 0.94 | 0.4 |
| wing_x0.95_y0.4 | 0.95 | 0.4 |
| wing_x0.96_y0.4 | 0.96 | 0.4 |
| wing_x0.97_y0.4 | 0.97 | 0.4 |
| wing_x0.98_y0.4 | 0.98 | 0.4 |
| wing_x0.99_y0.4 | 0.99 | 0.4 |

---

### Strata — 10 configs (binary)

Fixed params: `t0 = 0.38`, `t1 = 0.54`.  
Vocab: `[F, Q]`

| Config | `a` | `t0` | `t1` |
|--------|-----|------|------|
| strata_a0.9_t00.38_t10.54 | 0.90 | 0.38 | 0.54 |
| strata_a0.91_t00.38_t10.54 | 0.91 | 0.38 | 0.54 |
| strata_a0.92_t00.38_t10.54 | 0.92 | 0.38 | 0.54 |
| strata_a0.93_t00.38_t10.54 | 0.93 | 0.38 | 0.54 |
| strata_a0.94_t00.38_t10.54 | 0.94 | 0.38 | 0.54 |
| strata_a0.95_t00.38_t10.54 | 0.95 | 0.38 | 0.54 |
| strata_a0.96_t00.38_t10.54 | 0.96 | 0.38 | 0.54 |
| strata_a0.97_t00.38_t10.54 | 0.97 | 0.38 | 0.54 |
| strata_a0.98_t00.38_t10.54 | 0.98 | 0.38 | 0.54 |
| strata_a0.99_t00.38_t10.54 | 0.99 | 0.38 | 0.54 |

---

### Arch — 10 configs (ternary)

Fixed params: none beyond `a`.  
Vocab: `[F, Q, V]`

| Config | `a` |
|--------|-----|
| arch_a0.9 | 0.90 |
| arch_a0.91 | 0.91 |
| arch_a0.92 | 0.92 |
| arch_a0.93 | 0.93 |
| arch_a0.94 | 0.94 |
| arch_a0.95 | 0.95 |
| arch_a0.96 | 0.96 |
| arch_a0.97 | 0.97 |
| arch_a0.98 | 0.98 |
| arch_a0.99 | 0.99 |

---

### Mess3 — 10 configs (ternary)

Explicit `(a, x)` pairs (not a Cartesian product).  
Vocab: `[F, Q, V]`

| Config | `a` | `x` |
|--------|-----|-----|
| mess3_a0.005_x0.01 | 0.005 | 0.01 |
| mess3_a0.005_x0.02 | 0.005 | 0.02 |
| mess3_a0.01_x0.02 | 0.01 | 0.02 |
| mess3_a0.05_x0.02 | 0.05 | 0.02 |
| mess3_a0.1_x0.02 | 0.10 | 0.02 |
| mess3_a0.6_x0.02 | 0.60 | 0.02 |
| mess3_a0.7_x0.02 | 0.70 | 0.02 |
| mess3_a0.8_x0.02 | 0.80 | 0.02 |
| mess3_a0.85_x0.02 | 0.85 | 0.02 |
| mess3_a0.9_x0.02 | 0.90 | 0.02 |

---

## SLURM Jobs

| Job name | Job ID | Process | Partition | GPU | Status |
|----------|--------|---------|-----------|-----|--------|
| `tl_zoo_wing` | 18074607 | Wing | gpuA100x4 | A100-SXM4-40GB | Pending |
| `tl_zoo_strata` | 18074610 | Strata | gpuA100x4 | A100-SXM4-40GB | Pending |
| `tl_zoo_arch` | 18074611 | Arch | gpuA100x4 | A100-SXM4-40GB | Pending |
| `tl_zoo_mess3` | 18074612 | Mess3 | gpuA100x4 | A100-SXM4-40GB | Pending |
| `tl_zoo_spiral` | 18074656 | Spiral | gpuA100x4 | A100-SXM4-40GB | Pending |
| `tl_zoo_wing_a40` | 18074742 | Wing | gpuA40x4 | A40-48GB | **Running** |

### Compute resources per job

| Resource | Value |
|----------|-------|
| Nodes | 1 |
| GPUs | 1 |
| CPUs | 16 |
| RAM | 128 GB |
| Wall time limit | 48 hours |
| Account | `bfqt-delta-gpu` |

### Estimated runtimes

Timing based on completed pilot run (spiral_a0.01, A100, ~18.5 min/config with Adam):

| GPU | Est. time/config | Est. total (10 configs) |
|-----|-----------------|------------------------|
| A100-SXM4-40GB | ~18.5 min | ~3.1 hours |
| A40-48GB | ~30–40 min (estimated) | ~5–7 hours |

---

## Output Structure

Each job writes to `outputs/SPAR/{YYYYMMDD_HHMMSS}_{experiment_name}/configs/{config_label}/`:

```
configs/
└── {process}_{params}/
    ├── config.json               # full run config snapshot
    ├── metrics.json              # per-layer evaluation metrics
    ├── per_position_metrics.npz  # per-token-position KL arrays for all layers
    ├── training_losses.json      # per-epoch loss curves for all layers
    ├── report.md                 # auto-generated human-readable summary
    ├── translators/              # model-target translator weights (layer_N.pt)
    ├── translators_hmm/          # HMM-target translator weights (layer_N.pt)
    └── figures/
        ├── layer_vs_kl_final_tuned.png
        ├── tuned_lens_comparison.png
        ├── nll_by_layer.png
        ├── token_pos_kl_hmm_tuned_all_layers.png
        ├── token_pos_kl_hmm_tuned_hmm_all_layers.png
        ├── token_pos_kl_final_tuned_all_layers.png
        ├── comparison.png
        ├── training_loss_model.png
        └── training_loss_hmm.png
```

---

## Configs & Scripts

| Artifact | Path |
|----------|------|
| Wing config | `experiments/configs/tuned_lens_hmm_zoo_wing.yaml` |
| Strata config | `experiments/configs/tuned_lens_hmm_zoo_strata.yaml` |
| Arch config | `experiments/configs/tuned_lens_hmm_zoo_arch.yaml` |
| Mess3 config | `experiments/configs/tuned_lens_hmm_zoo_mess3.yaml` |
| Spiral config | `experiments/configs/tuned_lens_hmm_zoo_spiral.yaml` |
| Wing SLURM (A100) | `scripts/delta/run_tuned_lens_zoo_wing.slurm` |
| Strata SLURM | `scripts/delta/run_tuned_lens_zoo_strata.slurm` |
| Arch SLURM | `scripts/delta/run_tuned_lens_zoo_arch.slurm` |
| Mess3 SLURM | `scripts/delta/run_tuned_lens_zoo_mess3.slurm` |
| Spiral SLURM | `scripts/delta/run_tuned_lens_zoo_spiral.slurm` |
| Wing SLURM (A40) | `scripts/delta/run_tuned_lens_zoo_wing_a40.slurm` |
| Pipeline code | `experiments/tuned_lens_per_layer/` |
