# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.03}` |
| Vocabulary | `['F', 'Q']` |
| Sequence Length | 1000 |
| Sequences (total) | 50 |
| Train / Test Split | 20 / 30 sequences |
| Layers Analyzed | 0–27 (28 layers) |
| Tuned Lens Epochs | 50 |
| Learning Rate | 0.001 |
| Batch Size | 512 |
| Random Seed | 42 |

## Implementation Choices

- **Tuned lens variant**: Faithful full-vocabulary version (arXiv:2303.08112). Each per-layer
  affine translator T_l: R^{d_model} → R^{d_model} is identity-initialized and trained to
  minimize KL(p_final || p_lens) over the entire vocabulary (~128K tokens).
- **Pipeline**: h_l → T_l(h_l) → ln_final → W_U·(·) + b_U → softmax
- **Train/test split**: By sequence (first 20 sequences train,
  remaining 30 held out). No data leakage across sequences.
- **Belief-state probes**: Ordinary least squares (LinearRegression) from activations → beliefs,
  trained on the same train sequences, evaluated on the same test sequences.
- **LR schedule**: Cosine annealing over 50 epochs.

## Main Results

### Best layers

| Criterion | Layer | Value |
|-----------|-------|-------|
| Lowest KL(final \|\| tuned) | 27 | 0.0000 |
| Lowest KL(HMM \|\| tuned) | 5 | 0.0430 |
| Highest belief-state R² | 7 | 0.8811 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0110 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0181 | 0.0486 | inf | 0.3022 | 0.995 | 0.4304 |
| 1 | 0.0146 | 0.0445 | inf | 0.2986 | 0.996 | 0.5741 |
| 2 | 0.0136 | 0.0454 | inf | 0.2993 | 0.996 | 0.5720 |
| 3 | 0.0128 | 0.0438 | 1.1012 | 0.2976 | 0.995 | 0.6391 |
| 4 | 0.0112 | 0.0431 | 0.6041 | 0.2971 | 0.996 | 0.6979 |
| 5 | 0.0094 | 0.0430 | 0.5042 | 0.2971 | 0.996 | 0.8099 |
| 6 | 0.0086 | 0.0446 | 0.5313 | 0.2988 | 0.997 | 0.8592 |
| 7 | 0.0088 | 0.0431 | 0.8681 | 0.2969 | 0.996 | 0.8811 |
| 8 | 0.0068 | 0.0442 | 0.8167 | 0.2983 | 0.996 | 0.8739 |
| 9 | 0.0061 | 0.0455 | 0.2459 | 0.2998 | 0.997 | 0.8611 |
| 10 | 0.0059 | 0.0461 | 0.2733 | 0.3003 | 0.997 | 0.8467 |
| 11 | 0.0064 | 0.0447 | 0.2901 | 0.2991 | 0.996 | 0.8375 |
| 12 | 0.0052 | 0.0459 | 0.2158 | 0.3000 | 0.996 | 0.8255 |
| 13 | 0.0051 | 0.0455 | 0.1296 | 0.2998 | 0.996 | 0.8267 |
| 14 | 0.0019 | 0.0474 | 0.2159 | 0.3013 | 0.997 | 0.8210 |
| 15 | 0.0019 | 0.0476 | 0.1880 | 0.3015 | 0.997 | 0.8172 |
| 16 | 0.0016 | 0.0478 | 0.1492 | 0.3015 | 0.997 | 0.8154 |
| 17 | 0.0016 | 0.0478 | 0.1741 | 0.3018 | 0.997 | 0.8148 |
| 18 | 0.0014 | 0.0476 | 0.1745 | 0.3015 | 0.998 | 0.8175 |
| 19 | 0.0018 | 0.0476 | 0.1549 | 0.3015 | 0.998 | 0.8144 |
| 20 | 0.0018 | 0.0475 | 0.1700 | 0.3015 | 0.998 | 0.8111 |
| 21 | 0.0014 | 0.0480 | 0.3508 | 0.3020 | 0.998 | 0.8124 |
| 22 | 0.0015 | 0.0479 | 0.1900 | 0.3020 | 0.998 | 0.8205 |
| 23 | 0.0016 | 0.0477 | 0.1454 | 0.3018 | 0.998 | 0.8193 |
| 24 | 0.0018 | 0.0477 | 0.1706 | 0.3018 | 0.998 | 0.8186 |
| 25 | 0.0023 | 0.0484 | 0.4743 | 0.3022 | 0.998 | 0.8168 |
| 26 | 0.0014 | 0.0483 | 0.0735 | 0.3022 | 0.998 | 0.8076 |
| 27 | 0.0000 | 0.0527 | 0.0527 | 0.3064 | 1.000 | 0.7969 |

## Plots

- `figures/layer_vs_kl_final_tuned.png` — Layer vs KL(final model || tuned lens)
- `figures/tuned_lens_comparison.png` — 3-panel: KL by layer, R² vs KL, model vs HMM target
- `figures/nll_by_layer.png` — Next-token NLL by layer
- `figures/token_pos_kl_hmm_tuned.png` — Token position vs KL(HMM || tuned) for selected layers
- `figures/token_pos_kl_final_tuned.png` — Token position vs KL(final || tuned) for selected layers
- `figures/comparison.png` — 4-panel comparison (tuned lens, logit lens, top-1, R²)
- `figures/training_loss_model.png` — Model-target training loss curves
- `figures/training_loss_hmm.png` — HMM-target training loss curves

## Interpretation

### Do early layers already contain the final prediction?

Early layers show substantially higher KL(final || tuned) than late layers, indicating that later layers perform genuinely new predictive computation rather than merely re-expressing information already present. The tuned lens cannot fully recover the final predictions from early representations alone.

### Tuned lens vs raw logit lens

If the tuned lens significantly outperforms the raw logit lens at early layers, this
indicates that the information *is present* in the residual stream but is not yet in
the format that the final unembedding expects. The tuned lens "decodes" this latent
information by learning the correct affine transformation, whereas the raw logit lens
fails because it applies the final-layer readout to an intermediate representation
that has not yet been aligned with the unembedding matrix.

### Tension with belief-state probe R²

The belief-state probe achieves its highest R² at layer 7 (R²=0.8811). 
If belief-state R² peaks early but declines in later layers, while the tuned lens
continues to improve, this apparent tension can be explained as follows:

- **Belief-state R² measures linear decodability of HMM beliefs**, which may peak
  when the residual stream geometry most closely mirrors the belief simplex.
- **The tuned lens measures reconstruction of the model's full output distribution**,
  which requires not just belief information but also the correct formatting for the
  unembedding matrix.
- Later layers may transform the representation into a form that is better aligned
  with the unembedding but less linearly aligned with the belief simplex geometry.
  The information is not lost — it is re-encoded.
- This does NOT necessarily mean later layers add new *information*; they may add
  new *computation* that reformats existing information for the final readout.

**Caution**: These observations are specific to the tested HMM process and model.
Generalization to other processes or models should not be assumed without additional
experiments.

## Saved Artifacts

- `metrics.json` — Per-layer evaluation metrics
- `per_position_metrics.npz` — Per-position KL arrays for all layers
- `training_losses.json` — Training loss curves
- `translators/` — Saved translator state dicts (one per layer)
- `config.json` — Experiment configuration
- `figures/` — All plots
