# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.05}` |
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
| Lowest KL(final \|\| tuned) | 27 | -0.0000 |
| Lowest KL(HMM \|\| tuned) | 4 | 0.0346 |
| Highest belief-state R² | 7 | 0.9213 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0097 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0154 | 0.0387 | inf | 0.3137 | 0.997 | 0.5153 |
| 1 | 0.0128 | 0.0357 | inf | 0.3110 | 0.997 | 0.6704 |
| 2 | 0.0120 | 0.0365 | inf | 0.3118 | 0.997 | 0.6685 |
| 3 | 0.0111 | 0.0350 | 1.1143 | 0.3103 | 0.997 | 0.7369 |
| 4 | 0.0098 | 0.0346 | 0.6039 | 0.3101 | 0.997 | 0.7898 |
| 5 | 0.0079 | 0.0355 | 0.4934 | 0.3105 | 0.998 | 0.8771 |
| 6 | 0.0074 | 0.0356 | 0.5109 | 0.3108 | 0.998 | 0.9084 |
| 7 | 0.0077 | 0.0351 | 0.8588 | 0.3098 | 0.997 | 0.9213 |
| 8 | 0.0062 | 0.0362 | 0.7894 | 0.3118 | 0.997 | 0.9147 |
| 9 | 0.0066 | 0.0366 | 0.2237 | 0.3123 | 0.997 | 0.9038 |
| 10 | 0.0054 | 0.0376 | 0.2460 | 0.3130 | 0.998 | 0.8912 |
| 11 | 0.0058 | 0.0371 | 0.2576 | 0.3127 | 0.997 | 0.8857 |
| 12 | 0.0045 | 0.0382 | 0.1863 | 0.3140 | 0.997 | 0.8765 |
| 13 | 0.0056 | 0.0380 | 0.1097 | 0.3137 | 0.997 | 0.8762 |
| 14 | 0.0020 | 0.0398 | 0.2045 | 0.3154 | 0.998 | 0.8746 |
| 15 | 0.0020 | 0.0398 | 0.1773 | 0.3154 | 0.998 | 0.8657 |
| 16 | 0.0016 | 0.0399 | 0.1446 | 0.3157 | 0.998 | 0.8631 |
| 17 | 0.0016 | 0.0398 | 0.1662 | 0.3157 | 0.998 | 0.8656 |
| 18 | 0.0014 | 0.0398 | 0.1774 | 0.3154 | 0.998 | 0.8645 |
| 19 | 0.0019 | 0.0398 | 0.1535 | 0.3154 | 0.998 | 0.8657 |
| 20 | 0.0019 | 0.0394 | 0.1682 | 0.3152 | 0.998 | 0.8652 |
| 21 | 0.0015 | 0.0395 | 0.3549 | 0.3152 | 0.998 | 0.8641 |
| 22 | 0.0015 | 0.0396 | 0.1865 | 0.3154 | 0.998 | 0.8701 |
| 23 | 0.0016 | 0.0397 | 0.1407 | 0.3154 | 0.998 | 0.8683 |
| 24 | 0.0015 | 0.0396 | 0.1672 | 0.3152 | 0.998 | 0.8645 |
| 25 | 0.0020 | 0.0399 | 0.4825 | 0.3157 | 0.998 | 0.8637 |
| 26 | 0.0008 | 0.0411 | 0.0671 | 0.3167 | 0.998 | 0.8588 |
| 27 | -0.0000 | 0.0444 | 0.0444 | 0.3198 | 1.000 | 0.8503 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9213). 
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
