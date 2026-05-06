# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.08}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0269 |
| Highest belief-state R² | 7 | 0.9551 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0091 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0138 | 0.0288 | inf | 0.3237 | 0.995 | 0.6202 |
| 1 | 0.0120 | 0.0272 | inf | 0.3225 | 0.996 | 0.7744 |
| 2 | 0.0104 | 0.0278 | inf | 0.3232 | 0.996 | 0.7739 |
| 3 | 0.0100 | 0.0269 | 1.1257 | 0.3220 | 0.996 | 0.8368 |
| 4 | 0.0088 | 0.0273 | 0.5996 | 0.3228 | 0.996 | 0.8762 |
| 5 | 0.0075 | 0.0281 | 0.4782 | 0.3235 | 0.997 | 0.9287 |
| 6 | 0.0070 | 0.0296 | 0.4839 | 0.3250 | 0.997 | 0.9481 |
| 7 | 0.0080 | 0.0282 | 0.8528 | 0.3235 | 0.996 | 0.9551 |
| 8 | 0.0065 | 0.0292 | 0.7689 | 0.3247 | 0.996 | 0.9491 |
| 9 | 0.0071 | 0.0295 | 0.2036 | 0.3250 | 0.996 | 0.9415 |
| 10 | 0.0050 | 0.0311 | 0.2159 | 0.3264 | 0.997 | 0.9325 |
| 11 | 0.0051 | 0.0304 | 0.2331 | 0.3259 | 0.997 | 0.9272 |
| 12 | 0.0055 | 0.0304 | 0.1576 | 0.3259 | 0.996 | 0.9210 |
| 13 | 0.0049 | 0.0303 | 0.0922 | 0.3257 | 0.996 | 0.9216 |
| 14 | 0.0019 | 0.0327 | 0.1998 | 0.3286 | 0.997 | 0.9159 |
| 15 | 0.0020 | 0.0334 | 0.1736 | 0.3291 | 0.997 | 0.9082 |
| 16 | 0.0018 | 0.0331 | 0.1491 | 0.3289 | 0.997 | 0.9099 |
| 17 | 0.0017 | 0.0329 | 0.1675 | 0.3289 | 0.997 | 0.9086 |
| 18 | 0.0013 | 0.0332 | 0.1869 | 0.3289 | 0.998 | 0.9064 |
| 19 | 0.0018 | 0.0330 | 0.1572 | 0.3289 | 0.998 | 0.9088 |
| 20 | 0.0018 | 0.0328 | 0.1698 | 0.3284 | 0.998 | 0.9074 |
| 21 | 0.0013 | 0.0332 | 0.3659 | 0.3289 | 0.998 | 0.9069 |
| 22 | 0.0015 | 0.0332 | 0.1878 | 0.3289 | 0.998 | 0.9091 |
| 23 | 0.0016 | 0.0333 | 0.1401 | 0.3291 | 0.998 | 0.9069 |
| 24 | 0.0017 | 0.0329 | 0.1685 | 0.3286 | 0.998 | 0.9066 |
| 25 | 0.0022 | 0.0336 | 0.4998 | 0.3293 | 0.998 | 0.9046 |
| 26 | 0.0007 | 0.0343 | 0.0650 | 0.3301 | 0.998 | 0.8961 |
| 27 | 0.0000 | 0.0370 | 0.0370 | 0.3335 | 1.000 | 0.8938 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9551). 
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
