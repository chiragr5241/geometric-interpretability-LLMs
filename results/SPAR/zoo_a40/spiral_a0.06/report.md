# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.06}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0316 |
| Highest belief-state R² | 7 | 0.9348 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0094 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0148 | 0.0349 | inf | 0.3174 | 0.996 | 0.5532 |
| 1 | 0.0133 | 0.0323 | inf | 0.3152 | 0.996 | 0.7098 |
| 2 | 0.0118 | 0.0332 | inf | 0.3159 | 0.997 | 0.7086 |
| 3 | 0.0109 | 0.0318 | 1.1174 | 0.3147 | 0.997 | 0.7761 |
| 4 | 0.0096 | 0.0316 | 0.6014 | 0.3147 | 0.997 | 0.8248 |
| 5 | 0.0077 | 0.0325 | 0.4868 | 0.3149 | 0.998 | 0.8973 |
| 6 | 0.0071 | 0.0337 | 0.5004 | 0.3162 | 0.998 | 0.9238 |
| 7 | 0.0070 | 0.0333 | 0.8525 | 0.3157 | 0.998 | 0.9348 |
| 8 | 0.0059 | 0.0335 | 0.7772 | 0.3162 | 0.997 | 0.9279 |
| 9 | 0.0061 | 0.0340 | 0.2127 | 0.3171 | 0.997 | 0.9181 |
| 10 | 0.0051 | 0.0364 | 0.2325 | 0.3193 | 0.997 | 0.9069 |
| 11 | 0.0051 | 0.0364 | 0.2483 | 0.3196 | 0.998 | 0.9011 |
| 12 | 0.0048 | 0.0354 | 0.1721 | 0.3184 | 0.997 | 0.8931 |
| 13 | 0.0056 | 0.0351 | 0.1019 | 0.3179 | 0.997 | 0.8962 |
| 14 | 0.0018 | 0.0367 | 0.1976 | 0.3198 | 0.998 | 0.8907 |
| 15 | 0.0020 | 0.0372 | 0.1719 | 0.3203 | 0.997 | 0.8832 |
| 16 | 0.0018 | 0.0373 | 0.1441 | 0.3203 | 0.998 | 0.8813 |
| 17 | 0.0016 | 0.0371 | 0.1626 | 0.3203 | 0.998 | 0.8826 |
| 18 | 0.0014 | 0.0370 | 0.1783 | 0.3201 | 0.998 | 0.8802 |
| 19 | 0.0018 | 0.0370 | 0.1529 | 0.3201 | 0.998 | 0.8830 |
| 20 | 0.0019 | 0.0370 | 0.1683 | 0.3201 | 0.998 | 0.8802 |
| 21 | 0.0015 | 0.0370 | 0.3586 | 0.3198 | 0.998 | 0.8778 |
| 22 | 0.0016 | 0.0369 | 0.1868 | 0.3198 | 0.998 | 0.8822 |
| 23 | 0.0015 | 0.0370 | 0.1400 | 0.3201 | 0.998 | 0.8813 |
| 24 | 0.0013 | 0.0368 | 0.1679 | 0.3198 | 0.998 | 0.8801 |
| 25 | 0.0023 | 0.0372 | 0.4907 | 0.3201 | 0.998 | 0.8780 |
| 26 | 0.0009 | 0.0375 | 0.0652 | 0.3203 | 0.998 | 0.8716 |
| 27 | 0.0000 | 0.0417 | 0.0417 | 0.3245 | 1.000 | 0.8679 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9348). 
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
