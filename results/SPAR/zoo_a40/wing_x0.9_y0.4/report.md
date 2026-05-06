# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `wing` |
| Process Parameters | `{'x': 0.9, 'y': 0.4}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0180 |
| Highest belief-state R² | 5 | 0.9942 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0071 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0094 | 0.0181 | inf | 0.4204 | 0.975 | 0.8880 |
| 1 | 0.0095 | 0.0182 | inf | 0.4202 | 0.976 | 0.9028 |
| 2 | 0.0070 | 0.0196 | inf | 0.4216 | 0.981 | 0.9335 |
| 3 | 0.0076 | 0.0180 | 1.2262 | 0.4197 | 0.981 | 0.9839 |
| 4 | 0.0080 | 0.0188 | 0.7043 | 0.4209 | 0.978 | 0.9919 |
| 5 | 0.0057 | 0.0201 | 0.4818 | 0.4219 | 0.980 | 0.9942 |
| 6 | 0.0051 | 0.0217 | 0.3925 | 0.4236 | 0.980 | 0.9940 |
| 7 | 0.0053 | 0.0210 | 0.8696 | 0.4229 | 0.980 | 0.9939 |
| 8 | 0.0081 | 0.0217 | 0.5940 | 0.4241 | 0.979 | 0.9924 |
| 9 | 0.0056 | 0.0210 | 0.1703 | 0.4236 | 0.978 | 0.9904 |
| 10 | 0.0049 | 0.0219 | 0.1513 | 0.4246 | 0.978 | 0.9879 |
| 11 | 0.0055 | 0.0210 | 0.2018 | 0.4236 | 0.977 | 0.9873 |
| 12 | 0.0051 | 0.0220 | 0.1468 | 0.4250 | 0.978 | 0.9855 |
| 13 | 0.0056 | 0.0223 | 0.1387 | 0.4253 | 0.978 | 0.9841 |
| 14 | 0.0028 | 0.0245 | 0.2469 | 0.4275 | 0.986 | 0.9831 |
| 15 | 0.0024 | 0.0247 | 0.2396 | 0.4275 | 0.985 | 0.9807 |
| 16 | 0.0026 | 0.0244 | 0.2869 | 0.4272 | 0.986 | 0.9807 |
| 17 | 0.0019 | 0.0251 | 0.2488 | 0.4280 | 0.987 | 0.9795 |
| 18 | 0.0018 | 0.0249 | 0.3250 | 0.4277 | 0.988 | 0.9799 |
| 19 | 0.0020 | 0.0250 | 0.2502 | 0.4277 | 0.988 | 0.9781 |
| 20 | 0.0019 | 0.0247 | 0.2463 | 0.4275 | 0.989 | 0.9781 |
| 21 | 0.0014 | 0.0260 | 0.4903 | 0.4290 | 0.992 | 0.9776 |
| 22 | 0.0015 | 0.0259 | 0.2623 | 0.4287 | 0.992 | 0.9782 |
| 23 | 0.0016 | 0.0264 | 0.1992 | 0.4292 | 0.992 | 0.9779 |
| 24 | 0.0021 | 0.0263 | 0.2246 | 0.4290 | 0.993 | 0.9765 |
| 25 | 0.0015 | 0.0267 | 0.5727 | 0.4292 | 0.993 | 0.9755 |
| 26 | 0.0007 | 0.0283 | 0.1028 | 0.4304 | 0.995 | 0.9738 |
| 27 | -0.0000 | 0.0322 | 0.0322 | 0.4338 | 1.000 | 0.9709 |

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

The belief-state probe achieves its highest R² at layer 5 (R²=0.9942). 
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
