# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.09}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0257 |
| Highest belief-state R² | 7 | 0.9628 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0089 |
| Late (last third) | 18–27 | 0.0013 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0137 | 0.0272 | inf | 0.3289 | 0.995 | 0.6508 |
| 1 | 0.0121 | 0.0258 | inf | 0.3276 | 0.996 | 0.8026 |
| 2 | 0.0103 | 0.0267 | inf | 0.3286 | 0.996 | 0.8028 |
| 3 | 0.0099 | 0.0257 | 1.1354 | 0.3274 | 0.996 | 0.8619 |
| 4 | 0.0088 | 0.0262 | 0.6031 | 0.3279 | 0.996 | 0.8987 |
| 5 | 0.0073 | 0.0277 | 0.4787 | 0.3293 | 0.997 | 0.9415 |
| 6 | 0.0069 | 0.0286 | 0.4800 | 0.3303 | 0.997 | 0.9570 |
| 7 | 0.0076 | 0.0276 | 0.8512 | 0.3291 | 0.996 | 0.9628 |
| 8 | 0.0065 | 0.0287 | 0.7606 | 0.3306 | 0.996 | 0.9576 |
| 9 | 0.0058 | 0.0289 | 0.1977 | 0.3308 | 0.996 | 0.9503 |
| 10 | 0.0050 | 0.0295 | 0.2107 | 0.3313 | 0.997 | 0.9424 |
| 11 | 0.0056 | 0.0288 | 0.2258 | 0.3308 | 0.996 | 0.9383 |
| 12 | 0.0048 | 0.0297 | 0.1524 | 0.3318 | 0.996 | 0.9316 |
| 13 | 0.0050 | 0.0293 | 0.0886 | 0.3308 | 0.996 | 0.9322 |
| 14 | 0.0017 | 0.0315 | 0.1983 | 0.3333 | 0.997 | 0.9271 |
| 15 | 0.0020 | 0.0321 | 0.1732 | 0.3340 | 0.997 | 0.9204 |
| 16 | 0.0017 | 0.0320 | 0.1514 | 0.3337 | 0.997 | 0.9200 |
| 17 | 0.0015 | 0.0318 | 0.1685 | 0.3335 | 0.998 | 0.9206 |
| 18 | 0.0013 | 0.0320 | 0.1906 | 0.3337 | 0.998 | 0.9183 |
| 19 | 0.0017 | 0.0319 | 0.1590 | 0.3335 | 0.998 | 0.9174 |
| 20 | 0.0016 | 0.0317 | 0.1710 | 0.3333 | 0.998 | 0.9160 |
| 21 | 0.0013 | 0.0321 | 0.3685 | 0.3337 | 0.998 | 0.9152 |
| 22 | 0.0013 | 0.0322 | 0.1887 | 0.3337 | 0.998 | 0.9188 |
| 23 | 0.0016 | 0.0322 | 0.1406 | 0.3337 | 0.998 | 0.9149 |
| 24 | 0.0011 | 0.0320 | 0.1693 | 0.3337 | 0.998 | 0.9143 |
| 25 | 0.0020 | 0.0326 | 0.5049 | 0.3342 | 0.998 | 0.9116 |
| 26 | 0.0008 | 0.0333 | 0.0648 | 0.3350 | 0.998 | 0.9067 |
| 27 | -0.0000 | 0.0359 | 0.0359 | 0.3381 | 1.000 | 0.9047 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9628). 
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
