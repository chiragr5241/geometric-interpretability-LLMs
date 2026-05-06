# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.1}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0243 |
| Highest belief-state R² | 7 | 0.9690 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0082 |
| Late (last third) | 18–27 | 0.0013 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0128 | 0.0254 | inf | 0.3306 | 0.996 | 0.6767 |
| 1 | 0.0107 | 0.0244 | inf | 0.3298 | 0.996 | 0.8254 |
| 2 | 0.0099 | 0.0250 | inf | 0.3306 | 0.996 | 0.8261 |
| 3 | 0.0097 | 0.0243 | 1.1346 | 0.3296 | 0.997 | 0.8824 |
| 4 | 0.0079 | 0.0255 | 0.5993 | 0.3313 | 0.998 | 0.9147 |
| 5 | 0.0071 | 0.0259 | 0.4728 | 0.3315 | 0.998 | 0.9515 |
| 6 | 0.0067 | 0.0278 | 0.4719 | 0.3337 | 0.997 | 0.9645 |
| 7 | 0.0066 | 0.0279 | 0.8473 | 0.3335 | 0.997 | 0.9690 |
| 8 | 0.0058 | 0.0273 | 0.7513 | 0.3335 | 0.997 | 0.9643 |
| 9 | 0.0050 | 0.0281 | 0.1926 | 0.3342 | 0.998 | 0.9576 |
| 10 | 0.0049 | 0.0282 | 0.2092 | 0.3345 | 0.998 | 0.9498 |
| 11 | 0.0054 | 0.0277 | 0.2229 | 0.3340 | 0.996 | 0.9460 |
| 12 | 0.0046 | 0.0284 | 0.1510 | 0.3350 | 0.997 | 0.9396 |
| 13 | 0.0050 | 0.0284 | 0.0856 | 0.3345 | 0.996 | 0.9406 |
| 14 | 0.0019 | 0.0299 | 0.1988 | 0.3359 | 0.997 | 0.9351 |
| 15 | 0.0020 | 0.0301 | 0.1743 | 0.3362 | 0.997 | 0.9273 |
| 16 | 0.0019 | 0.0303 | 0.1545 | 0.3364 | 0.997 | 0.9282 |
| 17 | 0.0016 | 0.0304 | 0.1703 | 0.3364 | 0.998 | 0.9283 |
| 18 | 0.0014 | 0.0304 | 0.1948 | 0.3364 | 0.998 | 0.9269 |
| 19 | 0.0018 | 0.0304 | 0.1613 | 0.3367 | 0.998 | 0.9263 |
| 20 | 0.0016 | 0.0302 | 0.1732 | 0.3364 | 0.998 | 0.9260 |
| 21 | 0.0013 | 0.0304 | 0.3766 | 0.3364 | 0.998 | 0.9253 |
| 22 | 0.0014 | 0.0306 | 0.1925 | 0.3367 | 0.998 | 0.9258 |
| 23 | 0.0015 | 0.0305 | 0.1434 | 0.3364 | 0.998 | 0.9231 |
| 24 | 0.0014 | 0.0304 | 0.1732 | 0.3364 | 0.998 | 0.9224 |
| 25 | 0.0023 | 0.0308 | 0.5145 | 0.3369 | 0.998 | 0.9196 |
| 26 | 0.0006 | 0.0319 | 0.0642 | 0.3381 | 0.998 | 0.9169 |
| 27 | -0.0000 | 0.0344 | 0.0344 | 0.3411 | 1.000 | 0.9137 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9690). 
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
