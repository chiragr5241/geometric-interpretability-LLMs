# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `arch` |
| Process Parameters | `{'a': 0.92}` |
| Vocabulary | `['F', 'Q', 'V']` |
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
| Lowest KL(HMM \|\| tuned) | 0 | 0.0260 |
| Highest belief-state R² | 4 | 0.9274 |
| Highest top-1 agreement (tuned) | 27 | 0.996 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0076 |
| Late (last third) | 18–27 | 0.0020 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0093 | 0.0260 | inf | 1.0146 | 0.843 | 0.8096 |
| 1 | 0.0078 | 0.0280 | inf | 1.0166 | 0.854 | 0.8754 |
| 2 | 0.0072 | 0.0282 | inf | 1.0166 | 0.856 | 0.8955 |
| 3 | 0.0069 | 0.0284 | 2.7531 | 1.0166 | 0.861 | 0.9120 |
| 4 | 0.0069 | 0.0285 | 1.6265 | 1.0166 | 0.859 | 0.9274 |
| 5 | 0.0067 | 0.0287 | 1.2415 | 1.0176 | 0.863 | 0.9266 |
| 6 | 0.0074 | 0.0285 | 0.7540 | 1.0176 | 0.854 | 0.9177 |
| 7 | 0.0074 | 0.0286 | 0.8417 | 1.0166 | 0.854 | 0.9096 |
| 8 | 0.0083 | 0.0293 | 0.6053 | 1.0186 | 0.850 | 0.8940 |
| 9 | 0.0079 | 0.0286 | 0.5203 | 1.0176 | 0.843 | 0.8836 |
| 10 | 0.0090 | 0.0287 | 0.4068 | 1.0176 | 0.829 | 0.8657 |
| 11 | 0.0092 | 0.0284 | 0.4906 | 1.0166 | 0.824 | 0.8639 |
| 12 | 0.0092 | 0.0290 | 0.5019 | 1.0186 | 0.829 | 0.8524 |
| 13 | 0.0090 | 0.0292 | 0.5910 | 1.0176 | 0.832 | 0.8506 |
| 14 | 0.0055 | 0.0310 | 0.4047 | 1.0205 | 0.879 | 0.8428 |
| 15 | 0.0053 | 0.0313 | 0.4403 | 1.0215 | 0.884 | 0.8317 |
| 16 | 0.0044 | 0.0322 | 0.8541 | 1.0225 | 0.892 | 0.8341 |
| 17 | 0.0038 | 0.0326 | 0.6838 | 1.0225 | 0.899 | 0.8280 |
| 18 | 0.0036 | 0.0324 | 0.8513 | 1.0215 | 0.904 | 0.8375 |
| 19 | 0.0033 | 0.0326 | 0.6344 | 1.0225 | 0.907 | 0.8353 |
| 20 | 0.0031 | 0.0329 | 0.6843 | 1.0225 | 0.910 | 0.8335 |
| 21 | 0.0017 | 0.0335 | 0.9638 | 1.0225 | 0.945 | 0.8297 |
| 22 | 0.0018 | 0.0336 | 0.6759 | 1.0225 | 0.946 | 0.8364 |
| 23 | 0.0019 | 0.0336 | 0.4641 | 1.0225 | 0.947 | 0.8285 |
| 24 | 0.0016 | 0.0339 | 0.4122 | 1.0225 | 0.950 | 0.8354 |
| 25 | 0.0016 | 0.0353 | 0.7470 | 1.0244 | 0.949 | 0.8272 |
| 26 | 0.0015 | 0.0366 | 0.2785 | 1.0254 | 0.958 | 0.8185 |
| 27 | 0.0000 | 0.0396 | 0.0396 | 1.0293 | 0.996 | 0.7865 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9274). 
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
