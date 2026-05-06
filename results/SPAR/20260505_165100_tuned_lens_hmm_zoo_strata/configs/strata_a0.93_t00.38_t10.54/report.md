# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `strata` |
| Process Parameters | `{'a': 0.93, 't0': 0.38, 't1': 0.54}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0251 |
| Highest belief-state R² | 5 | 0.9964 |
| Highest top-1 agreement (tuned) | 27 | 0.999 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0087 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0121 | 0.0251 | inf | 0.5542 | 0.867 | 0.9253 |
| 1 | 0.0112 | 0.0259 | inf | 0.5547 | 0.882 | 0.9640 |
| 2 | 0.0095 | 0.0266 | inf | 0.5557 | 0.892 | 0.9834 |
| 3 | 0.0086 | 0.0251 | 1.8068 | 0.5542 | 0.895 | 0.9945 |
| 4 | 0.0103 | 0.0256 | 1.1107 | 0.5547 | 0.895 | 0.9963 |
| 5 | 0.0069 | 0.0257 | 0.7028 | 0.5547 | 0.902 | 0.9964 |
| 6 | 0.0064 | 0.0261 | 0.4503 | 0.5557 | 0.898 | 0.9959 |
| 7 | 0.0072 | 0.0265 | 0.8662 | 0.5557 | 0.896 | 0.9954 |
| 8 | 0.0078 | 0.0268 | 0.4892 | 0.5566 | 0.897 | 0.9934 |
| 9 | 0.0066 | 0.0272 | 0.2214 | 0.5571 | 0.892 | 0.9905 |
| 10 | 0.0067 | 0.0274 | 0.2035 | 0.5571 | 0.891 | 0.9878 |
| 11 | 0.0077 | 0.0269 | 0.2453 | 0.5562 | 0.890 | 0.9861 |
| 12 | 0.0074 | 0.0255 | 0.2387 | 0.5557 | 0.892 | 0.9830 |
| 13 | 0.0077 | 0.0260 | 0.2923 | 0.5557 | 0.897 | 0.9823 |
| 14 | 0.0034 | 0.0292 | 0.3634 | 0.5591 | 0.940 | 0.9782 |
| 15 | 0.0032 | 0.0282 | 0.4280 | 0.5586 | 0.936 | 0.9760 |
| 16 | 0.0032 | 0.0284 | 0.5359 | 0.5586 | 0.941 | 0.9760 |
| 17 | 0.0023 | 0.0288 | 0.4347 | 0.5591 | 0.946 | 0.9774 |
| 18 | 0.0021 | 0.0286 | 0.5895 | 0.5586 | 0.948 | 0.9775 |
| 19 | 0.0020 | 0.0290 | 0.4595 | 0.5591 | 0.951 | 0.9759 |
| 20 | 0.0022 | 0.0289 | 0.4642 | 0.5591 | 0.953 | 0.9755 |
| 21 | 0.0014 | 0.0319 | 0.7524 | 0.5620 | 0.968 | 0.9758 |
| 22 | 0.0015 | 0.0319 | 0.4176 | 0.5620 | 0.968 | 0.9760 |
| 23 | 0.0015 | 0.0324 | 0.3283 | 0.5625 | 0.970 | 0.9755 |
| 24 | 0.0019 | 0.0325 | 0.3415 | 0.5625 | 0.971 | 0.9755 |
| 25 | 0.0021 | 0.0337 | 0.6898 | 0.5635 | 0.971 | 0.9749 |
| 26 | 0.0005 | 0.0348 | 0.1723 | 0.5640 | 0.979 | 0.9716 |
| 27 | -0.0000 | 0.0371 | 0.0371 | 0.5659 | 0.999 | 0.9702 |

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

The belief-state probe achieves its highest R² at layer 5 (R²=0.9964). 
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
