# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `strata` |
| Process Parameters | `{'a': 0.92, 't0': 0.38, 't1': 0.54}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0239 |
| Highest belief-state R² | 4 | 0.9973 |
| Highest top-1 agreement (tuned) | 27 | 0.999 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0082 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0112 | 0.0239 | inf | 0.5527 | 0.872 | 0.9338 |
| 1 | 0.0109 | 0.0245 | inf | 0.5532 | 0.888 | 0.9694 |
| 2 | 0.0089 | 0.0249 | inf | 0.5542 | 0.897 | 0.9868 |
| 3 | 0.0082 | 0.0239 | 1.7852 | 0.5532 | 0.901 | 0.9958 |
| 4 | 0.0091 | 0.0243 | 1.0882 | 0.5537 | 0.900 | 0.9973 |
| 5 | 0.0063 | 0.0249 | 0.6886 | 0.5542 | 0.905 | 0.9973 |
| 6 | 0.0061 | 0.0254 | 0.4418 | 0.5547 | 0.903 | 0.9968 |
| 7 | 0.0083 | 0.0256 | 0.8636 | 0.5547 | 0.902 | 0.9963 |
| 8 | 0.0067 | 0.0257 | 0.4864 | 0.5552 | 0.905 | 0.9945 |
| 9 | 0.0063 | 0.0270 | 0.2146 | 0.5562 | 0.900 | 0.9921 |
| 10 | 0.0066 | 0.0263 | 0.1980 | 0.5557 | 0.896 | 0.9894 |
| 11 | 0.0070 | 0.0258 | 0.2395 | 0.5552 | 0.895 | 0.9881 |
| 12 | 0.0071 | 0.0242 | 0.2256 | 0.5542 | 0.899 | 0.9862 |
| 13 | 0.0072 | 0.0247 | 0.2789 | 0.5542 | 0.901 | 0.9849 |
| 14 | 0.0034 | 0.0277 | 0.3544 | 0.5576 | 0.941 | 0.9826 |
| 15 | 0.0030 | 0.0269 | 0.4158 | 0.5571 | 0.938 | 0.9802 |
| 16 | 0.0031 | 0.0268 | 0.5222 | 0.5571 | 0.944 | 0.9799 |
| 17 | 0.0022 | 0.0274 | 0.4225 | 0.5576 | 0.949 | 0.9810 |
| 18 | 0.0020 | 0.0273 | 0.5744 | 0.5576 | 0.952 | 0.9810 |
| 19 | 0.0020 | 0.0278 | 0.4465 | 0.5576 | 0.954 | 0.9802 |
| 20 | 0.0021 | 0.0278 | 0.4543 | 0.5581 | 0.957 | 0.9791 |
| 21 | 0.0013 | 0.0305 | 0.7464 | 0.5605 | 0.969 | 0.9792 |
| 22 | 0.0015 | 0.0303 | 0.4119 | 0.5601 | 0.969 | 0.9795 |
| 23 | 0.0015 | 0.0308 | 0.3232 | 0.5605 | 0.970 | 0.9794 |
| 24 | 0.0016 | 0.0308 | 0.3391 | 0.5605 | 0.973 | 0.9786 |
| 25 | 0.0016 | 0.0319 | 0.7010 | 0.5615 | 0.973 | 0.9775 |
| 26 | 0.0004 | 0.0330 | 0.1662 | 0.5625 | 0.978 | 0.9758 |
| 27 | -0.0000 | 0.0352 | 0.0352 | 0.5640 | 0.999 | 0.9749 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9973). 
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
