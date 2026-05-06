# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.01}` |
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
| Lowest KL(HMM \|\| tuned) | 5 | 0.0598 |
| Highest belief-state R² | 7 | 0.8204 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0135 |
| Late (last third) | 18–27 | 0.0016 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0214 | 0.0698 | inf | 0.2891 | 0.994 | 0.3292 |
| 1 | 0.0176 | 0.0640 | inf | 0.2842 | 0.994 | 0.4505 |
| 2 | 0.0165 | 0.0650 | inf | 0.2852 | 0.994 | 0.4444 |
| 3 | 0.0158 | 0.0628 | 1.0948 | 0.2830 | 0.994 | 0.5028 |
| 4 | 0.0147 | 0.0613 | 0.6121 | 0.2812 | 0.994 | 0.5655 |
| 5 | 0.0116 | 0.0598 | 0.5240 | 0.2800 | 0.995 | 0.7040 |
| 6 | 0.0104 | 0.0610 | 0.5612 | 0.2812 | 0.995 | 0.7807 |
| 7 | 0.0104 | 0.0627 | 0.8982 | 0.2830 | 0.995 | 0.8204 |
| 8 | 0.0091 | 0.0605 | 0.8643 | 0.2805 | 0.994 | 0.8119 |
| 9 | 0.0075 | 0.0609 | 0.2863 | 0.2812 | 0.995 | 0.7941 |
| 10 | 0.0071 | 0.0633 | 0.3174 | 0.2834 | 0.995 | 0.7697 |
| 11 | 0.0073 | 0.0637 | 0.3441 | 0.2839 | 0.996 | 0.7579 |
| 12 | 0.0057 | 0.0621 | 0.2657 | 0.2820 | 0.995 | 0.7385 |
| 13 | 0.0056 | 0.0620 | 0.1624 | 0.2817 | 0.995 | 0.7427 |
| 14 | 0.0020 | 0.0646 | 0.2375 | 0.2849 | 0.996 | 0.7333 |
| 15 | 0.0020 | 0.0646 | 0.2125 | 0.2849 | 0.997 | 0.7254 |
| 16 | 0.0018 | 0.0644 | 0.1592 | 0.2847 | 0.997 | 0.7265 |
| 17 | 0.0019 | 0.0649 | 0.1930 | 0.2852 | 0.997 | 0.7256 |
| 18 | 0.0016 | 0.0647 | 0.1724 | 0.2852 | 0.997 | 0.7236 |
| 19 | 0.0020 | 0.0648 | 0.1599 | 0.2852 | 0.997 | 0.7251 |
| 20 | 0.0019 | 0.0645 | 0.1755 | 0.2849 | 0.997 | 0.7242 |
| 21 | 0.0015 | 0.0650 | 0.3415 | 0.2856 | 0.998 | 0.7221 |
| 22 | 0.0015 | 0.0649 | 0.1920 | 0.2854 | 0.997 | 0.7313 |
| 23 | 0.0019 | 0.0648 | 0.1524 | 0.2854 | 0.997 | 0.7281 |
| 24 | 0.0022 | 0.0647 | 0.1756 | 0.2852 | 0.997 | 0.7201 |
| 25 | 0.0021 | 0.0651 | 0.4637 | 0.2854 | 0.998 | 0.7184 |
| 26 | 0.0016 | 0.0654 | 0.0872 | 0.2856 | 0.998 | 0.7143 |
| 27 | 0.0000 | 0.0699 | 0.0699 | 0.2903 | 1.000 | 0.7062 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.8204). 
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
