# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.04}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0377 |
| Highest belief-state R² | 7 | 0.9033 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0102 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0166 | 0.0427 | inf | 0.3081 | 0.996 | 0.4748 |
| 1 | 0.0134 | 0.0392 | inf | 0.3049 | 0.996 | 0.6231 |
| 2 | 0.0125 | 0.0400 | inf | 0.3057 | 0.997 | 0.6236 |
| 3 | 0.0114 | 0.0386 | 1.1084 | 0.3044 | 0.997 | 0.6896 |
| 4 | 0.0109 | 0.0377 | 0.6044 | 0.3035 | 0.996 | 0.7476 |
| 5 | 0.0082 | 0.0383 | 0.4988 | 0.3037 | 0.998 | 0.8472 |
| 6 | 0.0075 | 0.0398 | 0.5203 | 0.3054 | 0.998 | 0.8864 |
| 7 | 0.0097 | 0.0382 | 0.8622 | 0.3032 | 0.997 | 0.9033 |
| 8 | 0.0065 | 0.0396 | 0.8005 | 0.3054 | 0.997 | 0.8954 |
| 9 | 0.0055 | 0.0405 | 0.2332 | 0.3064 | 0.998 | 0.8834 |
| 10 | 0.0052 | 0.0412 | 0.2539 | 0.3074 | 0.998 | 0.8708 |
| 11 | 0.0052 | 0.0409 | 0.2679 | 0.3074 | 0.998 | 0.8641 |
| 12 | 0.0048 | 0.0407 | 0.1971 | 0.3071 | 0.997 | 0.8531 |
| 13 | 0.0048 | 0.0406 | 0.1178 | 0.3071 | 0.997 | 0.8565 |
| 14 | 0.0016 | 0.0430 | 0.2078 | 0.3096 | 0.998 | 0.8526 |
| 15 | 0.0016 | 0.0431 | 0.1800 | 0.3093 | 0.998 | 0.8453 |
| 16 | 0.0015 | 0.0431 | 0.1451 | 0.3093 | 0.998 | 0.8449 |
| 17 | 0.0013 | 0.0432 | 0.1676 | 0.3096 | 0.998 | 0.8444 |
| 18 | 0.0014 | 0.0429 | 0.1747 | 0.3093 | 0.998 | 0.8462 |
| 19 | 0.0016 | 0.0430 | 0.1533 | 0.3093 | 0.998 | 0.8478 |
| 20 | 0.0018 | 0.0427 | 0.1689 | 0.3091 | 0.998 | 0.8454 |
| 21 | 0.0014 | 0.0432 | 0.3517 | 0.3096 | 0.998 | 0.8437 |
| 22 | 0.0015 | 0.0430 | 0.1865 | 0.3096 | 0.998 | 0.8492 |
| 23 | 0.0016 | 0.0432 | 0.1417 | 0.3096 | 0.998 | 0.8475 |
| 24 | 0.0016 | 0.0431 | 0.1675 | 0.3096 | 0.998 | 0.8447 |
| 25 | 0.0021 | 0.0435 | 0.4770 | 0.3098 | 0.998 | 0.8419 |
| 26 | 0.0013 | 0.0437 | 0.0696 | 0.3101 | 0.998 | 0.8371 |
| 27 | 0.0000 | 0.0477 | 0.0477 | 0.3140 | 1.000 | 0.8283 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9033). 
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
