# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `mess3` |
| Process Parameters | `{'x': 0.05, 'a': 0.85}` |
| Vocabulary | `['A', 'B', 'C']` |
| Sequence Length | 2000 |
| Sequences (total) | 10 |
| Train / Test Split | 8 / 2 sequences |
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
- **Train/test split**: By sequence (first 8 sequences train,
  remaining 2 held out). No data leakage across sequences.
- **Belief-state probes**: Ordinary least squares (LinearRegression) from activations → beliefs,
  trained on the same train sequences, evaluated on the same test sequences.
- **LR schedule**: Cosine annealing over 50 epochs.

## Main Results

### Best layers

| Criterion | Layer | Value |
|-----------|-------|-------|
| Lowest KL(final \|\| tuned) | 26 | 0.0008 |
| Lowest KL(HMM \|\| tuned) | 0 | 0.0354 |
| Highest belief-state R² | 2 | 0.9990 |
| Highest top-1 agreement (tuned) | 27 | 0.994 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0093 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0115 | 0.0354 | inf | 0.8306 | 0.969 | 0.9934 |
| 1 | 0.0099 | 0.0389 | inf | 0.8340 | 0.974 | 0.9986 |
| 2 | 0.0087 | 0.0407 | 1.5486 | 0.8359 | 0.973 | 0.9990 |
| 3 | 0.0088 | 0.0398 | 0.9141 | 0.8340 | 0.974 | 0.9989 |
| 4 | 0.0089 | 0.0397 | 0.9383 | 0.8340 | 0.973 | 0.9990 |
| 5 | 0.0086 | 0.0405 | 0.6636 | 0.8350 | 0.973 | 0.9985 |
| 6 | 0.0090 | 0.0402 | 0.5302 | 0.8350 | 0.976 | 0.9980 |
| 7 | 0.0091 | 0.0406 | 0.5844 | 0.8354 | 0.973 | 0.9977 |
| 8 | 0.0091 | 0.0405 | 0.5334 | 0.8364 | 0.974 | 0.9965 |
| 9 | 0.0095 | 0.0424 | 0.7169 | 0.8379 | 0.973 | 0.9945 |
| 10 | 0.0095 | 0.0424 | 0.7730 | 0.8379 | 0.970 | 0.9933 |
| 11 | 0.0098 | 0.0412 | 0.8219 | 0.8345 | 0.969 | 0.9925 |
| 12 | 0.0089 | 0.0429 | 0.4939 | 0.8369 | 0.972 | 0.9910 |
| 13 | 0.0086 | 0.0416 | 0.2870 | 0.8345 | 0.972 | 0.9910 |
| 14 | 0.0051 | 0.0434 | 0.1612 | 0.8369 | 0.982 | 0.9902 |
| 15 | 0.0046 | 0.0459 | 0.1544 | 0.8389 | 0.983 | 0.9905 |
| 16 | 0.0037 | 0.0473 | 0.2212 | 0.8408 | 0.983 | 0.9895 |
| 17 | 0.0032 | 0.0482 | 0.1423 | 0.8408 | 0.984 | 0.9904 |
| 18 | 0.0026 | 0.0498 | 0.1355 | 0.8433 | 0.984 | 0.9907 |
| 19 | 0.0022 | 0.0501 | 0.1173 | 0.8433 | 0.986 | 0.9907 |
| 20 | 0.0023 | 0.0503 | 0.6732 | 0.8433 | 0.984 | 0.9903 |
| 21 | 0.0013 | 0.0489 | 0.6451 | 0.8428 | 0.989 | 0.9883 |
| 22 | 0.0011 | 0.0492 | 0.2700 | 0.8428 | 0.989 | 0.9891 |
| 23 | 0.0011 | 0.0491 | 0.1524 | 0.8433 | 0.991 | 0.9874 |
| 24 | 0.0012 | 0.0486 | 0.1675 | 0.8428 | 0.990 | 0.9884 |
| 25 | 0.0011 | 0.0492 | 0.6076 | 0.8433 | 0.989 | 0.9873 |
| 26 | 0.0008 | 0.0473 | 0.5724 | 0.8423 | 0.993 | 0.9877 |
| 27 | 0.0009 | 0.0446 | 0.0491 | 0.8394 | 0.994 | 0.9854 |

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

The belief-state probe achieves its highest R² at layer 2 (R²=0.9990). 
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
