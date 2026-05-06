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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0382 |
| Highest belief-state R² | 7 | 0.9033 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0100 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0167 | 0.0428 | inf | 0.3083 | 0.996 | 0.4747 |
| 1 | 0.0139 | 0.0394 | inf | 0.3052 | 0.997 | 0.6231 |
| 2 | 0.0123 | 0.0401 | inf | 0.3059 | 0.997 | 0.6235 |
| 3 | 0.0114 | 0.0387 | 1.1083 | 0.3044 | 0.997 | 0.6898 |
| 4 | 0.0097 | 0.0382 | 0.6044 | 0.3040 | 0.998 | 0.7478 |
| 5 | 0.0081 | 0.0387 | 0.4988 | 0.3042 | 0.998 | 0.8473 |
| 6 | 0.0075 | 0.0392 | 0.5201 | 0.3047 | 0.998 | 0.8865 |
| 7 | 0.0076 | 0.0384 | 0.8624 | 0.3035 | 0.997 | 0.9033 |
| 8 | 0.0072 | 0.0395 | 0.8004 | 0.3052 | 0.997 | 0.8955 |
| 9 | 0.0053 | 0.0417 | 0.2331 | 0.3076 | 0.998 | 0.8833 |
| 10 | 0.0060 | 0.0401 | 0.2538 | 0.3062 | 0.997 | 0.8709 |
| 11 | 0.0054 | 0.0401 | 0.2678 | 0.3064 | 0.998 | 0.8642 |
| 12 | 0.0046 | 0.0408 | 0.1969 | 0.3071 | 0.997 | 0.8531 |
| 13 | 0.0047 | 0.0407 | 0.1177 | 0.3071 | 0.997 | 0.8566 |
| 14 | 0.0016 | 0.0431 | 0.2077 | 0.3096 | 0.998 | 0.8525 |
| 15 | 0.0017 | 0.0430 | 0.1800 | 0.3093 | 0.998 | 0.8453 |
| 16 | 0.0015 | 0.0430 | 0.1451 | 0.3093 | 0.998 | 0.8452 |
| 17 | 0.0013 | 0.0432 | 0.1676 | 0.3096 | 0.998 | 0.8445 |
| 18 | 0.0013 | 0.0430 | 0.1747 | 0.3093 | 0.998 | 0.8464 |
| 19 | 0.0015 | 0.0431 | 0.1533 | 0.3093 | 0.998 | 0.8477 |
| 20 | 0.0017 | 0.0428 | 0.1689 | 0.3091 | 0.998 | 0.8458 |
| 21 | 0.0015 | 0.0432 | 0.3518 | 0.3096 | 0.998 | 0.8439 |
| 22 | 0.0015 | 0.0431 | 0.1865 | 0.3096 | 0.998 | 0.8493 |
| 23 | 0.0016 | 0.0431 | 0.1417 | 0.3096 | 0.998 | 0.8479 |
| 24 | 0.0016 | 0.0430 | 0.1675 | 0.3093 | 0.998 | 0.8446 |
| 25 | 0.0022 | 0.0435 | 0.4770 | 0.3098 | 0.998 | 0.8421 |
| 26 | 0.0016 | 0.0437 | 0.0696 | 0.3098 | 0.998 | 0.8373 |
| 27 | 0.0000 | 0.0477 | 0.0477 | 0.3140 | 1.000 | 0.8286 |

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
