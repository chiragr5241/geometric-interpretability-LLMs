# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `wing` |
| Process Parameters | `{'x': 0.98, 'y': 0.4}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0201 |
| Highest belief-state R² | 6 | 0.9831 |
| Highest top-1 agreement (tuned) | 27 | 0.999 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0100 |
| Late (last third) | 18–27 | 0.0018 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0152 | 0.0270 | inf | 0.3052 | 0.930 | 0.8419 |
| 1 | 0.0157 | 0.0247 | inf | 0.3025 | 0.933 | 0.8515 |
| 2 | 0.0119 | 0.0239 | inf | 0.3020 | 0.949 | 0.8729 |
| 3 | 0.0116 | 0.0211 | 0.9526 | 0.2986 | 0.951 | 0.9320 |
| 4 | 0.0112 | 0.0201 | 0.6017 | 0.2976 | 0.948 | 0.9606 |
| 5 | 0.0082 | 0.0212 | 0.4432 | 0.2988 | 0.953 | 0.9804 |
| 6 | 0.0066 | 0.0222 | 0.4846 | 0.2998 | 0.952 | 0.9831 |
| 7 | 0.0067 | 0.0240 | 1.0179 | 0.3020 | 0.951 | 0.9821 |
| 8 | 0.0069 | 0.0224 | 0.8408 | 0.3005 | 0.949 | 0.9799 |
| 9 | 0.0063 | 0.0244 | 0.2444 | 0.3030 | 0.949 | 0.9763 |
| 10 | 0.0070 | 0.0226 | 0.1541 | 0.3008 | 0.945 | 0.9726 |
| 11 | 0.0066 | 0.0238 | 0.2638 | 0.3013 | 0.945 | 0.9722 |
| 12 | 0.0076 | 0.0222 | 0.2456 | 0.3008 | 0.945 | 0.9684 |
| 13 | 0.0074 | 0.0226 | 0.1890 | 0.3005 | 0.945 | 0.9677 |
| 14 | 0.0038 | 0.0249 | 0.2353 | 0.3027 | 0.965 | 0.9650 |
| 15 | 0.0034 | 0.0247 | 0.2360 | 0.3027 | 0.964 | 0.9618 |
| 16 | 0.0035 | 0.0252 | 0.2651 | 0.3030 | 0.968 | 0.9619 |
| 17 | 0.0024 | 0.0253 | 0.2437 | 0.3035 | 0.971 | 0.9604 |
| 18 | 0.0026 | 0.0255 | 0.2995 | 0.3035 | 0.973 | 0.9601 |
| 19 | 0.0025 | 0.0255 | 0.2478 | 0.3035 | 0.975 | 0.9571 |
| 20 | 0.0025 | 0.0258 | 0.2316 | 0.3037 | 0.976 | 0.9570 |
| 21 | 0.0016 | 0.0269 | 0.3899 | 0.3044 | 0.984 | 0.9584 |
| 22 | 0.0018 | 0.0261 | 0.2191 | 0.3040 | 0.983 | 0.9574 |
| 23 | 0.0024 | 0.0265 | 0.1693 | 0.3042 | 0.983 | 0.9551 |
| 24 | 0.0022 | 0.0268 | 0.1761 | 0.3044 | 0.984 | 0.9543 |
| 25 | 0.0019 | 0.0271 | 0.3754 | 0.3044 | 0.985 | 0.9529 |
| 26 | 0.0006 | 0.0326 | 0.0988 | 0.3098 | 0.990 | 0.9496 |
| 27 | -0.0000 | 0.0333 | 0.0333 | 0.3096 | 0.999 | 0.9480 |

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

The belief-state probe achieves its highest R² at layer 6 (R²=0.9831). 
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
