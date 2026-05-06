# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `arch` |
| Process Parameters | `{'a': 0.9}` |
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
| Lowest KL(HMM \|\| tuned) | 0 | 0.0242 |
| Highest belief-state R² | 4 | 0.9278 |
| Highest top-1 agreement (tuned) | 27 | 0.995 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0070 |
| Late (last third) | 18–27 | 0.0018 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0087 | 0.0242 | inf | 1.0215 | 0.825 | 0.8249 |
| 1 | 0.0072 | 0.0261 | inf | 1.0234 | 0.840 | 0.8847 |
| 2 | 0.0066 | 0.0258 | inf | 1.0234 | 0.846 | 0.9003 |
| 3 | 0.0065 | 0.0261 | 2.7677 | 1.0234 | 0.850 | 0.9152 |
| 4 | 0.0064 | 0.0263 | 1.6280 | 1.0234 | 0.846 | 0.9278 |
| 5 | 0.0065 | 0.0265 | 1.2384 | 1.0234 | 0.848 | 0.9273 |
| 6 | 0.0067 | 0.0263 | 0.7504 | 1.0234 | 0.838 | 0.9198 |
| 7 | 0.0074 | 0.0267 | 0.8254 | 1.0234 | 0.835 | 0.9118 |
| 8 | 0.0070 | 0.0265 | 0.5926 | 1.0244 | 0.834 | 0.8956 |
| 9 | 0.0074 | 0.0267 | 0.5066 | 1.0234 | 0.823 | 0.8824 |
| 10 | 0.0087 | 0.0266 | 0.3948 | 1.0234 | 0.814 | 0.8714 |
| 11 | 0.0085 | 0.0262 | 0.4705 | 1.0234 | 0.807 | 0.8698 |
| 12 | 0.0084 | 0.0266 | 0.4904 | 1.0244 | 0.813 | 0.8597 |
| 13 | 0.0087 | 0.0272 | 0.5757 | 1.0244 | 0.812 | 0.8587 |
| 14 | 0.0050 | 0.0288 | 0.3979 | 1.0273 | 0.862 | 0.8491 |
| 15 | 0.0049 | 0.0284 | 0.4384 | 1.0273 | 0.869 | 0.8346 |
| 16 | 0.0041 | 0.0293 | 0.8606 | 1.0273 | 0.880 | 0.8392 |
| 17 | 0.0036 | 0.0297 | 0.6886 | 1.0273 | 0.883 | 0.8315 |
| 18 | 0.0032 | 0.0297 | 0.8551 | 1.0273 | 0.892 | 0.8372 |
| 19 | 0.0031 | 0.0299 | 0.6365 | 1.0273 | 0.896 | 0.8364 |
| 20 | 0.0027 | 0.0301 | 0.6918 | 1.0273 | 0.899 | 0.8359 |
| 21 | 0.0017 | 0.0307 | 0.9641 | 1.0283 | 0.933 | 0.8366 |
| 22 | 0.0017 | 0.0305 | 0.6751 | 1.0283 | 0.933 | 0.8383 |
| 23 | 0.0019 | 0.0308 | 0.4604 | 1.0283 | 0.935 | 0.8335 |
| 24 | 0.0016 | 0.0309 | 0.4070 | 1.0283 | 0.942 | 0.8279 |
| 25 | 0.0015 | 0.0322 | 0.7320 | 1.0293 | 0.943 | 0.8172 |
| 26 | 0.0010 | 0.0334 | 0.2696 | 1.0312 | 0.952 | 0.7941 |
| 27 | 0.0000 | 0.0377 | 0.0377 | 1.0352 | 0.995 | 0.7817 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9278). 
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
