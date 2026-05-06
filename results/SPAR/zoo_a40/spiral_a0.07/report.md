# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.07}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0290 |
| Highest belief-state R² | 7 | 0.9456 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0092 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0141 | 0.0314 | inf | 0.3203 | 0.996 | 0.5879 |
| 1 | 0.0122 | 0.0292 | inf | 0.3184 | 0.996 | 0.7440 |
| 2 | 0.0109 | 0.0301 | inf | 0.3196 | 0.996 | 0.7438 |
| 3 | 0.0103 | 0.0290 | 1.1213 | 0.3184 | 0.996 | 0.8087 |
| 4 | 0.0090 | 0.0292 | 0.6003 | 0.3186 | 0.996 | 0.8525 |
| 5 | 0.0074 | 0.0305 | 0.4825 | 0.3196 | 0.997 | 0.9137 |
| 6 | 0.0070 | 0.0310 | 0.4913 | 0.3203 | 0.997 | 0.9362 |
| 7 | 0.0075 | 0.0303 | 0.8535 | 0.3193 | 0.997 | 0.9456 |
| 8 | 0.0062 | 0.0314 | 0.7730 | 0.3210 | 0.996 | 0.9392 |
| 9 | 0.0076 | 0.0318 | 0.2075 | 0.3215 | 0.996 | 0.9301 |
| 10 | 0.0051 | 0.0340 | 0.2213 | 0.3237 | 0.997 | 0.9196 |
| 11 | 0.0071 | 0.0322 | 0.2366 | 0.3218 | 0.997 | 0.9134 |
| 12 | 0.0044 | 0.0329 | 0.1634 | 0.3228 | 0.996 | 0.9060 |
| 13 | 0.0050 | 0.0325 | 0.0966 | 0.3223 | 0.996 | 0.9075 |
| 14 | 0.0018 | 0.0346 | 0.1996 | 0.3250 | 0.997 | 0.9026 |
| 15 | 0.0019 | 0.0351 | 0.1748 | 0.3257 | 0.997 | 0.8973 |
| 16 | 0.0017 | 0.0351 | 0.1505 | 0.3257 | 0.997 | 0.8965 |
| 17 | 0.0016 | 0.0349 | 0.1688 | 0.3254 | 0.997 | 0.8934 |
| 18 | 0.0014 | 0.0351 | 0.1874 | 0.3254 | 0.998 | 0.8929 |
| 19 | 0.0019 | 0.0349 | 0.1587 | 0.3254 | 0.998 | 0.8950 |
| 20 | 0.0019 | 0.0348 | 0.1705 | 0.3252 | 0.998 | 0.8931 |
| 21 | 0.0015 | 0.0350 | 0.3652 | 0.3254 | 0.998 | 0.8922 |
| 22 | 0.0016 | 0.0349 | 0.1900 | 0.3254 | 0.998 | 0.8956 |
| 23 | 0.0015 | 0.0350 | 0.1423 | 0.3254 | 0.998 | 0.8925 |
| 24 | 0.0014 | 0.0348 | 0.1698 | 0.3254 | 0.998 | 0.8918 |
| 25 | 0.0018 | 0.0353 | 0.4959 | 0.3257 | 0.998 | 0.8921 |
| 26 | 0.0022 | 0.0359 | 0.0659 | 0.3262 | 0.998 | 0.8842 |
| 27 | 0.0000 | 0.0392 | 0.0392 | 0.3301 | 1.000 | 0.8770 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.9456). 
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
