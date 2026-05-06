# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `wing` |
| Process Parameters | `{'x': 0.91, 'y': 0.4}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0181 |
| Highest belief-state R² | 5 | 0.9931 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0076 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0102 | 0.0185 | inf | 0.4124 | 0.973 | 0.8841 |
| 1 | 0.0096 | 0.0188 | inf | 0.4121 | 0.976 | 0.8984 |
| 2 | 0.0079 | 0.0200 | inf | 0.4136 | 0.981 | 0.9277 |
| 3 | 0.0082 | 0.0181 | 1.1996 | 0.4116 | 0.981 | 0.9814 |
| 4 | 0.0080 | 0.0190 | 0.6942 | 0.4126 | 0.979 | 0.9905 |
| 5 | 0.0061 | 0.0203 | 0.4756 | 0.4136 | 0.981 | 0.9931 |
| 6 | 0.0053 | 0.0219 | 0.3936 | 0.4153 | 0.980 | 0.9930 |
| 7 | 0.0066 | 0.0209 | 0.8757 | 0.4143 | 0.978 | 0.9930 |
| 8 | 0.0073 | 0.0217 | 0.6017 | 0.4153 | 0.979 | 0.9913 |
| 9 | 0.0067 | 0.0213 | 0.1752 | 0.4148 | 0.978 | 0.9892 |
| 10 | 0.0051 | 0.0232 | 0.1514 | 0.4170 | 0.978 | 0.9865 |
| 11 | 0.0061 | 0.0214 | 0.2080 | 0.4148 | 0.977 | 0.9858 |
| 12 | 0.0053 | 0.0225 | 0.1533 | 0.4163 | 0.977 | 0.9840 |
| 13 | 0.0058 | 0.0224 | 0.1434 | 0.4163 | 0.977 | 0.9824 |
| 14 | 0.0029 | 0.0250 | 0.2482 | 0.4187 | 0.986 | 0.9811 |
| 15 | 0.0026 | 0.0255 | 0.2392 | 0.4192 | 0.985 | 0.9778 |
| 16 | 0.0027 | 0.0251 | 0.2839 | 0.4189 | 0.986 | 0.9774 |
| 17 | 0.0019 | 0.0257 | 0.2474 | 0.4194 | 0.986 | 0.9771 |
| 18 | 0.0019 | 0.0257 | 0.3208 | 0.4197 | 0.987 | 0.9780 |
| 19 | 0.0020 | 0.0260 | 0.2474 | 0.4197 | 0.988 | 0.9763 |
| 20 | 0.0020 | 0.0254 | 0.2422 | 0.4192 | 0.988 | 0.9767 |
| 21 | 0.0014 | 0.0269 | 0.4791 | 0.4207 | 0.992 | 0.9759 |
| 22 | 0.0016 | 0.0266 | 0.2573 | 0.4204 | 0.992 | 0.9765 |
| 23 | 0.0016 | 0.0270 | 0.1952 | 0.4207 | 0.992 | 0.9758 |
| 24 | 0.0018 | 0.0272 | 0.2191 | 0.4209 | 0.993 | 0.9752 |
| 25 | 0.0013 | 0.0275 | 0.5561 | 0.4209 | 0.993 | 0.9733 |
| 26 | 0.0006 | 0.0293 | 0.1038 | 0.4221 | 0.995 | 0.9705 |
| 27 | -0.0000 | 0.0337 | 0.0337 | 0.4255 | 1.000 | 0.9685 |

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

The belief-state probe achieves its highest R² at layer 5 (R²=0.9931). 
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
