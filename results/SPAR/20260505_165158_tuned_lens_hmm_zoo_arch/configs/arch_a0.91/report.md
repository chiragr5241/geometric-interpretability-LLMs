# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `arch` |
| Process Parameters | `{'a': 0.91}` |
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
| Lowest KL(HMM \|\| tuned) | 0 | 0.0248 |
| Highest belief-state R² | 4 | 0.9277 |
| Highest top-1 agreement (tuned) | 27 | 0.996 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0074 |
| Late (last third) | 18–27 | 0.0018 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0091 | 0.0248 | inf | 1.0186 | 0.834 | 0.8186 |
| 1 | 0.0077 | 0.0266 | inf | 1.0205 | 0.846 | 0.8819 |
| 2 | 0.0069 | 0.0267 | inf | 1.0195 | 0.850 | 0.8996 |
| 3 | 0.0067 | 0.0268 | 2.7589 | 1.0195 | 0.855 | 0.9142 |
| 4 | 0.0066 | 0.0270 | 1.6269 | 1.0205 | 0.853 | 0.9277 |
| 5 | 0.0067 | 0.0274 | 1.2392 | 1.0205 | 0.856 | 0.9265 |
| 6 | 0.0073 | 0.0272 | 0.7515 | 1.0205 | 0.845 | 0.9188 |
| 7 | 0.0074 | 0.0272 | 0.8322 | 1.0195 | 0.844 | 0.9116 |
| 8 | 0.0075 | 0.0276 | 0.5991 | 1.0205 | 0.842 | 0.8948 |
| 9 | 0.0080 | 0.0279 | 0.5122 | 1.0205 | 0.832 | 0.8825 |
| 10 | 0.0090 | 0.0282 | 0.4009 | 1.0215 | 0.822 | 0.8700 |
| 11 | 0.0087 | 0.0273 | 0.4800 | 1.0205 | 0.822 | 0.8663 |
| 12 | 0.0088 | 0.0280 | 0.4957 | 1.0215 | 0.823 | 0.8604 |
| 13 | 0.0089 | 0.0289 | 0.5813 | 1.0215 | 0.824 | 0.8571 |
| 14 | 0.0052 | 0.0303 | 0.3989 | 1.0244 | 0.871 | 0.8468 |
| 15 | 0.0049 | 0.0300 | 0.4368 | 1.0244 | 0.879 | 0.8298 |
| 16 | 0.0041 | 0.0310 | 0.8541 | 1.0254 | 0.896 | 0.8362 |
| 17 | 0.0035 | 0.0311 | 0.6842 | 1.0244 | 0.896 | 0.8288 |
| 18 | 0.0032 | 0.0312 | 0.8508 | 1.0254 | 0.903 | 0.8331 |
| 19 | 0.0029 | 0.0313 | 0.6338 | 1.0254 | 0.908 | 0.8321 |
| 20 | 0.0026 | 0.0316 | 0.6850 | 1.0254 | 0.911 | 0.8287 |
| 21 | 0.0017 | 0.0322 | 0.9617 | 1.0254 | 0.940 | 0.8276 |
| 22 | 0.0017 | 0.0323 | 0.6739 | 1.0254 | 0.940 | 0.8353 |
| 23 | 0.0018 | 0.0324 | 0.4611 | 1.0254 | 0.940 | 0.8326 |
| 24 | 0.0016 | 0.0325 | 0.4082 | 1.0254 | 0.945 | 0.8325 |
| 25 | 0.0015 | 0.0339 | 0.7360 | 1.0273 | 0.944 | 0.8237 |
| 26 | 0.0014 | 0.0352 | 0.2748 | 1.0283 | 0.951 | 0.8194 |
| 27 | 0.0000 | 0.0389 | 0.0389 | 1.0322 | 0.996 | 0.7944 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9277). 
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
