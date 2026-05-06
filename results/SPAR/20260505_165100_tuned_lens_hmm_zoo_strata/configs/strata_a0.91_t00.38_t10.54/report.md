# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `strata` |
| Process Parameters | `{'a': 0.91, 't0': 0.38, 't1': 0.54}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0231 |
| Highest belief-state R² | 4 | 0.9980 |
| Highest top-1 agreement (tuned) | 27 | 0.999 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0078 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0109 | 0.0231 | inf | 0.5537 | 0.879 | 0.9415 |
| 1 | 0.0101 | 0.0235 | inf | 0.5532 | 0.890 | 0.9739 |
| 2 | 0.0085 | 0.0240 | inf | 0.5547 | 0.901 | 0.9893 |
| 3 | 0.0079 | 0.0231 | 1.7755 | 0.5537 | 0.903 | 0.9968 |
| 4 | 0.0081 | 0.0234 | 1.0758 | 0.5542 | 0.903 | 0.9980 |
| 5 | 0.0062 | 0.0242 | 0.6830 | 0.5547 | 0.913 | 0.9979 |
| 6 | 0.0060 | 0.0244 | 0.4378 | 0.5552 | 0.908 | 0.9975 |
| 7 | 0.0070 | 0.0252 | 0.8606 | 0.5557 | 0.907 | 0.9970 |
| 8 | 0.0073 | 0.0252 | 0.4820 | 0.5562 | 0.907 | 0.9954 |
| 9 | 0.0061 | 0.0263 | 0.2103 | 0.5571 | 0.903 | 0.9934 |
| 10 | 0.0064 | 0.0269 | 0.1962 | 0.5576 | 0.901 | 0.9910 |
| 11 | 0.0067 | 0.0275 | 0.2340 | 0.5581 | 0.901 | 0.9899 |
| 12 | 0.0062 | 0.0244 | 0.2172 | 0.5552 | 0.900 | 0.9879 |
| 13 | 0.0067 | 0.0251 | 0.2686 | 0.5557 | 0.904 | 0.9871 |
| 14 | 0.0033 | 0.0277 | 0.3509 | 0.5591 | 0.945 | 0.9851 |
| 15 | 0.0029 | 0.0267 | 0.4117 | 0.5586 | 0.942 | 0.9826 |
| 16 | 0.0032 | 0.0266 | 0.5187 | 0.5581 | 0.948 | 0.9828 |
| 17 | 0.0022 | 0.0274 | 0.4198 | 0.5591 | 0.954 | 0.9846 |
| 18 | 0.0020 | 0.0269 | 0.5699 | 0.5586 | 0.955 | 0.9832 |
| 19 | 0.0019 | 0.0275 | 0.4422 | 0.5591 | 0.957 | 0.9821 |
| 20 | 0.0021 | 0.0275 | 0.4486 | 0.5591 | 0.959 | 0.9819 |
| 21 | 0.0014 | 0.0300 | 0.7466 | 0.5615 | 0.969 | 0.9821 |
| 22 | 0.0014 | 0.0299 | 0.4114 | 0.5615 | 0.970 | 0.9824 |
| 23 | 0.0016 | 0.0302 | 0.3236 | 0.5615 | 0.971 | 0.9820 |
| 24 | 0.0016 | 0.0303 | 0.3407 | 0.5615 | 0.974 | 0.9820 |
| 25 | 0.0022 | 0.0316 | 0.7121 | 0.5630 | 0.973 | 0.9803 |
| 26 | 0.0004 | 0.0338 | 0.1665 | 0.5649 | 0.981 | 0.9781 |
| 27 | -0.0000 | 0.0350 | 0.0350 | 0.5654 | 0.999 | 0.9777 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9980). 
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
