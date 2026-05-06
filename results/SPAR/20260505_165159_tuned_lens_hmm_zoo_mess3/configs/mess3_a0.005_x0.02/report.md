# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `mess3` |
| Process Parameters | `{'a': 0.005, 'x': 0.02}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0702 |
| Highest belief-state R² | 4 | 0.9859 |
| Highest top-1 agreement (tuned) | 27 | 0.996 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0190 |
| Late (last third) | 18–27 | 0.0043 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0237 | 0.0762 | inf | 0.9404 | 0.804 | 0.9310 |
| 1 | 0.0193 | 0.0725 | inf | 0.9365 | 0.824 | 0.9592 |
| 2 | 0.0180 | 0.0733 | inf | 0.9370 | 0.839 | 0.9701 |
| 3 | 0.0166 | 0.0716 | 2.6817 | 0.9355 | 0.836 | 0.9806 |
| 4 | 0.0166 | 0.0702 | 1.6449 | 0.9341 | 0.838 | 0.9859 |
| 5 | 0.0159 | 0.0720 | 1.2331 | 0.9355 | 0.841 | 0.9853 |
| 6 | 0.0183 | 0.0733 | 0.7726 | 0.9370 | 0.830 | 0.9811 |
| 7 | 0.0185 | 0.0739 | 1.0315 | 0.9375 | 0.825 | 0.9774 |
| 8 | 0.0196 | 0.0762 | 0.7351 | 0.9395 | 0.819 | 0.9702 |
| 9 | 0.0229 | 0.0793 | 0.5198 | 0.9424 | 0.813 | 0.9618 |
| 10 | 0.0245 | 0.0819 | 0.4362 | 0.9463 | 0.798 | 0.9530 |
| 11 | 0.0247 | 0.0824 | 0.4914 | 0.9458 | 0.791 | 0.9506 |
| 12 | 0.0231 | 0.0818 | 0.4647 | 0.9443 | 0.796 | 0.9448 |
| 13 | 0.0217 | 0.0816 | 0.5672 | 0.9448 | 0.800 | 0.9469 |
| 14 | 0.0149 | 0.0849 | 0.5388 | 0.9492 | 0.841 | 0.9459 |
| 15 | 0.0130 | 0.0850 | 0.5364 | 0.9487 | 0.855 | 0.9395 |
| 16 | 0.0105 | 0.0844 | 0.8243 | 0.9478 | 0.871 | 0.9444 |
| 17 | 0.0089 | 0.0845 | 0.6878 | 0.9482 | 0.880 | 0.9469 |
| 18 | 0.0077 | 0.0837 | 0.8752 | 0.9478 | 0.893 | 0.9470 |
| 19 | 0.0073 | 0.0839 | 0.6624 | 0.9487 | 0.897 | 0.9460 |
| 20 | 0.0068 | 0.0830 | 0.6715 | 0.9473 | 0.902 | 0.9466 |
| 21 | 0.0043 | 0.0836 | 1.0638 | 0.9473 | 0.929 | 0.9474 |
| 22 | 0.0037 | 0.0819 | 0.7142 | 0.9458 | 0.936 | 0.9492 |
| 23 | 0.0038 | 0.0823 | 0.5208 | 0.9458 | 0.937 | 0.9472 |
| 24 | 0.0038 | 0.0829 | 0.5124 | 0.9468 | 0.935 | 0.9448 |
| 25 | 0.0039 | 0.0862 | 0.9401 | 0.9497 | 0.936 | 0.9447 |
| 26 | 0.0013 | 0.0876 | 0.3019 | 0.9512 | 0.953 | 0.9401 |
| 27 | 0.0000 | 0.0896 | 0.0896 | 0.9526 | 0.996 | 0.9315 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9859). 
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
