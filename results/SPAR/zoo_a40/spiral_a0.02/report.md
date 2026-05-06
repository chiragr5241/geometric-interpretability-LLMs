# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `spiral` |
| Process Parameters | `{'a': 0.02}` |
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
| Lowest KL(HMM \|\| tuned) | 7 | 0.0490 |
| Highest belief-state R² | 7 | 0.8535 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0124 |
| Late (last third) | 18–27 | 0.0015 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0205 | 0.0572 | inf | 0.2964 | 0.994 | 0.3824 |
| 1 | 0.0161 | 0.0523 | inf | 0.2920 | 0.995 | 0.5181 |
| 2 | 0.0158 | 0.0532 | inf | 0.2930 | 0.995 | 0.5139 |
| 3 | 0.0144 | 0.0513 | 1.0972 | 0.2910 | 0.995 | 0.5781 |
| 4 | 0.0124 | 0.0511 | 0.6068 | 0.2910 | 0.996 | 0.6392 |
| 5 | 0.0105 | 0.0493 | 0.5127 | 0.2893 | 0.996 | 0.7626 |
| 6 | 0.0097 | 0.0510 | 0.5451 | 0.2908 | 0.996 | 0.8245 |
| 7 | 0.0098 | 0.0490 | 0.8823 | 0.2886 | 0.995 | 0.8535 |
| 8 | 0.0076 | 0.0507 | 0.8345 | 0.2900 | 0.995 | 0.8455 |
| 9 | 0.0067 | 0.0530 | 0.2593 | 0.2925 | 0.996 | 0.8302 |
| 10 | 0.0066 | 0.0537 | 0.2909 | 0.2932 | 0.996 | 0.8075 |
| 11 | 0.0068 | 0.0545 | 0.3162 | 0.2942 | 0.996 | 0.8027 |
| 12 | 0.0055 | 0.0528 | 0.2395 | 0.2922 | 0.995 | 0.7833 |
| 13 | 0.0060 | 0.0523 | 0.1420 | 0.2917 | 0.995 | 0.7867 |
| 14 | 0.0020 | 0.0546 | 0.2239 | 0.2939 | 0.997 | 0.7833 |
| 15 | 0.0020 | 0.0546 | 0.1961 | 0.2939 | 0.997 | 0.7757 |
| 16 | 0.0017 | 0.0544 | 0.1510 | 0.2937 | 0.997 | 0.7792 |
| 17 | 0.0017 | 0.0547 | 0.1794 | 0.2942 | 0.997 | 0.7765 |
| 18 | 0.0016 | 0.0545 | 0.1719 | 0.2939 | 0.997 | 0.7752 |
| 19 | 0.0021 | 0.0544 | 0.1556 | 0.2939 | 0.997 | 0.7765 |
| 20 | 0.0020 | 0.0543 | 0.1709 | 0.2939 | 0.997 | 0.7697 |
| 21 | 0.0016 | 0.0544 | 0.3468 | 0.2939 | 0.998 | 0.7724 |
| 22 | 0.0017 | 0.0544 | 0.1896 | 0.2939 | 0.997 | 0.7804 |
| 23 | 0.0018 | 0.0543 | 0.1470 | 0.2939 | 0.997 | 0.7765 |
| 24 | 0.0021 | 0.0542 | 0.1720 | 0.2937 | 0.997 | 0.7740 |
| 25 | 0.0018 | 0.0547 | 0.4679 | 0.2942 | 0.998 | 0.7729 |
| 26 | 0.0008 | 0.0548 | 0.0791 | 0.2944 | 0.998 | 0.7698 |
| 27 | 0.0000 | 0.0593 | 0.0593 | 0.2988 | 1.000 | 0.7578 |

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

The belief-state probe achieves its highest R² at layer 7 (R²=0.8535). 
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
