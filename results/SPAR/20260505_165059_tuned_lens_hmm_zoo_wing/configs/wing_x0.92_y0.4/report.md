# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `wing` |
| Process Parameters | `{'x': 0.92, 'y': 0.4}` |
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
| Lowest KL(HMM \|\| tuned) | 3 | 0.0189 |
| Highest belief-state R² | 5 | 0.9922 |
| Highest top-1 agreement (tuned) | 27 | 1.000 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0075 |
| Late (last third) | 18–27 | 0.0016 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0105 | 0.0196 | inf | 0.3992 | 0.970 | 0.8776 |
| 1 | 0.0099 | 0.0197 | inf | 0.3987 | 0.973 | 0.8920 |
| 2 | 0.0084 | 0.0211 | inf | 0.4001 | 0.979 | 0.9203 |
| 3 | 0.0088 | 0.0189 | 1.1565 | 0.3977 | 0.979 | 0.9781 |
| 4 | 0.0084 | 0.0197 | 0.6733 | 0.3987 | 0.976 | 0.9891 |
| 5 | 0.0066 | 0.0211 | 0.4636 | 0.3997 | 0.979 | 0.9922 |
| 6 | 0.0054 | 0.0227 | 0.3933 | 0.4014 | 0.978 | 0.9920 |
| 7 | 0.0055 | 0.0232 | 0.8818 | 0.4019 | 0.978 | 0.9917 |
| 8 | 0.0064 | 0.0224 | 0.6183 | 0.4014 | 0.976 | 0.9901 |
| 9 | 0.0052 | 0.0225 | 0.1773 | 0.4014 | 0.976 | 0.9877 |
| 10 | 0.0053 | 0.0231 | 0.1507 | 0.4019 | 0.976 | 0.9850 |
| 11 | 0.0055 | 0.0227 | 0.2100 | 0.4011 | 0.976 | 0.9842 |
| 12 | 0.0055 | 0.0232 | 0.1568 | 0.4021 | 0.975 | 0.9817 |
| 13 | 0.0056 | 0.0234 | 0.1454 | 0.4023 | 0.975 | 0.9810 |
| 14 | 0.0030 | 0.0253 | 0.2445 | 0.4041 | 0.985 | 0.9795 |
| 15 | 0.0025 | 0.0255 | 0.2362 | 0.4041 | 0.984 | 0.9778 |
| 16 | 0.0028 | 0.0253 | 0.2802 | 0.4038 | 0.985 | 0.9763 |
| 17 | 0.0019 | 0.0256 | 0.2448 | 0.4041 | 0.986 | 0.9760 |
| 18 | 0.0020 | 0.0259 | 0.3166 | 0.4043 | 0.987 | 0.9760 |
| 19 | 0.0020 | 0.0259 | 0.2450 | 0.4043 | 0.987 | 0.9755 |
| 20 | 0.0020 | 0.0255 | 0.2384 | 0.4038 | 0.988 | 0.9753 |
| 21 | 0.0015 | 0.0267 | 0.4702 | 0.4050 | 0.992 | 0.9747 |
| 22 | 0.0018 | 0.0265 | 0.2544 | 0.4050 | 0.992 | 0.9745 |
| 23 | 0.0019 | 0.0267 | 0.1940 | 0.4053 | 0.992 | 0.9739 |
| 24 | 0.0022 | 0.0270 | 0.2169 | 0.4053 | 0.993 | 0.9726 |
| 25 | 0.0024 | 0.0275 | 0.5422 | 0.4055 | 0.993 | 0.9721 |
| 26 | 0.0003 | 0.0301 | 0.1023 | 0.4077 | 0.996 | 0.9691 |
| 27 | -0.0000 | 0.0339 | 0.0339 | 0.4106 | 1.000 | 0.9682 |

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

The belief-state probe achieves its highest R² at layer 5 (R²=0.9922). 
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
