# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `mess3` |
| Process Parameters | `{'a': 0.05, 'x': 0.02}` |
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
| Lowest KL(HMM \|\| tuned) | 1 | 0.0563 |
| Highest belief-state R² | 4 | 0.9669 |
| Highest top-1 agreement (tuned) | 27 | 0.996 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0123 |
| Late (last third) | 18–27 | 0.0031 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0142 | 0.0576 | inf | 1.0225 | 0.813 | 0.9277 |
| 1 | 0.0120 | 0.0563 | inf | 1.0205 | 0.825 | 0.9462 |
| 2 | 0.0112 | 0.0569 | inf | 1.0215 | 0.836 | 0.9548 |
| 3 | 0.0105 | 0.0567 | 2.8810 | 1.0215 | 0.840 | 0.9627 |
| 4 | 0.0103 | 0.0565 | 1.7438 | 1.0215 | 0.837 | 0.9669 |
| 5 | 0.0112 | 0.0582 | 1.2924 | 1.0225 | 0.841 | 0.9648 |
| 6 | 0.0118 | 0.0576 | 0.7928 | 1.0225 | 0.828 | 0.9580 |
| 7 | 0.0122 | 0.0578 | 1.0086 | 1.0225 | 0.823 | 0.9530 |
| 8 | 0.0143 | 0.0598 | 0.6933 | 1.0234 | 0.816 | 0.9430 |
| 9 | 0.0149 | 0.0604 | 0.4639 | 1.0234 | 0.807 | 0.9302 |
| 10 | 0.0168 | 0.0622 | 0.3801 | 1.0264 | 0.793 | 0.9164 |
| 11 | 0.0180 | 0.0628 | 0.4278 | 1.0264 | 0.785 | 0.9119 |
| 12 | 0.0170 | 0.0609 | 0.4299 | 1.0254 | 0.787 | 0.9017 |
| 13 | 0.0156 | 0.0611 | 0.5424 | 1.0254 | 0.794 | 0.9027 |
| 14 | 0.0099 | 0.0637 | 0.5173 | 1.0293 | 0.848 | 0.9057 |
| 15 | 0.0096 | 0.0648 | 0.5384 | 1.0303 | 0.850 | 0.8899 |
| 16 | 0.0077 | 0.0642 | 0.8809 | 1.0293 | 0.868 | 0.8933 |
| 17 | 0.0061 | 0.0644 | 0.7213 | 1.0293 | 0.885 | 0.8960 |
| 18 | 0.0057 | 0.0635 | 0.9233 | 1.0283 | 0.894 | 0.8959 |
| 19 | 0.0054 | 0.0639 | 0.6921 | 1.0293 | 0.897 | 0.9003 |
| 20 | 0.0048 | 0.0636 | 0.7051 | 1.0283 | 0.900 | 0.8946 |
| 21 | 0.0031 | 0.0645 | 1.0741 | 1.0293 | 0.932 | 0.9043 |
| 22 | 0.0027 | 0.0636 | 0.7281 | 1.0283 | 0.934 | 0.9068 |
| 23 | 0.0028 | 0.0639 | 0.5215 | 1.0283 | 0.935 | 0.9052 |
| 24 | 0.0026 | 0.0643 | 0.5012 | 1.0293 | 0.936 | 0.9022 |
| 25 | 0.0025 | 0.0663 | 0.9048 | 1.0303 | 0.935 | 0.8967 |
| 26 | 0.0010 | 0.0673 | 0.2967 | 1.0312 | 0.954 | 0.8945 |
| 27 | 0.0000 | 0.0701 | 0.0701 | 1.0332 | 0.996 | 0.8709 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9669). 
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
