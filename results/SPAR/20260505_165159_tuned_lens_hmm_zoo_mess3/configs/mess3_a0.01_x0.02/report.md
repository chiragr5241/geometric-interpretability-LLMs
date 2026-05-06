# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `mess3` |
| Process Parameters | `{'a': 0.01, 'x': 0.02}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0686 |
| Highest belief-state R² | 4 | 0.9835 |
| Highest top-1 agreement (tuned) | 27 | 0.997 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0171 |
| Late (last third) | 18–27 | 0.0038 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0211 | 0.0738 | inf | 0.9536 | 0.807 | 0.9373 |
| 1 | 0.0172 | 0.0704 | inf | 0.9502 | 0.828 | 0.9620 |
| 2 | 0.0161 | 0.0714 | inf | 0.9512 | 0.840 | 0.9716 |
| 3 | 0.0150 | 0.0700 | 2.7063 | 0.9497 | 0.838 | 0.9790 |
| 4 | 0.0147 | 0.0686 | 1.6561 | 0.9487 | 0.839 | 0.9835 |
| 5 | 0.0148 | 0.0705 | 1.2404 | 0.9502 | 0.842 | 0.9830 |
| 6 | 0.0162 | 0.0713 | 0.7732 | 0.9512 | 0.832 | 0.9786 |
| 7 | 0.0167 | 0.0717 | 1.0255 | 0.9512 | 0.825 | 0.9755 |
| 8 | 0.0187 | 0.0750 | 0.7284 | 0.9546 | 0.822 | 0.9680 |
| 9 | 0.0207 | 0.0774 | 0.5101 | 0.9565 | 0.812 | 0.9600 |
| 10 | 0.0227 | 0.0808 | 0.4246 | 0.9614 | 0.801 | 0.9503 |
| 11 | 0.0223 | 0.0806 | 0.4763 | 0.9600 | 0.796 | 0.9478 |
| 12 | 0.0214 | 0.0801 | 0.4536 | 0.9595 | 0.796 | 0.9418 |
| 13 | 0.0195 | 0.0788 | 0.5587 | 0.9585 | 0.807 | 0.9432 |
| 14 | 0.0132 | 0.0815 | 0.5297 | 0.9609 | 0.850 | 0.9413 |
| 15 | 0.0119 | 0.0817 | 0.5331 | 0.9609 | 0.861 | 0.9317 |
| 16 | 0.0093 | 0.0803 | 0.8259 | 0.9600 | 0.881 | 0.9366 |
| 17 | 0.0075 | 0.0807 | 0.6880 | 0.9614 | 0.891 | 0.9398 |
| 18 | 0.0065 | 0.0798 | 0.8783 | 0.9604 | 0.903 | 0.9431 |
| 19 | 0.0064 | 0.0804 | 0.6626 | 0.9609 | 0.906 | 0.9435 |
| 20 | 0.0059 | 0.0797 | 0.6714 | 0.9604 | 0.905 | 0.9427 |
| 21 | 0.0040 | 0.0796 | 1.0583 | 0.9595 | 0.928 | 0.9423 |
| 22 | 0.0034 | 0.0781 | 0.7103 | 0.9585 | 0.935 | 0.9455 |
| 23 | 0.0034 | 0.0783 | 0.5167 | 0.9585 | 0.936 | 0.9446 |
| 24 | 0.0034 | 0.0793 | 0.5077 | 0.9595 | 0.935 | 0.9436 |
| 25 | 0.0035 | 0.0819 | 0.9332 | 0.9619 | 0.936 | 0.9425 |
| 26 | 0.0012 | 0.0833 | 0.3004 | 0.9634 | 0.956 | 0.9386 |
| 27 | 0.0000 | 0.0862 | 0.0862 | 0.9658 | 0.997 | 0.9319 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9835). 
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
