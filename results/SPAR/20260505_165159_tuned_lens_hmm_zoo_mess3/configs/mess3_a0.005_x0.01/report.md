# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `mess3` |
| Process Parameters | `{'a': 0.005, 'x': 0.01}` |
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
| Lowest KL(HMM \|\| tuned) | 4 | 0.0612 |
| Highest belief-state R² | 4 | 0.9835 |
| Highest top-1 agreement (tuned) | 27 | 0.997 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0256 |
| Late (last third) | 18–27 | 0.0066 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0324 | 0.0703 | inf | 0.8804 | 0.765 | 0.9402 |
| 1 | 0.0265 | 0.0641 | inf | 0.8735 | 0.791 | 0.9617 |
| 2 | 0.0241 | 0.0660 | inf | 0.8755 | 0.806 | 0.9707 |
| 3 | 0.0229 | 0.0637 | 2.6814 | 0.8730 | 0.798 | 0.9794 |
| 4 | 0.0223 | 0.0612 | 1.6760 | 0.8711 | 0.797 | 0.9835 |
| 5 | 0.0214 | 0.0636 | 1.2804 | 0.8726 | 0.802 | 0.9825 |
| 6 | 0.0240 | 0.0655 | 0.8043 | 0.8740 | 0.795 | 0.9795 |
| 7 | 0.0238 | 0.0659 | 1.0702 | 0.8745 | 0.792 | 0.9755 |
| 8 | 0.0285 | 0.0714 | 0.7813 | 0.8789 | 0.785 | 0.9680 |
| 9 | 0.0298 | 0.0741 | 0.5735 | 0.8823 | 0.778 | 0.9594 |
| 10 | 0.0332 | 0.0786 | 0.4915 | 0.8867 | 0.768 | 0.9505 |
| 11 | 0.0336 | 0.0769 | 0.5422 | 0.8848 | 0.750 | 0.9472 |
| 12 | 0.0347 | 0.0800 | 0.5062 | 0.8867 | 0.752 | 0.9401 |
| 13 | 0.0328 | 0.0791 | 0.6068 | 0.8867 | 0.752 | 0.9416 |
| 14 | 0.0228 | 0.0808 | 0.5605 | 0.8896 | 0.810 | 0.9429 |
| 15 | 0.0205 | 0.0806 | 0.5432 | 0.8896 | 0.817 | 0.9328 |
| 16 | 0.0161 | 0.0759 | 0.7896 | 0.8853 | 0.837 | 0.9374 |
| 17 | 0.0146 | 0.0754 | 0.6722 | 0.8843 | 0.844 | 0.9455 |
| 18 | 0.0126 | 0.0730 | 0.8517 | 0.8828 | 0.861 | 0.9459 |
| 19 | 0.0118 | 0.0728 | 0.6475 | 0.8823 | 0.866 | 0.9497 |
| 20 | 0.0100 | 0.0706 | 0.6551 | 0.8799 | 0.876 | 0.9492 |
| 21 | 0.0065 | 0.0696 | 1.0346 | 0.8789 | 0.906 | 0.9510 |
| 22 | 0.0058 | 0.0668 | 0.6805 | 0.8765 | 0.916 | 0.9546 |
| 23 | 0.0056 | 0.0667 | 0.4890 | 0.8765 | 0.918 | 0.9552 |
| 24 | 0.0058 | 0.0675 | 0.4742 | 0.8774 | 0.916 | 0.9544 |
| 25 | 0.0062 | 0.0698 | 0.8458 | 0.8794 | 0.913 | 0.9524 |
| 26 | 0.0020 | 0.0715 | 0.2789 | 0.8804 | 0.941 | 0.9504 |
| 27 | 0.0000 | 0.0741 | 0.0741 | 0.8838 | 0.997 | 0.9461 |

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
