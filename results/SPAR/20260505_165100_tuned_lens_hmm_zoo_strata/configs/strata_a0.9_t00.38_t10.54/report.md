# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-3B` |
| HMM Process | `strata` |
| Process Parameters | `{'a': 0.9, 't0': 0.38, 't1': 0.54}` |
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
| Lowest KL(HMM \|\| tuned) | 0 | 0.0224 |
| Highest belief-state R² | 4 | 0.9983 |
| Highest top-1 agreement (tuned) | 27 | 0.999 |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | 0–9 | 0.0079 |
| Late (last third) | 18–27 | 0.0014 |

### Per-layer metrics table

| Layer | KL(final\|\|tuned) | KL(HMM\|\|tuned) | KL(HMM\|\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.0110 | 0.0224 | inf | 0.5557 | 0.881 | 0.9477 |
| 1 | 0.0104 | 0.0232 | inf | 0.5557 | 0.893 | 0.9776 |
| 2 | 0.0085 | 0.0236 | inf | 0.5571 | 0.904 | 0.9912 |
| 3 | 0.0079 | 0.0229 | 1.7739 | 0.5562 | 0.908 | 0.9974 |
| 4 | 0.0089 | 0.0231 | 1.0693 | 0.5562 | 0.906 | 0.9983 |
| 5 | 0.0063 | 0.0237 | 0.6806 | 0.5566 | 0.913 | 0.9982 |
| 6 | 0.0062 | 0.0239 | 0.4365 | 0.5571 | 0.909 | 0.9979 |
| 7 | 0.0064 | 0.0246 | 0.8600 | 0.5576 | 0.907 | 0.9974 |
| 8 | 0.0077 | 0.0248 | 0.4777 | 0.5581 | 0.908 | 0.9960 |
| 9 | 0.0062 | 0.0251 | 0.2070 | 0.5581 | 0.908 | 0.9941 |
| 10 | 0.0064 | 0.0252 | 0.1933 | 0.5581 | 0.902 | 0.9921 |
| 11 | 0.0069 | 0.0249 | 0.2284 | 0.5576 | 0.903 | 0.9911 |
| 12 | 0.0071 | 0.0236 | 0.2110 | 0.5566 | 0.903 | 0.9897 |
| 13 | 0.0067 | 0.0247 | 0.2609 | 0.5576 | 0.908 | 0.9885 |
| 14 | 0.0033 | 0.0275 | 0.3501 | 0.5610 | 0.949 | 0.9866 |
| 15 | 0.0030 | 0.0268 | 0.4091 | 0.5601 | 0.947 | 0.9852 |
| 16 | 0.0031 | 0.0264 | 0.5154 | 0.5596 | 0.950 | 0.9855 |
| 17 | 0.0022 | 0.0272 | 0.4173 | 0.5605 | 0.958 | 0.9862 |
| 18 | 0.0020 | 0.0270 | 0.5644 | 0.5605 | 0.958 | 0.9853 |
| 19 | 0.0020 | 0.0275 | 0.4372 | 0.5610 | 0.961 | 0.9842 |
| 20 | 0.0021 | 0.0273 | 0.4432 | 0.5610 | 0.963 | 0.9841 |
| 21 | 0.0014 | 0.0301 | 0.7429 | 0.5635 | 0.973 | 0.9844 |
| 22 | 0.0015 | 0.0300 | 0.4084 | 0.5635 | 0.972 | 0.9845 |
| 23 | 0.0015 | 0.0302 | 0.3207 | 0.5635 | 0.974 | 0.9847 |
| 24 | 0.0013 | 0.0304 | 0.3391 | 0.5640 | 0.976 | 0.9840 |
| 25 | 0.0016 | 0.0317 | 0.7172 | 0.5649 | 0.976 | 0.9826 |
| 26 | 0.0004 | 0.0330 | 0.1649 | 0.5659 | 0.981 | 0.9813 |
| 27 | -0.0000 | 0.0353 | 0.0353 | 0.5674 | 0.999 | 0.9801 |

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

The belief-state probe achieves its highest R² at layer 4 (R²=0.9983). 
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
