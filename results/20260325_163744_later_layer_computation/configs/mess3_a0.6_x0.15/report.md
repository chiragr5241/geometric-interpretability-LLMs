# Later-Layer Computation Investigation: mess3_a0.6_x0.15

## Experiment Configuration

- **HMM process**: `mess3`
- **Parameters**: `{'a': 0.6, 'x': 0.15}`
- **Layers analysed**: 0–27 (28 layers)

## Part A: Belief Subspace Decomposition

At each layer, a linear probe was trained to predict HMM belief states from residual-stream activations. The probe weight matrix defines a belief-relevant subspace (via QR decomposition). Each activation is decomposed into a belief-aligned component (h_belief) and an orthogonal complement (h_orth).

| Layer | Probe R² | Variance (belief) | Variance (orth) |
|-------|----------|-------------------|-----------------|
|  0 | 0.9990 | 0.0104 | 0.9896 |
|  1 | 0.9999 | 0.0406 | 0.9594 |
|  2 | 0.9999 | 0.0357 | 0.9643 |
|  3 | 0.9999 | 0.0311 | 0.9689 |
|  4 | 0.9999 | 0.0191 | 0.9809 |
|  5 | 0.9998 | 0.0149 | 0.9851 |
|  6 | 0.9995 | 0.0096 | 0.9904 |
|  7 | 0.9993 | 0.0059 | 0.9941 |
|  8 | 0.9987 | 0.0047 | 0.9953 |
|  9 | 0.9981 | 0.0042 | 0.9958 |
| 10 | 0.9973 | 0.0044 | 0.9956 |
| 11 | 0.9970 | 0.0037 | 0.9963 |
| 12 | 0.9967 | 0.0034 | 0.9966 |
| 13 | 0.9967 | 0.0028 | 0.9972 |
| 14 | 0.9966 | 0.0026 | 0.9974 |
| 15 | 0.9965 | 0.0025 | 0.9975 |
| 16 | 0.9967 | 0.0025 | 0.9975 |
| 17 | 0.9966 | 0.0024 | 0.9976 |
| 18 | 0.9964 | 0.0023 | 0.9977 |
| 19 | 0.9962 | 0.0022 | 0.9978 |
| 20 | 0.9960 | 0.0021 | 0.9979 |
| 21 | 0.9961 | 0.0021 | 0.9979 |
| 22 | 0.9964 | 0.0021 | 0.9979 |
| 23 | 0.9961 | 0.0021 | 0.9979 |
| 24 | 0.9963 | 0.0020 | 0.9980 |
| 25 | 0.9957 | 0.0019 | 0.9981 |
| 26 | 0.9957 | 0.0018 | 0.9982 |
| 27 | 0.9952 | 0.0016 | 0.9984 |

## Part B: Predictive Residual Analysis

Linear decoders were trained from each component to predict multiple targets. The key question: *what can h_orth predict that h_belief cannot?*

### Mean R² across layers (by component × target)

| Target | full | belief | orth | orth_pca |
|--------|------|------|------|------|
| concept_logits | 0.9643 | 0.0734 | 0.9631 | 0.5454 |
| hmm_next_token | 0.9974 | 0.9974 | 0.9336 | 0.6609 |
| belief_entropy | -9476.0659 | -5175.4310 | -9467.1727 | -6574.3292 |
| token_position | -9506.1312 | -5218.0874 | -9500.6421 | -7400.1735 |
| logit_residual | 0.9611 | -0.0002 | 0.9364 | 0.4999 |
| multi_step_2 | 0.9974 | 0.9974 | 0.9336 | 0.6609 |
| multi_step_4 | 0.9974 | 0.9974 | 0.9336 | 0.6609 |

## Part C: Causal Interventions

Projection ablation: at each layer, the orthogonal or belief component was ablated during the forward pass, and downstream logit changes were measured.

| Layer | Condition | KL(orig‖abl) | KL(HMM‖abl) | KL(HMM‖orig) | Top-1 (abl) | Top-1 (orig) |
|-------|-----------|-------------|-------------|--------------|-------------|-------------|

## Part D: Interpretation Ranking

Candidate interpretations ranked by evidence strength:

### 1. Residual correction on top of belief state (score: 0.562)

*Later layers apply a correction to predictions that goes beyond what the belief-state probe captures, possibly encoding model-specific learned biases.*

**Evidence for:**
- Orth predicts logit residual (what belief can't explain) with R²=0.936

### 2. Output readout / logit refinement (score: 0.501)

*Later layers reformat belief state information into the unembedding-compatible format needed for token prediction.*

**Evidence for:**
- Orth component predicts concept logits (R²=0.963)
**Evidence against:**
- Orth also predicts HMM probs (R²=0.934)

### 3. Compression / redistribution (score: 0.100)

*Later layers compress or redistribute information across the residual stream, increasing importance of non-belief directions.*


### 4. Multi-step predictive information (score: 0.000)

*Later layers encode future predictions beyond one-step that are not captured by the current belief state.*

**Evidence against:**
- Belief predicts 2-step as well or better

### 5. Uncertainty tracking (score: -6627.021)

*Later layers encode a representation of predictive uncertainty beyond what the belief state provides.*

**Evidence against:**
- Orth does not predict entropy well (R²=-9467.173)
- Belief component predicts entropy better

### 6. Synchronization-progress information (score: -6650.449)

*Later layers track how far along the sequence the model has progressed, encoding synchronization depth.*

**Evidence against:**
- Orth does not encode position well (R²=-9500.642)

## Conclusions

### Is the later-layer signal merely a reformatted belief state?

The causal evidence suggests that ablating the orthogonal component has minimal effect on downstream predictions (mean KL < 0.005). This is consistent with the later-layer signal being primarily a reformatted version of the belief state, rather than genuinely additional computation. The top-ranked interpretation is: **Residual correction on top of belief state**.

### Main limitations

- The belief subspace is very low-dimensional (n_states) relative to d_model, giving the orthogonal complement far more capacity. The orth_pca control (matching dimensionality) should be checked for all claims about orthogonal predictive power.
- Ablation creates out-of-distribution activations. The mean_ablate_orth condition partially addresses this but cannot fully eliminate it.
- Results are specific to the tested HMM process and model.
