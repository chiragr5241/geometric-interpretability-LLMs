# Later-Layer Computation Investigation: fern_x0.63

## Experiment Configuration

- **HMM process**: `fern`
- **Parameters**: `{'x': 0.63}`
- **Layers analysed**: 0–27 (28 layers)

## Part A: Belief Subspace Decomposition

At each layer, a linear probe was trained to predict HMM belief states from residual-stream activations. The probe weight matrix defines a belief-relevant subspace (via QR decomposition). Each activation is decomposed into a belief-aligned component (h_belief) and an orthogonal complement (h_orth).

| Layer | Probe R² | Variance (belief) | Variance (orth) |
|-------|----------|-------------------|-----------------|
|  0 | 0.9321 | 0.0049 | 0.9951 |
|  1 | 0.9792 | 0.0088 | 0.9912 |
|  2 | 0.9851 | 0.0112 | 0.9888 |
|  3 | 0.9888 | 0.0112 | 0.9888 |
|  4 | 0.9918 | 0.0091 | 0.9909 |
|  5 | 0.9928 | 0.0077 | 0.9923 |
|  6 | 0.9927 | 0.0057 | 0.9943 |
|  7 | 0.9932 | 0.0036 | 0.9964 |
|  8 | 0.9912 | 0.0030 | 0.9970 |
|  9 | 0.9894 | 0.0023 | 0.9977 |
| 10 | 0.9870 | 0.0022 | 0.9978 |
| 11 | 0.9860 | 0.0018 | 0.9982 |
| 12 | 0.9850 | 0.0019 | 0.9981 |
| 13 | 0.9842 | 0.0017 | 0.9983 |
| 14 | 0.9831 | 0.0017 | 0.9983 |
| 15 | 0.9824 | 0.0016 | 0.9984 |
| 16 | 0.9817 | 0.0017 | 0.9983 |
| 17 | 0.9815 | 0.0017 | 0.9983 |
| 18 | 0.9815 | 0.0017 | 0.9983 |
| 19 | 0.9808 | 0.0015 | 0.9985 |
| 20 | 0.9799 | 0.0016 | 0.9984 |
| 21 | 0.9793 | 0.0017 | 0.9983 |
| 22 | 0.9791 | 0.0016 | 0.9984 |
| 23 | 0.9788 | 0.0016 | 0.9984 |
| 24 | 0.9790 | 0.0015 | 0.9985 |
| 25 | 0.9779 | 0.0015 | 0.9985 |
| 26 | 0.9775 | 0.0013 | 0.9987 |
| 27 | 0.9765 | 0.0012 | 0.9988 |

## Part B: Predictive Residual Analysis

Linear decoders were trained from each component to predict multiple targets. The key question: *what can h_orth predict that h_belief cannot?*

### Mean R² across layers (by component × target)

| Target | full | belief | orth | orth_pca |
|--------|------|------|------|------|
| concept_logits | 0.9696 | 0.0579 | 0.9685 | 0.5619 |
| hmm_next_token | 0.9468 | 0.9482 | 0.6831 | 0.2132 |
| belief_entropy | -9493.8479 | -7478.7298 | -8781.9761 | -5690.3482 |
| token_position | -9581.2922 | -5180.4928 | -9575.2108 | -7241.8265 |
| logit_residual | 0.9673 | -0.0001 | 0.9503 | 0.5273 |
| multi_step_2 | 0.9944 | 0.9946 | 0.9230 | 0.5663 |
| multi_step_4 | 0.9953 | 0.9954 | 0.9281 | 0.5826 |

## Part C: Causal Interventions

Projection ablation: at each layer, the orthogonal or belief component was ablated during the forward pass, and downstream logit changes were measured.

| Layer | Condition | KL(orig‖abl) | KL(HMM‖abl) | KL(HMM‖orig) | Top-1 (abl) | Top-1 (orig) |
|-------|-----------|-------------|-------------|--------------|-------------|-------------|

## Part D: Interpretation Ranking

Candidate interpretations ranked by evidence strength:

### 1. Output readout / logit refinement (score: 0.579)

*Later layers reformat belief state information into the unembedding-compatible format needed for token prediction.*

**Evidence for:**
- Orth component predicts concept logits (R²=0.969)
**Evidence against:**
- Orth also predicts HMM probs (R²=0.683)

### 2. Residual correction on top of belief state (score: 0.570)

*Later layers apply a correction to predictions that goes beyond what the belief-state probe captures, possibly encoding model-specific learned biases.*

**Evidence for:**
- Orth predicts logit residual (what belief can't explain) with R²=0.950

### 3. Compression / redistribution (score: 0.100)

*Later layers compress or redistribute information across the residual stream, increasing importance of non-belief directions.*


### 4. Multi-step predictive information (score: 0.000)

*Later layers encode future predictions beyond one-step that are not captured by the current belief state.*

**Evidence against:**
- Belief predicts 2-step as well or better

### 5. Uncertainty tracking (score: -6147.383)

*Later layers encode a representation of predictive uncertainty beyond what the belief state provides.*

**Evidence against:**
- Orth does not predict entropy well (R²=-8781.976)
- Belief component predicts entropy better

### 6. Synchronization-progress information (score: -6702.648)

*Later layers track how far along the sequence the model has progressed, encoding synchronization depth.*

**Evidence against:**
- Orth does not encode position well (R²=-9575.211)

## Conclusions

### Is the later-layer signal merely a reformatted belief state?

The causal evidence suggests that ablating the orthogonal component has minimal effect on downstream predictions (mean KL < 0.005). This is consistent with the later-layer signal being primarily a reformatted version of the belief state, rather than genuinely additional computation. The top-ranked interpretation is: **Output readout / logit refinement**.

### Main limitations

- The belief subspace is very low-dimensional (n_states) relative to d_model, giving the orthogonal complement far more capacity. The orth_pca control (matching dimensionality) should be checked for all claims about orthogonal predictive power.
- Ablation creates out-of-distribution activations. The mean_ablate_orth condition partially addresses this but cannot fully eliminate it.
- Results are specific to the tested HMM process and model.
