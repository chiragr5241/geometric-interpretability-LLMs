"""Part B: Predictive residual analysis.

Trains linear decoders from {full, belief, orthogonal} components to predict
multiple targets, quantifying what the orthogonal complement encodes beyond
the belief state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment_utils import compute_r2

from .decomposition import DecompositionResult


COMPONENT_NAMES = ("full", "belief", "orth", "orth_pca")
TARGET_NAMES = (
    "concept_logits",
    "hmm_next_token",
    "belief_entropy",
    "token_position",
    "logit_residual",
    "multi_step_2",
    "multi_step_4",
)


@dataclass
class DecoderScore:
    r2: float
    mse: float


@dataclass
class LayerDecoderResults:
    """Decoder results for one layer across all components and targets."""
    layer: int
    scores: dict[str, dict[str, DecoderScore]]  # component -> target -> score


@dataclass
class DecoderResults:
    """All decoder results across layers."""
    layers: dict[int, LayerDecoderResults]


def compute_targets(
    concept_logits: np.ndarray,
    obs_probs: np.ndarray,
    belief_states: np.ndarray,
    token_positions: np.ndarray,
    hmm=None,
) -> dict[str, np.ndarray]:
    """Build all prediction target arrays.

    Parameters
    ----------
    concept_logits : (n_points, n_concepts) raw logits for HMM vocab tokens
    obs_probs : (n_points, n_vocab) P(next_token | belief)
    belief_states : (n_points, n_states)
    token_positions : (n_points,) position within each sequence
    hmm : simplexity HMM object (for multi-step targets)
    """
    targets = {}

    targets["concept_logits"] = concept_logits
    targets["hmm_next_token"] = obs_probs

    entropy = -np.sum(
        belief_states * np.log(belief_states + 1e-10), axis=-1, keepdims=True
    )
    targets["belief_entropy"] = entropy

    targets["token_position"] = token_positions.reshape(-1, 1).astype(np.float32)

    if hmm is not None:
        T = np.array(hmm.transition_matrices)  # (n_vocab, n_states, n_states)
        M = T.sum(axis=0)  # (n_states, n_states): one-step state transition

        # k-step: belief @ M^k, then marginalise to observation probs
        for k in (2, 4):
            Mk = np.linalg.matrix_power(M, k)
            future_beliefs = belief_states @ Mk
            future_beliefs = future_beliefs / (
                future_beliefs.sum(axis=-1, keepdims=True) + 1e-10
            )
            future_obs = np.einsum("ns,vst->nv", future_beliefs, T)
            targets[f"multi_step_{k}"] = future_obs

    return targets


def run_decoders(
    decomposition: DecompositionResult,
    targets: dict[str, np.ndarray],
    layer_indices: list[int],
    test_size: float = 0.2,
    n_pca_components: int | None = None,
) -> DecoderResults:
    """Train linear decoders for every (component, target, layer) combination.

    Uses the same train/test split as the decomposition to avoid leakage.
    """
    train_idx = decomposition.train_idx
    test_idx = decomposition.test_idx

    all_layers = {}
    import logging
    log = logging.getLogger("later_layer_computation")

    for li, layer in enumerate(layer_indices):
        ld = decomposition.layers[layer]

        # Use low-dimensional coordinates for belief component (n_states dims)
        # instead of full 3072D projected vector — avoids expensive Ridge on rank-deficient input
        belief_coords = (ld.h_belief + ld.h_orth) @ ld.Q  # (n_points, n_states) belief subspace coords

        # Dimensionality-matched PCA of orthogonal component
        n_states = ld.Q.shape[1]
        n_pca = n_pca_components if n_pca_components is not None else n_states
        n_pca_wide = min(50, n_pca * 10)  # wider PCA for orth
        orth_train = ld.h_orth[train_idx]

        pca_matched = PCA(n_components=n_pca)
        pca_matched.fit(orth_train)

        pca_wide = PCA(n_components=n_pca_wide)
        pca_wide.fit(orth_train)

        # Components: use low-dimensional representations to keep Ridge fast
        components = {
            "full": np.hstack([belief_coords, pca_wide.transform(ld.h_orth)]),
            "belief": belief_coords,           # n_states-dimensional
            "orth": pca_wide.transform(ld.h_orth),  # n_pca_wide-dimensional
            "orth_pca": pca_matched.transform(ld.h_orth),  # matched dim
        }

        # Compute logit_residual target using belief coords
        if "concept_logits" in targets:
            from sklearn.linear_model import LinearRegression
            _lr = LinearRegression()
            _lr.fit(belief_coords[train_idx], targets["concept_logits"][train_idx])
            belief_predicted_logits = _lr.predict(belief_coords)
            targets_with_residual = dict(targets)
            targets_with_residual["logit_residual"] = (
                targets["concept_logits"] - belief_predicted_logits
            )
        else:
            targets_with_residual = targets

        scores: dict[str, dict[str, DecoderScore]] = {}
        for comp_name, comp_data in components.items():
            scores[comp_name] = {}
            X_train = comp_data[train_idx]
            X_test = comp_data[test_idx]

            for tgt_name, tgt_data in targets_with_residual.items():
                if tgt_data is None:
                    continue
                y_train = tgt_data[train_idx]
                y_test = tgt_data[test_idx]

                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                r2 = compute_r2(y_test, y_pred)
                mse = float(np.mean((y_pred - y_test) ** 2))
                scores[comp_name][tgt_name] = DecoderScore(r2=r2, mse=mse)

        all_layers[layer] = LayerDecoderResults(layer=layer, scores=scores)

        if (li + 1) % 7 == 0 or li == len(layer_indices) - 1:
            log.info(
                f"  Decoder progress: {li+1}/{len(layer_indices)} layers"
            )

    return DecoderResults(layers=all_layers)
