"""Part A: Belief subspace construction and activation decomposition.

For each layer, trains a linear probe (activations -> belief states),
extracts the probe weight matrix, orthogonalises it via QR decomposition,
and projects activations into belief-aligned and orthogonal components.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment_utils import compute_r2


@dataclass
class LayerDecomposition:
    """Decomposition artifacts for a single layer."""
    layer: int
    probe_r2: float
    probe_mse: float
    probe_weights: np.ndarray       # (n_states, d_model)
    probe_bias: np.ndarray          # (n_states,)
    Q: np.ndarray                   # (d_model, n_states) orthonormal basis for belief subspace
    h_belief: np.ndarray            # (n_points, d_model) belief-aligned component
    h_orth: np.ndarray              # (n_points, d_model) orthogonal component
    var_belief: float               # fraction of total activation variance in belief subspace
    var_orth: float


@dataclass
class DecompositionResult:
    """Decompositions across all layers."""
    layers: dict[int, LayerDecomposition]
    train_idx: np.ndarray
    test_idx: np.ndarray


def build_decompositions(
    activations: dict[int, np.ndarray],
    belief_states: np.ndarray,
    layer_indices: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DecompositionResult:
    """Train probes and decompose activations at every layer.

    Parameters
    ----------
    activations : dict mapping layer index -> (n_points, d_model)
    belief_states : (n_points, n_states)
    layer_indices : which layers to process
    test_size, random_state : train/test split parameters

    Returns
    -------
    DecompositionResult with per-layer decompositions sharing the same
    train/test split indices.
    """
    n_points = belief_states.shape[0]
    indices = np.arange(n_points)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
    )

    layers = {}
    for i, layer in enumerate(layer_indices):
        acts = activations[layer]
        decomp = _decompose_layer(
            acts, belief_states, layer, train_idx, test_idx,
        )
        layers[layer] = decomp
        if (i + 1) % 7 == 0 or i == len(layer_indices) - 1:
            import logging
            logging.getLogger("later_layer_computation").info(
                f"  Decomposition progress: {i+1}/{len(layer_indices)} layers "
                f"(latest R²={decomp.probe_r2:.4f})"
            )

    return DecompositionResult(
        layers=layers,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def _decompose_layer(
    acts: np.ndarray,
    beliefs: np.ndarray,
    layer: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> LayerDecomposition:
    """Probe, orthogonalise, and project for one layer."""
    X_train, X_test = acts[train_idx], acts[test_idx]
    y_train, y_test = beliefs[train_idx], beliefs[test_idx]

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2 = compute_r2(y_test, y_pred)
    mse = float(np.mean((y_pred - y_test) ** 2))

    W = reg.coef_          # (n_states, d_model)
    b = reg.intercept_     # (n_states,)

    # QR decomposition to get orthonormal basis for column space of W^T
    Q, _ = np.linalg.qr(W.T.astype(np.float32))
    Q = Q.astype(np.float32)  # (d_model, n_states)

    # Project via Q: h_belief = (acts @ Q) @ Q.T avoids forming (d_model, d_model)
    coords = acts @ Q              # (n_points, n_states) — cheap
    h_belief = coords @ Q.T        # (n_points, d_model) — cheap
    h_orth = acts - h_belief

    total_var = np.var(acts, axis=0).sum()
    belief_var = np.var(h_belief, axis=0).sum()
    orth_var = np.var(h_orth, axis=0).sum()

    return LayerDecomposition(
        layer=layer,
        probe_r2=r2,
        probe_mse=mse,
        probe_weights=W,
        probe_bias=b,
        Q=Q,
        h_belief=h_belief,
        h_orth=h_orth,
        var_belief=float(belief_var / (total_var + 1e-10)),
        var_orth=float(orth_var / (total_var + 1e-10)),
    )
