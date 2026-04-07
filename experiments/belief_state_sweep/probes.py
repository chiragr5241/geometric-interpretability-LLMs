"""Linear probe training and evaluation for belief state decoding."""
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
class ProbeResults:
    """Results from training linear probes across all layers."""
    r2_per_layer: dict[int, float]
    mse_per_layer: dict[int, float]
    best_layer: int
    best_layer_predicted: np.ndarray   # predicted beliefs from best layer probe
    best_layer_gt: np.ndarray          # ground-truth beliefs for the test set


def train_probes(
    all_activations_per_seq: dict[int, list[np.ndarray]],
    beliefs_per_seq: list[np.ndarray],
    layer_indices: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> ProbeResults:
    """Train a LinearRegression probe per layer, within each sequence independently.

    To avoid inflated R² from cross-sequence structure, each probe is trained
    and evaluated on a single sequence.  Results are averaged across sequences.

    Parameters
    ----------
    all_activations_per_seq : dict mapping layer -> list of per-sequence arrays,
        each of shape (n_positions_i, d_model).
    beliefs_per_seq : list of per-sequence belief arrays,
        each of shape (n_positions_i, n_states).
    """
    n_sequences = len(beliefs_per_seq)
    r2_per_layer: dict[int, float] = {}
    mse_per_layer: dict[int, float] = {}
    best_r2 = -np.inf
    best_layer = layer_indices[0]
    best_predicted: np.ndarray | None = None
    best_gt: np.ndarray | None = None

    for layer in layer_indices:
        seq_r2s = []
        seq_mses = []
        layer_predicted_parts = []
        layer_gt_parts = []

        for seq_idx in range(n_sequences):
            acts = all_activations_per_seq[layer][seq_idx]
            beliefs = beliefs_per_seq[seq_idx]

            X_train, X_test, y_train, y_test = train_test_split(
                acts,
                beliefs,
                test_size=test_size,
                random_state=random_state + seq_idx,
            )

            reg = LinearRegression()
            reg.fit(X_train, y_train)

            y_pred_test = reg.predict(X_test)

            seq_mses.append(float(np.mean((y_pred_test - y_test) ** 2)))
            seq_r2s.append(compute_r2(y_test, y_pred_test))

            layer_predicted_parts.append(y_pred_test)
            layer_gt_parts.append(y_test)

        r2_test = float(np.mean(seq_r2s))
        mse_test = float(np.mean(seq_mses))
        r2_per_layer[layer] = r2_test
        mse_per_layer[layer] = mse_test

        if r2_test > best_r2:
            best_r2 = r2_test
            best_layer = layer
            best_predicted = np.concatenate(layer_predicted_parts, axis=0)
            best_gt = np.concatenate(layer_gt_parts, axis=0)

    return ProbeResults(
        r2_per_layer=r2_per_layer,
        mse_per_layer=mse_per_layer,
        best_layer=best_layer,
        best_layer_predicted=best_predicted,
        best_layer_gt=best_gt,
    )
