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
    all_activations: dict[int, np.ndarray],
    belief_states_flat: np.ndarray,
    layer_indices: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> ProbeResults:
    """Train a LinearRegression probe per layer and return results.

    Returns
    -------
    ProbeResults
        Contains R²/MSE per layer, the best layer index, and the predicted
        belief states from the best layer's probe (on the test set).
    """
    r2_per_layer: dict[int, float] = {}
    mse_per_layer: dict[int, float] = {}
    best_r2 = -np.inf
    best_layer = layer_indices[0]
    best_predicted: np.ndarray | None = None
    best_gt: np.ndarray | None = None

    for layer in layer_indices:
        acts = all_activations[layer]

        X_train, X_test, y_train, y_test = train_test_split(
            acts,
            belief_states_flat,
            test_size=test_size,
            random_state=random_state,
        )

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        y_pred_test = reg.predict(X_test)

        mse_test = float(np.mean((y_pred_test - y_test) ** 2))
        r2_test = compute_r2(y_test, y_pred_test)

        r2_per_layer[layer] = r2_test
        mse_per_layer[layer] = mse_test

        if r2_test > best_r2:
            best_r2 = r2_test
            best_layer = layer
            best_predicted = y_pred_test
            best_gt = y_test

    return ProbeResults(
        r2_per_layer=r2_per_layer,
        mse_per_layer=mse_per_layer,
        best_layer=best_layer,
        best_layer_predicted=best_predicted,
        best_layer_gt=best_gt,
    )
