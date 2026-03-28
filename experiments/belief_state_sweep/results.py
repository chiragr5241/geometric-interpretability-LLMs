"""Result container dataclass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigResult:
    """Results from a single HMM parameter configuration."""
    process_name: str
    process_params: dict[str, float]
    label: str
    vocab_tokens: list[str]
    belief_states_flat: np.ndarray      # (total_points, n_states)
    kl_mean: np.ndarray                 # (seq_len,)
    kl_std: np.ndarray                  # (seq_len,)
    r2_per_layer: dict[int, float]
    mse_per_layer: dict[int, float]
    predicted_beliefs: np.ndarray       # (n_test, n_states) — best layer probe predictions
    predicted_beliefs_gt: np.ndarray    # (n_test, n_states) — ground truth for test set
