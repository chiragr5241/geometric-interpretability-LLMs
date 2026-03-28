"""Configuration for the per-layer tuned lens experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TunedLensConfig:
    experiment_name: str = "tuned_lens_per_layer"
    model_name: str = "meta-llama/Llama-3.2-3B"
    n_ctx_override: int | None = 4098

    # HMM parameters
    process_name: str = "mess3"
    process_params: dict[str, float] = field(default_factory=lambda: {"x": 0.05, "a": 0.85})
    vocab_tokens: list[str] = field(default_factory=lambda: ["A", "B", "C"])

    # Sequence generation
    seq_length: int = 2000
    n_sequences: int = 10
    random_seed: int = 42

    # Layer selection
    layer_indices: list[int] = field(default_factory=lambda: list(range(28)))

    # Train/test split (by sequence index to avoid data leakage)
    n_train_sequences: int = 8  # first 8 sequences for training
    # remaining sequences are held-out test

    # Tuned lens training
    tuned_lens_epochs: int = 50
    tuned_lens_lr: float = 1e-3
    tuned_lens_batch_size: int = 512

    # Output
    output_name: str = "tuned_lens_per_layer"
    results_dir: str = "results/tuned_lens_per_layer"


def load_config(path: str | Path) -> TunedLensConfig:
    """Load config from a YAML file, merging with defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = TunedLensConfig()
    for key, val in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
