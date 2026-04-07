"""Configuration for the per-layer tuned lens experiment."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SweepEntry:
    """One process + parameter grid for sweeping."""
    process_name: str
    param_grid: dict[str, list[float]]
    vocab_tokens: list[str] | None = None
    seq_length: int | None = None
    n_sequences: int | None = None


@dataclass
class TunedLensConfig:
    experiment_name: str = "tuned_lens_per_layer"
    model_name: str = "meta-llama/Llama-3.2-3B"
    n_ctx_override: int | None = 4098

    # HMM parameters (used when running a single config without sweeps)
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
    output_user: str = "SPAR"
    output_name: str = "tuned_lens_per_layer"
    results_dir: str = "results/tuned_lens_per_layer"

    # Sweep support
    sweeps: list[SweepEntry] = field(default_factory=list)
    default_vocab_tokens: dict[int, list[str]] = field(
        default_factory=lambda: {2: ["A", "B"], 3: ["A", "B", "C"]}
    )


def expand_param_grid(entry: SweepEntry) -> list[dict[str, float]]:
    """Cartesian product of param_grid values."""
    keys = sorted(entry.param_grid.keys())
    values = [entry.param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def make_config_label(process_name: str, params: dict[str, float]) -> str:
    """e.g. 'mess3_a0.6_x0.15'."""
    parts = [process_name]
    for k in sorted(params.keys()):
        parts.append(f"{k}{params[k]}")
    return "_".join(parts)


def load_config(path: str | Path) -> TunedLensConfig:
    """Load config from a YAML file, merging with defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    sweeps_raw = raw.pop("sweeps", [])
    default_vocab_raw = raw.pop("default_vocab_tokens", None)

    cfg = TunedLensConfig()
    for key, val in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    if sweeps_raw:
        cfg.sweeps = [SweepEntry(**s) for s in sweeps_raw]
    if default_vocab_raw is not None:
        cfg.default_vocab_tokens = {int(k): v for k, v in default_vocab_raw.items()}

    return cfg
