"""Configuration for the per-layer tuned lens experiment."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SweepEntry:
    """One process + parameter grid for sweeping.

    Supply EITHER ``param_grid`` (cartesian product over per-key value lists)
    OR ``param_combos`` (explicit list of pre-built parameter dicts). The
    explicit form is required when parameter combinations are not a clean
    cartesian product (e.g. paired (a, x) values).
    """
    process_name: str
    param_grid: dict[str, list[float]] | None = None
    param_combos: list[dict[str, float]] | None = None
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
    # Optimizer for tuned-lens translators: "adam" (default) or "muon".
    # The Tuned Lens paper (arXiv:2303.08112) recommends Muon as an alternative
    # to Adam for the per-layer affine translators.
    tuned_lens_optimizer: str = "adam"
    # If True: model-target translator is trained on the FULL model output
    # distribution (canonical tuned lens, arXiv:2303.08112).
    # If False: model-target translator is trained on concept-token logits only.
    model_target_full_vocab: bool = True
    # Whether to also train an HMM-ground-truth target translator
    # (KL against true HMM next-token distribution).
    train_hmm_target: bool = True

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
    """Expand a SweepEntry into the list of parameter dicts to run.

    Uses ``param_combos`` verbatim if supplied; otherwise takes the cartesian
    product of ``param_grid``.
    """
    if entry.param_combos:
        return [dict(combo) for combo in entry.param_combos]
    if not entry.param_grid:
        raise ValueError(
            f"SweepEntry for {entry.process_name!r} has neither param_grid "
            f"nor param_combos."
        )
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
