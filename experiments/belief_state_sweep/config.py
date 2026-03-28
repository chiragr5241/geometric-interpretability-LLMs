"""Config dataclasses, YAML loader, and sweep-grid helpers."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import yaml

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment import ExperimentConfig, HMMConfig


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class SweepEntry:
    process_name: str
    param_grid: dict[str, list[float]]
    vocab_tokens: list[str] | None = None
    seq_length: int | None = None
    n_sequences: int | None = None


@dataclass
class BeliefStateSweepConfig(ExperimentConfig):
    seq_length: int = 2000
    n_sequences: int = 10
    random_seed: int = 42
    layer_indices: list[int] = field(default_factory=lambda: list(range(28)))
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sweeps: list[SweepEntry] = field(default_factory=list)
    default_vocab_tokens: dict[int, list[str]] = field(
        default_factory=lambda: {2: ["A", "B"], 3: ["A", "B", "C"]}
    )
    n_ctx_override: int | None = None


# ── Config loader ─────────────────────────────────────────────────────────────

def load_sweep_config(path: str) -> BeliefStateSweepConfig:
    """Custom config loader that handles nested sweep entries."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    hmm_raw = raw.pop("hmm", {"process_name": "sweep", "process_params": {}})
    probe_raw = raw.pop("probe", {})
    sweeps_raw = raw.pop("sweeps", [])
    default_vocab_raw = raw.pop("default_vocab_tokens", {2: ["A", "B"], 3: ["A", "B", "C"]})

    default_vocab = {int(k): v for k, v in default_vocab_raw.items()}
    hmm = HMMConfig(**hmm_raw)
    probe = ProbeConfig(**probe_raw)
    sweeps = [SweepEntry(**s) for s in sweeps_raw]

    return BeliefStateSweepConfig(
        hmm=hmm,
        probe=probe,
        sweeps=sweeps,
        default_vocab_tokens=default_vocab,
        **raw,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def resolve_vocab_tokens(
    hmm, entry: SweepEntry, defaults: dict[int, list[str]]
) -> list[str]:
    """Determine token labels for this HMM's vocabulary."""
    if entry.vocab_tokens is not None:
        assert len(entry.vocab_tokens) == hmm.vocab_size
        return entry.vocab_tokens
    if hmm.vocab_size in defaults:
        return defaults[hmm.vocab_size]
    return [chr(65 + i) for i in range(hmm.vocab_size)]
