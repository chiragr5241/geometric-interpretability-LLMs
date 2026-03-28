"""Config dataclass and YAML loader for later-layer computation experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment import ExperimentConfig, HMMConfig


@dataclass
class HMMEntry:
    """One HMM configuration to analyse."""
    process_name: str
    process_params: dict[str, float]
    vocab_tokens: list[str] | None = None
    seq_length: int | None = None
    n_sequences: int | None = None


@dataclass
class LaterLayerConfig(ExperimentConfig):
    seq_length: int = 500
    n_sequences: int = 50
    random_seed: int = 42
    layer_indices: list[int] = field(default_factory=lambda: list(range(28)))
    n_ctx_override: int | None = None

    # probe settings
    test_size: float = 0.2
    random_state: int = 42

    # HMM entries to analyse
    hmm_entries: list[HMMEntry] = field(default_factory=list)
    default_vocab_tokens: dict[int, list[str]] = field(
        default_factory=lambda: {2: ["A", "B"], 3: ["A", "B", "C"]}
    )

    # causal intervention settings
    causal_n_sequences: int = 20
    causal_batch_size: int = 4

    # decoder settings
    decoder_test_size: float = 0.2


def load_config(path: str) -> LaterLayerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    hmm_raw = raw.pop("hmm", {"process_name": "placeholder", "process_params": {}})
    entries_raw = raw.pop("hmm_entries", [])
    default_vocab_raw = raw.pop("default_vocab_tokens", {2: ["A", "B"], 3: ["A", "B", "C"]})

    hmm = HMMConfig(**hmm_raw)
    entries = [HMMEntry(**e) for e in entries_raw]
    default_vocab = {int(k): v for k, v in default_vocab_raw.items()}

    return LaterLayerConfig(
        hmm=hmm,
        hmm_entries=entries,
        default_vocab_tokens=default_vocab,
        **raw,
    )


def resolve_vocab_tokens(
    hmm, entry: HMMEntry, defaults: dict[int, list[str]]
) -> list[str]:
    if entry.vocab_tokens is not None:
        return entry.vocab_tokens
    if hmm.vocab_size in defaults:
        return defaults[hmm.vocab_size]
    return [chr(65 + i) for i in range(hmm.vocab_size)]
