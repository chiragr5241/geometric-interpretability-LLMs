from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HMMConfig:
    process_name: str
    process_params: dict[str, Any]


@dataclass
class ExperimentConfig:
    experiment_name: str
    hmm: HMMConfig
    model_name: str
    layer_indices: list[int]
    seq_length: int
    kl_params: dict[str, float]
    vocab_mapping: dict[str, int]


def load_config(path: str) -> ExperimentConfig:
    p = Path(path)
    with open(p) as f:
        raw = yaml.safe_load(f)
    hmm_raw = raw.pop("hmm")
    return ExperimentConfig(hmm=HMMConfig(**hmm_raw), **raw)


def setup_output_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "dani" / f"{timestamp}_{config.experiment_name}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "probes").mkdir(exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(
            {
                "experiment_name": config.experiment_name,
                "hmm": {
                    "process_name": config.hmm.process_name,
                    "process_params": config.hmm.process_params,
                },
                "model_name": config.model_name,
                "layer_indices": config.layer_indices,
                "seq_length": config.seq_length,
                "kl_params": config.kl_params,
                "vocab_mapping": config.vocab_mapping,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    return out_dir
