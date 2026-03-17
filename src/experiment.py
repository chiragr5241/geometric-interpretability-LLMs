from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

C = TypeVar("C", bound="ExperimentConfig")


@dataclass
class HMMConfig:
    process_name: str
    process_params: dict[str, Any]


@dataclass
class ExperimentConfig:
    experiment_name: str
    hmm: HMMConfig
    model_name: str
    output_user: str = field(default="dani", kw_only=True)



def load_config(path: str, cls: type[C]) -> C:
    with open(path) as f:
        raw = yaml.safe_load(f)
    hmm_raw = raw.pop("hmm")
    return cls(hmm=HMMConfig(**hmm_raw), **raw)


def apply_runtime_overrides(config: ExperimentConfig, output_user: str | None = None) -> ExperimentConfig:
    if output_user is not None:
        config.output_user = output_user
    return config


def setup_output_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / config.output_user / f"{timestamp}_{config.experiment_name}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "probes").mkdir(exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False, sort_keys=False)
    return out_dir
