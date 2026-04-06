"""CSV and JSON export utilities."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .results import ConfigResult


def save_results_csv(results: list[ConfigResult], out_dir: Path) -> None:
    """Save R² and KL divergence results to CSV files."""
    if not results:
        return

    all_param_keys = sorted(
        {k for r in results for k in r.process_params}
    )
    layers = sorted(results[0].r2_per_layer.keys())

    # sweep_results.csv: one row per config
    results_path = out_dir / "sweep_results.csv"
    with open(results_path, "w", newline="") as f:
        param_cols = [f"param_{k}" for k in all_param_keys]
        r2_cols = [f"r2_layer_{l}" for l in layers]
        header = (
            ["label", "process_name"]
            + param_cols
            + r2_cols
            + ["best_r2", "best_r2_layer", "mean_kl", "std_kl",
               "mean_kl_all_vocab", "std_kl_all_vocab"]
        )
        writer = csv.writer(f)
        writer.writerow(header)

        for r in results:
            best_layer = max(r.r2_per_layer, key=r.r2_per_layer.get)
            row = [
                r.label,
                r.process_name,
                *[r.process_params.get(k, "") for k in all_param_keys],
                *[r.r2_per_layer[l] for l in layers],
                r.r2_per_layer[best_layer],
                best_layer,
                float(r.kl_mean.mean()),
                float(r.kl_mean.std()),
                float(r.kl_all_vocab_mean.mean()),
                float(r.kl_all_vocab_mean.std()),
            ]
            writer.writerow(row)

    # kl_by_position.csv: long format, one row per (config, position)
    kl_path = out_dir / "kl_by_position.csv"
    with open(kl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "position", "kl_mean", "kl_std",
                         "kl_all_vocab_mean", "kl_all_vocab_std"])
        for r in results:
            for pos in range(len(r.kl_mean)):
                writer.writerow([
                    r.label,
                    pos,
                    float(r.kl_mean[pos]),
                    float(r.kl_std[pos]),
                    float(r.kl_all_vocab_mean[pos]),
                    float(r.kl_all_vocab_std[pos]),
                ])
