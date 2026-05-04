#!/usr/bin/env python3
"""Post-hoc generator for (layer × k) intervention effect plots.

Reads `plot_data/aggregated_plot_data.json` from one or more existing
single-sequence-intervention output directories and writes the new plots
into each run's `figures/` directory using the helpers in
`single_seq_interventions.py`.

Usage:
    python experiments/replot_intervention_k_dependence.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plot_titles import format_hmm_process  # type: ignore
from single_seq_interventions import (  # type: ignore
    PATCH_CONDITIONS,
    STEER_CONDITIONS,
    _emit_layer_k_effects,
)


def _reindex(by_cond: dict) -> dict[str, dict[int, dict[int, dict]]]:
    """Convert {cond: {str(k): {str(layer): agg}}} → {cond: {k: {layer: agg}}}."""
    out: dict[str, dict[int, dict[int, dict]]] = {}
    for cond, by_k in by_cond.items():
        out[cond] = {
            int(k): {int(l): agg for l, agg in by_layer.items()}
            for k, by_layer in by_k.items()
        }
    return out


def _read_run_meta(run_dir: Path) -> tuple[str, bool]:
    """Read config.yaml from a run dir and return (hmm_subtitle, save_html)."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return "", False
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return "", False
    hmm = cfg.get("hmm", {}) or {}
    name = hmm.get("process_name", "")
    params = hmm.get("process_params", {}) or {}
    subtitle = ""
    if name:
        try:
            subtitle = format_hmm_process(name, params)
        except Exception:
            subtitle = name
    save_html = bool(cfg.get("save_html", False))
    return subtitle, save_html


def _process(run_dir: Path) -> None:
    data_path = run_dir / "plot_data" / "aggregated_plot_data.json"
    if not data_path.exists():
        print(f"[skip] {run_dir}: missing {data_path}")
        return
    with open(data_path) as f:
        data = json.load(f)

    layer_indices = [int(l) for l in data["layer_indices"]]
    k_values = [int(k) for k in data["k_values"]]
    metrics = data["metrics"]
    hmm_subtitle, save_html = _read_run_meta(run_dir)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {run_dir}")
    print(f"  layers={len(layer_indices)} k_values={len(k_values)} hmm='{hmm_subtitle}' → {fig_dir}")

    _emit_layer_k_effects(
        label="patching",
        conditions=PATCH_CONDITIONS,
        agg_to_target=_reindex(metrics["patching_to_target"]),
        agg_baseline_to_target=_reindex(metrics["baseline_kl_to_target_patch"]),
        layer_indices=layer_indices,
        k_values=k_values,
        fig_dir=fig_dir,
        hmm_subtitle=hmm_subtitle,
        save_html=save_html,
    )
    _emit_layer_k_effects(
        label="steering",
        conditions=STEER_CONDITIONS,
        agg_to_target=_reindex(metrics["steering_to_target"]),
        agg_baseline_to_target=_reindex(metrics["baseline_kl_to_target_steer"]),
        layer_indices=layer_indices,
        k_values=k_values,
        fig_dir=fig_dir,
        hmm_subtitle=hmm_subtitle,
        save_html=save_html,
    )
    print(f"  done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        _process(run_dir.resolve())


if __name__ == "__main__":
    main()
