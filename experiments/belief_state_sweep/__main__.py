#!/usr/bin/env python3
"""
Belief state geometry sweep — parameter grid search across HMM processes.

Sweeps over multiple HMM processes (mess3, leopard, etc.) with cartesian product
parameter grids. For each configuration, generates sequences, runs them through
an LLM, computes KL divergence, trains linear regression probes (evaluated by R²),
and produces per-config plots plus a cross-config comparison.

Usage:
    python -m experiments.belief_state_sweep experiments/configs/belief_state_sweep.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment import setup_output_dir
from experiment_utils import get_device, load_model, setup_logging

from simplexity.generative_processes.builder import build_hidden_markov_model

from .config import (
    BeliefStateSweepConfig,
    SweepEntry,
    expand_param_grid,
    load_sweep_config,
    make_config_label,
    resolve_vocab_tokens,
)
from .export import save_results_csv
from .pipeline import run_single_config
from .plotting import (
    plot_belief_simplex,
    plot_comparison,
    plot_kl_divergence,
    plot_predicted_simplex,
    plot_r2_by_layer,
)
from .results import ConfigResult


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python -m experiments.belief_state_sweep <config.yaml>")
        sys.exit(1)

    config = load_sweep_config(sys.argv[1])
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="belief_state_sweep")

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Device: {device}")

    # Load model once (shared across all configs)
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    # Enumerate all configs from sweep entries
    all_config_runs: list[tuple[str, dict[str, float], SweepEntry]] = []
    for entry in config.sweeps:
        param_combos = expand_param_grid(entry)
        for params in param_combos:
            all_config_runs.append((entry.process_name, params, entry))

    total_configs = len(all_config_runs)
    logger.info(f"Total configurations to sweep: {total_configs}")
    for i, (pname, params, _) in enumerate(all_config_runs):
        logger.info(f"  [{i + 1}/{total_configs}] {pname}: {params}")

    # Run each configuration
    all_results: list[ConfigResult] = []

    for i, (process_name, params, entry) in enumerate(all_config_runs):
        label = make_config_label(process_name, params)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Config [{i + 1}/{total_configs}]: {label}")
        logger.info(f"{'=' * 60}")

        # Resolve vocab tokens
        hmm_temp = build_hidden_markov_model(
            process_name, process_params=params, device=None,
        )
        vocab_tokens = resolve_vocab_tokens(
            hmm_temp, entry, config.default_vocab_tokens
        )
        logger.info(
            f"Process: {process_name}, params: {params}, "
            f"vocab: {vocab_tokens} ({hmm_temp.vocab_size} symbols, "
            f"{hmm_temp.num_states} states)"
        )

        # Per-config output subdirectory
        config_dir = out_dir / "configs" / label
        (config_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Run pipeline
        result = run_single_config(
            process_name=process_name,
            process_params=params,
            vocab_tokens=vocab_tokens,
            model=model,
            config=config,
            logger=logger,
            pca_plot_path=config_dir / "figures" / "pca_by_layer.png",
            pca_3d_plot_path=config_dir / "figures" / "pca_3d_by_layer.png",
            seq_length=entry.seq_length,
            n_sequences=entry.n_sequences,
        )

        # Save per-config plots
        plot_belief_simplex(
            result.belief_states_flat,
            title=f"Ground Truth Beliefs — {label}",
            path=config_dir / "figures" / "belief_simplex.png",
        )
        plot_predicted_simplex(
            result.predicted_beliefs,
            result.predicted_beliefs_gt,
            title=f"Predicted Beliefs (Best Probe, Layer {max(result.r2_per_layer, key=result.r2_per_layer.get)}) — {label}",
            path=config_dir / "figures" / "predicted_simplex.png",
        )
        plot_kl_divergence(
            result.kl_mean,
            result.kl_std,
            title=f"KL(HMM || LLM) — {label}",
            path=config_dir / "figures" / "kl_divergence.png",
        )
        plot_r2_by_layer(
            result.r2_per_layer,
            title=f"Belief State R² by Layer — {label}",
            path=config_dir / "figures" / "r2_by_layer.png",
        )

        # Save per-config metrics
        metrics = {
            "process_name": process_name,
            "process_params": params,
            "vocab_tokens": vocab_tokens,
            "r2_per_layer": {str(k): v for k, v in result.r2_per_layer.items()},
            "mse_per_layer": {str(k): v for k, v in result.mse_per_layer.items()},
            "mean_kl": float(result.kl_mean.mean()),
        }
        with open(config_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_results.append(result)
        logger.info(
            f"Config {label} complete. "
            f"Best R²={max(result.r2_per_layer.values()):.4f}"
        )

    # Cross-config comparison
    logger.info(f"\n{'=' * 60}")
    logger.info("Generating cross-config comparison plots...")
    logger.info(f"{'=' * 60}")

    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(all_results, path=comparison_dir / "comparison.png")

    # Aggregate summary
    summary = {
        "total_configs": total_configs,
        "model_name": config.model_name,
        "seq_length": config.seq_length,
        "n_sequences": config.n_sequences,
        "configs": [
            {
                "label": r.label,
                "process_name": r.process_name,
                "process_params": r.process_params,
                "best_r2": float(max(r.r2_per_layer.values())),
                "best_r2_layer": int(max(r.r2_per_layer, key=r.r2_per_layer.get)),
                "mean_kl": float(r.kl_mean.mean()),
            }
            for r in all_results
        ],
    }
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_results_csv(all_results, out_dir)

    logger.info(f"\nAll outputs written to {out_dir}")
    logger.info("Sweep summary:")
    for entry in summary["configs"]:
        logger.info(
            f"  {entry['label']}: best R²={entry['best_r2']:.4f} "
            f"(layer {entry['best_r2_layer']}), mean KL={entry['mean_kl']:.4f}"
        )


if __name__ == "__main__":
    main()
