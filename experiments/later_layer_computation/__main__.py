#!/usr/bin/env python3
"""Later-layer computation mechanism investigation.

Investigates what later layers compute beyond the belief state by
decomposing activations, training predictive decoders, and performing
causal ablation experiments.

Usage:
    python -m experiments.later_layer_computation experiments/configs/later_layer_computation.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment import setup_output_dir
from experiment_utils import get_device, load_model, setup_logging

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m experiments.later_layer_computation <config.yaml>"
        )
        sys.exit(1)

    config = load_config(sys.argv[1])
    device = get_device()
    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="later_layer_computation")

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"HMM entries to analyse: {len(config.hmm_entries)}")

    model = load_model(
        config.model_name, device, logger, n_ctx=config.n_ctx_override,
    )

    all_summaries = []

    for i, entry in enumerate(config.hmm_entries):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Entry [{i + 1}/{len(config.hmm_entries)}]: "
                     f"{entry.process_name} {entry.process_params}")
        logger.info(f"{'#' * 60}")

        summary = run_pipeline(
            entry=entry,
            model=model,
            config=config,
            out_dir=out_dir,
            logger=logger,
        )
        all_summaries.append(summary)

    # Write aggregate summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"\nAll outputs written to {out_dir}")
    logger.info("Top interpretations per config:")
    for s in all_summaries:
        top = s["interpretations"][0] if s["interpretations"] else None
        if top:
            logger.info(
                f"  {s['label']}: #{1} {top['name']} (score={top['score']:.3f})"
            )


if __name__ == "__main__":
    main()
