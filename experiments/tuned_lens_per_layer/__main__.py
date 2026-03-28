"""Entry point: python -m experiments.tuned_lens_per_layer [config.yaml]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment_utils import get_device, load_model, setup_logging

from .config import TunedLensConfig, load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer tuned lens experiment")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file. If omitted, uses defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TunedLensConfig()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, name="tuned_lens")
    logging.getLogger("experiments.tuned_lens_per_layer").setLevel(logging.INFO)
    for handler in logger.handlers:
        logging.getLogger("experiments.tuned_lens_per_layer").addHandler(handler)

    logger.info(f"Config: {config}")
    logger.info(f"Output: {output_dir}")

    # Load model
    device = get_device()
    model = load_model(
        config.model_name,
        device,
        logger,
        n_ctx=config.n_ctx_override,
    )

    # Run pipeline
    summary = run_pipeline(model, config, output_dir)

    logger.info("Experiment complete.")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
