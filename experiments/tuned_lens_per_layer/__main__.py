"""Entry point: python -m experiments.tuned_lens_per_layer [config.yaml]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment_utils import get_device, load_model, setup_logging
from simplexity.generative_processes.builder import build_hidden_markov_model

from .config import TunedLensConfig, expand_param_grid, load_config, make_config_label
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

    if not config.sweeps:
        # Single config mode (backwards compatible)
        summary = run_pipeline(model, config, output_dir)
        logger.info("Experiment complete.")
        logger.info(f"Results saved to: {output_dir}")
    else:
        # Sweep mode: iterate over all process + param combinations
        all_configs: list[tuple[str, dict[str, float], list[str]]] = []
        for entry in config.sweeps:
            param_combos = expand_param_grid(entry)
            for params in param_combos:
                # Resolve vocab tokens
                hmm_temp = build_hidden_markov_model(
                    entry.process_name, process_params=params, device=None,
                )
                if entry.vocab_tokens is not None:
                    vocab = entry.vocab_tokens
                elif hmm_temp.vocab_size in config.default_vocab_tokens:
                    vocab = config.default_vocab_tokens[hmm_temp.vocab_size]
                else:
                    vocab = [chr(65 + i) for i in range(hmm_temp.vocab_size)]
                all_configs.append((entry.process_name, params, vocab))

        total = len(all_configs)
        logger.info(f"Sweep: {total} configurations")
        for i, (pname, params, vocab) in enumerate(all_configs):
            logger.info(f"  [{i+1}/{total}] {pname}: {params} vocab={vocab}")

        for i, (process_name, params, vocab) in enumerate(all_configs):
            label = make_config_label(process_name, params)
            logger.info(f"\n{'='*60}")
            logger.info(f"Config [{i+1}/{total}]: {label}")
            logger.info(f"{'='*60}")

            # Create a per-config copy of the config
            cfg = TunedLensConfig(
                experiment_name=config.experiment_name,
                model_name=config.model_name,
                n_ctx_override=config.n_ctx_override,
                process_name=process_name,
                process_params=params,
                vocab_tokens=vocab,
                seq_length=config.seq_length,
                n_sequences=config.n_sequences,
                random_seed=config.random_seed,
                layer_indices=config.layer_indices,
                n_train_sequences=config.n_train_sequences,
                tuned_lens_epochs=config.tuned_lens_epochs,
                tuned_lens_lr=config.tuned_lens_lr,
                tuned_lens_batch_size=config.tuned_lens_batch_size,
                results_dir=config.results_dir,
            )

            config_dir = output_dir / "configs" / label
            summary = run_pipeline(model, cfg, config_dir)

            logger.info(f"Config {label} complete.")

        logger.info(f"\nAll {total} configurations complete.")
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
