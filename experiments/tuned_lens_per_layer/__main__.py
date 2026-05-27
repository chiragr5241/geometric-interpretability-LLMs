"""Entry point: python -m experiments.tuned_lens_per_layer [config.yaml]"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from experiment_utils import get_device, load_model, setup_logging
from simplexity.generative_processes.builder import build_hidden_markov_model

from .config import TunedLensConfig, expand_param_grid, load_config, make_config_label
from .pipeline import run_pipeline


def _setup_output_dir(config: TunedLensConfig, override: str | None = None) -> Path:
    """Create timestamped output directory, matching belief_state_sweep convention."""
    if override:
        output_dir = Path(override)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "outputs" / config.output_user / f"{timestamp}_{config.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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

    # Setup output directory (timestamped like belief_state_sweep)
    output_dir = _setup_output_dir(config, args.output_dir)

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
        # Single config mode: release backbone after training to free GPU memory.
        t0 = time.time()
        summary = run_pipeline(model, config, output_dir, release_backbone=True)
        logger.info(f"Experiment complete. Total: {time.time() - t0:.1f}s")
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

        sweep_t0 = time.time()
        per_config_times: list[float] = []
        for i, (process_name, params, vocab) in enumerate(all_configs):
            label = make_config_label(process_name, params)
            cfg_t0 = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Config [{i+1}/{total}]: {label}  (started at {time.strftime('%H:%M:%S')})")
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
                tuned_lens_optimizer=config.tuned_lens_optimizer,
                use_bf16=config.use_bf16,
                forward_chunk_size=config.forward_chunk_size,
                train_pos_window=config.train_pos_window,
                train_tuned_full=config.train_tuned_full,
                train_tuned_concept=config.train_tuned_concept,
                train_tuned_hmm=config.train_tuned_hmm,
                train_hmm_target=config.train_hmm_target,
                model_target_full_vocab=config.model_target_full_vocab,
                results_dir=config.results_dir,
            )

            config_dir = output_dir / "configs" / label
            # Keep backbone on GPU between configs (forward pass of next config needs it).
            run_pipeline(model, cfg, config_dir, release_backbone=False)
            gc.collect()
            torch.cuda.empty_cache()

            cfg_dt = time.time() - cfg_t0
            per_config_times.append(cfg_dt)
            avg = sum(per_config_times) / len(per_config_times)
            remaining = avg * (total - i - 1)
            logger.info(
                f"Config {label} complete in {cfg_dt:.1f}s "
                f"(avg {avg:.1f}s/cfg, ETA {remaining/60:.1f}m for remaining {total - i - 1})"
            )

        logger.info(f"\nAll {total} configurations complete in {(time.time() - sweep_t0)/60:.1f} min.")
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
