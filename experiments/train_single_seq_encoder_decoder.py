#!/usr/bin/env python3
"""SPAR-29 Phase 1+2 — Per-sequence encoder-decoder training.

Phase 1: Generate N independent sequences from the HMM and compute belief states.
         Saves tokens + beliefs to {out_dir}/hmm_data.npz (skipped if it already exists).

Phase 2: For each sequence, train a linear encoder (probe) and affine decoder at every
         target layer using post-convergence positions [P-1, T_split_act).  Saves only
         weights — no activations — to {out_dir}/seq_{i}/layer_{l}/ (checkpoints per
         sequence so a crash after seq 7/10 leaves seqs 0-6 reusable).

Alignment (no BOS):
    cache position t  ──▶  beliefs[t+1]

Usage:
    python experiments/train_single_seq_encoder_decoder.py \\
        experiments/configs/train_single_seq_encoder_decoder.yaml
    python experiments/train_single_seq_encoder_decoder.py \\
        experiments/configs/train_single_seq_encoder_decoder.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decoder import DecoderInput, DecoderResult
from decoder import train_batched as train_decoders_batched
from encoder_decoder_utils import (
    decoder_loss_curves,
    encoder_mse_curves,
    evaluate_decoder,
    evaluate_encoder,
    evaluate_roundtrip,
    layer_line_plot,
    r2,
    simplex_scatter,
)
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import get_device, load_model, setup_logging
from hmm.hmm import Mess3HMM
from probes import ProbeInput, ProbeResult, train_probes_batched

DRY_RUN_N_SEQ = 2
DRY_RUN_SEQ_LEN = 80
DRY_RUN_TRANSIENT = 30
DRY_RUN_LAYERS = [0, 2, 10, 17, 27]


@dataclass
class TrainSingleSeqEncoderDecoderConfig(ExperimentConfig):
    n_sequences: int = 10
    seq_length: int = 1500
    post_convergence_start: int = 30
    train_eval_split: float = 0.7
    layer_indices: list[int] = field(default_factory=list)
    vocab_mapping: dict[str, int] = field(default_factory=dict)
    encoder_params: dict[str, Any] = field(default_factory=dict)
    decoder_params: dict[str, Any] = field(default_factory=dict)
    simplex_layers: list[int] = field(default_factory=list)
    max_probes_per_batch: int = 22
    max_decoders_per_batch: int = 4
    n_ctx_override: int | None = None
    random_seed: int = 42


def _seq_complete(seq_dir: Path, layer_indices: list[int]) -> bool:
    for layer in layer_indices:
        layer_dir = seq_dir / f"layer_{layer}"
        if not (layer_dir / "probe.pt").exists():
            return False
        if not (layer_dir / "decoder.pt").exists():
            return False
    return True


def _eval_split_idx(seq_length: int, post_convergence_start: int, train_eval_split: float) -> int:
    n_post_conv = seq_length - post_convergence_start + 1
    return int(n_post_conv * train_eval_split)


def _aggregate_layer_metrics(
    per_seq: list[dict[int, dict[str, float]]],
    layer_indices: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer in layer_indices:
        vals: dict[str, list[float]] = {}
        for seq_metrics in per_seq:
            for k, v in seq_metrics[layer].items():
                vals.setdefault(k, []).append(v)
        out[str(layer)] = {
            k: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "per_seq": vs}
            for k, vs in vals.items()
        }
    return out


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Per-sequence encoder-decoder training (SPAR-29)")
    parser.add_argument("config", type=str)
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, TrainSingleSeqEncoderDecoderConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.n_sequences = DRY_RUN_N_SEQ
        config.seq_length = DRY_RUN_SEQ_LEN
        config.post_convergence_start = DRY_RUN_TRANSIENT
        config.layer_indices = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.experiment_name = f"{config.experiment_name}_dry_run"
        config.encoder_params = {**config.encoder_params, "epochs": 60}
        config.decoder_params = {
            **config.decoder_params,
            "max_epochs": 300,
            "patience": 30,
        }

    rng = np.random.default_rng(config.random_seed)
    device = get_device()
    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="train_singleseq")

    L = config.seq_length
    P = config.post_convergence_start
    split_idx = _eval_split_idx(L, P, config.train_eval_split)
    n_post_conv = L - P + 1

    logger.info(f"Output dir       : {out_dir}")
    logger.info(f"Device           : {device}")
    logger.info(f"Dry run          : {args.dry_run}")
    logger.info(f"N sequences      : {config.n_sequences}")
    logger.info(f"Seq length       : {L}")
    logger.info(f"Post-conv start  : {P}")
    logger.info(f"Train split idx  : {split_idx} / {n_post_conv} post-conv positions")
    logger.info(f"Layers           : {config.layer_indices}")

    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype

    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_vocab = len(config.vocab_mapping)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Phase 1: data generation ─────────────────────────────────────────────
    npz_path = out_dir / "hmm_data.npz"
    if npz_path.exists():
        logger.info(f"Phase 1: hmm_data.npz already exists, skipping generation")
        arr = np.load(npz_path)
        all_tokens = arr["tokens"]    # (N, L) int64
        all_beliefs = arr["beliefs"]  # (N, L+1, n_states) float32
    else:
        logger.info(f"Phase 1: generating {config.n_sequences} sequences of length {L} ...")
        # generate_dataset returns (tokens, tokens_y, states) — we only need tokens
        tok_batch, _, _ = hmm.generate_dataset(config.n_sequences, L, return_states=True)
        bel_batch = hmm.compute_belief_state(tok_batch)
        all_tokens = tok_batch.cpu().numpy().astype(np.int64)    # (N, L)
        all_beliefs = bel_batch.cpu().numpy().astype(np.float32)  # (N, L+1, n_states)
        np.savez_compressed(npz_path, tokens=all_tokens, beliefs=all_beliefs)
        logger.info(f"  Saved {npz_path}")

    N = all_tokens.shape[0]

    # ── Phase 2: per-sequence training ───────────────────────────────────────
    logger.info("Phase 2: per-sequence encoder-decoder training ...")
    enc_p = config.encoder_params
    dec_p = config.decoder_params

    per_seq_metrics: list[dict[int, dict[str, float]]] = []

    for seq_i in range(N):
        seq_dir = out_dir / f"seq_{seq_i}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        (seq_dir / "figures").mkdir(exist_ok=True)

        if _seq_complete(seq_dir, config.layer_indices):
            logger.info(f"  Seq {seq_i}: already complete, loading metrics")
            metrics_i = json.loads((seq_dir / "metrics.json").read_text()) if (seq_dir / "metrics.json").exists() else {}
            per_seq_metrics.append({
                layer: metrics_i.get(str(layer), {})
                for layer in config.layer_indices
            })
            continue

        logger.info(f"  Seq {seq_i}/{N-1}: forward pass ...")
        seq_tokens = all_tokens[seq_i]    # (L,) int64
        seq_beliefs = all_beliefs[seq_i]  # (L+1, n_states)

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L, f"Expected {L} LLM tokens, got {llm_tokens.shape[1]}"

        with torch.no_grad():
            _, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type=None,
            )

        # Training slice: act positions [P-1, P-1+split_idx), beliefs [P, P+split_idx)
        # Eval slice:     act positions [P-1+split_idx, L),   beliefs [P+split_idx, L+1)
        #   alignment: cache position t → beliefs[t+1]
        train_act_start = P - 1
        train_act_end = P - 1 + split_idx
        eval_act_start = train_act_end

        # Extract all per-layer act arrays from the single forward pass
        all_layer_acts: dict[int, np.ndarray] = {
            layer: cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            for layer in config.layer_indices
        }
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        encoder_inputs: dict[int, ProbeInput] = {}
        decoder_inputs: dict[int, DecoderInput] = {}
        for layer in config.layer_indices:
            acts_full = all_layer_acts[layer]
            train_acts = acts_full[train_act_start:train_act_end]
            train_beliefs = seq_beliefs[P : P + split_idx]
            encoder_inputs[layer] = ProbeInput(
                activations=train_acts,
                gt_belief_states=train_beliefs,
                tokens=seq_tokens[train_act_start:train_act_end],
                gt_next_token_preds=np.zeros((split_idx, n_vocab), dtype=np.float32),
                computed_next_token_preds=np.zeros((split_idx, n_vocab), dtype=np.float32),
            )
            decoder_inputs[layer] = DecoderInput(
                activations=train_acts,
                belief_states=train_beliefs,
            )

        logger.info(f"  Seq {seq_i}: training encoders ...")
        probe_results: dict[int, ProbeResult] = train_probes_batched(
            encoder_inputs,
            split=1.0,  # sliced manually above
            lr=enc_p.get("lr", 1e-3),
            epochs=enc_p.get("epochs", 1000),
            max_probes_per_batch=config.max_probes_per_batch,
        )

        logger.info(f"  Seq {seq_i}: training decoders ...")
        decoder_results: dict[int, DecoderResult] = train_decoders_batched(
            decoder_inputs,
            split=1.0,  # sliced manually above
            lr=dec_p.get("lr", 1e-3),
            max_epochs=dec_p.get("max_epochs", 50000),
            patience=dec_p.get("patience", 400),
            min_relative_improvement=dec_p.get("min_relative_improvement", 0.01),
            max_decoders_per_batch=config.max_decoders_per_batch,
        )

        # ── Evaluate on held-out eval slice (re-use all_layer_acts) ──────────
        layer_metrics: dict[int, dict[str, float]] = {}
        eval_beliefs_arr = seq_beliefs[P + split_idx : L + 1]

        for layer in config.layer_indices:
            eval_acts = all_layer_acts[layer][eval_act_start:]
            pr = probe_results[layer]
            dr = decoder_results[layer]

            enc_mse, enc_r2 = evaluate_encoder(pr.probe, eval_acts, eval_beliefs_arr)
            dec_loss, dec_norm = evaluate_decoder(dr.decoder, eval_beliefs_arr, eval_acts)
            rt_loss = evaluate_roundtrip(pr.probe, dr.decoder, eval_acts)

            layer_metrics[layer] = {
                "train_mse": float(pr.test_mse),
                "eval_enc_mse": enc_mse,
                "eval_enc_r2": enc_r2,
                "eval_dec_loss": dec_loss,
                "eval_dec_norm_loss": dec_norm,
                "eval_roundtrip_loss": rt_loss,
            }
            logger.info(
                f"    Layer {layer:2d}: enc_r2={enc_r2:.3f}  dec_loss={dec_loss:.3e}  rt={rt_loss:.3e}"
            )

        # Use all_layer_acts as the "eval_cache" for simplex scatter below
        eval_cache = all_layer_acts

        # ── Per-sequence simplex scatter ──────────────────────────────────────
        fig_dir = seq_dir / "figures"
        for layer in config.simplex_layers:
            if layer not in config.layer_indices:
                continue
            pr = probe_results[layer]
            enc_dev = next(pr.probe.parameters()).device
            acts_s = eval_cache[layer][eval_act_start:]
            with torch.no_grad():
                pred = pr.probe(torch.from_numpy(acts_s).float().to(enc_dev)).cpu().numpy()
            simplex_scatter(pred, eval_beliefs_arr, layer, fig_dir / f"simplex_layer_{layer}")

        del all_layer_acts

        per_seq_metrics.append(layer_metrics)

        # ── Save weights ─────────────────────────────────────────────────────
        logger.info(f"  Seq {seq_i}: saving weights ...")
        for layer in config.layer_indices:
            layer_dir = seq_dir / f"layer_{layer}"
            probe_results[layer].save_weights_only(layer_dir)
            decoder_results[layer].save(layer_dir)

        with open(seq_dir / "metrics.json", "w") as f:
            json.dump({str(l): v for l, v in layer_metrics.items()}, f, indent=2)

        # ── Per-sequence diagnostic plots ─────────────────────────────────────
        decoder_loss_curves(decoder_results, config.layer_indices, fig_dir / "decoder_loss_curves")
        encoder_mse_curves(probe_results, config.layer_indices, fig_dir / "encoder_mse_curves")
        layer_line_plot(
            train_vals=[layer_metrics[l]["train_mse"] for l in config.layer_indices],
            eval_vals=[layer_metrics[l]["eval_enc_mse"] for l in config.layer_indices],
            layer_indices=config.layer_indices,
            y_title="MSE",
            title=f"Encoder MSE by layer — Seq {seq_i}",
            path=fig_dir / "encoder_mse_per_layer",
        )
        layer_line_plot(
            eval_vals=[layer_metrics[l]["eval_enc_r2"] for l in config.layer_indices],
            layer_indices=config.layer_indices,
            y_title="R²",
            title=f"Encoder R² by layer — Seq {seq_i}",
            path=fig_dir / "encoder_r2_per_layer",
        )
        layer_line_plot(
            eval_vals=[layer_metrics[l]["eval_dec_loss"] for l in config.layer_indices],
            layer_indices=config.layer_indices,
            y_title="MSE",
            title=f"Decoder reconstruction loss by layer — Seq {seq_i}",
            path=fig_dir / "decoder_recon_loss_per_layer",
            log_y=True,
        )
        layer_line_plot(
            eval_vals=[layer_metrics[l]["eval_roundtrip_loss"] for l in config.layer_indices],
            layer_indices=config.layer_indices,
            y_title="Round-trip MSE",
            title=f"Round-trip loss by layer — Seq {seq_i}",
            path=fig_dir / "roundtrip_loss_per_layer",
            log_y=True,
        )

    # ── Aggregate metrics + plots ─────────────────────────────────────────────
    logger.info("Aggregating metrics across sequences ...")
    agg_metrics = _aggregate_layer_metrics(per_seq_metrics, config.layer_indices)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(agg_metrics, f, indent=2)

    agg_fig_dir = out_dir / "figures"

    layer_line_plot(
        eval_vals=[agg_metrics[str(l)]["eval_enc_r2"]["mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="R² (mean ± std)",
        title="Encoder R² by layer — aggregated across sequences",
        path=agg_fig_dir / "encoder_r2_per_layer",
    )
    layer_line_plot(
        eval_vals=[agg_metrics[str(l)]["eval_dec_loss"]["mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Reconstruction MSE",
        title="Decoder reconstruction loss by layer — aggregated",
        path=agg_fig_dir / "decoder_recon_loss_per_layer",
        log_y=True,
    )
    layer_line_plot(
        eval_vals=[agg_metrics[str(l)]["eval_roundtrip_loss"]["mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Round-trip MSE",
        title="Round-trip loss by layer — aggregated",
        path=agg_fig_dir / "roundtrip_loss_per_layer",
        log_y=True,
    )

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
