#!/usr/bin/env python3
"""SPAR-29 Phase 1+2 — Per-sequence encoder-decoder training (multi-layer batched).

Phase 1:  Generate N independent sequences from the HMM and compute belief states.
          Saves tokens + beliefs to {out_dir}/hmm_data.npz (skipped if it already exists).

Phase 2a: Run all N sequences through the LLM and save per-(seq, layer) post-convergence
          activations to disk as {out_dir}/activations/seq_{i}_layer_{l}.npy.
          Then delete the model to free VRAM for training.

Phase 2b: Train encoders and decoders in a single flat batch keyed by (layer, seq_i).
          All untrained (layer, seq_i) pairs are batched together (subject to
          max_probes_per_batch / max_decoders_per_batch chunking) to maximise GPU
          utilisation on larger hardware.
          Saves weights to {out_dir}/seq_{i}/layer_{l}/ (same layout as Phase 3 expects).

Phase 2c: Evaluate trained encoders/decoders on held-out positions and generate
          per-sequence + aggregate diagnostic plots (including normalised decoder
          reconstruction loss: MSE / mean_act_norm²).

Alignment (no BOS):
    cache position t  ──▶  beliefs[t+1]

Usage:
    python experiments/train_single_seq_encoder_decoder.py \\
        experiments/configs/train_single_seq_encoder_decoder.yaml
    python experiments/train_single_seq_encoder_decoder.py \\
        experiments/configs/train_single_seq_encoder_decoder.yaml --dry-run
    python experiments/train_single_seq_encoder_decoder.py \\
        experiments/configs/train_single_seq_encoder_decoder.yaml \\
        --resume outputs/dani/20260407_191612_train_single_seq_encoder_decoder
"""
from __future__ import annotations

import argparse
import json
import shutil
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
    simplex_scatter,
)
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import build_emission_matrix, get_device, load_model, setup_logging
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
    balance: bool = True
    train_frac: float = 0.8
    layer_indices: list[int] = field(default_factory=list)
    vocab_mapping: dict[str, int] = field(default_factory=dict)
    encoder_params: dict[str, Any] = field(default_factory=dict)
    decoder_params: dict[str, Any] = field(default_factory=dict)
    simplex_layers: list[int] = field(default_factory=list)
    max_probes_per_batch: int = 22
    max_decoders_per_batch: int = 10
    n_ctx_override: int | None = None
    random_seed: int = 42
    keep_activations: bool = False


def _bin_label(p_t: float) -> int:
    if p_t < 0.3:
        return 0
    if p_t < 0.6:
        return 1
    return 2


def _compute_balanced_indices(
    tokens: np.ndarray,
    beliefs: np.ndarray,
    emit_np: np.ndarray,
    P: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    """Bin post-convergence positions by HMM token probability p_t, subsample to equal
    bin sizes, then stratify into train/eval sets.

    p_t for post-conv position k is:  belief[P-1+k] · emit_row[token[P-1+k]]
    (probability of observing the token given the prior belief state).

    Returns:
        train_idx    – indices into the post-conv slice for training (sorted)
        eval_idx     – indices into the post-conv slice for evaluation (sorted)
        p_t          – per-position token probabilities (length n_post_conv)
        counts_before – bin counts before subsampling [surprise, ambiguous, expected]
        min_bin_cap  – the per-bin size after subsampling (= min of counts_before)
    """
    L = len(tokens)
    n_post_conv = L - P + 1
    p_t = np.array(
        [float((beliefs[P - 1 + k] * emit_np[int(tokens[P - 1 + k])]).sum()) for k in range(n_post_conv)],
        dtype=np.float32,
    )

    bin_indices: list[list[int]] = [[], [], []]
    for k, p in enumerate(p_t):
        bin_indices[_bin_label(float(p))].append(k)

    counts_before = [len(b) for b in bin_indices]
    min_cap = min(counts_before)

    train_list: list[int] = []
    eval_list: list[int] = []
    for b in range(3):
        idx = np.array(bin_indices[b], dtype=np.int64)
        rng.shuffle(idx)
        idx = idx[:min_cap]
        n_tr = max(1, int(len(idx) * train_frac))
        train_list.extend(idx[:n_tr].tolist())
        eval_list.extend(idx[n_tr:].tolist())

    return (
        np.array(sorted(train_list), dtype=np.int64),
        np.array(sorted(eval_list), dtype=np.int64),
        p_t,
        counts_before,
        min_cap,
    )


def _fwd_pass_complete(act_dir: Path, seq_i: int, layer_indices: list[int]) -> bool:
    return all(
        (act_dir / f"seq_{seq_i}_layer_{l}.npy").exists()
        for l in layer_indices
    )


def _layer_training_complete(out_dir: Path, layer: int, n_sequences: int) -> bool:
    for seq_i in range(n_sequences):
        layer_dir = out_dir / f"seq_{seq_i}" / f"layer_{layer}"
        if not (layer_dir / "probe.pt").exists() or not (layer_dir / "decoder.pt").exists():
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing output directory to resume from")
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

    device = get_device()
    if args.resume:
        out_dir = Path(args.resume).resolve()
        if not out_dir.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {out_dir}")
        (out_dir / "figures").mkdir(exist_ok=True)
    else:
        out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="train_singleseq")

    L = config.seq_length
    P = config.post_convergence_start
    n_post_conv = L - P + 1

    logger.info(f"Output dir       : {out_dir}")
    logger.info(f"Device           : {device}")
    logger.info(f"Dry run          : {args.dry_run}")
    logger.info(f"Balance          : {config.balance}")
    logger.info(f"N sequences      : {config.n_sequences}")
    logger.info(f"Seq length       : {L}")
    logger.info(f"Post-conv start  : {P}")
    logger.info(f"Post-conv positions/seq: {n_post_conv}")
    logger.info(f"Layers           : {config.layer_indices}")

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
        logger.info("Phase 1: hmm_data.npz already exists, skipping generation")
        arr = np.load(npz_path)
        all_tokens = arr["tokens"]
        all_beliefs = arr["beliefs"]
    else:
        logger.info(f"Phase 1: generating {config.n_sequences} sequences of length {L} ...")
        tok_batch, _, _ = hmm.generate_dataset(config.n_sequences, L, return_states=True)
        bel_batch = hmm.compute_belief_state(tok_batch)
        all_tokens = tok_batch.cpu().numpy().astype(np.int64)
        all_beliefs = bel_batch.cpu().numpy().astype(np.float32)
        np.savez_compressed(npz_path, tokens=all_tokens, beliefs=all_beliefs)
        logger.info(f"  Saved {npz_path}")

    N = all_tokens.shape[0]

    # ── Train / eval index computation ───────────────────────────────────────
    # When balance=True: bin each post-conv position by p_t, subsample to equal
    # bin sizes, then stratify into train/eval.  When balance=False: use a
    # contiguous split at split_idx (original behaviour).
    seq_train_idx: list[np.ndarray] = []
    seq_eval_idx: list[np.ndarray] = []
    split_idx: int = -1  # used only when balance=False

    if config.balance:
        emit_np = build_emission_matrix(hmm)
        balance_rng = np.random.default_rng(config.random_seed)
        balance_stats: list[dict] = []
        logger.info("Computing per-sequence balanced train/eval indices ...")
        for seq_i in range(N):
            tr_idx, ev_idx, _p_t, counts_before, min_cap = _compute_balanced_indices(
                all_tokens[seq_i], all_beliefs[seq_i], emit_np,
                P, config.train_frac, balance_rng,
            )
            if min_cap == 0:
                raise RuntimeError(
                    f"Seq {seq_i}: smallest p_t bin is empty. "
                    "Increase seq_length or post_convergence_start."
                )
            seq_train_idx.append(tr_idx)
            seq_eval_idx.append(ev_idx)
            balance_stats.append({
                "seq_i": seq_i,
                "counts_before_balance": counts_before,
                "min_bin_cap": min_cap,
                "n_train": int(len(tr_idx)),
                "n_eval": int(len(ev_idx)),
            })
            logger.info(
                f"  Seq {seq_i}: bins (surprise/ambiguous/expected)={counts_before}, "
                f"cap={min_cap}, train={len(tr_idx)}, eval={len(ev_idx)}"
            )
        with open(out_dir / "balance_stats.json", "w") as f:
            json.dump(balance_stats, f, indent=2)
    else:
        split_idx = _eval_split_idx(L, P, config.train_eval_split)
        logger.info(f"Unbalanced split idx: {split_idx} / {n_post_conv} post-conv positions")
        for seq_i in range(N):
            seq_train_idx.append(np.arange(split_idx, dtype=np.int64))
            seq_eval_idx.append(np.arange(split_idx, n_post_conv, dtype=np.int64))

    # ── Phase 2a: forward passes ─────────────────────────────────────────────
    act_dir = out_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    needs_forward = any(
        not _fwd_pass_complete(act_dir, seq_i, config.layer_indices)
        for seq_i in range(N)
    )

    if needs_forward:
        logger.info("Phase 2a: running forward passes ...")
        model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

        for seq_i in range(N):
            if _fwd_pass_complete(act_dir, seq_i, config.layer_indices):
                logger.info(f"  Seq {seq_i}: activations already cached, skipping")
                continue

            logger.info(f"  Seq {seq_i}/{N-1}: forward pass ...")
            seq_tokens = all_tokens[seq_i]
            text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
            llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
            assert llm_tokens.shape[1] == L, f"Expected {L} LLM tokens, got {llm_tokens.shape[1]}"

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    llm_tokens, names_filter=hook_names, return_type=None,
                )

            for layer in config.layer_indices:
                acts = cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
                np.save(act_dir / f"seq_{seq_i}_layer_{layer}.npy", acts[P - 1 :])

            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        logger.info("Phase 2a: model offloaded, VRAM freed")
    else:
        logger.info("Phase 2a: all activations already cached, skipping forward passes")

    # ── Phase 2b: multi-layer batched training ───────────────────────────────
    logger.info("Phase 2b: multi-layer batched encoder-decoder training ...")
    enc_p = config.encoder_params
    dec_p = config.decoder_params

    for seq_i in range(N):
        (out_dir / f"seq_{seq_i}").mkdir(parents=True, exist_ok=True)

    pending_layers = [
        layer for layer in config.layer_indices
        if not _layer_training_complete(out_dir, layer, N)
    ]
    skipped = [l for l in config.layer_indices if l not in pending_layers]
    if skipped:
        logger.info(f"  Skipping already-complete layers: {skipped}")

    if pending_layers:
        n_pairs = len(pending_layers) * N
        logger.info(
            f"  Loading activations for {len(pending_layers)} layers × {N} sequences"
            f" = {n_pairs} (layer, seq) pairs ..."
        )
        encoder_inputs: dict[tuple[int, int], ProbeInput] = {}
        decoder_inputs: dict[tuple[int, int], DecoderInput] = {}

        for layer in pending_layers:
            for seq_i in range(N):
                acts_post_conv = np.load(act_dir / f"seq_{seq_i}_layer_{layer}.npy")
                tr_idx = seq_train_idx[seq_i]
                train_acts = acts_post_conv[tr_idx]
                train_beliefs = all_beliefs[seq_i][P + tr_idx]
                train_tokens = all_tokens[seq_i][(P - 1) + tr_idx]
                n_tr = len(tr_idx)

                encoder_inputs[(layer, seq_i)] = ProbeInput(
                    activations=train_acts,
                    gt_belief_states=train_beliefs,
                    tokens=train_tokens,
                    gt_next_token_preds=np.zeros((n_tr, n_vocab), dtype=np.float32),
                    computed_next_token_preds=np.zeros((n_tr, n_vocab), dtype=np.float32),
                )
                decoder_inputs[(layer, seq_i)] = DecoderInput(
                    activations=train_acts,
                    belief_states=train_beliefs,
                )

        logger.info(f"  Training {n_pairs} encoders (max_per_batch={config.max_probes_per_batch}) ...")
        probe_results: dict[tuple[int, int], ProbeResult] = train_probes_batched(
            encoder_inputs,
            split=1.0,
            lr=enc_p.get("lr", 1e-3),
            epochs=enc_p.get("epochs", 1000),
            max_probes_per_batch=config.max_probes_per_batch,
        )

        logger.info(f"  Training {n_pairs} decoders (max_per_batch={config.max_decoders_per_batch}) ...")
        decoder_results: dict[tuple[int, int], DecoderResult] = train_decoders_batched(
            decoder_inputs,
            split=1.0,
            lr=dec_p.get("lr", 1e-3),
            max_epochs=dec_p.get("max_epochs", 50000),
            patience=dec_p.get("patience", 400),
            min_relative_improvement=dec_p.get("min_relative_improvement", 0.01),
            max_decoders_per_batch=config.max_decoders_per_batch,
        )

        for layer in pending_layers:
            for seq_i in range(N):
                layer_dir = out_dir / f"seq_{seq_i}" / f"layer_{layer}"
                probe_results[(layer, seq_i)].save_weights_only(layer_dir)
                decoder_results[(layer, seq_i)].save(layer_dir)

        logger.info(f"  Saved {n_pairs} encoder/decoder pairs")

    # ── Phase 2c: evaluation + plots ─────────────────────────────────────────
    logger.info("Phase 2c: evaluation and plotting ...")
    per_seq_metrics: list[dict[int, dict[str, float]]] = []

    for seq_i in range(N):
        seq_dir = out_dir / f"seq_{seq_i}"
        fig_dir = seq_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        if (seq_dir / "metrics.json").exists():
            logger.info(f"  Seq {seq_i}: metrics already exist, loading")
            metrics_i = json.loads((seq_dir / "metrics.json").read_text())
            per_seq_metrics.append({
                layer: metrics_i.get(str(layer), {})
                for layer in config.layer_indices
            })
            continue

        logger.info(f"  Seq {seq_i}/{N-1}: evaluating ...")
        ev_idx = seq_eval_idx[seq_i]
        eval_beliefs_arr = all_beliefs[seq_i][P + ev_idx]

        probe_results_seq: dict[int, ProbeResult] = {}
        decoder_results_seq: dict[int, DecoderResult] = {}
        layer_metrics: dict[int, dict[str, float]] = {}

        for layer in config.layer_indices:
            layer_dir = seq_dir / f"layer_{layer}"
            pr = ProbeResult.load_weights_only(layer_dir)
            dr = DecoderResult.load(layer_dir)
            probe_results_seq[layer] = pr
            decoder_results_seq[layer] = dr

            acts_post_conv = np.load(act_dir / f"seq_{seq_i}_layer_{layer}.npy")
            eval_acts = acts_post_conv[ev_idx]

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

        for layer in config.simplex_layers:
            if layer not in config.layer_indices:
                continue
            pr = probe_results_seq[layer]
            enc_dev = next(pr.probe.parameters()).device
            acts_post_conv = np.load(act_dir / f"seq_{seq_i}_layer_{layer}.npy")
            simplex_eval_acts = acts_post_conv[ev_idx]
            with torch.no_grad():
                pred = pr.probe(torch.from_numpy(simplex_eval_acts).float().to(enc_dev)).cpu().numpy()
            simplex_scatter(pred, eval_beliefs_arr, layer, fig_dir / f"simplex_layer_{layer}")

        per_seq_metrics.append(layer_metrics)

        with open(seq_dir / "metrics.json", "w") as f:
            json.dump({str(l): v for l, v in layer_metrics.items()}, f, indent=2)

        decoder_loss_curves(decoder_results_seq, config.layer_indices, fig_dir / "decoder_loss_curves")
        encoder_mse_curves(probe_results_seq, config.layer_indices, fig_dir / "encoder_mse_curves")
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
            eval_vals=[layer_metrics[l]["eval_dec_norm_loss"] for l in config.layer_indices],
            layer_indices=config.layer_indices,
            y_title="MSE / mean_norm²",
            title=f"Normalised decoder reconstruction loss by layer — Seq {seq_i}",
            path=fig_dir / "decoder_recon_norm_loss_per_layer",
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
        eval_vals=[agg_metrics[str(l)]["eval_dec_norm_loss"]["mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="MSE / mean_norm² (mean across sequences)",
        title="Normalised decoder reconstruction loss by layer — aggregated",
        path=agg_fig_dir / "decoder_recon_norm_loss_per_layer",
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

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if not config.keep_activations:
        logger.info("Cleaning up activation cache ...")
        shutil.rmtree(act_dir, ignore_errors=True)

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
