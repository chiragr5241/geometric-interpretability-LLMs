#!/usr/bin/env python3
"""Train encoder-decoder pairs for activation patching (SPAR-14).

Trains a linear encoder (belief-state probe) and an affine decoder
(activation reconstructor) at each target layer.  Both are trained on
pooled post-convergence activations from N_train HMM sequences and
evaluated on N_eval held-out sequences.

No BOS token is used (prepend_bos=False).  Alignment:
    cache position t  →  beliefs[t+1]
  i.e., for post_convergence_start P:
    acts[P-1:]  paired with  beliefs[P:]

Phases:
  1. Forward passes  — generate HMM sequences, run model, cache activations.
  2. Pool train data — concatenate post-convergence positions from train seqs.
  3. Train encoders  — pooled linear probe per layer (belief-state prediction).
  4. Train decoders  — pooled affine decoder per layer (activation reconstruction).
  5. Evaluate        — encoder R², decoder loss, round-trip on eval sequences.
  6. Artifacts       — save weights, metrics JSON, charts.

Usage:
    python experiments/train_encoder_decoder.py experiments/configs/train_encoder_decoder.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decoder import Decoder, DecoderInput, DecoderResult
from decoder import train_batched as train_decoders_batched
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import get_device, load_model, setup_logging
from hmm.hmm import Mess3HMM
from probes import Probe, ProbeInput, ProbeResult, train_probes_batched


@dataclass
class TrainEncoderDecoderConfig(ExperimentConfig):
    layer_indices: list[int]
    seq_length: int
    n_train_sequences: int
    n_eval_sequences: int
    post_convergence_start: int
    pooled_probes: bool
    vocab_mapping: dict[str, int]
    encoder_params: dict[str, Any]
    decoder_params: dict[str, Any]
    simplex_layers: list[int]
    max_probes_per_batch: int = 22
    max_decoders_per_batch: int = 4
    n_ctx_override: int | None = None


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _r2(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def _evaluate_encoder(
    probe: Probe,
    acts: np.ndarray,
    beliefs: np.ndarray,
) -> tuple[float, float]:
    dev = next(probe.parameters()).device
    probe.eval()
    with torch.no_grad():
        pred = probe(torch.from_numpy(acts).float().to(dev)).cpu().numpy()
    mse = float(np.mean((pred - beliefs) ** 2))
    return mse, _r2(pred, beliefs)


def _evaluate_decoder(
    decoder: Decoder,
    beliefs: np.ndarray,
    acts: np.ndarray,
) -> tuple[float, float]:
    dev = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.from_numpy(beliefs).float().to(dev)).cpu().numpy()
    mse = float(np.mean((pred - acts) ** 2))
    mean_act_norm = float(np.linalg.norm(acts, axis=-1).mean())
    return mse, mse / (mean_act_norm ** 2 + 1e-10)


def _evaluate_roundtrip(
    probe: Probe,
    decoder: Decoder,
    acts: np.ndarray,
) -> float:
    enc_dev = next(probe.parameters()).device
    dec_dev = next(decoder.parameters()).device
    with torch.no_grad():
        encoded = probe(torch.from_numpy(acts).float().to(enc_dev))
        reconstructed = decoder(encoded.to(dec_dev)).cpu().numpy()
    return float(np.mean((acts - reconstructed) ** 2))


# ── Plotting helpers ────────────────────────────────────────────────────────

def _simplex_scatter(
    predicted_beliefs: np.ndarray,
    gt_beliefs: np.ndarray,
    layer: int,
    path: Path,
) -> None:
    sqrt3 = np.sqrt(3.0)
    pred = np.clip(predicted_beliefs, 0, None)
    pred = pred / (pred.sum(axis=-1, keepdims=True) + 1e-10)
    x = pred[:, 1] + 0.5 * pred[:, 2]
    y = (sqrt3 / 2) * pred[:, 2]
    colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in np.clip(gt_beliefs, 0, 1)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=colors, size=3, opacity=0.6),
        )
    )
    corners = [
        (0, 0, "S0", "bottom center"),
        (1, 0, "S1", "bottom center"),
        (0.5, sqrt3 / 2, "SR", "top center"),
    ]
    for cx, cy, lbl, pos in corners:
        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(color="black", size=8),
                text=[lbl], textposition=pos, showlegend=False,
            )
        )
    for (x0, y0), (x1, y1) in [
        ((0, 0), (1, 0)), ((1, 0), (0.5, sqrt3 / 2)), ((0.5, sqrt3 / 2), (0, 0))
    ]:
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color="black", width=1), showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Encoder predictions — Layer {layer} (eval, N={len(x)})",
        xaxis=dict(visible=False, range=[-0.15, 1.15]),
        yaxis=dict(
            visible=False,
            range=[-0.15, sqrt3 / 2 + 0.15],
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
        height=500,
        width=520,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    fig.write_image(str(path.with_suffix(".png")))


def _decoder_loss_curves(
    decoder_results: dict[int, DecoderResult],
    layer_indices: list[int],
    path: Path,
) -> None:
    n = len(layer_indices)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layer_indices],
    )
    for idx, layer in enumerate(layer_indices):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        dr = decoder_results[layer]
        eps = list(range(len(dr.train_loss_curve)))
        show = idx == 0
        fig.add_trace(
            go.Scatter(
                x=eps, y=dr.train_loss_curve, name="Train",
                showlegend=show, line=dict(color="#1f77b4"),
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=eps, y=dr.eval_loss_curve, name="Eval",
                showlegend=show, line=dict(color="#ff7f0e"),
            ),
            row=row, col=col,
        )
        fig.update_yaxes(type="log", row=row, col=col)
    fig.update_layout(
        title="Decoder training loss curves (log y)",
        height=260 * n_rows,
        width=320 * n_cols,
    )
    fig.write_image(str(path.with_suffix(".png")))


def _layer_line_plot(
    eval_vals: list[float],
    layer_indices: list[int],
    y_title: str,
    title: str,
    path: Path,
    train_vals: list[float] | None = None,
    log_y: bool = False,
) -> None:
    layers = [str(l) for l in layer_indices]
    traces: list[go.BaseTraceType] = []
    if train_vals is not None:
        traces.append(
            go.Scatter(x=layers, y=train_vals, name="Train", mode="lines+markers")
        )
    traces.append(
        go.Scatter(x=layers, y=eval_vals, name="Eval", mode="lines+markers")
    )
    fig = go.Figure(traces)
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=y_title,
        height=420,
        width=720,
        margin=dict(t=70, b=60, l=70, r=40),
    )
    fig.write_image(str(path.with_suffix(".png")))


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder-decoder pairs")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-user",
        type=str,
        default=None,
        help="Override output_user from the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config, TrainEncoderDecoderConfig)
    apply_runtime_overrides(config, output_user=args.output_user)
    N_train = config.n_train_sequences
    N_eval = config.n_eval_sequences
    N_total = N_train + N_eval
    L = config.seq_length
    P = config.post_convergence_start  # first belief index used; acts[P-1:] ↔ beliefs[P:]
    device = get_device()

    out_dir = setup_output_dir(config)
    (out_dir / "probes" / "pooled").mkdir(parents=True, exist_ok=True)
    (out_dir / "decoders" / "pooled").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, name="enc_dec")

    logger.info(f"Output dir        : {out_dir}")
    logger.info(f"Device            : {device}")
    logger.info(f"Train / eval seqs : {N_train} / {N_eval}")
    logger.info(f"Seq length        : {L}")
    logger.info(f"Post-conv start   : {P}  ({L - P + 1} positions/seq)")
    logger.info(f"Layers            : {config.layer_indices}")

    # ── Model + HMM ────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    n_vocab = len(config.vocab_mapping)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Phase 1: Forward passes ─────────────────────────────────────────────
    logger.info("Phase 1: forward passes (no BOS) ...")
    all_acts: list[dict[int, np.ndarray]] = []
    all_beliefs: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []

    for seq_idx in range(N_total):
        label = "train" if seq_idx < N_train else "eval"
        logger.info(f"  Sequence {seq_idx + 1}/{N_total} [{label}]")

        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
        seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy()  # (L+1, n_states)

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L, (
            f"Expected {L} LLM tokens (prepend_bos=False), got {llm_tokens.shape[1]}"
        )

        with torch.no_grad():
            _, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type=None,
            )

        seq_acts = {
            layer: cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            for layer in config.layer_indices
        }
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_acts.append(seq_acts)
        all_beliefs.append(seq_beliefs)
        all_tokens.append(seq_tokens)

    # ── Phase 2: Pool post-convergence train data ───────────────────────────
    # No BOS: acts[t] ↔ beliefs[t+1].  For ≥ P tokens seen: acts[P-1:] ↔ beliefs[P:].
    logger.info("Phase 2: pooling post-convergence training data ...")
    n_pts_per_seq = L - P + 1
    logger.info(f"  {n_pts_per_seq} positions/seq × {N_train} seqs = {N_train * n_pts_per_seq} total")

    pooled_encoder_inputs: dict[int, ProbeInput] = {}
    pooled_decoder_inputs: dict[int, DecoderInput] = {}

    for layer in config.layer_indices:
        acts_list, beliefs_list, tokens_list = [], [], []
        for seq_idx in range(N_train):
            acts_list.append(all_acts[seq_idx][layer][P - 1:])   # (n_pts_per_seq, d_model)
            beliefs_list.append(all_beliefs[seq_idx][P:])        # (n_pts_per_seq, n_states)
            tokens_list.append(all_tokens[seq_idx][P - 1:])      # (n_pts_per_seq,)

        acts_pooled = np.concatenate(acts_list, axis=0)
        beliefs_pooled = np.concatenate(beliefs_list, axis=0)
        tokens_pooled = np.concatenate(tokens_list, axis=0)
        n_total = acts_pooled.shape[0]
        logger.info(f"  Layer {layer}: {n_total} pooled points")

        pooled_encoder_inputs[layer] = ProbeInput(
            activations=acts_pooled,
            gt_belief_states=beliefs_pooled,
            tokens=tokens_pooled,
            gt_next_token_preds=np.zeros((n_total, n_vocab), dtype=np.float32),
            computed_next_token_preds=np.zeros((n_total, n_vocab), dtype=np.float32),
        )
        pooled_decoder_inputs[layer] = DecoderInput(
            activations=acts_pooled,
            belief_states=beliefs_pooled,
        )

    # ── Phase 3: Train encoders ─────────────────────────────────────────────
    encoder_params = config.encoder_params
    logger.info(
        f"Phase 3: training encoders (lr={encoder_params.get('lr', 1e-3)}, "
        f"epochs={encoder_params.get('epochs', 1000)}) ..."
    )
    pooled_encoder_results: dict[int, ProbeResult] = train_probes_batched(
        pooled_encoder_inputs,
        lr=encoder_params.get("lr", 1e-3),
        epochs=encoder_params.get("epochs", 1000),
        max_probes_per_batch=config.max_probes_per_batch,
    )

    # ── Phase 4: Train decoders ─────────────────────────────────────────────
    decoder_params = config.decoder_params
    logger.info(
        f"Phase 4: training decoders (lr={decoder_params.get('lr', 1e-3)}, "
        f"max_epochs={decoder_params.get('max_epochs', 1000)}, "
        f"patience={decoder_params.get('patience', 200)}, "
        f"min_relative_improvement={decoder_params.get('min_relative_improvement', 1e-3)}) ..."
    )
    pooled_decoder_results: dict[int, DecoderResult] = train_decoders_batched(
        pooled_decoder_inputs,
        lr=decoder_params.get("lr", 1e-3),
        max_epochs=decoder_params.get("max_epochs", 1000),
        patience=decoder_params.get("patience", 200),
        min_relative_improvement=decoder_params.get("min_relative_improvement", 1e-3),
        max_decoders_per_batch=config.max_decoders_per_batch,
    )

    # ── Phase 5: Evaluate ───────────────────────────────────────────────────
    logger.info("Phase 5: evaluating on eval sequences ...")
    metrics: dict[str, dict] = {}

    for layer in config.layer_indices:
        pr = pooled_encoder_results[layer]
        dr = pooled_decoder_results[layer]

        # Encoder: internal train/eval from ProbeResult
        split_idx = pr.test_split_idx
        train_pred = pr.computed_belief_states[:split_idx]
        train_gt = pr.gt_belief_states[:split_idx]
        test_pred = pr.computed_belief_states[split_idx:]
        test_gt = pr.gt_belief_states[split_idx:]

        enc_internal_train_mse = float(np.mean((train_pred - train_gt) ** 2))
        enc_internal_train_r2 = _r2(train_pred, train_gt)
        enc_internal_test_mse = pr.test_mse
        enc_internal_test_r2 = _r2(test_pred, test_gt)

        # Eval on held-out sequences
        eval_enc_mses: list[float] = []
        eval_enc_r2s: list[float] = []
        eval_dec_losses: list[float] = []
        eval_dec_norm_losses: list[float] = []
        eval_rt_losses: list[float] = []

        for seq_idx in range(N_train, N_total):
            acts_e = all_acts[seq_idx][layer][P - 1:]
            beliefs_e = all_beliefs[seq_idx][P:]

            enc_mse, enc_r2 = _evaluate_encoder(pr.probe, acts_e, beliefs_e)
            dec_loss, dec_norm = _evaluate_decoder(dr.decoder, beliefs_e, acts_e)
            rt_loss = _evaluate_roundtrip(pr.probe, dr.decoder, acts_e)

            eval_enc_mses.append(enc_mse)
            eval_enc_r2s.append(enc_r2)
            eval_dec_losses.append(dec_loss)
            eval_dec_norm_losses.append(dec_norm)
            eval_rt_losses.append(rt_loss)

        logger.info(
            f"Layer {layer:2d}: "
            f"enc_r2 int={enc_internal_test_r2:.3f} eval={np.mean(eval_enc_r2s):.3f}  "
            f"dec_loss int={dr.eval_loss:.4e} eval={np.mean(eval_dec_losses):.4e}  "
            f"rt_eval={np.mean(eval_rt_losses):.4e}"
        )

        metrics[str(layer)] = {
            "encoder": {
                "internal_train_mse": enc_internal_train_mse,
                "internal_train_r2": enc_internal_train_r2,
                "internal_test_mse": enc_internal_test_mse,
                "internal_test_r2": enc_internal_test_r2,
                "eval_mse_per_seq": [float(v) for v in eval_enc_mses],
                "eval_r2_per_seq": [float(v) for v in eval_enc_r2s],
                "eval_mse_mean": float(np.mean(eval_enc_mses)),
                "eval_r2_mean": float(np.mean(eval_enc_r2s)),
            },
            "decoder": {
                "internal_train_loss": dr.train_loss,
                "internal_eval_loss": dr.eval_loss,
                "internal_train_normalized_loss": dr.normalized_train_loss,
                "internal_eval_normalized_loss": dr.normalized_eval_loss,
                "eval_loss_per_seq": [float(v) for v in eval_dec_losses],
                "eval_normalized_loss_per_seq": [float(v) for v in eval_dec_norm_losses],
                "eval_loss_mean": float(np.mean(eval_dec_losses)),
                "eval_normalized_loss_mean": float(np.mean(eval_dec_norm_losses)),
            },
            "roundtrip": {
                "eval_loss_per_seq": [float(v) for v in eval_rt_losses],
                "eval_loss_mean": float(np.mean(eval_rt_losses)),
            },
        }

    # ── Save artifacts ──────────────────────────────────────────────────────
    logger.info("Saving encoder/decoder weights ...")
    for layer in config.layer_indices:
        pooled_encoder_results[layer].save(out_dir / "probes" / "pooled" / f"layer_{layer}")
        pooled_decoder_results[layer].save(out_dir / "decoders" / "pooled" / f"layer_{layer}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ───────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    fig_dir = out_dir / "figures"

    _decoder_loss_curves(
        pooled_decoder_results,
        config.layer_indices,
        fig_dir / "decoder_loss_curves",
    )

    _layer_line_plot(
        train_vals=[metrics[str(l)]["encoder"]["internal_train_r2"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["encoder"]["eval_r2_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="R²",
        title="Encoder R² by layer",
        path=fig_dir / "encoder_r2_per_layer",
    )

    _layer_line_plot(
        train_vals=[metrics[str(l)]["decoder"]["internal_train_loss"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["decoder"]["eval_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Reconstruction loss (MSE)",
        title="Decoder reconstruction loss by layer (log scale)",
        path=fig_dir / "decoder_recon_loss_per_layer",
        log_y=True,
    )

    _layer_line_plot(
        train_vals=[metrics[str(l)]["decoder"]["internal_train_normalized_loss"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["decoder"]["eval_normalized_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Normalized reconstruction loss",
        title="Decoder normalized reconstruction loss by layer (log scale)",
        path=fig_dir / "decoder_normalized_loss_per_layer",
        log_y=True,
    )

    _layer_line_plot(
        eval_vals=[metrics[str(l)]["roundtrip"]["eval_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Round-trip loss (MSE)",
        title="Round-trip loss ‖act − D(E(act))‖² by layer (eval)",
        path=fig_dir / "roundtrip_loss_per_layer",
        log_y=True,
    )

    # Simplex scatter for selected layers (pooled eval sequences)
    for layer in config.simplex_layers:
        if layer not in config.layer_indices:
            continue
        pr = pooled_encoder_results[layer]
        enc_dev = next(pr.probe.parameters()).device
        all_pred, all_gt = [], []
        for seq_idx in range(N_train, N_total):
            acts_e = all_acts[seq_idx][layer][P - 1:]
            beliefs_e = all_beliefs[seq_idx][P:]
            with torch.no_grad():
                pred = pr.probe(torch.from_numpy(acts_e).float().to(enc_dev)).cpu().numpy()
            all_pred.append(pred)
            all_gt.append(beliefs_e)
        _simplex_scatter(
            np.concatenate(all_pred, axis=0),
            np.concatenate(all_gt, axis=0),
            layer,
            fig_dir / f"simplex_layer_{layer}",
        )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
