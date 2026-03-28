#!/usr/bin/env python3
"""SPAR-20 - Decoder fidelity diagnostic: MSE vs token probability.

For each (sequence, position) pair after the transient cutoff, compute:
  - Decoder-only MSE: ||a_t - decoder(b_t)||²
  - Round-trip MSE:   ||a_t - decoder(encoder(a_t))||²
both in raw and activation-normalized form.

Scatter each metric against p_t = P(x_t | b_{t-1}), the HMM predictive
probability of the observed token, to test whether decoder infidelity is
systematically higher for unlikely emissions (low p_t).

Usage:
    python experiments/decoder_fidelity.py experiments/configs/decoder_fidelity.yaml
    python experiments/decoder_fidelity.py experiments/configs/decoder_fidelity.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DRY_RUN_LAYERS = [0, 2, 6, 10, 17, 25]
DRY_RUN_N_SEQ = 3
DRY_RUN_SEQ_LEN = 100
DRY_RUN_TRANSIENT_CUTOFF = 20

from decoder import Decoder, DecoderResult
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import (
    build_emission_matrix,
    get_device,
    load_model,
    setup_logging,
)
from hmm.hmm import Mess3HMM
from probes import Probe, ProbeResult


@dataclass
class DecoderFidelityConfig(ExperimentConfig):
    encoder_decoder_dir: str
    layer_indices: list[int]
    n_sequences: int
    seq_length: int
    transient_cutoff: int
    batch_size: int
    vocab_mapping: dict[str, int]
    n_ctx_override: int | None = None
    n_bins: int = 20


def _binned_mean(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 0:
            means[i] = y[mask].mean()
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


def _make_grid_figure(
    layer_indices: list[int],
    x_all: np.ndarray,
    y_by_layer: dict[int, np.ndarray],
    x_label: str,
    y_label: str,
    title: str,
    n_bins: int,
    n_cols: int = 7,
) -> go.Figure:
    n_layers = len(layer_indices)
    n_cols = min(n_cols, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layer_indices],
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
    )

    n_pts = len(x_all)
    stride = max(1, n_pts // 2000)

    for panel_idx, layer in enumerate(layer_indices):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        y = y_by_layer[layer]

        fig.add_trace(
            go.Scatter(
                x=x_all[::stride],
                y=y[::stride],
                mode="markers",
                marker=dict(size=2, color="steelblue", opacity=0.35),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        bx, by = _binned_mean(x_all, y, n_bins)
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color="crimson", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text=title,
        height=320 * n_rows + 80,
        width=340 * n_cols + 60,
        showlegend=False,
    )
    fig.update_xaxes(title_text=x_label, title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text=y_label, title_font=dict(size=10), tickfont=dict(size=9))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="SPAR-20 decoder fidelity diagnostic")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            f"Quick test: layers {DRY_RUN_LAYERS}, "
            f"seq_length={DRY_RUN_SEQ_LEN}, n_sequences={DRY_RUN_N_SEQ}"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config, DecoderFidelityConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.layer_indices = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.n_sequences = DRY_RUN_N_SEQ
        config.seq_length = DRY_RUN_SEQ_LEN
        config.transient_cutoff = DRY_RUN_TRANSIENT_CUTOFF
        config.experiment_name = f"{config.experiment_name}_dry_run"

    P = config.transient_cutoff
    L = config.seq_length
    n_usable = L - P  # positions per sequence used for metrics

    device = get_device()
    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="dec_fidelity")

    logger.info(f"Output dir       : {out_dir}")
    logger.info(f"Device           : {device}")
    logger.info(f"Dry run          : {args.dry_run}")
    logger.info(f"Sequences        : {config.n_sequences}")
    logger.info(f"Seq length       : {L}")
    logger.info(
        f"Transient cutoff : {P}  "
        f"({n_usable} positions/seq, {config.n_sequences * n_usable} total)"
    )
    logger.info(f"Layers           : {config.layer_indices}")

    # ── Model + HMM ───────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    d_model = model.cfg.d_model

    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # emit_np[token_idx, state_idx] = P(token | state)  shape: (n_tokens, n_states)
    emit_np: np.ndarray = build_emission_matrix(hmm)

    # ── Phase 1: Generate sequences and run forward passes ────────────────────
    logger.info("Phase 1: generating sequences and running forward passes ...")

    all_acts: dict[int, np.ndarray] = {
        layer: np.empty((config.n_sequences, L, d_model), dtype=np.float32)
        for layer in config.layer_indices
    }
    all_beliefs = np.empty((config.n_sequences, L + 1, hmm.num_states), dtype=np.float32)
    all_tokens = np.empty((config.n_sequences, L), dtype=np.int64)

    for seq_idx in range(config.n_sequences):
        logger.info(f"  Sequence {seq_idx + 1}/{config.n_sequences}")

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

        for layer in config.layer_indices:
            all_acts[layer][seq_idx] = (
                cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            )
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_beliefs[seq_idx] = seq_beliefs
        all_tokens[seq_idx] = seq_tokens

    logger.info("Forward passes complete.")

    # ── Phase 2: Compute token probabilities for usable positions ─────────────
    # Alignment (no BOS): activation at LLM position t encodes beliefs[t+1].
    # For positions t in [P, L-1]:
    #   - beliefs_pred[t]  = beliefs[t]    (before observing x_t)
    #   - beliefs_true[t]  = beliefs[t+1]  (after observing x_t)
    #   - p_t = Σ_s beliefs_pred[t, s] · P(x_t | s)
    logger.info("Phase 2: computing token probabilities ...")

    beliefs_pred = all_beliefs[:, P:L, :]       # (n_seq, n_usable, n_states)
    beliefs_true = all_beliefs[:, P + 1:L + 1, :]  # (n_seq, n_usable, n_states)
    tokens_slice = all_tokens[:, P:L]           # (n_seq, n_usable)

    # emit_np[token_idx, state_idx] → emit_np[tokens_slice]: (n_seq, n_usable, n_states)
    emit_at_token = emit_np[tokens_slice]       # P(x_t | each state)
    token_probs = (beliefs_pred * emit_at_token).sum(axis=-1)  # (n_seq, n_usable)

    token_probs_flat: np.ndarray = token_probs.reshape(-1)          # (n_pts,)
    beliefs_true_flat: np.ndarray = beliefs_true.reshape(-1, hmm.num_states)  # (n_pts, n_states)

    n_pts = len(token_probs_flat)
    logger.info(
        f"  {n_pts} data points, "
        f"token prob range [{token_probs_flat.min():.4f}, {token_probs_flat.max():.4f}]"
    )

    # ── Phase 3: Per-layer MSE metrics ───────────────────────────────────────
    logger.info("Phase 3: computing per-layer MSE metrics ...")

    enc_dec_dir = Path(config.encoder_decoder_dir)
    decoder_base = enc_dec_dir / "decoders" / "pooled"
    probe_base = enc_dec_dir / "probes" / "pooled"

    mse_dec_by_layer: dict[int, np.ndarray] = {}
    mse_dec_norm_by_layer: dict[int, np.ndarray] = {}
    mse_rt_by_layer: dict[int, np.ndarray] = {}
    mse_rt_norm_by_layer: dict[int, np.ndarray] = {}
    summary: dict[str, dict] = {}

    beliefs_true_t = torch.from_numpy(beliefs_true_flat).float().to(device)

    for layer in config.layer_indices:
        logger.info(f"  Layer {layer} ...")

        decoder_result = DecoderResult.load(decoder_base / f"layer_{layer}")
        probe_result = ProbeResult.load(probe_base / f"layer_{layer}")

        decoder = decoder_result.decoder.to(device)
        encoder = probe_result.probe.to(device)
        decoder.eval()
        encoder.eval()

        acts = all_acts[layer][:, P:L, :].reshape(-1, d_model)  # (n_pts, d_model)
        acts_t = torch.from_numpy(acts).float().to(device)

        with torch.no_grad():
            # Decoder-only: map true belief → activation space
            a_hat = decoder(beliefs_true_t).cpu().numpy()  # (n_pts, d_model)

            # Round-trip: encode activation → clip/renorm → decode back
            b_hat_raw = encoder(acts_t).cpu().numpy()      # (n_pts, n_states)
            b_hat = np.clip(b_hat_raw, 0.0, None)
            b_hat /= b_hat.sum(axis=-1, keepdims=True) + 1e-10
            a_hat_rt = decoder(
                torch.from_numpy(b_hat).float().to(device)
            ).cpu().numpy()                                # (n_pts, d_model)

        norm_sq = (acts ** 2).sum(axis=-1) + 1e-10        # (n_pts,)

        diff_dec = acts - a_hat
        mse_dec = (diff_dec ** 2).sum(axis=-1)            # (n_pts,)
        mse_dec_norm = mse_dec / norm_sq

        diff_rt = acts - a_hat_rt
        mse_rt = (diff_rt ** 2).sum(axis=-1)              # (n_pts,)
        mse_rt_norm = mse_rt / norm_sq

        mse_dec_by_layer[layer] = mse_dec
        mse_dec_norm_by_layer[layer] = mse_dec_norm
        mse_rt_by_layer[layer] = mse_rt
        mse_rt_norm_by_layer[layer] = mse_rt_norm

        # Pearson correlation between token_prob and each MSE metric
        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            a_z = a - a.mean()
            b_z = b - b.mean()
            denom = np.sqrt((a_z ** 2).sum() * (b_z ** 2).sum()) + 1e-10
            return float((a_z * b_z).sum() / denom)

        summary[str(layer)] = {
            "pearson_dec":      _pearson(token_probs_flat, mse_dec),
            "pearson_dec_norm": _pearson(token_probs_flat, mse_dec_norm),
            "pearson_rt":       _pearson(token_probs_flat, mse_rt),
            "pearson_rt_norm":  _pearson(token_probs_flat, mse_rt_norm),
            "mean_mse_dec":     float(mse_dec.mean()),
            "mean_mse_dec_norm": float(mse_dec_norm.mean()),
            "mean_mse_rt":      float(mse_rt.mean()),
            "mean_mse_rt_norm": float(mse_rt_norm.mean()),
        }
        logger.info(
            f"    pearson(p_t, dec_norm)={summary[str(layer)]['pearson_dec_norm']:.3f}  "
            f"pearson(p_t, rt_norm)={summary[str(layer)]['pearson_rt_norm']:.3f}"
        )

    # ── Phase 4: Save metrics JSON ────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # ── Phase 5: Generate 4 scatter-grid figures ──────────────────────────────
    logger.info("Phase 5: generating figures ...")

    plot_specs = [
        (
            "dec_unnorm",
            mse_dec_by_layer,
            "Token probability p_t",
            "Decoder MSE  ||a_t − â||²",
            "Decoder-only MSE vs Token Probability (un-normalized)",
        ),
        (
            "dec_norm",
            mse_dec_norm_by_layer,
            "Token probability p_t",
            "Decoder MSE / ||a_t||²",
            "Decoder-only MSE vs Token Probability (normalized)",
        ),
        (
            "rt_unnorm",
            mse_rt_by_layer,
            "Token probability p_t",
            "Round-trip MSE  ||a_t − â_rt||²",
            "Round-trip MSE vs Token Probability (un-normalized)",
        ),
        (
            "rt_norm",
            mse_rt_norm_by_layer,
            "Token probability p_t",
            "Round-trip MSE / ||a_t||²",
            "Round-trip MSE vs Token Probability (normalized)",
        ),
    ]

    for name, y_by_layer, x_label, y_label, title in plot_specs:
        fig = _make_grid_figure(
            layer_indices=config.layer_indices,
            x_all=token_probs_flat,
            y_by_layer=y_by_layer,
            x_label=x_label,
            y_label=y_label,
            title=title,
            n_bins=config.n_bins,
        )
        png_path = out_dir / "figures" / f"{name}.png"
        fig.write_image(str(png_path))
        logger.info(f"  Saved {name}.png")

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
