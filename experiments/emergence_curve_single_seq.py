#!/usr/bin/env python3
"""
SPAR-35 — Belief geometry emergence curve: per-sequence probes.

Loads per-sequence probes from a prior `train_single_seq_encoder_decoder` output
directory, reuses that directory's tokens + beliefs (hmm_data.npz), and evaluates
each sequence's own probe across all positions to reveal when belief geometry
emerges relative to KL convergence.  Unlike emergence_curve.py (which uses pooled
probes on freshly generated sequences), here each sequence is assessed with the
probe trained on its post-convergence activations, so the transient (positions
0–50) is genuinely held-out.

Usage:
    python experiments/emergence_curve_single_seq.py \\
        experiments/configs/emergence_curve_single_seq.yaml
    python experiments/emergence_curve_single_seq.py \\
        experiments/configs/emergence_curve_single_seq.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import (
    build_emission_matrix,
    compute_optimal_probs,
    get_device,
    get_model_probs_projected,
    load_model,
    resolve_hmm_token_ids,
    setup_logging,
)
from hmm import build_process_hmm
from metrics.probe_metrics import (
    compute_kl,
    find_kl_convergence_patience,
    find_r2_emergence_patience,
)
from plot_titles import format_hmm_process
from probes import Probe, ProbeResult

DRY_RUN_N_SEQ = 2
DRY_RUN_LAYERS = [0, 10, 27]


@dataclass
class EmergenceCurveSingleSeqConfig(ExperimentConfig):
    probe_dir: str
    layer_indices: list[int]
    sliding_window: int
    kl_params: dict
    r2_params: dict
    vocab_mapping: dict[str, int]
    max_plot_position: int = 75
    n_ctx_override: int | None = None
    strip_bos: bool = True
    seq_indices: list[int] | None = None  # None = use all sequences in the probe dir


# ── Probe loading ─────────────────────────────────────────────────────────────

def _load_per_seq_probes(
    probe_dir: Path,
    seq_indices: list[int],
    layer_indices: list[int],
    device: torch.device,
    logger,
) -> dict[int, dict[int, ProbeResult]]:
    """Load per-sequence per-layer probes.

    Returns: {seq_i: {layer: ProbeResult}}
    """
    probes: dict[int, dict[int, ProbeResult]] = {}
    for seq_i in seq_indices:
        probes[seq_i] = {}
        for layer in layer_indices:
            path = probe_dir / f"seq_{seq_i}" / f"layer_{layer}"
            pr = ProbeResult.load_weights_only(path)
            pr.probe.to(device)
            probes[seq_i][layer] = pr
    logger.info(
        f"Loaded {len(seq_indices)} × {len(layer_indices)} per-sequence probes"
    )
    return probes


# ── Config validation ─────────────────────────────────────────────────────────

def _validate_against_probe_dir(
    config: EmergenceCurveSingleSeqConfig,
    probe_dir_config: dict,
    logger,
) -> None:
    """Error out if key params don't match the probe dir's config."""
    probe_hmm = probe_dir_config.get("hmm", {})
    checks = {
        "n_sequences": (None, probe_dir_config.get("n_sequences")),
        "seq_length": (None, probe_dir_config.get("seq_length")),
        "vocab_mapping": (config.vocab_mapping, probe_dir_config.get("vocab_mapping")),
        "hmm_process_name": (config.hmm.process_name, probe_hmm.get("process_name")),
        "hmm_process_params": (config.hmm.process_params, probe_hmm.get("process_params")),
    }
    for key, (mine, theirs) in checks.items():
        if theirs is None:
            continue
        if mine is not None and mine != theirs:
            raise ValueError(
                f"Config mismatch for '{key}': this config has {mine!r}, "
                f"probe_dir has {theirs!r}"
            )
    logger.info("Probe dir config validation passed.")


# ── Computation helpers ───────────────────────────────────────────────────────

def _apply_probe(
    probe: Probe,
    acts: np.ndarray,   # (T, d_model)
    beliefs: np.ndarray,  # (T, n_states)
) -> tuple[np.ndarray, np.ndarray]:
    """Apply probe at every position. Returns (se, preds).

    se[t] = ||pred_t - belief_t||² summed over states.
    preds: (T, n_states).
    """
    device = next(probe.parameters()).device
    probe.eval()
    with torch.no_grad():
        preds = probe(torch.from_numpy(acts).float().to(device)).cpu().numpy()
    se = ((preds - beliefs) ** 2).sum(axis=-1)
    return se, preds


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_fig(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")))
    except Exception:
        pass


_SS_TOT_MIN = 1e-3  # below this the window beliefs are near-constant; R² is undefined


def _sliding_r2(se: np.ndarray, beliefs: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window R²(t) for t = 0 .. T-1, with variable-width edges.

    Returns nan at positions where beliefs have near-zero variance in the window
    (long same-state run), because R² is undefined in that case.
    """
    T = len(se)
    half = window // 2
    cs_se = np.concatenate([[0.0], np.cumsum(se)])
    norm_sq = (beliefs ** 2).sum(axis=1)
    cs_norm_sq = np.concatenate([[0.0], np.cumsum(norm_sq)])
    cs_b = np.vstack([np.zeros((1, beliefs.shape[1])), np.cumsum(beliefs, axis=0)])
    r2 = np.full(T, np.nan)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        n = hi - lo
        ss_res = float(cs_se[hi] - cs_se[lo])
        sum_norm_sq = float(cs_norm_sq[hi] - cs_norm_sq[lo])
        sum_b = cs_b[hi] - cs_b[lo]
        ss_tot = sum_norm_sq - float((sum_b ** 2).sum()) / n
        if ss_tot < _SS_TOT_MIN:
            continue  # beliefs near-constant in window; leave r2[t] = nan
        r2[t] = 1.0 - ss_res / ss_tot
    return r2



# ── Plotting helpers ──────────────────────────────────────────────────────────

def _layer_colors(layer_indices: list[int]) -> list[str]:
    import plotly.colors as pc
    n = len(layer_indices)
    return pc.sample_colorscale("Plasma", [i / max(n - 1, 1) for i in range(n)])


def _band_trace(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color_rgba: str,
    yaxis: str = "y",
    clip_positive: bool = False,
) -> go.Scatter:
    valid = ~np.isnan(mean)
    xv, mv, sv = x[valid], mean[valid], std[valid]
    upper = mv + sv
    lower = mv - sv
    if clip_positive:
        lower = np.clip(lower, 1e-8, None)
        upper = np.clip(upper, 1e-8, None)
    return go.Scatter(
        x=np.concatenate([xv, xv[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor=color_rgba,
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        mode="lines",
        yaxis=yaxis,
    )


def _plot_convergence(
    positions: np.ndarray,
    kl_positions: np.ndarray,
    metric_mean: dict[int, np.ndarray],  # {layer: (T,)}
    metric_std: dict[int, np.ndarray],
    kl_mean: np.ndarray,
    kl_std: np.ndarray,
    layer_indices: list[int],
    metric_name: str,
    log_kl: bool,
    t_kl_rep: int,
    sliding_window: int,
    hmm_subtitle: str,
    path: Path,
) -> None:
    """One convergence figure: one line per layer (metric) + one KL line."""
    colors = _layer_colors(layer_indices)
    kl_type = "log" if log_kl else "linear"
    kl_axis_type = "log" if log_kl else "linear"

    fig = go.Figure()

    # KL band + mean (right y-axis, yaxis2)
    kl_clip = log_kl
    fig.add_trace(_band_trace(kl_positions, kl_mean, kl_std,
                               "rgba(180,34,34,0.07)", "y2", clip_positive=kl_clip))
    fig.add_trace(go.Scatter(
        x=kl_positions, y=kl_mean,
        name="KL (smoothed)",
        line=dict(color="firebrick", width=2),
        mode="lines", yaxis="y2",
    ))

    # Metric per layer (left y-axis, yaxis)
    for color, layer in zip(colors, layer_indices):
        m = metric_mean[layer]
        s = metric_std[layer]
        rgba = color.replace("rgb", "rgba").replace(")", ",0.10)")
        fig.add_trace(_band_trace(positions, m, s, rgba, "y"))
        fig.add_trace(go.Scatter(
            x=positions, y=m,
            name=f"Layer {layer}",
            line=dict(color=color, width=1.8),
            mode="lines", yaxis="y",
        ))

    # Vertical KL threshold marker (x-axis is 1-indexed)
    t_kl_x = t_kl_rep + 1
    if kl_positions[0] <= t_kl_x <= kl_positions[-1]:
        fig.add_shape(
            type="line", x0=t_kl_x, x1=t_kl_x, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="firebrick", width=1.5, dash="dash"),
        )

    ylabel = f"Sliding R² (W={sliding_window})" if metric_name == "R²" else "MSE (‖ŷ−b‖²)"
    fig.update_layout(
        title=(
            f"{metric_name} + KL ({kl_type}) — per-sequence probes<br>"
            f"<sup>{hmm_subtitle} | Mean ± std across sequences | t_KL={t_kl_rep}</sup>"
        ),
        height=500, width=1020,
        margin=dict(t=80, b=110, l=70, r=140),
        legend=dict(
            orientation="v", x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
        ),
        xaxis=dict(title="HMM tokens seen (BOS stripped)", domain=[0.0, 0.82]),
        yaxis=dict(title=ylabel),
        yaxis2=dict(
            title="KL (nats)", overlaying="y", side="right", anchor="x",
            showgrid=False, type=kl_axis_type,
        ),
    )
    _save_fig(fig, path)


def _plot_lag_summary(
    layer_indices: list[int],
    lag_means: list[float | None],
    lag_stds: list[float | None],
    n_seqs_reached: list[int],
    r2_patience: int,
    kl_patience: int,
    hmm_subtitle: str,
    path: Path,
) -> None:
    labels = [str(l) for l in layer_indices]
    y_vals, err_vals, colors, texts = [], [], [], []
    for lm, ls, n in zip(lag_means, lag_stds, n_seqs_reached):
        if lm is None:
            y_vals.append(0.0)
            err_vals.append(0.0)
            colors.append("lightgray")
            texts.append("N/A")
        else:
            y_vals.append(lm)
            err_vals.append(ls if ls is not None else 0.0)
            colors.append("firebrick" if lm > 0 else "steelblue")
            texts.append(f"{lm:+.0f} (n={n})")

    fig = go.Figure(go.Bar(
        x=labels, y=y_vals,
        error_y=dict(type="data", array=err_vals, visible=True),
        marker_color=colors,
        text=texts,
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=(
            "Lag: t_R² − t_KL per layer (patience-based)<br>"
            f"<sup>{hmm_subtitle} | R² patience={r2_patience}, KL patience={kl_patience} | "
            "Positive = geometry lags output; Negative = geometry leads</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Lag (tokens)",
        height=420, width=760,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Belief geometry emergence curve (per-sequence probes)")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with 2 sequences and 3 layers to verify the pipeline")
    args = parser.parse_args()

    config = load_config(args.config, EmergenceCurveSingleSeqConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.layer_indices = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.experiment_name = f"{config.experiment_name}_dry_run"

    device = get_device(config.device)

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="emergence_curve_single_seq")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = out_dir / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    probe_dir = Path(config.probe_dir)
    if not probe_dir.is_absolute():
        probe_dir = Path(__file__).resolve().parent.parent / probe_dir

    logger.info(f"Output dir  : {out_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Probe dir   : {probe_dir}")
    logger.info(f"Layers      : {config.layer_indices}")

    # ── Load and validate probe dir config ───────────────────────────────────
    probe_dir_config_path = probe_dir / "config.yaml"
    with open(probe_dir_config_path) as f:
        probe_dir_cfg = yaml.safe_load(f)

    n_sequences: int = probe_dir_cfg["n_sequences"]
    seq_length: int = probe_dir_cfg["seq_length"]

    # Resolve which sequence indices to run
    all_indices = list(range(n_sequences))
    if config.seq_indices is not None:
        invalid = [i for i in config.seq_indices if i not in all_indices]
        if invalid:
            raise ValueError(
                f"seq_indices contains out-of-range indices {invalid} "
                f"(probe dir has {n_sequences} sequences: 0–{n_sequences - 1})"
            )
        seq_run_indices = list(config.seq_indices)
    else:
        seq_run_indices = all_indices

    if args.dry_run:
        seq_run_indices = seq_run_indices[:DRY_RUN_N_SEQ]

    logger.info(f"Probe dir   : n_seq_total={n_sequences}, seq_len={seq_length}")
    logger.info(f"Sequences   : {seq_run_indices}")
    logger.info(f"Dry run     : {args.dry_run}")

    _validate_against_probe_dir(config, probe_dir_cfg, logger)

    # ── Load HMM data ─────────────────────────────────────────────────────────
    npz_path = probe_dir / "hmm_data.npz"
    arr = np.load(npz_path)
    all_tokens = arr["tokens"][:n_sequences]   # (N, L)  int64
    all_beliefs = arr["beliefs"][:n_sequences] # (N, L+1, n_states)  float32
    assert all_tokens.shape == (n_sequences, seq_length), (
        f"hmm_data tokens shape {all_tokens.shape} != ({n_sequences}, {seq_length})"
    )
    logger.info(f"Loaded hmm_data.npz: tokens={all_tokens.shape}, beliefs={all_beliefs.shape}")

    # ── Load per-sequence probes ──────────────────────────────────────────────
    probes = _load_per_seq_probes(probe_dir, seq_run_indices, config.layer_indices, device, logger)

    # ── Model + HMM ──────────────────────────────────────────────────────────
    model = load_model(
        config.model_name,
        device,
        logger,
        n_ctx=config.n_ctx_override,
        model_n_devices=config.model_n_devices,
    )

    hmm = build_process_hmm(
        config.hmm.process_name,
        config.hmm.process_params,
        hmm_device=device,
    )
    logger.info(f"HMM         : {config.hmm.process_name} {config.hmm.process_params}")
    hmm_subtitle = format_hmm_process(config.hmm.process_name, config.hmm.process_params)

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    emit = build_emission_matrix(hmm)

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_hmm_tokens, logger)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    kl_smooth_window: int = int(config.kl_params.get("smooth_window", 5))
    kl_patience: int = int(config.kl_params.get("patience", 50))
    kl_min_rel_improvement: float = float(config.kl_params.get("min_rel_improvement", 0.01))
    kl_min_position: int = int(config.kl_params.get("min_position", 0))

    r2_patience: int = int(config.r2_params.get("patience", 50))
    r2_min_rel_improvement: float = float(config.r2_params.get("min_rel_improvement", 0.01))
    r2_min_position: int = int(config.r2_params.get("min_position", 0))

    # ── Per-sequence storage ──────────────────────────────────────────────────
    per_seq_sw_r2: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    per_seq_mse: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    t_r2_per_seq: dict[int, list[int]] = {l: [] for l in config.layer_indices}
    r2_crossed_per_seq: dict[int, list[bool]] = {l: [] for l in config.layer_indices}
    per_seq_kl_smooth: list[np.ndarray] = []
    t_kl_per_seq: list[int] = []
    kl_crossed_per_seq: list[bool] = []

    n_run = len(seq_run_indices)

    # ── Forward passes ────────────────────────────────────────────────────────
    for run_idx, seq_i in enumerate(seq_run_indices):
        logger.info(f"Sequence {seq_i} ({run_idx + 1}/{n_run})")

        seq_tokens = all_tokens[seq_i]   # (L,)
        seq_beliefs = all_beliefs[seq_i] # (L+1, n_states)

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=True, truncate=False)
        assert llm_tokens.shape[1] == seq_length + 1, (
            f"Seq {seq_i}: expected {seq_length + 1} LLM tokens, got {llm_tokens.shape[1]}"
        )

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                llm_tokens, names_filter=hook_names, return_type="logits",
            )

        # KL computation
        model_probs = get_model_probs_projected(
            logits[0:1], first_tok_id, mid_tok_ids, seq_length,
        )  # (L, n_hmm_tokens+1)
        optimal_probs = compute_optimal_probs(seq_beliefs, emit)  # (L, n_tokens)

        _, kl_smooth = compute_kl(
            model_probs, optimal_probs,
            smooth_window=kl_smooth_window,
            include_junk=True,
        )
        t_kl, kl_crossed = find_kl_convergence_patience(
            kl_smooth,
            patience=kl_patience,
            min_rel_improvement=kl_min_rel_improvement,
            min_position=kl_min_position,
        )
        per_seq_kl_smooth.append(kl_smooth)
        t_kl_per_seq.append(t_kl)
        kl_crossed_per_seq.append(kl_crossed)

        if kl_crossed:
            logger.info(f"  KL t* = {t_kl}/{seq_length}")
        else:
            logger.warning(f"  KL t* = {t_kl}/{seq_length} (argmin fallback)")

        # Beliefs aligned to activations (drop prior b_0)
        if config.strip_bos:
            beliefs_aligned = seq_beliefs[1:]  # (L, n_states) — b_1..b_L
        else:
            beliefs_aligned = seq_beliefs[:-1]  # (L, n_states) — b_0..b_{L-1}

        for layer in config.layer_indices:
            acts = cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            if config.strip_bos:
                acts = acts[1:]  # (L, d_model) — drop BOS position

            se, _ = _apply_probe(probes[seq_i][layer].probe, acts, beliefs_aligned)
            sw_r2 = _sliding_r2(se, beliefs_aligned, config.sliding_window)
            t_r2, r2_crossed = find_r2_emergence_patience(
                sw_r2,
                patience=r2_patience,
                min_rel_improvement=r2_min_rel_improvement,
                min_position=r2_min_position,
            )

            per_seq_sw_r2[layer].append(sw_r2)
            per_seq_mse[layer].append(se)
            t_r2_per_seq[layer].append(t_r2)
            r2_crossed_per_seq[layer].append(r2_crossed)

            r2_at_tkl = sw_r2[t_kl]
            r2_at_tr2 = sw_r2[t_r2]
            logger.info(
                f"  Layer {layer}: t_R²={t_r2} (crossed={r2_crossed})  "
                f"sw_r2@t_kl={'nan' if np.isnan(r2_at_tkl) else f'{r2_at_tkl:.3f}'}  "
                f"r2@t_r2={'nan' if np.isnan(r2_at_tr2) else f'{r2_at_tr2:.3f}'}  "
                f"mse@0={se[0]:.4f}"
            )

        del logits, cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    L = seq_length
    end = min(config.max_plot_position, L)
    # x-axis is 1-indexed: position k = "k HMM tokens seen, BOS stripped"
    positions = np.arange(1, end + 1)
    kl_positions = np.arange(1, min(end, L) + 1)

    kl_smooth_arr = np.stack(per_seq_kl_smooth)   # (N, L)
    kl_mean = kl_smooth_arr[:, :end].mean(axis=0)
    kl_std = kl_smooth_arr[:, :end].std(axis=0)

    t_kl_arr = np.array(t_kl_per_seq)
    t_kl_rep = int(np.round(t_kl_arr.mean()))

    sw_r2_mean: dict[int, np.ndarray] = {}
    sw_r2_std: dict[int, np.ndarray] = {}
    mse_mean: dict[int, np.ndarray] = {}
    mse_std: dict[int, np.ndarray] = {}
    lag_means: list[float | None] = []
    lag_stds: list[float | None] = []
    n_seqs_reached: list[int] = []

    metrics_out: dict = {
        "t_kl_per_seq": t_kl_per_seq,
        "t_kl_mean": float(t_kl_arr.mean()),
        "t_kl_std": float(t_kl_arr.std()),
        "layers": {},
    }

    logger.info("Computing per-layer metrics ...")
    for layer in config.layer_indices:
        sw_r2_stacked = np.stack(per_seq_sw_r2[layer])   # (N, L)
        mse_stacked = np.stack(per_seq_mse[layer])        # (N, L)

        sw_r2_mean[layer] = np.nanmean(sw_r2_stacked[:, :end], axis=0)
        sw_r2_std[layer] = np.nanstd(sw_r2_stacked[:, :end], axis=0)
        mse_mean[layer] = np.nanmean(mse_stacked[:, :end], axis=0)
        mse_std[layer] = np.nanstd(mse_stacked[:, :end], axis=0)

        t_r2_list = t_r2_per_seq[layer]
        crossed_list = r2_crossed_per_seq[layer]

        # Lag only counts sequences where both KL and R² patience criteria fired
        valid_lags = [
            (t_r2 - t_kl)
            for t_r2, t_kl, r2_ok, kl_ok in zip(
                t_r2_list, t_kl_per_seq, crossed_list, kl_crossed_per_seq
            )
            if r2_ok and kl_ok
        ]
        n_reached = sum(crossed_list)
        n_seqs_reached.append(n_reached)

        if valid_lags:
            lag_mean = float(np.mean(valid_lags))
            lag_std = float(np.std(valid_lags))
        else:
            lag_mean = None
            lag_std = None

        lag_means.append(lag_mean)
        lag_stds.append(lag_std)

        if lag_mean is not None:
            logger.info(
                f"Layer {layer}: lag={lag_mean:+.0f}±{lag_std:.0f} tok  "
                f"(n_patience_fired={n_reached}/{n_run})"
            )
        else:
            logger.info(f"Layer {layer}: R² patience never fired  t_KL={t_kl_rep}")

        metrics_out["layers"][str(layer)] = {
            "seq_indices": seq_run_indices,
            "t_r2_per_seq": t_r2_list,
            "r2_crossed_per_seq": crossed_list,
            "t_r2_mean": float(np.mean(t_r2_list)),
            "t_r2_std": float(np.std(t_r2_list)),
            "lag_mean": lag_mean,
            "lag_std": lag_std,
            "n_r2_patience_fired": n_reached,
        }

    # ── Convergence plots (4 variants) ────────────────────────────────────────
    logger.info("Generating convergence plots ...")
    for metric_name, metric_m, metric_s in [
        ("R²", sw_r2_mean, sw_r2_std),
        ("MSE", mse_mean, mse_std),
    ]:
        for log_kl in [False, True]:
            suffix = f"{'r2' if metric_name == 'R²' else 'mse'}_{'logkl' if log_kl else 'linkl'}"
            _plot_convergence(
                positions=positions,
                kl_positions=kl_positions,
                metric_mean=metric_m,
                metric_std=metric_s,
                kl_mean=kl_mean,
                kl_std=kl_std,
                layer_indices=config.layer_indices,
                metric_name=metric_name,
                log_kl=log_kl,
                t_kl_rep=t_kl_rep,
                sliding_window=config.sliding_window,
                hmm_subtitle=hmm_subtitle,
                path=fig_dir / f"emergence_{suffix}",
            )

    # ── Lag plot ──────────────────────────────────────────────────────────────
    logger.info("Generating lag plot ...")
    _plot_lag_summary(
        layer_indices=config.layer_indices,
        lag_means=lag_means,
        lag_stds=lag_stds,
        n_seqs_reached=n_seqs_reached,
        r2_patience=r2_patience,
        kl_patience=kl_patience,
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / "lag_summary",
    )

    # ── Save JSON outputs ─────────────────────────────────────────────────────
    _write_json(
        plot_data_dir / "emergence_curves.json",
        {
            "seq_indices": seq_run_indices,
            "positions": positions.tolist(),
            "kl_positions": kl_positions.tolist(),
            "kl_mean": kl_mean.tolist(),
            "kl_std": kl_std.tolist(),
            "t_kl_per_seq": t_kl_per_seq,
            "kl_crossed_per_seq": kl_crossed_per_seq,
            "layers": {
                str(layer): {
                    "sw_r2_mean": sw_r2_mean[layer].tolist(),
                    "sw_r2_std": sw_r2_std[layer].tolist(),
                    "mse_mean": mse_mean[layer].tolist(),
                    "mse_std": mse_std[layer].tolist(),
                    "t_r2_per_seq": t_r2_per_seq[layer],
                    "r2_crossed_per_seq": r2_crossed_per_seq[layer],
                    "lag_mean": metrics_out["layers"][str(layer)]["lag_mean"],
                    "lag_std": metrics_out["layers"][str(layer)]["lag_std"],
                }
                for layer in config.layer_indices
            },
        },
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    lag_summary = {
        "seq_indices": seq_run_indices,
        "n_sequences": n_run,
        "kl_patience": kl_patience,
        "kl_min_rel_improvement": kl_min_rel_improvement,
        "r2_patience": r2_patience,
        "r2_min_rel_improvement": r2_min_rel_improvement,
        "t_kl_mean": float(t_kl_arr.mean()),
        "t_kl_std": float(t_kl_arr.std()),
        "n_kl_patience_fired": int(sum(kl_crossed_per_seq)),
        "per_layer": {
            str(l): {
                "t_r2_mean": metrics_out["layers"][str(l)]["t_r2_mean"],
                "t_r2_std": metrics_out["layers"][str(l)]["t_r2_std"],
                "lag_mean": metrics_out["layers"][str(l)]["lag_mean"],
                "lag_std": metrics_out["layers"][str(l)]["lag_std"],
                "n_r2_patience_fired": metrics_out["layers"][str(l)]["n_r2_patience_fired"],
            }
            for l in config.layer_indices
        },
    }
    with open(out_dir / "lag_summary.json", "w") as f:
        json.dump(lag_summary, f, indent=2)

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
