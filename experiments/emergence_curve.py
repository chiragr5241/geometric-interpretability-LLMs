#!/usr/bin/env python3
"""
SPAR-12 — Belief geometry emergence curve: R² vs KL during transient.

Loads pooled probes from a prior experiment directory, generates held-out
sequences of length 2,000, and evaluates the probe at every position to
reveal when belief geometry (linear decodability) emerges relative to KL
convergence of the model's output distribution.

Usage:
    python experiments/emergence_curve.py experiments/configs/emergence_curve.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import (
    build_emission_matrix,
    compute_optimal_probs,
    get_device,
    get_model_probs,
    get_model_probs_projected,
    load_model,
    resolve_hmm_token_ids,
    setup_logging,
)
from hmm.hmm import Mess3HMM
from metrics.probe_metrics import (
    compute_auc_quantile_positions,
    compute_kl,
    compute_r2_kl_ccf,
    find_kl_threshold,
)
from probes import Probe, ProbeResult


@dataclass
class EmergenceCurveConfig(ExperimentConfig):
    pooled_probe_dir: str
    layer_indices: list[int]
    seq_length: int
    n_sequences: int
    batch_size: int
    sliding_window: int
    min_reporting_position: int
    r2_threshold: float
    simplex_n_sequences: int
    simplex_positions: list[int]
    kl_params: dict
    vocab_mapping: dict[str, int]
    ccf_max_lag: int = 200
    ccf_smooth_window: int = 15
    ccf_n_tokens: int = 150
    auc_quantiles: list[float] = None
    n_ctx_override: int | None = None
    strip_bos: bool = True

    def __post_init__(self) -> None:
        if self.auc_quantiles is None:
            self.auc_quantiles = [0.50, 0.90, 0.95, 0.99]


# ── Computation helpers ──────────────────────────────────────────────────────────

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


def _cumulative_r2(se: np.ndarray, beliefs: np.ndarray) -> np.ndarray:
    """Cumulative R²(t) for t = 0 .. T-1, vectorized O(T)."""
    T = len(se)
    cum_ss_res = np.cumsum(se)
    norm_sq = (beliefs ** 2).sum(axis=1)
    cum_norm_sq = np.cumsum(norm_sq)
    cum_b = np.cumsum(beliefs, axis=0)
    counts = np.arange(1, T + 1, dtype=np.float64)
    mean_t = cum_b / counts[:, None]
    mean_norm_sq_t = (mean_t ** 2).sum(axis=1)
    ss_tot = cum_norm_sq - counts * mean_norm_sq_t
    return 1.0 - cum_ss_res / np.clip(ss_tot, 1e-10, None)


def _sliding_r2(se: np.ndarray, beliefs: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window R²(t) for t = 0 .. T-1, with variable-width edges."""
    T = len(se)
    half = window // 2
    cs_se = np.concatenate([[0.0], np.cumsum(se)])
    norm_sq = (beliefs ** 2).sum(axis=1)
    cs_norm_sq = np.concatenate([[0.0], np.cumsum(norm_sq)])
    cs_b = np.vstack([np.zeros((1, beliefs.shape[1])), np.cumsum(beliefs, axis=0)])
    r2 = np.zeros(T)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        n = hi - lo
        ss_res = float(cs_se[hi] - cs_se[lo])
        sum_norm_sq = float(cs_norm_sq[hi] - cs_norm_sq[lo])
        sum_b = cs_b[hi] - cs_b[lo]
        ss_tot = sum_norm_sq - float((sum_b ** 2).sum()) / n
        r2[t] = 1.0 - ss_res / (ss_tot + 1e-10)
    return r2


def _find_r2_emergence(
    sw_r2: np.ndarray,
    threshold: float,
    min_pos: int,
) -> int | None:
    """Return first position >= min_pos where sliding R² >= threshold, or None."""
    above = np.where(sw_r2[min_pos:] >= threshold)[0]
    return int(above[0]) + min_pos if len(above) > 0 else None


# ── Plotting helpers ─────────────────────────────────────────────────────────────

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


def _plot_emergence_curve(
    L: int,
    cum_r2_mean: np.ndarray,
    cum_r2_std: np.ndarray,
    sw_r2_mean: np.ndarray,
    sw_r2_std: np.ndarray,
    kl_smooth_mean: np.ndarray,
    kl_smooth_std: np.ndarray,
    mse_mean: np.ndarray,
    mse_std: np.ndarray,
    layer: int,
    t_r2: int | None,
    t_kl: int,
    lag_mean: float | None,
    lag_std: float | None,
    r2_threshold: float,
    sliding_window: int,
    min_report: int,
    path: Path,
    max_pos: int | None = None,
) -> None:
    """Triple-axis R² / KL / MSE emergence curve. max_pos=None → full sequence."""
    end = max_pos if max_pos is not None else (L + 1)
    end_kl = min(end, L)

    pos_r2 = np.arange(end)
    pos_kl = np.arange(end_kl)

    cr2_m = cum_r2_mean[:end].copy()
    cr2_m[:min_report] = np.nan
    cr2_s = cum_r2_std[:end].copy()
    cr2_s[:min_report] = np.nan

    sr2_m = sw_r2_mean[:end]
    sr2_s = sw_r2_std[:end]
    kl_m = kl_smooth_mean[:end_kl]
    kl_s = kl_smooth_std[:end_kl]
    mse_m = mse_mean[:end]
    mse_s = mse_std[:end]

    fig = go.Figure()

    # ── KL (yaxis2, right-inner, log scale) ─────────────────────────────────
    fig.add_trace(_band_trace(pos_kl, kl_m, kl_s, "rgba(180,34,34,0.07)", "y2",
                              clip_positive=True))
    fig.add_trace(go.Scatter(
        x=pos_kl, y=kl_m, name="KL (smoothed)",
        line=dict(color="firebrick", width=2), mode="lines", yaxis="y2",
    ))

    # ── MSE (yaxis3, right-outer, log scale) ─────────────────────────────────
    fig.add_trace(_band_trace(pos_r2, mse_m, mse_s, "rgba(60,160,100,0.07)", "y3",
                              clip_positive=True))
    fig.add_trace(go.Scatter(
        x=pos_r2, y=mse_m, name="MSE (‖ŷ−b‖²)",
        line=dict(color="seagreen", width=2), mode="lines", yaxis="y3",
    ))

    # ── Cumulative R² (yaxis, left) ──────────────────────────────────────────
    fig.add_trace(_band_trace(pos_r2, cr2_m, cr2_s, "rgba(70,130,180,0.07)", "y"))
    fig.add_trace(go.Scatter(
        x=pos_r2, y=cr2_m, name="Cumulative R²",
        line=dict(color="steelblue", width=2), mode="lines", yaxis="y",
    ))

    # ── Sliding R² (yaxis, left) ─────────────────────────────────────────────
    fig.add_trace(_band_trace(pos_r2, sr2_m, sr2_s, "rgba(255,140,0,0.07)", "y"))
    fig.add_trace(go.Scatter(
        x=pos_r2, y=sr2_m, name=f"Sliding R² (W={sliding_window})",
        line=dict(color="darkorange", width=2), mode="lines", yaxis="y",
    ))

    # R² threshold dotted line
    fig.add_trace(go.Scatter(
        x=[pos_r2[0], pos_r2[-1]], y=[r2_threshold, r2_threshold],
        mode="lines", line=dict(color="darkorange", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip", yaxis="y",
    ))

    # Vertical markers (paper coords span full height regardless of axis scales)
    if end_kl > t_kl >= 0:
        fig.add_shape(
            type="line", x0=t_kl, x1=t_kl, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="firebrick", width=1.5, dash="dash"),
        )
    if t_r2 is not None and (max_pos is None or t_r2 < max_pos):
        fig.add_shape(
            type="line", x0=t_r2, x1=t_r2, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="darkorange", width=1.5, dash="dash"),
        )

    lag_str = (
        f"lag={lag_mean:+.0f}±{lag_std:.0f} tok"
        if lag_mean is not None
        else "R² threshold not reached"
    )
    sfx = f" | first {max_pos} tokens" if max_pos is not None else ""
    fig.update_layout(
        title=(
            f"Layer {layer} — Emergence curve{sfx}<br>"
            f"<sup>t_KL={t_kl}, t_R²={t_r2} | {lag_str}</sup>"
        ),
        height=500, width=1020,
        margin=dict(t=80, b=110, l=70, r=160),
        legend=dict(
            orientation="h",
            x=0.0, xanchor="left",
            y=-0.22, yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
        ),
        xaxis=dict(title="Position (tokens seen)", domain=[0.0, 0.78]),
        yaxis=dict(title="R²", range=[None, 1.05]),
        yaxis2=dict(
            title="KL (nats)", overlaying="y", side="right", anchor="x",
            showgrid=False, type="log",
        ),
        yaxis3=dict(
            title="MSE (‖ŷ−b‖²)", overlaying="y", side="right",
            anchor="free", position=0.90, showgrid=False, type="log",
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_lag_summary(
    layer_indices: list[int],
    lag_means: list[float | None],
    lag_stds: list[float | None],
    r2_threshold: float,
    path: Path,
) -> None:
    labels = [str(l) for l in layer_indices]
    y_vals, err_vals, colors = [], [], []
    for lm, ls in zip(lag_means, lag_stds):
        if lm is None:
            y_vals.append(0.0)
            err_vals.append(0.0)
            colors.append("lightgray")
        else:
            y_vals.append(lm)
            err_vals.append(ls if ls is not None else 0.0)
            colors.append("firebrick" if lm > 0 else "steelblue")

    fig = go.Figure(go.Bar(
        x=labels, y=y_vals,
        error_y=dict(type="data", array=err_vals, visible=True),
        marker_color=colors,
        text=[f"{v:+.0f}" if lm is not None else "N/A"
              for v, lm in zip(y_vals, lag_means)],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=(
            f"Lag: t_R²(R²>{r2_threshold}) − t_KL per layer<br>"
            "<sup>Positive = geometry lags output; Negative = geometry leads output</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Lag (tokens)",
        height=420, width=760,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    fig.write_image(str(path.with_suffix(".png")))


def _layer_colors(layer_indices: list[int]) -> list[str]:
    import plotly.colors as pc
    n = len(layer_indices)
    return pc.sample_colorscale("Plasma", [i / max(n - 1, 1) for i in range(n)])


_SYMLOG_TICK_VALS: list[float] = [-1000, -100, -10, -1, 0, 1]


def _symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def _symlog_ticks() -> tuple[list[float], list[str]]:
    positions = [float(np.sign(v) * np.log10(1.0 + abs(v))) for v in _SYMLOG_TICK_VALS]
    labels = [str(int(v)) for v in _SYMLOG_TICK_VALS]
    return positions, labels


def _plot_auc_curves(
    n_tok: int,
    layer_indices: list[int],
    r2_raw_mean: dict[int, np.ndarray],
    r2_raw_std: dict[int, np.ndarray],
    r2_norm_mean: dict[int, np.ndarray],
    r2_norm_std: dict[int, np.ndarray],
    kl_raw_mean: np.ndarray,
    kl_raw_std: np.ndarray,
    kl_norm_mean: np.ndarray,
    kl_norm_std: np.ndarray,
    path: Path,
) -> None:
    """2-panel figure: unnormalized (top) and [0,1]-normalized (bottom) convergence curves.

    Shows R² per layer and −KL (shared) over the first n_tok tokens.
    These are the raw inputs to the AUC quantile computation.
    The unnormalized panel uses a symlog y-axis (sign(x)*log10(1+|x|)) so that
    the deep-negative transient is compressed without hiding positive values.
    """
    colors = _layer_colors(layer_indices)
    tokens = np.arange(1, n_tok + 1)
    kl_color = "firebrick"
    symlog_tick_pos, symlog_tick_lab = _symlog_ticks()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Unnormalized (symlog y)", "Normalized [0, 1]"],
        vertical_spacing=0.10,
    )

    for panel, (r2_means, r2_stds, kl_m, kl_s) in enumerate([
        (r2_raw_mean, r2_raw_std, kl_raw_mean, kl_raw_std),
        (r2_norm_mean, r2_norm_std, kl_norm_mean, kl_norm_std),
    ], start=1):
        use_symlog = panel == 1
        _t = _symlog if use_symlog else (lambda x: x)

        # KL band + mean (dashed, no legend duplication)
        fig.add_trace(go.Scatter(
            x=np.concatenate([tokens, tokens[::-1]]),
            y=np.concatenate([_t(kl_m + kl_s), _t((kl_m - kl_s))[::-1]]),
            fill="toself",
            fillcolor="rgba(180,34,34,0.10)",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip", mode="lines",
        ), row=panel, col=1)
        fig.add_trace(go.Scatter(
            x=tokens, y=_t(kl_m),
            name="−KL",
            line=dict(color=kl_color, width=2, dash="dash"),
            mode="lines",
            showlegend=(panel == 1),
            legendgroup="kl",
        ), row=panel, col=1)

        # R² per layer
        for color, layer in zip(colors, layer_indices):
            m = r2_means[layer]
            s = r2_stds[layer]
            fig.add_trace(go.Scatter(
                x=np.concatenate([tokens, tokens[::-1]]),
                y=np.concatenate([_t(m + s), _t((m - s))[::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.10)"),
                line=dict(width=0),
                showlegend=False, hoverinfo="skip", mode="lines",
            ), row=panel, col=1)
            fig.add_trace(go.Scatter(
                x=tokens, y=_t(m),
                name=f"Layer {layer}",
                line=dict(color=color, width=1.6),
                mode="lines",
                showlegend=(panel == 1),
                legendgroup=str(layer),
            ), row=panel, col=1)

    fig.update_xaxes(title_text="Position (tokens)", row=2, col=1)
    fig.update_yaxes(
        title_text="Value (symlog)",
        tickvals=symlog_tick_pos,
        ticktext=symlog_tick_lab,
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Normalized value", row=2, col=1)
    fig.update_layout(
        title=(
            f"AUC convergence curves — first {n_tok} tokens<br>"
            "<sup>R² per layer (solid) and −KL (dashed); "
            "shaded = ±1 std across sequences</sup>"
        ),
        height=650, width=900,
        margin=dict(t=90, b=60, l=70, r=40),
        legend=dict(
            orientation="v", x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_cross_correlation_by_layer(
    lags: np.ndarray,
    layer_indices: list[int],
    ccf_raw_mean: dict[int, np.ndarray],
    ccf_raw_std: dict[int, np.ndarray],
    ccf_deriv_mean: dict[int, np.ndarray],
    ccf_deriv_std: dict[int, np.ndarray],
    path: Path,
) -> None:
    """2-panel CCF figure: raw signals (top), derivatives (bottom), one series per layer."""
    colors = _layer_colors(layer_indices)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["CCF — raw signals (R² vs −KL)", "CCF — derivatives"],
        vertical_spacing=0.12,
    )

    for color, layer in zip(colors, layer_indices):
        raw_m = ccf_raw_mean[layer]
        raw_s = ccf_raw_std[layer]
        deriv_m = ccf_deriv_mean[layer]
        deriv_s = ccf_deriv_std[layer]

        show_legend = True
        for row, (m, s) in enumerate([(raw_m, raw_s), (deriv_m, deriv_s)], start=1):
            # Shaded band
            fig.add_trace(go.Scatter(
                x=np.concatenate([lags, lags[::-1]]),
                y=np.concatenate([m + s, (m - s)[::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)"),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                mode="lines",
            ), row=row, col=1)
            # Mean curve
            fig.add_trace(go.Scatter(
                x=lags, y=m,
                name=f"Layer {layer}",
                line=dict(color=color, width=1.8),
                mode="lines",
                showlegend=show_legend,
                legendgroup=str(layer),
            ), row=row, col=1)
            show_legend = False

    fig.add_vline(x=0, line_color="black", line_width=1, line_dash="dot", row="all", col=1)

    fig.update_xaxes(title_text="Lag τ (tokens)", row=2, col=1)
    fig.update_yaxes(title_text="CCF", row=1, col=1)
    fig.update_yaxes(title_text="CCF (deriv)", row=2, col=1)
    fig.update_layout(
        title=(
            "Cross-correlation R²(t) vs −KL(t) per layer<br>"
            "<sup>Positive peak τ → geometry lags KL; negative → geometry leads</sup>"
        ),
        height=600, width=900,
        margin=dict(t=90, b=60, l=70, r=40),
        legend=dict(
            orientation="v", x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_combined_lag(
    layer_indices: list[int],
    ccf_peak_lags: list[float],
    auc_lag_means: dict[int, np.ndarray],
    auc_lag_stds: dict[int, np.ndarray],
    auc_quantiles: list[float],
    path: Path,
) -> None:
    """Multi-series lag plot: CCF peak lag + AUC-quantile lags per layer."""
    x_labels = [str(l) for l in layer_indices]

    series_colors = ["black", "steelblue", "darkorange", "seagreen", "firebrick"]
    series_names = ["CCF peak"] + [f"AUC-{int(q * 100)}" for q in auc_quantiles]

    ccf_y = ccf_peak_lags
    auc_ys = [
        [float(auc_lag_means[l][i]) for l in layer_indices]
        for i in range(len(auc_quantiles))
    ]
    auc_errs = [
        [float(auc_lag_stds[l][i]) for l in layer_indices]
        for i in range(len(auc_quantiles))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=ccf_y,
        name=series_names[0],
        mode="lines+markers",
        line=dict(color=series_colors[0], width=2),
        marker=dict(size=7),
    ))
    for i, (name, color, ys, errs) in enumerate(
        zip(series_names[1:], series_colors[1:], auc_ys, auc_errs)
    ):
        fig.add_trace(go.Scatter(
            x=x_labels, y=ys,
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
            error_y=dict(type="data", array=errs, visible=True,
                         color=color, thickness=1.2, width=4),
        ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=(
            "Lag per layer — threshold-free metrics<br>"
            "<sup>Positive = geometry lags KL; Negative = geometry leads KL</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Lag (tokens)",
        height=700, width=820,
        margin=dict(t=80, b=80, l=70, r=40),
        legend=dict(
            orientation="h",
            x=0.0, xanchor="left",
            y=-0.12, yanchor="top",
            bgcolor="rgba(255,255,255,0.0)",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


# ── Simplex trajectory helpers ───────────────────────────────────────────────────

def _to_barycentric(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return b[:, 1] + 0.5 * b[:, 2], (np.sqrt(3) / 2) * b[:, 2]


def _belief_colors(b: np.ndarray) -> list[str]:
    c = np.clip(b, 0.0, 1.0)
    return [f"rgb({int(r*255)},{int(g*255)},{int(b_*255)})" for r, g, b_ in c]


def _simplex_frame_traces() -> list[go.BaseTraceType]:
    sqrt3 = np.sqrt(3)
    outline = go.Scatter(
        x=[0, 1, 0.5, 0], y=[0, 0, sqrt3 / 2, 0],
        mode="lines", line=dict(color="black", width=1),
        showlegend=False, hoverinfo="skip",
    )
    labels = go.Scatter(
        x=[-0.08, 1.08, 0.5], y=[-0.05, -0.05, sqrt3 / 2 + 0.06],
        mode="text", text=["S0", "S1", "S2"],
        showlegend=False, hoverinfo="skip", textfont=dict(size=8),
    )
    return [outline, labels]


def _plot_simplex_trajectories(
    simplex_preds_all: list[dict[int, np.ndarray]],
    simplex_beliefs_all: list[dict[int, np.ndarray]],
    positions: list[int],
    layer_indices: list[int],
    fig_dir: Path,
) -> None:
    sqrt3 = np.sqrt(3)
    n_seqs = len(simplex_preds_all)
    n_pos = len(positions)
    sizes = np.linspace(5, 14, n_pos).tolist()

    for layer in layer_indices:
        fig = make_subplots(
            rows=1, cols=n_seqs,
            subplot_titles=[f"Seq {s}" for s in range(n_seqs)],
            horizontal_spacing=0.05,
        )
        for s in range(n_seqs):
            preds_layer = simplex_preds_all[s].get(layer)
            beliefs_layer = simplex_beliefs_all[s].get(layer)
            if preds_layer is None or beliefs_layer is None:
                continue

            for trace in _simplex_frame_traces():
                fig.add_trace(trace, row=1, col=s + 1)

            px, py = _to_barycentric(preds_layer)   # (n_pos,)
            gt_colors = _belief_colors(beliefs_layer)  # per-point color = gt belief

            # Trajectory line (thin gray)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines",
                line=dict(color="rgba(120,120,120,0.4)", width=1),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=s + 1)

            # Points: fill = gt belief color, size increases with position
            for i, (pos_t, gc, sz) in enumerate(zip(positions, gt_colors, sizes)):
                fig.add_trace(go.Scatter(
                    x=[px[i]], y=[py[i]],
                    mode="markers+text",
                    marker=dict(
                        color=gc, size=sz,
                        line=dict(color="rgba(60,60,60,0.6)", width=1),
                    ),
                    text=[str(pos_t)],
                    textposition="top center",
                    textfont=dict(size=7),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=s + 1)

        fig.update_xaxes(range=[-0.15, 1.15], showticklabels=False,
                         showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-0.12, sqrt3 / 2 + 0.14], showticklabels=False,
                         showgrid=False, zeroline=False)
        fig.update_layout(
            title=(
                f"Layer {layer} — Simplex trajectory of probe predictions<br>"
                "<sup>Color = gt belief state (RGB = S0,S1,S2); "
                "size increases with position; label = token position</sup>"
            ),
            height=380, width=max(600, 300 * n_seqs),
            showlegend=False,
            margin=dict(t=80, b=30, l=20, r=20),
        )
        fig.write_image(str(fig_dir / f"simplex_trajectory_layer_{layer}.png"))


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Belief geometry emergence curve")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-user",
        type=str,
        default=None,
        help="Override output_user from the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config, EmergenceCurveConfig)
    apply_runtime_overrides(config, output_user=args.output_user)
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="emergence_curve")
    fig_dir = out_dir / "figures"

    logger.info(f"Output dir      : {out_dir}")
    logger.info(f"Device          : {device}")
    logger.info(f"N sequences     : {config.n_sequences}")
    logger.info(f"Seq length      : {config.seq_length}")
    logger.info(f"Layers          : {config.layer_indices}")
    logger.info(f"Pooled probe dir: {config.pooled_probe_dir}")

    L: int = config.seq_length
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]
    use_projected: bool = bool(config.kl_params.get("include_junk", False))

    kl_params_find = {k: v for k, v in config.kl_params.items()}
    kl_smooth_window: int = int(config.kl_params.get("smooth_window", 5))

    # ── Load pooled probes ────────────────────────────────────────────────────
    probe_base = Path(config.pooled_probe_dir) / "probes" / "pooled"
    probes: dict[int, ProbeResult] = {}
    for layer in config.layer_indices:
        pr = ProbeResult.load(probe_base / f"layer_{layer}")
        pr.probe.to(device)
        probes[layer] = pr
        logger.info(f"Loaded probe: layer {layer}  d_model={pr.probe.W.shape[0]}")

    # ── Model + HMM ──────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    if config.seq_length >= model.cfg.n_ctx:
        raise ValueError(
            f"seq_length={config.seq_length} >= model n_ctx={model.cfg.n_ctx}"
        )

    hmm = Mess3HMM()
    p = config.hmm.process_params
    if "x" in p and "alpha" in p:
        hmm.create_hmm(p["x"], p["alpha"])
        logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    emit = build_emission_matrix(hmm)

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_hmm_tokens, logger)

    # Precompute simplex snapshot positions.
    # Config positions are 1-indexed HMM token positions.
    # simplex_positions_valid: the position labels used in plots.
    # simplex_indices: the array indices used to index into acts/beliefs arrays.
    if config.strip_bos:
        simplex_positions_valid: list[int] = [p for p in config.simplex_positions if 1 <= p <= L]
        simplex_indices: list[int] = [p - 1 for p in simplex_positions_valid]
    else:
        simplex_positions_valid = [p for p in config.simplex_positions if 0 <= p <= L]
        simplex_indices = simplex_positions_valid

    # ── Per-sequence storage ──────────────────────────────────────────────────
    per_seq_cum_r2: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    per_seq_sw_r2: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    per_seq_mse: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    t_r2_per_seq: dict[int, list[int | None]] = {l: [] for l in config.layer_indices}
    t_kl_per_seq: list[int] = []
    per_seq_kl_smooth: list[np.ndarray] = []

    # CCF per sequence per layer: {layer: list of (ccf_raw, ccf_deriv)}
    per_seq_ccf: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        l: [] for l in config.layer_indices
    }
    # AUC quantile token positions: r2 is per-layer, kl is shared across layers
    per_seq_auc_r2: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    per_seq_auc_kl: list[np.ndarray] = []
    # [0,1] normalized curves used for AUC: r2 is per-layer, kl is shared
    per_seq_r2_norm: dict[int, list[np.ndarray]] = {l: [] for l in config.layer_indices}
    per_seq_kl_norm: list[np.ndarray] = []

    # Simplex trajectory data: [seq_idx] -> {layer: (n_pos, n_states)}
    simplex_preds_all: list[dict[int, np.ndarray]] = []
    simplex_beliefs_all: list[dict[int, np.ndarray]] = []

    # ── Forward passes (batched) ──────────────────────────────────────────────
    for chunk_start in range(0, config.n_sequences, config.batch_size):
        B = min(config.batch_size, config.n_sequences - chunk_start)
        chunk_seed = int(torch.randint(2**31, (1,)).item())
        torch.manual_seed(chunk_seed)
        logger.info(
            f"Sequences {chunk_start + 1}–{chunk_start + B}/{config.n_sequences}"
            f"  batch={B}  seed={chunk_seed}"
        )

        tokens_batch, _, _ = hmm.generate_dataset(B, L, return_states=True)  # (B, L)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)               # (B, L+1, n_states)

        llm_tokens_list: list[torch.Tensor] = []
        for b in range(B):
            seq_tokens_b = tokens_batch[b].cpu().numpy()
            text_b = " ".join(idx_to_token[int(t)] for t in seq_tokens_b)
            llm_tokens_b = model.to_tokens(text_b, prepend_bos=True, truncate=False)
            assert llm_tokens_b.shape[1] == L + 1, (
                f"Unexpected token length {llm_tokens_b.shape[1]}, expected {L + 1}"
            )
            llm_tokens_list.append(llm_tokens_b)
        llm_tokens = torch.cat(llm_tokens_list, dim=0)  # (B, L+1)

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                llm_tokens, names_filter=hook_names, return_type="logits",
            )

        for b in range(B):
            seq_idx = chunk_start + b
            seq_beliefs = beliefs_batch[b].cpu().numpy()  # (L+1, n_states)

            model_probs = (
                get_model_probs_projected(logits[b : b + 1], first_tok_id, mid_tok_ids, L)
                if use_projected
                else get_model_probs(logits[b : b + 1], first_tok_id, mid_tok_ids, L)
            )
            optimal_probs = compute_optimal_probs(seq_beliefs, emit)  # (L, n_tokens)
            if config.strip_bos:
                seq_beliefs = seq_beliefs[1:]  # (L, n_states) — drop prior/BOS position

            _, kl_smooth = compute_kl(
                model_probs, optimal_probs,
                smooth_window=kl_smooth_window,
                include_junk=use_projected,
            )
            t_kl, kl_crossed = find_kl_threshold(
                model_probs, optimal_probs, **kl_params_find, logger=logger,
            )
            per_seq_kl_smooth.append(kl_smooth)
            t_kl_per_seq.append(t_kl)

            if kl_crossed:
                logger.info(f"  Seq {seq_idx + 1}: KL t* = {t_kl}/{L}")
            else:
                logger.warning(f"  Seq {seq_idx + 1}: KL t* = {t_kl}/{L} (argmin fallback)")

            if seq_idx < config.simplex_n_sequences:
                simplex_preds_seq: dict[int, np.ndarray] = {}
                simplex_beliefs_seq: dict[int, np.ndarray] = {}

            # KL truncation and AUC computed once per sequence (shared across layers)
            n_tok = config.ccf_n_tokens
            kl_trunc = kl_smooth[:n_tok]   # shape (n_tok,)
            auc_kl, kl_norm = compute_auc_quantile_positions(-kl_trunc, config.auc_quantiles)
            per_seq_auc_kl.append(auc_kl)
            per_seq_kl_norm.append(kl_norm)

            for layer in config.layer_indices:
                acts = cache[f"blocks.{layer}.hook_resid_post"][b].float().cpu().numpy()
                if config.strip_bos:
                    acts = acts[1:]  # (L, d_model) — drop BOS position

                se, preds = _apply_probe(probes[layer].probe, acts, seq_beliefs)
                cum_r2 = _cumulative_r2(se, seq_beliefs)
                sw_r2 = _sliding_r2(se, seq_beliefs, config.sliding_window)
                t_r2 = _find_r2_emergence(sw_r2, config.r2_threshold, config.min_reporting_position)

                per_seq_cum_r2[layer].append(cum_r2)
                per_seq_sw_r2[layer].append(sw_r2)
                per_seq_mse[layer].append(se)
                t_r2_per_seq[layer].append(t_r2)

                r2_aligned = sw_r2[:n_tok]   # shape (n_tok,)
                _, ccf_raw, ccf_deriv = compute_r2_kl_ccf(
                    r2_aligned, kl_trunc,
                    max_lag=config.ccf_max_lag,
                    smooth_window=config.ccf_smooth_window,
                )
                per_seq_ccf[layer].append((ccf_raw, ccf_deriv))

                # AUC quantile positions and normalized curve for R²
                auc_r2, r2_norm = compute_auc_quantile_positions(r2_aligned, config.auc_quantiles)
                per_seq_auc_r2[layer].append(auc_r2)
                per_seq_r2_norm[layer].append(r2_norm)

                logger.info(
                    f"  Seq {seq_idx + 1} Layer {layer}: t_R²={t_r2}  "
                    f"sw_r2@t_kl={sw_r2[t_kl]:.3f}  cum_r2@t_kl={cum_r2[t_kl]:.3f}"
                )

                if seq_idx < config.simplex_n_sequences:
                    simplex_preds_seq[layer] = preds[simplex_indices]
                    simplex_beliefs_seq[layer] = seq_beliefs[simplex_indices]

            if seq_idx < config.simplex_n_sequences:
                simplex_preds_all.append(simplex_preds_seq)
                simplex_beliefs_all.append(simplex_beliefs_seq)

        del logits, cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Aggregate across sequences ────────────────────────────────────────────
    kl_smooth_mean = np.mean(per_seq_kl_smooth, axis=0)    # (L,)
    kl_smooth_std = np.std(per_seq_kl_smooth, axis=0)
    t_kl_arr = np.array(t_kl_per_seq)

    # Use mean t_kl across sequences as the representative threshold position
    t_kl_rep = int(np.round(t_kl_arr.mean()))

    metrics_out: dict = {
        "t_kl_per_seq": t_kl_per_seq,
        "t_kl_mean": float(t_kl_arr.mean()),
        "t_kl_std": float(t_kl_arr.std()),
        "layers": {},
    }

    lag_means: list[float | None] = []
    lag_stds: list[float | None] = []
    t_r2_reps: list[int | None] = []

    # CCF summary: lags shared, per-layer mean/std arrays
    ccf_lags: np.ndarray | None = None
    ccf_raw_mean: dict[int, np.ndarray] = {}
    ccf_raw_std: dict[int, np.ndarray] = {}
    ccf_deriv_mean: dict[int, np.ndarray] = {}
    ccf_deriv_std: dict[int, np.ndarray] = {}
    ccf_peak_lags: list[float] = []

    # AUC lag summary: {layer: (n_quantiles,) mean lag array}
    auc_lag_means: dict[int, np.ndarray] = {}
    auc_lag_stds: dict[int, np.ndarray] = {}

    # AUC curve aggregates for plotting: {layer: (n_tok,) arrays}
    r2_raw_mean: dict[int, np.ndarray] = {}
    r2_raw_std: dict[int, np.ndarray] = {}
    r2_norm_mean: dict[int, np.ndarray] = {}
    r2_norm_std: dict[int, np.ndarray] = {}
    # KL curves are layer-independent; computed once after the loop
    kl_raw_mean: np.ndarray | None = None
    kl_raw_std: np.ndarray | None = None
    kl_norm_mean: np.ndarray | None = None
    kl_norm_std: np.ndarray | None = None

    # ── Per-layer metrics + plots ─────────────────────────────────────────────
    logger.info("Computing per-layer metrics ...")
    for layer in config.layer_indices:
        cum_r2_stacked = np.stack(per_seq_cum_r2[layer])   # (N, L+1)
        sw_r2_stacked = np.stack(per_seq_sw_r2[layer])     # (N, L+1)
        mse_stacked = np.stack(per_seq_mse[layer])         # (N, L+1)

        cum_r2_mean = cum_r2_stacked.mean(axis=0)
        cum_r2_std = cum_r2_stacked.std(axis=0)
        sw_r2_mean = sw_r2_stacked.mean(axis=0)
        sw_r2_std = sw_r2_stacked.std(axis=0)
        mse_mean_arr = mse_stacked.mean(axis=0)
        mse_std_arr = mse_stacked.std(axis=0)

        t_r2_list = t_r2_per_seq[layer]
        valid_lags = [
            (t_r2 - t_kl)
            for t_r2, t_kl in zip(t_r2_list, t_kl_per_seq)
            if t_r2 is not None
        ]

        if valid_lags:
            lag_mean = float(np.mean(valid_lags))
            lag_std = float(np.std(valid_lags))
        else:
            lag_mean = None
            lag_std = None

        lag_means.append(lag_mean)
        lag_stds.append(lag_std)

        valid_t_r2 = [t for t in t_r2_list if t is not None]
        t_r2_rep = int(np.round(np.mean(valid_t_r2))) if valid_t_r2 else None
        t_r2_reps.append(t_r2_rep)

        # CCF aggregation (actual lag range may be shorter than ccf_max_lag
        # because compute_cross_correlation clamps to T-1)
        ccf_raws = np.stack([pair[0] for pair in per_seq_ccf[layer]])   # (N, 2*actual_lag+1)
        ccf_derivs = np.stack([pair[1] for pair in per_seq_ccf[layer]])
        actual_half = (ccf_raws.shape[1] - 1) // 2
        ccf_lags = np.arange(-actual_half, actual_half + 1)
        ccf_raw_mean[layer] = ccf_raws.mean(axis=0)
        ccf_raw_std[layer] = ccf_raws.std(axis=0)
        ccf_deriv_mean[layer] = ccf_derivs.mean(axis=0)
        ccf_deriv_std[layer] = ccf_derivs.std(axis=0)
        ccf_peak_lags.append(float(ccf_lags[int(ccf_raw_mean[layer].argmax())]))

        # AUC quantile lag aggregation (kl is shared across layers)
        auc_r2_stacked = np.stack(per_seq_auc_r2[layer])  # (N, n_quantiles)
        auc_kl_stacked = np.stack(per_seq_auc_kl)          # (N, n_quantiles)
        auc_lag_arr = auc_r2_stacked - auc_kl_stacked       # (N, n_quantiles)
        auc_lag_means[layer] = auc_lag_arr.mean(axis=0)
        auc_lag_stds[layer] = auc_lag_arr.std(axis=0)

        # AUC curve aggregation for plotting
        r2_raw_arr = np.stack(
            [sw_r2[:config.ccf_n_tokens] for sw_r2 in per_seq_sw_r2[layer]]
        )  # (N, n_tok)
        r2_norm_arr = np.stack(per_seq_r2_norm[layer])  # (N, n_tok)
        r2_raw_mean[layer] = r2_raw_arr.mean(axis=0)
        r2_raw_std[layer] = r2_raw_arr.std(axis=0)
        r2_norm_mean[layer] = r2_norm_arr.mean(axis=0)
        r2_norm_std[layer] = r2_norm_arr.std(axis=0)

        metrics_out["layers"][str(layer)] = {
            "t_r2_per_seq": t_r2_list,
            "t_r2_mean": float(np.mean(valid_t_r2)) if valid_t_r2 else None,
            "t_r2_std": float(np.std(valid_t_r2)) if valid_t_r2 else None,
            "lag_mean": lag_mean,
            "lag_std": lag_std,
            "n_r2_reached": len(valid_lags),
            "ccf_peak_lag": ccf_peak_lags[-1],
            "auc_quantile_lags": {
                str(q): float(auc_lag_means[layer][i])
                for i, q in enumerate(config.auc_quantiles)
            },
            "auc_quantile_lag_stds": {
                str(q): float(auc_lag_stds[layer][i])
                for i, q in enumerate(config.auc_quantiles)
            },
        }

        if lag_mean is not None:
            logger.info(
                f"Layer {layer}: t_R²={t_r2_rep}  t_KL={t_kl_rep}  "
                f"lag={lag_mean:+.0f}±{lag_std:.0f} tok"
            )
        else:
            logger.info(f"Layer {layer}: R² threshold not reached  t_KL={t_kl_rep}")

        # Zoomed emergence curve (first 200 tokens)
        _plot_emergence_curve(
            L=L,
            cum_r2_mean=cum_r2_mean, cum_r2_std=cum_r2_std,
            sw_r2_mean=sw_r2_mean, sw_r2_std=sw_r2_std,
            kl_smooth_mean=kl_smooth_mean, kl_smooth_std=kl_smooth_std,
            mse_mean=mse_mean_arr, mse_std=mse_std_arr,
            layer=layer,
            t_r2=t_r2_rep, t_kl=t_kl_rep,
            lag_mean=lag_mean, lag_std=lag_std,
            r2_threshold=config.r2_threshold,
            sliding_window=config.sliding_window,
            min_report=config.min_reporting_position,
            path=fig_dir / f"emergence_curve_layer_{layer}_zoom150",
            max_pos=150,
        )

    # ── KL curve aggregation (layer-independent) ─────────────────────────────
    kl_raw_arr = np.stack(
        [-kl[:config.ccf_n_tokens] for kl in per_seq_kl_smooth]
    )  # (N, n_tok), using −KL so it increases like R²
    kl_norm_arr = np.stack(per_seq_kl_norm)  # (N, n_tok)
    kl_raw_mean = kl_raw_arr.mean(axis=0)
    kl_raw_std = kl_raw_arr.std(axis=0)
    kl_norm_mean = kl_norm_arr.mean(axis=0)
    kl_norm_std = kl_norm_arr.std(axis=0)

    # ── Lag summary (hard-threshold) ──────────────────────────────────────────
    _plot_lag_summary(
        config.layer_indices, lag_means, lag_stds,
        config.r2_threshold, fig_dir / "lag_summary",
    )

    # ── AUC convergence curves figure ────────────────────────────────────────
    logger.info("Generating AUC convergence curves figure ...")
    _plot_auc_curves(
        n_tok=config.ccf_n_tokens,
        layer_indices=config.layer_indices,
        r2_raw_mean=r2_raw_mean, r2_raw_std=r2_raw_std,
        r2_norm_mean=r2_norm_mean, r2_norm_std=r2_norm_std,
        kl_raw_mean=kl_raw_mean, kl_raw_std=kl_raw_std,
        kl_norm_mean=kl_norm_mean, kl_norm_std=kl_norm_std,
        path=fig_dir / "auc_curves",
    )

    # ── Cross-correlation figure ──────────────────────────────────────────────
    if ccf_lags is not None:
        logger.info("Generating cross-correlation figure ...")
        _plot_cross_correlation_by_layer(
            lags=ccf_lags,
            layer_indices=config.layer_indices,
            ccf_raw_mean=ccf_raw_mean,
            ccf_raw_std=ccf_raw_std,
            ccf_deriv_mean=ccf_deriv_mean,
            ccf_deriv_std=ccf_deriv_std,
            path=fig_dir / "ccf_by_layer",
        )

    # ── Combined lag figure (threshold-free) ─────────────────────────────────
    logger.info("Generating combined lag figure ...")
    _plot_combined_lag(
        layer_indices=config.layer_indices,
        ccf_peak_lags=ccf_peak_lags,
        auc_lag_means=auc_lag_means,
        auc_lag_stds=auc_lag_stds,
        auc_quantiles=config.auc_quantiles,
        path=fig_dir / "lag_combined",
    )

    # ── Simplex trajectory plots ──────────────────────────────────────────────
    if simplex_preds_all:
        logger.info("Generating simplex trajectory plots ...")
        _plot_simplex_trajectories(
            simplex_preds_all, simplex_beliefs_all,
            simplex_positions_valid, config.layer_indices, fig_dir,
        )

    # ── Save JSON metrics ─────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    lag_summary = {
        "r2_threshold": config.r2_threshold,
        "ccf_max_lag": config.ccf_max_lag,
        "ccf_smooth_window": config.ccf_smooth_window,
        "auc_quantiles": config.auc_quantiles,
        "t_kl_mean": float(t_kl_arr.mean()),
        "t_kl_std": float(t_kl_arr.std()),
        "per_layer": {
            str(l): {
                "t_r2_mean": metrics_out["layers"][str(l)]["t_r2_mean"],
                "t_r2_std": metrics_out["layers"][str(l)]["t_r2_std"],
                "lag_mean": metrics_out["layers"][str(l)]["lag_mean"],
                "lag_std": metrics_out["layers"][str(l)]["lag_std"],
                "n_r2_reached": metrics_out["layers"][str(l)]["n_r2_reached"],
                "ccf_peak_lag": metrics_out["layers"][str(l)]["ccf_peak_lag"],
                "auc_quantile_lags": metrics_out["layers"][str(l)]["auc_quantile_lags"],
                "auc_quantile_lag_stds": metrics_out["layers"][str(l)]["auc_quantile_lag_stds"],
            }
            for l in config.layer_indices
        },
    }
    with open(out_dir / "lag_summary.json", "w") as f:
        json.dump(lag_summary, f, indent=2)

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
