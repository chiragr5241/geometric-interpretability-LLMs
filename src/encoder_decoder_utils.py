from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from decoder import Decoder, DecoderResult
from probes import Probe, ProbeResult

logger = logging.getLogger(__name__)


def _with_subtitle(title: str, subtitle: str | None) -> str:
    if subtitle is None:
        return title
    return f"{title}<br><sup>{subtitle}</sup>"


def _save_fig(fig: go.Figure, path: Path) -> None:
    """Save a Plotly figure as PNG."""
    fig.write_image(str(path.with_suffix(".png")))


def r2(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def evaluate_encoder(
    probe: Probe,
    acts: np.ndarray,
    beliefs: np.ndarray,
) -> tuple[float, float]:
    dev = next(probe.parameters()).device
    probe.eval()
    with torch.no_grad():
        pred = probe(torch.from_numpy(acts).float().to(dev)).cpu().numpy()
    mse = float(np.mean((pred - beliefs) ** 2))
    return mse, r2(pred, beliefs)


def evaluate_decoder(
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


def evaluate_roundtrip(
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


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_z = a - a.mean()
    b_z = b - b.mean()
    denom = float(np.sqrt((a_z ** 2).sum() * (b_z ** 2).sum()) + 1e-10)
    return float((a_z * b_z).sum() / denom)


def fidelity_arrays(
    acts: np.ndarray,
    beliefs_true: np.ndarray,
    probe: Probe,
    decoder: Decoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probe = probe.to(device)
    decoder = decoder.to(device)
    probe.eval()
    decoder.eval()
    acts_t = torch.from_numpy(acts).float().to(device)
    beliefs_t = torch.from_numpy(beliefs_true).float().to(device)
    with torch.no_grad():
        a_hat = decoder(beliefs_t).cpu().numpy()
        b_hat_raw = probe(acts_t).cpu().numpy()
        b_hat = np.clip(b_hat_raw, 0.0, None)
        b_hat /= b_hat.sum(axis=-1, keepdims=True) + 1e-10
        a_hat_rt = decoder(torch.from_numpy(b_hat).float().to(device)).cpu().numpy()
    norm_sq = (acts ** 2).sum(axis=-1) + 1e-10
    mse_dec = ((acts - a_hat) ** 2).sum(axis=-1)
    mse_dec_norm = mse_dec / norm_sq
    mse_rt = ((acts - a_hat_rt) ** 2).sum(axis=-1)
    mse_rt_norm = mse_rt / norm_sq
    return mse_dec, mse_dec_norm, mse_rt, mse_rt_norm


def binned_mean(
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


def simplex_scatter(
    predicted_beliefs: np.ndarray,
    gt_beliefs: np.ndarray,
    layer: int,
    path: Path,
    title: str | None = None,
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
        title=title or f"Encoder predictions — Layer {layer} (eval, N={len(x)})",
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
    _save_fig(fig, path)


def decoder_loss_curves(
    decoder_results: dict[int, DecoderResult],
    layer_indices: list[int],
    path: Path,
    subtitle: str | None = None,
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
        title=_with_subtitle("Decoder training loss curves (log y)", subtitle),
        height=260 * n_rows,
        width=320 * n_cols,
    )
    _save_fig(fig, path)


def encoder_mse_curves(
    probe_results: dict[int, ProbeResult],
    layer_indices: list[int],
    path: Path,
    subtitle: str | None = None,
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
        pr = probe_results[layer]
        eps = list(range(len(pr.train_mse_curve)))
        show = idx == 0
        fig.add_trace(
            go.Scatter(
                x=eps, y=pr.train_mse_curve, name="Train MSE",
                showlegend=show, line=dict(color="#2ca02c"),
            ),
            row=row, col=col,
        )
        if pr.test_mse_curve:
            eval_eps = list(range(len(pr.test_mse_curve)))
            fig.add_trace(
                go.Scatter(
                    x=eval_eps, y=pr.test_mse_curve, name="Eval MSE",
                    showlegend=show, line=dict(color="#ff7f0e"),
                ),
                row=row, col=col,
            )
        fig.update_yaxes(type="log", row=row, col=col)
    fig.update_layout(
        title=_with_subtitle("Encoder train/eval MSE curves (log y)", subtitle),
        height=260 * n_rows,
        width=320 * n_cols,
    )
    _save_fig(fig, path)


def layer_line_plot(
    eval_vals: list[float],
    layer_indices: list[int],
    y_title: str,
    title: str,
    path: Path,
    train_vals: list[float] | None = None,
    log_y: bool = False,
    subtitle: str | None = None,
) -> None:
    layers = [str(l) for l in layer_indices]
    traces: list[Any] = []
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
        title=_with_subtitle(title, subtitle),
        xaxis_title="Layer",
        yaxis_title=y_title,
        height=420,
        width=720,
        margin=dict(t=70, b=60, l=70, r=40),
    )
    _save_fig(fig, path)


def make_grid_figure(
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
    for panel_idx, layer_val in enumerate(layer_indices):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        y = y_by_layer[layer_val]
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
        bx, by = binned_mean(x_all, y, n_bins)
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
