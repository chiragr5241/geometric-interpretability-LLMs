#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--title",
        type=str,
        default="Encoder MSE by layer",
    )
    return parser.parse_args()


def default_output_path(metrics_path: Path) -> Path:
    return metrics_path.parent / "figures" / "encoder_mse_per_layer.png"


def load_metrics(metrics_path: Path) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    metrics = json.loads(metrics_path.read_text())
    layers = sorted(int(layer) for layer in metrics.keys())
    train_mse = np.array(
        [metrics[str(layer)]["encoder"]["internal_train_mse"] for layer in layers],
        dtype=float,
    )
    eval_mean_mse = np.array(
        [metrics[str(layer)]["encoder"]["eval_mse_mean"] for layer in layers],
        dtype=float,
    )
    eval_std_mse = np.array(
        [
            np.std(metrics[str(layer)]["encoder"]["eval_mse_per_seq"], ddof=1)
            for layer in layers
        ],
        dtype=float,
    )
    return layers, train_mse, eval_mean_mse, eval_std_mse


def make_figure(
    layers: list[int],
    train_mse: np.ndarray,
    eval_mean_mse: np.ndarray,
    eval_std_mse: np.ndarray,
    title: str,
) -> go.Figure:
    upper = eval_mean_mse + eval_std_mse
    lower = np.maximum(eval_mean_mse - eval_std_mse, 0.0)
    x = np.array(layers, dtype=int)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            name="Held-out eval MSE ± 1 std",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=train_mse,
            mode="lines+markers",
            name="Train MSE",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=eval_mean_mse,
            mode="lines+markers",
            name="Held-out eval MSE",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="MSE",
        height=440,
        width=760,
        margin=dict(t=70, b=60, l=80, r=40),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_path.resolve()
    output_path = (args.output.resolve() if args.output is not None else default_output_path(metrics_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers, train_mse, eval_mean_mse, eval_std_mse = load_metrics(metrics_path)
    fig = make_figure(layers, train_mse, eval_mean_mse, eval_std_mse, args.title)
    fig.write_image(str(output_path))
    print(output_path)


if __name__ == "__main__":
    main()
