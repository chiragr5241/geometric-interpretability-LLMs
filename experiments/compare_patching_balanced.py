#!/usr/bin/env python3
"""One-off comparison of activation-patching results from unbalanced vs
probability-balanced encoder-decoder training.

Generates three figures:
  1. overlay_kl_vs_layer  — both experiments on a shared log-scale y-axis;
                            old = dashed, new = solid
  2. delta_kl_abs         — KL_new − KL_old per condition per layer
                            (negative = balanced improved)
  3. delta_kl_rel         — (KL_old − KL_new) / KL_old × 100 %
                            (positive = % improvement from balanced training)

Usage:
    python experiments/compare_patching_balanced.py \
        results/20260316_200705_activation_patching/metrics.json \
        outputs/dani/20260330_204238_activation_patching/metrics.json \
        --out outputs/dani/compare_patching_balanced
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

CONDITIONS = ["factual", "counterfactual", "garbage_valid", "garbage_random"]

_COLORS: dict[str, str] = {
    "factual":        "#1f77b4",
    "counterfactual": "#ff7f0e",
    "garbage_valid":  "#2ca02c",
    "garbage_random": "#d62728",
}


def _load(path: Path) -> tuple[list[int], dict[str, dict[int, dict]]]:
    with path.open() as f:
        raw = json.load(f)
    conds = raw["conditions"]
    layers = sorted({int(k) for cond in conds.values() for k in cond})
    parsed: dict[str, dict[int, dict]] = {
        cond: {int(layer): conds[cond][layer] for layer in conds[cond]}
        for cond in CONDITIONS
    }
    return layers, parsed


def _plot_overlay(
    layers: list[int],
    old: dict[str, dict[int, dict]],
    new: dict[str, dict[int, dict]],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layers]
    fig = go.Figure()

    for cond in CONDITIONS:
        color = _COLORS[cond]
        label = cond.replace("_", "-")

        # Old = dashed, no error band
        old_means = [old[cond][l]["mean"] for l in layers]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=old_means,
            name=f"{label} (unbalanced)",
            mode="lines+markers",
            line=dict(color=color, dash="dash", width=1.5),
            marker=dict(size=4, symbol="circle-open"),
        ))

        # New = solid, with ±stderr band
        new_means   = [new[cond][l]["mean"]   for l in layers]
        new_stderrs = [new[cond][l]["stderr"] for l in layers]
        upper = [m + e for m, e in zip(new_means, new_stderrs)]
        lower = [m - e for m, e in zip(new_means, new_stderrs)]
        rgba = _hex_to_rgba(color, 0.10)

        fig.add_trace(go.Scatter(
            x=layers_str + layers_str[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=rgba,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=new_means,
            name=f"{label} (balanced)",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=(
            "Activation patching: KL vs layer — unbalanced vs balanced decoders<br>"
            "<sup>KL(P_patched ‖ P_opt(η_target)) — log scale — "
            "dashed = unbalanced, solid ± stderr = balanced</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=520, width=940,
        margin=dict(t=90, b=60, l=70, r=20),
        legend=dict(tracegroupgap=0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))
    print(f"  Saved: {path.with_suffix('.png')}")


def _plot_delta_abs(
    layers: list[int],
    old: dict[str, dict[int, dict]],
    new: dict[str, dict[int, dict]],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layers]
    fig = go.Figure()

    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1))

    for cond in CONDITIONS:
        color = _COLORS[cond]
        label = cond.replace("_", "-")
        delta = [new[cond][l]["mean"] - old[cond][l]["mean"] for l in layers]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=delta,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=(
            "Δ KL (balanced − unbalanced) by condition and layer<br>"
            "<sup>Negative = balanced decoder improves (lower KL)</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats]",
        height=460, width=820,
        margin=dict(t=80, b=60, l=70, r=20),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))
    print(f"  Saved: {path.with_suffix('.png')}")


def _plot_delta_rel(
    layers: list[int],
    old: dict[str, dict[int, dict]],
    new: dict[str, dict[int, dict]],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layers]
    fig = go.Figure()

    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1))

    for cond in CONDITIONS:
        color = _COLORS[cond]
        label = cond.replace("_", "-")
        rel = [
            (old[cond][l]["mean"] - new[cond][l]["mean"]) / old[cond][l]["mean"] * 100
            for l in layers
        ]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=rel,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=(
            "Relative KL improvement from probability-balanced decoder training<br>"
            "<sup>(KL_unbalanced − KL_balanced) / KL_unbalanced × 100 %  —  "
            "positive = balanced is better</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Improvement [%]",
        height=460, width=820,
        margin=dict(t=80, b=60, l=70, r=20),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))
    print(f"  Saved: {path.with_suffix('.png')}")


def _print_summary(
    layers: list[int],
    old: dict[str, dict[int, dict]],
    new: dict[str, dict[int, dict]],
) -> None:
    print("\n─── Summary: mean KL averaged across all layers ───\n")
    header = f"{'Condition':<20} {'Unbalanced':>12} {'Balanced':>12} {'Δ (abs)':>12} {'Δ (%)':>10}"
    print(header)
    print("─" * len(header))
    for cond in CONDITIONS:
        old_avg = sum(old[cond][l]["mean"] for l in layers) / len(layers)
        new_avg = sum(new[cond][l]["mean"] for l in layers) / len(layers)
        delta   = new_avg - old_avg
        rel     = (old_avg - new_avg) / old_avg * 100
        sign    = "+" if rel >= 0 else ""
        label   = cond.replace("_", "-")
        print(f"{label:<20} {old_avg:>12.5f} {new_avg:>12.5f} {delta:>+12.5f} {sign}{rel:>9.1f}%")

    print("\n─── KL at peak (layers 5–15, where patching matters most) ───\n")
    peak_layers = [l for l in layers if 5 <= l <= 15]
    print(header)
    print("─" * len(header))
    for cond in CONDITIONS:
        old_avg = sum(old[cond][l]["mean"] for l in peak_layers) / len(peak_layers)
        new_avg = sum(new[cond][l]["mean"] for l in peak_layers) / len(peak_layers)
        delta   = new_avg - old_avg
        rel     = (old_avg - new_avg) / old_avg * 100
        sign    = "+" if rel >= 0 else ""
        label   = cond.replace("_", "-")
        print(f"{label:<20} {old_avg:>12.5f} {new_avg:>12.5f} {delta:>+12.5f} {sign}{rel:>9.1f}%")
    print()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_metrics", type=Path, help="metrics.json from unbalanced run")
    parser.add_argument("new_metrics", type=Path, help="metrics.json from balanced run")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/dani/compare_patching_balanced"),
                        help="Output directory for figures")
    args = parser.parse_args()

    print(f"Loading unbalanced: {args.old_metrics}")
    old_layers, old_data = _load(args.old_metrics)
    print(f"Loading balanced:   {args.new_metrics}")
    new_layers, new_data = _load(args.new_metrics)

    if old_layers != new_layers:
        raise ValueError(
            f"Layer indices differ: {old_layers} vs {new_layers}"
        )
    layers = old_layers

    print(f"\nGenerating figures → {args.out}/\n")
    _plot_overlay (layers, old_data, new_data, args.out / "overlay_kl_vs_layer")
    _plot_delta_abs(layers, old_data, new_data, args.out / "delta_kl_abs")
    _plot_delta_rel(layers, old_data, new_data, args.out / "delta_kl_rel")
    _print_summary (layers, old_data, new_data)


if __name__ == "__main__":
    main()
