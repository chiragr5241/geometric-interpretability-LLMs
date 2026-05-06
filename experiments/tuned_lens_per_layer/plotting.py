"""Plotting functions for the per-layer tuned lens experiment.

Naming convention (all figures live under ``figures/``):

    by-layer (one figure, all lenses overlaid):
        kl_hmm_by_layer.png            KL(HMM || lens)
        kl_final_by_layer.png          KL(final model || lens)
        nll_by_layer.png               next-token NLL
        top1_agreement_by_layer.png    top-1 vs final-model top-1
        r2_belief_by_layer.png         belief-state probe R²

    by-position (one figure per lens, all layers overlaid):
        kl_hmm_by_position_<lens>.png
        kl_final_by_position_<lens>.png

    training:
        training_loss_<lens>.png       per-layer KL training curves

    summary:
        summary.png                    multi-panel overview

``<lens>`` ∈ {``logit``, ``tuned_full``, ``tuned_concept``, ``tuned_hmm``}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import LayerMetrics


# Visual style: consistent color/marker/linestyle per lens across all plots.
LENS_STYLE = {
    "logit":          {"color": "gray",       "ls": "--", "marker": "^", "label": "Logit lens"},
    "tuned_full":     {"color": "darkorange", "ls": "-",  "marker": "s", "label": "Tuned (full vocab)"},
    "tuned_concept":  {"color": "steelblue",  "ls": "-",  "marker": "o", "label": "Tuned (concept vocab)"},
    "tuned_hmm":      {"color": "green",      "ls": "-",  "marker": "D", "label": "Tuned (HMM target)"},
}


def _present_lenses(metrics: list[LayerMetrics]) -> list[str]:
    """Return lens names that exist in the metrics, in the canonical display order."""
    if not metrics:
        return []
    available = set(metrics[0].lenses.keys())
    order = ["logit", "tuned_full", "tuned_concept", "tuned_hmm"]
    return [n for n in order if n in available]


# ─────────────────────────── by-layer scalar plots ──────────────────────────


def _plot_per_layer_metric(
    metrics: list[LayerMetrics],
    metric_key: str,
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool = False,
) -> None:
    layers = [m.layer for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    for lens_name in _present_lenses(metrics):
        style = LENS_STYLE[lens_name]
        vals = [m.lenses[lens_name][metric_key] for m in metrics]
        ax.plot(
            layers, vals,
            color=style["color"], linestyle=style["ls"], marker=style["marker"],
            linewidth=2, markersize=5, label=style["label"],
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kl_hmm_by_layer(metrics: list[LayerMetrics], path: Path) -> None:
    _plot_per_layer_metric(
        metrics, "kl_hmm",
        ylabel="KL(HMM || lens)",
        title="KL(HMM || lens) by Layer",
        path=path,
    )


def plot_kl_final_by_layer(metrics: list[LayerMetrics], path: Path) -> None:
    _plot_per_layer_metric(
        metrics, "kl_final",
        ylabel="KL(final model || lens)",
        title="KL(final model || lens) by Layer",
        path=path,
    )


def plot_nll_by_layer(metrics: list[LayerMetrics], path: Path) -> None:
    _plot_per_layer_metric(
        metrics, "nll",
        ylabel="NLL (next token)",
        title="Next-Token NLL by Layer",
        path=path,
    )


def plot_top1_agreement_by_layer(metrics: list[LayerMetrics], path: Path) -> None:
    _plot_per_layer_metric(
        metrics, "top1_agreement",
        ylabel="Top-1 agreement with final model",
        title="Top-1 Agreement by Layer",
        path=path,
    )


def plot_r2_belief_by_layer(r2_per_layer: dict[int, float], path: Path) -> None:
    layers = sorted(r2_per_layer.keys())
    vals = [r2_per_layer[l] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layers, vals, color="tab:purple", alpha=0.75)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Belief-State Probe R² by Layer")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── by-position curves ─────────────────────────────


def _plot_by_position_for_lens(
    metrics: list[LayerMetrics],
    selected_layers: list[int],
    lens_name: str,
    by_pos_key: str,
    title: str,
    path: Path,
    smooth_window: int = 0,
    log_x: bool = True,
) -> None:
    layer_to_metrics = {m.layer: m for m in metrics}
    n_layers = len(selected_layers)
    figsize = (14, 6) if n_layers > 5 else (12, 5)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(selected_layers)))

    for i, layer in enumerate(selected_layers):
        m = layer_to_metrics[layer]
        if lens_name not in m.lenses:
            continue
        kl_by_pos = m.lenses[lens_name][by_pos_key]
        positions = np.arange(len(kl_by_pos))
        if smooth_window > 0:
            kl_by_pos = np.convolve(
                kl_by_pos, np.ones(smooth_window) / smooth_window, mode="valid",
            )
            positions = positions[: len(kl_by_pos)]
        lw = 0.8 if n_layers > 10 else 1
        alpha = 0.6 if n_layers > 10 else 0.8
        ax.plot(positions, kl_by_pos, linewidth=lw, alpha=alpha, color=colors[i],
                label=f"Layer {layer}")

    ax.set_xlabel("Token position")
    ax.set_ylabel("KL divergence")
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ncol = min(7, max(2, len(selected_layers) // 4))
    ax.legend(ncol=ncol, fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kl_hmm_by_position(
    metrics: list[LayerMetrics],
    selected_layers: list[int],
    lens_name: str,
    path: Path,
) -> None:
    style_label = LENS_STYLE[lens_name]["label"]
    _plot_by_position_for_lens(
        metrics, selected_layers, lens_name,
        by_pos_key="kl_hmm_by_pos",
        title=f"KL(HMM || {style_label}) by Token Position",
        path=path,
    )


def plot_kl_final_by_position(
    metrics: list[LayerMetrics],
    selected_layers: list[int],
    lens_name: str,
    path: Path,
) -> None:
    style_label = LENS_STYLE[lens_name]["label"]
    _plot_by_position_for_lens(
        metrics, selected_layers, lens_name,
        by_pos_key="kl_final_by_pos",
        title=f"KL(final model || {style_label}) by Token Position",
        path=path,
    )


# ─────────────────────────────── training ───────────────────────────────────


def plot_training_loss(
    loss_curves: dict[int, list[float]],
    path: Path,
    label: str = "",
) -> None:
    """Per-layer KL training curves for one tuned-lens variant."""
    fig, ax = plt.subplots(figsize=(12, 4))
    layers = sorted(loss_curves.keys())
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=min(layers) - 5, vmax=max(layers))
    for layer in layers:
        ax.plot(loss_curves[layer], alpha=0.8, linewidth=1,
                label=f"L{layer}", color=cmap(norm(layer)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL loss")
    title = "Tuned Lens Training Loss per Layer"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=7, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────── summary ─────────────────────────────────────


def plot_summary(
    metrics: list[LayerMetrics],
    r2_per_layer: dict[int, float],
    path: Path,
    title: str = "Tuned Lens Per-Layer Analysis",
) -> None:
    """4-panel summary: KL(HMM||lens), KL(final||lens), top-1, R²."""
    layers = [m.layer for m in metrics]
    lenses = _present_lenses(metrics)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: KL(HMM || lens)
    ax = axes[0]
    for lens_name in lenses:
        style = LENS_STYLE[lens_name]
        vals = [m.lenses[lens_name]["kl_hmm"] for m in metrics]
        ax.plot(layers, vals, color=style["color"], linestyle=style["ls"],
                marker=style["marker"], linewidth=2, markersize=4, label=style["label"])
    ax.set_xlabel("Layer"); ax.set_ylabel("KL(HMM || lens)"); ax.set_title("Alignment with HMM")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: KL(final || lens)
    ax = axes[1]
    for lens_name in lenses:
        style = LENS_STYLE[lens_name]
        vals = [m.lenses[lens_name]["kl_final"] for m in metrics]
        ax.plot(layers, vals, color=style["color"], linestyle=style["ls"],
                marker=style["marker"], linewidth=2, markersize=4, label=style["label"])
    ax.set_xlabel("Layer"); ax.set_ylabel("KL(final || lens)")
    ax.set_title("Reconstruction of Final Predictions")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: top-1 agreement
    ax = axes[2]
    for lens_name in lenses:
        style = LENS_STYLE[lens_name]
        vals = [m.lenses[lens_name]["top1_agreement"] for m in metrics]
        ax.plot(layers, vals, color=style["color"], linestyle=style["ls"],
                marker=style["marker"], linewidth=2, markersize=4, label=style["label"])
    ax.set_xlabel("Layer"); ax.set_ylabel("Top-1 agreement")
    ax.set_title("Top-1 Agreement with Final Model")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: belief-state R²
    ax = axes[3]
    if r2_per_layer:
        r2_layers = sorted(r2_per_layer.keys())
        r2_vals = [r2_per_layer[l] for l in r2_layers]
        ax.bar(r2_layers, r2_vals, color="tab:purple", alpha=0.75)
        ax.set_xlabel("Layer"); ax.set_ylabel("R²"); ax.set_title("Belief-State Probe R²")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
