"""Plotting functions for the per-layer tuned lens experiment."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import LayerMetrics


def plot_layer_vs_kl_final(
    metrics: list[LayerMetrics],
    path: Path,
) -> None:
    """Plot: layer index vs KL(final model || tuned lens)."""
    layers = [m.layer for m in metrics]
    kl_vals = [m.kl_final_vs_tuned for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, kl_vals, "o-", color="tab:blue", linewidth=2, markersize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(final model || tuned lens)")
    ax.set_title("Per-Layer Tuned Lens: Ability to Reconstruct Final Predictions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_vs_kl_hmm(
    metrics: list[LayerMetrics],
    path: Path,
) -> None:
    """Plot: layer index vs KL(HMM || tuned lens) and KL(HMM || logit lens)."""
    layers = [m.layer for m in metrics]
    kl_tuned = [m.kl_hmm_vs_tuned for m in metrics]
    kl_logit = [m.kl_hmm_vs_logit for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, kl_tuned, "s-", color="darkorange", linewidth=2, markersize=5,
            label="Tuned lens")
    ax.plot(layers, kl_logit, "^--", color="gray", linewidth=1.5, markersize=5,
            label="Raw logit lens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(HMM || lens)")
    ax.set_title("KL Divergence from HMM Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_position_vs_kl(
    metrics: list[LayerMetrics],
    selected_layers: list[int],
    path: Path,
    metric_type: str = "kl_hmm_vs_tuned",
    smooth_window: int = 0,
) -> None:
    """Plot: token position vs KL for selected layers.

    metric_type: one of 'kl_hmm_vs_tuned', 'kl_final_vs_tuned', 'kl_hmm_vs_logit', 'kl_hmm_vs_tuned_hmm'
    smooth_window: if > 0, apply rolling mean with this window size
    """
    layer_to_metrics = {m.layer: m for m in metrics}
    attr_name = f"{metric_type}_by_pos"

    titles = {
        "kl_hmm_vs_tuned": "KL(HMM || Tuned Lens) by Token Position",
        "kl_final_vs_tuned": "KL(Final Model || Tuned Lens) by Token Position",
        "kl_hmm_vs_logit": "KL(HMM || Logit Lens) by Token Position",
        "kl_hmm_vs_tuned_hmm": "KL(HMM || HMM-Target Tuned Lens) by Token Position",
    }

    # Determine figsize based on number of layers
    n_layers = len(selected_layers)
    figsize = (14, 6) if n_layers > 5 else (12, 5)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(selected_layers)))

    for i, layer in enumerate(selected_layers):
        m = layer_to_metrics[layer]
        kl_by_pos = getattr(m, attr_name)
        positions = np.arange(len(kl_by_pos))

        # Apply smoothing if requested
        if smooth_window > 0:
            kl_by_pos = np.convolve(kl_by_pos, np.ones(smooth_window) / smooth_window, mode='valid')
            positions = positions[:len(kl_by_pos)]

        # For many layers, reduce linewidth and alpha
        lw = 0.8 if n_layers > 10 else 1
        alpha = 0.6 if n_layers > 10 else 0.8

        ax.plot(positions, kl_by_pos, linewidth=lw, alpha=alpha, color=colors[i],
                label=f"Layer {layer}")

    ax.set_xlabel("Token position")
    ax.set_ylabel("KL divergence")
    ax.set_title(titles.get(metric_type, metric_type))
    ax.set_xscale("log")
    ncol = min(7, max(2, len(selected_layers) // 4))
    ax.legend(ncol=ncol, fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    metrics: list[LayerMetrics],
    r2_per_layer: dict[int, float] | None,
    path: Path,
) -> None:
    """4-panel comparison: tuned lens vs logit lens vs belief-state probe R^2."""
    layers = [m.layer for m in metrics]

    n_panels = 4 if r2_per_layer is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 1: KL(final || tuned lens) by layer
    ax = axes[0]
    ax.plot(layers, [m.kl_final_vs_tuned for m in metrics], "o-", color="tab:blue", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(final || tuned lens)")
    ax.set_title("Tuned Lens → Final Predictions")
    ax.grid(True, alpha=0.3)

    # Panel 2: KL(HMM || lens) for tuned vs logit
    ax = axes[1]
    ax.plot(layers, [m.kl_hmm_vs_tuned for m in metrics], "s-",
            color="darkorange", linewidth=2, label="Tuned lens")
    ax.plot(layers, [m.kl_hmm_vs_logit for m in metrics], "^--",
            color="gray", linewidth=1.5, label="Logit lens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(HMM || lens)")
    ax.set_title("Alignment with HMM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Top-1 agreement with final model
    ax = axes[2]
    ax.plot(layers, [m.top1_agreement_tuned for m in metrics], "o-",
            color="tab:green", linewidth=2, label="Tuned lens")
    ax.plot(layers, [m.top1_agreement_logit for m in metrics], "^--",
            color="gray", linewidth=1.5, label="Logit lens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 agreement")
    ax.set_title("Top-1 Agreement with Final Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Belief-state probe R^2 (if available)
    if r2_per_layer is not None:
        ax = axes[3]
        r2_layers = sorted(r2_per_layer.keys())
        r2_vals = [r2_per_layer[l] for l in r2_layers]
        ax.bar(r2_layers, r2_vals, color="tab:purple", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("R²")
        ax.set_title("Belief-State Probe R²")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Tuned Lens Per-Layer Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(
    loss_curves: dict[int, list[float]],
    path: Path,
    label: str = "",
) -> None:
    """Plot tuned lens training loss curves per layer."""
    fig, ax = plt.subplots(figsize=(12, 4))

    layers = sorted(loss_curves.keys())
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=min(layers) - 5, vmax=max(layers))
    for layer in layers:
        ax.plot(loss_curves[layer], alpha=0.8, linewidth=1,
                label=f'L{layer}', color=cmap(norm(layer)))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL loss (concept tokens)")
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


def plot_tuned_lens_results(
    metrics: list[LayerMetrics],
    r2_per_layer: dict[int, float],
    path: Path,
    title: str = "Tuned Lens Results",
) -> None:
    """Three-panel tuned lens results plot matching the notebook's plot_tuned_lens_results.

    Panel 1: KL divergence by layer (4 KL curves + logit lens baseline)
    Panel 2: R² vs KL(HMM || lens) dual y-axis
    Panel 3: KL(model || lens) for model-target vs HMM-target
    """
    layers = [m.layer for m in metrics]

    kl_tuned = [m.kl_hmm_vs_tuned for m in metrics]
    kl_tuned_hmm = [m.kl_hmm_vs_tuned_hmm for m in metrics]
    kl_tuned_vs_model = [m.kl_final_vs_tuned for m in metrics]
    kl_tuned_hmm_vs_model = [m.kl_final_vs_tuned_hmm for m in metrics]
    kl_logit = [m.kl_hmm_vs_logit for m in metrics]
    r2_values = [r2_per_layer[l] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: KL divergence by layer
    ax = axes[0]
    ax.plot(layers, kl_tuned, "o-", color="steelblue", linewidth=2,
            label="KL(HMM || tuned_lens)")
    ax.plot(layers, kl_tuned_hmm, "o-", color="green", linewidth=2,
            label="KL(HMM || tuned_lens_hmm)")
    ax.plot(layers, kl_tuned_vs_model, "o-", color="darkorange", linewidth=2,
            label="KL(model || tuned_lens)")
    ax.plot(layers, kl_tuned_hmm_vs_model, "o-", color="red", linewidth=2,
            label="KL(model || tuned_lens_hmm)")
    ax.axhline(
        kl_logit[-1], color="grey", linestyle="--", linewidth=1,
        label=f"Logit lens (layer {layers[-1]})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL")
    ax.set_title("KL Divergence by Layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: R² vs KL(HMM || lens) dual y-axis
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(layers, r2_values, "o-", color="black", linewidth=2,
             label="R² (linear probe)")
    ax2.plot(layers, kl_tuned, "s--", color="steelblue", linewidth=2,
             label="KL(HMM || tuned_lens)")
    ax2.plot(layers, kl_tuned_hmm, "s--", color="green", linewidth=2,
             label="KL(HMM || tuned_lens_hmm)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("R² (linear probe → belief state)", color="steelblue")
    ax2.set_ylabel("KL(HMM || tuned lens)", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.set_title("Belief-State R² vs Tuned-Lens KL per Layer")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 3: KL(model || lens) for both translators
    ax = axes[2]
    ax.plot(layers, kl_tuned_vs_model, "o-", color="darkorange", linewidth=2,
            label="KL(model || tuned_lens)")
    ax.plot(layers, kl_tuned_hmm_vs_model, "o-", color="red", linewidth=2,
            label="KL(model || tuned_lens_hmm)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL")
    ax.set_title("KL(model || lens): Model-target vs HMM-target")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_nll_by_layer(
    metrics: list[LayerMetrics],
    path: Path,
) -> None:
    """Plot NLL of actual next token under tuned and logit lens."""
    layers = [m.layer for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, [m.nll_tuned for m in metrics], "o-",
            color="tab:blue", linewidth=2, label="Tuned lens")
    ax.plot(layers, [m.nll_logit for m in metrics], "^--",
            color="gray", linewidth=1.5, label="Logit lens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("NLL (next token)")
    ax.set_title("Next-Token NLL by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
