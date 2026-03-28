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
) -> None:
    """Plot: token position vs KL for selected layers.

    metric_type: one of 'kl_hmm_vs_tuned', 'kl_final_vs_tuned', 'kl_hmm_vs_logit'
    """
    layer_to_metrics = {m.layer: m for m in metrics}
    attr_name = f"{metric_type}_by_pos"

    titles = {
        "kl_hmm_vs_tuned": "KL(HMM || Tuned Lens) by Token Position",
        "kl_final_vs_tuned": "KL(Final Model || Tuned Lens) by Token Position",
        "kl_hmm_vs_logit": "KL(HMM || Logit Lens) by Token Position",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(selected_layers)))

    for i, layer in enumerate(selected_layers):
        m = layer_to_metrics[layer]
        kl_by_pos = getattr(m, attr_name)
        positions = np.arange(len(kl_by_pos))
        ax.plot(positions, kl_by_pos, linewidth=1, alpha=0.8, color=colors[i],
                label=f"Layer {layer}")

    ax.set_xlabel("Token position")
    ax.set_ylabel("KL divergence")
    ax.set_title(titles.get(metric_type, metric_type))
    ax.set_xscale("log")
    ax.legend(ncol=2, fontsize=7)
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
) -> None:
    """Plot tuned lens training loss curves per layer."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for layer, losses in sorted(loss_curves.items()):
        ax.plot(losses, alpha=0.6, linewidth=1, label=f"L{layer}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL loss (full vocabulary)")
    ax.set_title("Tuned Lens Training Loss per Layer")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=7, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
