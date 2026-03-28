"""Plotting utilities: belief simplex, KL divergence, R² bar chart, and cross-config comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from visualization import _to_barycentric, _belief_colors

from .results import ConfigResult


def plot_belief_simplex(
    belief_states: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot ground truth belief states on a 2D simplex (barycentric coords)."""
    x, y = _to_barycentric(belief_states)
    colors_rgb = _belief_colors(belief_states)
    mpl_colors = []
    for c in colors_rgb:
        parts = c.replace("rgb(", "").replace(")", "").split(",")
        mpl_colors.append((int(parts[0]) / 255, int(parts[1]) / 255, int(parts[2]) / 255))

    sqrt3 = np.sqrt(3)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, sqrt3 / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1)

    ax.scatter(x, y, c=mpl_colors, s=1, alpha=0.5)

    ax.text(-0.06, -0.04, "S0", fontsize=9, ha="center")
    ax.text(1.06, -0.04, "S1", fontsize=9, ha="center")
    ax.text(0.5, sqrt3 / 2 + 0.04, "S2", fontsize=9, ha="center")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, sqrt3 / 2 + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_simplex(
    predicted_beliefs: np.ndarray,
    gt_beliefs: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot predicted belief states from the linear probe on a 2D simplex.

    Points are the probe's predictions, colored by ground-truth belief state.
    """
    x, y = _to_barycentric(predicted_beliefs)
    colors_rgb = _belief_colors(gt_beliefs)
    mpl_colors = []
    for c in colors_rgb:
        parts = c.replace("rgb(", "").replace(")", "").split(",")
        mpl_colors.append((int(parts[0]) / 255, int(parts[1]) / 255, int(parts[2]) / 255))

    sqrt3 = np.sqrt(3)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, sqrt3 / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1)

    ax.scatter(x, y, c=mpl_colors, s=1, alpha=0.5)

    ax.text(-0.06, -0.04, "S0", fontsize=9, ha="center")
    ax.text(1.06, -0.04, "S1", fontsize=9, ha="center")
    ax.text(0.5, sqrt3 / 2 + 0.04, "S2", fontsize=9, ha="center")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, sqrt3 / 2 + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kl_divergence(
    kl_mean: np.ndarray,
    kl_std: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot KL divergence over sequence position."""
    seq_len = len(kl_mean)
    positions = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(positions, kl_mean, linewidth=0.8)
    ax.fill_between(
        positions, kl_mean - kl_std, kl_mean + kl_std, alpha=0.2
    )
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("KL(HMM || LLM)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlim(1, seq_len)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r2_by_layer(
    r2_per_layer: dict[int, float],
    title: str,
    path: Path,
) -> None:
    """Bar chart of R² values per layer."""
    layers = sorted(r2_per_layer.keys())
    values = [r2_per_layer[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(layers)), values, tick_label=[str(l) for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (test set)")
    ax.set_title(title)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results: list[ConfigResult], path: Path) -> None:
    """Multi-panel comparison across all configs."""
    process_names = sorted(set(r.process_name for r in results))
    process_cmaps = {
        process_names[i]: plt.cm.get_cmap(cmap)
        for i, cmap in enumerate(["Blues", "Oranges", "Greens", "Reds"])
        if i < len(process_names)
    }

    process_counts: dict[str, int] = {}
    result_colors = []
    for r in results:
        idx = process_counts.get(r.process_name, 0)
        process_counts[r.process_name] = idx + 1
        n_in_group = sum(1 for rr in results if rr.process_name == r.process_name)
        frac = 0.4 + 0.5 * (idx / max(1, n_in_group - 1))
        result_colors.append(process_cmaps[r.process_name](frac))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: R² by layer
    ax = axes[0]
    for i, r in enumerate(results):
        layers = sorted(r.r2_per_layer.keys())
        values = [r.r2_per_layer[l] for l in layers]
        ax.plot(layers, values, marker=".", markersize=3, linewidth=1.2,
                label=r.label, color=result_colors[i])
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Belief State Probe R² by Layer — All Configurations")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: KL divergence
    ax = axes[1]
    for i, r in enumerate(results):
        positions = np.arange(len(r.kl_mean))
        ax.plot(positions, r.kl_mean, linewidth=1.0,
                label=r.label, color=result_colors[i])
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("KL(HMM || LLM)")
    ax.set_title("KL Divergence Convergence — All Configurations")
    ax.set_xscale("log")
    ax.set_xlim(1, max(len(r.kl_mean) for r in results))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
