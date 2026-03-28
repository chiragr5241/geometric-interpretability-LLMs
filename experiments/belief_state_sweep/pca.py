"""PCA visualisation of residual-stream activations (2D and 3D grids)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(
    all_activations: dict[int, np.ndarray],
    belief_states_flat: np.ndarray,
    layers: list[int],
    title: str,
    path: Path,
    n_cols: int = 7,
) -> None:
    """PCA of residual stream per layer, colored by belief state. 4x7 grid for 28 layers."""
    n_layers = len(layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    beliefs_rgb = belief_states_flat / (
        belief_states_flat.sum(axis=1, keepdims=True) + 1e-10
    )

    for idx, layer in enumerate(sorted(layers)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        pca = PCA(n_components=6)
        pca_result = pca.fit_transform(all_activations[layer])
        evr = pca.explained_variance_ratio_

        ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=beliefs_rgb,
            s=2,
            alpha=0.4,
            rasterized=True,
        )
        ax.set_xlabel(f"PC0 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC1 ({evr[1]:.1%})")
        ax.set_title(f"Layer {layer}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for idx in range(n_layers, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_3d(
    all_activations: dict[int, np.ndarray],
    belief_states_flat: np.ndarray,
    layers: list[int],
    title: str,
    path: Path,
    n_cols: int = 7,
) -> None:
    """3D PCA of residual stream per layer, colored by belief state. Uses first 3 PCs."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_layers = len(layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows))

    beliefs_rgb = belief_states_flat / (
        belief_states_flat.sum(axis=1, keepdims=True) + 1e-10
    )

    for idx, layer in enumerate(sorted(layers)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")

        pca = PCA(n_components=6)
        pca_result = pca.fit_transform(all_activations[layer])
        evr = pca.explained_variance_ratio_

        ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            pca_result[:, 2],
            c=beliefs_rgb,
            s=2,
            alpha=0.4,
            rasterized=True,
        )
        ax.set_xlabel(f"PC0 ({evr[0]:.1%})", fontsize=6)
        ax.set_ylabel(f"PC1 ({evr[1]:.1%})", fontsize=6)
        ax.set_zlabel(f"PC2 ({evr[2]:.1%})", fontsize=6)
        ax.set_title(f"Layer {layer}", fontsize=8)
        ax.tick_params(labelsize=5)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
