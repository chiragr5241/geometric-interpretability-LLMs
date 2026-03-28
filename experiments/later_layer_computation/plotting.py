"""Visualization for the later-layer computation experiment."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .decoders import DecoderResults, COMPONENT_NAMES, TARGET_NAMES
from .decomposition import DecompositionResult
from .intervention import InterventionResults


COMP_COLORS = {
    "full": "#333333",
    "belief": "#2196F3",
    "orth": "#FF5722",
    "orth_pca": "#FF9800",
}
COMP_LABELS = {
    "full": "Full activation",
    "belief": "Belief component",
    "orth": "Orthogonal component",
    "orth_pca": "Orth (PCA-matched)",
}


def plot_probe_r2_and_variance(
    decomposition: DecompositionResult,
    layer_indices: list[int],
    path: Path,
) -> None:
    """Layer vs probe R² and variance decomposition (2-panel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    r2s = [decomposition.layers[l].probe_r2 for l in layer_indices]
    ax1.plot(layer_indices, r2s, "o-", color="#2196F3", linewidth=2, markersize=4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe R²")
    ax1.set_title("Belief State Probe R² by Layer")
    ax1.set_ylim(min(r2s) - 0.01, 1.005)
    ax1.grid(True, alpha=0.3)

    var_b = [decomposition.layers[l].var_belief for l in layer_indices]
    var_o = [decomposition.layers[l].var_orth for l in layer_indices]
    ax2.fill_between(layer_indices, 0, var_b, alpha=0.4, color="#2196F3", label="Belief subspace")
    ax2.fill_between(layer_indices, var_b, [b + o for b, o in zip(var_b, var_o)],
                     alpha=0.4, color="#FF5722", label="Orthogonal")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Fraction of Total Variance")
    ax2.set_title("Variance Decomposition by Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_r2(
    decoder_results: DecoderResults,
    layer_indices: list[int],
    path: Path,
) -> None:
    """Grid of (target × layer) showing R² for each component."""
    available_targets = set()
    for layer in layer_indices:
        lr = decoder_results.layers.get(layer)
        if lr is None:
            continue
        for comp_scores in lr.scores.values():
            available_targets.update(comp_scores.keys())
    targets = [t for t in TARGET_NAMES if t in available_targets]

    n_targets = len(targets)
    if n_targets == 0:
        return

    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, tgt in enumerate(targets):
        ax = axes[idx // n_cols, idx % n_cols]
        for comp in COMPONENT_NAMES:
            vals = []
            for layer in layer_indices:
                lr = decoder_results.layers.get(layer)
                if lr is None:
                    vals.append(float("nan"))
                    continue
                s = lr.scores.get(comp, {}).get(tgt)
                vals.append(s.r2 if s is not None else float("nan"))
            ax.plot(
                layer_indices, vals, "o-",
                color=COMP_COLORS[comp],
                label=COMP_LABELS[comp],
                linewidth=1.5, markersize=3, alpha=0.85,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel("R²")
        ax.set_title(tgt.replace("_", " ").title())
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_targets, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Predictive Residual Analysis: R² by Layer", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_sweep(
    intervention_results: InterventionResults,
    layer_indices: list[int],
    path: Path,
) -> None:
    """Layer vs causal effect for each ablation condition (3-panel)."""
    conditions = ["ablate_orth", "ablate_belief", "mean_ablate_orth"]
    cond_labels = {
        "ablate_orth": "Ablate orthogonal",
        "ablate_belief": "Ablate belief",
        "mean_ablate_orth": "Mean-replace orthogonal",
    }
    cond_colors = {
        "ablate_orth": "#FF5722",
        "ablate_belief": "#2196F3",
        "mean_ablate_orth": "#FF9800",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: KL(original || ablated)
    ax = axes[0]
    for cond in conditions:
        layers_c = [r.layer for r in intervention_results.results if r.condition == cond]
        kl_vals = [r.kl_vs_original_mean for r in intervention_results.results if r.condition == cond]
        kl_stds = [r.kl_vs_original_std for r in intervention_results.results if r.condition == cond]
        ax.errorbar(layers_c, kl_vals, yerr=kl_stds, fmt="o-",
                    color=cond_colors[cond], label=cond_labels[cond],
                    linewidth=1.5, markersize=4, capsize=2, alpha=0.85)
    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("KL(original ‖ ablated)")
    ax.set_title("Faithfulness Cost of Ablation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: KL(HMM || ablated) and baseline
    ax = axes[1]
    for cond in conditions:
        layers_c = [r.layer for r in intervention_results.results if r.condition == cond]
        kl_hmm = [r.kl_vs_hmm_mean for r in intervention_results.results if r.condition == cond]
        ax.plot(layers_c, kl_hmm, "o-",
                color=cond_colors[cond], label=f"{cond_labels[cond]}",
                linewidth=1.5, markersize=4, alpha=0.85)
    baseline = [r.kl_vs_hmm_baseline for r in intervention_results.results if r.condition == "ablate_orth"]
    baseline_layers = [r.layer for r in intervention_results.results if r.condition == "ablate_orth"]
    if baseline:
        ax.plot(baseline_layers, baseline, "k--", label="Baseline (no ablation)",
                linewidth=1, alpha=0.6)
    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("KL(HMM ‖ model)")
    ax.set_title("Prediction Quality After Ablation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Top-1 accuracy
    ax = axes[2]
    for cond in conditions:
        layers_c = [r.layer for r in intervention_results.results if r.condition == cond]
        top1 = [r.top1_accuracy_ablated for r in intervention_results.results if r.condition == cond]
        ax.plot(layers_c, top1, "o-",
                color=cond_colors[cond], label=cond_labels[cond],
                linewidth=1.5, markersize=4, alpha=0.85)
    if baseline_layers:
        orig_top1 = [r.top1_accuracy_original for r in intervention_results.results if r.condition == "ablate_orth"]
        ax.plot(baseline_layers, orig_top1, "k--", label="No ablation",
                linewidth=1, alpha=0.6)
    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("Top-1 Accuracy (vs HMM)")
    ax.set_title("Top-1 Accuracy After Ablation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Causal Intervention: Projection Ablation by Layer", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_position_effects(
    decoder_results: DecoderResults,
    decomposition: DecompositionResult,
    token_positions: np.ndarray,
    layer_indices: list[int],
    path: Path,
    n_bins: int = 20,
) -> None:
    """Show how decoder R² varies with token position (early vs late in sequence)."""
    sample_layers = [
        layer_indices[0],
        layer_indices[len(layer_indices) // 4],
        layer_indices[len(layer_indices) // 2],
        layer_indices[-1],
    ]

    pos_bins = np.linspace(token_positions.min(), token_positions.max() + 1, n_bins + 1)
    bin_centers = 0.5 * (pos_bins[:-1] + pos_bins[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, layer in enumerate(sample_layers):
        ax = axes[ax_idx]
        ld = decomposition.layers[layer]

        belief_norms = np.linalg.norm(ld.h_belief, axis=-1)
        orth_norms = np.linalg.norm(ld.h_orth, axis=-1)
        ratio = orth_norms / (belief_norms + 1e-10)

        bin_means = []
        for i in range(n_bins):
            mask = (token_positions >= pos_bins[i]) & (token_positions < pos_bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(ratio[mask].mean())
            else:
                bin_means.append(float("nan"))

        ax.plot(bin_centers, bin_means, "o-", color="#FF5722", markersize=3)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("‖h_orth‖ / ‖h_belief‖")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Orthogonal / Belief Norm Ratio vs Token Position", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_comparison(
    decoder_results: DecoderResults,
    intervention_results: InterventionResults,
    decomposition: DecompositionResult,
    layer_indices: list[int],
    path: Path,
) -> None:
    """4-panel summary: R², decoder gap, ablation cost, variance decomposition."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Probe R²
    ax = axes[0, 0]
    r2s = [decomposition.layers[l].probe_r2 for l in layer_indices]
    ax.plot(layer_indices, r2s, "o-", color="#2196F3", linewidth=2, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe R²")
    ax.set_title("Belief State Probe R²")
    ax.grid(True, alpha=0.3)

    # Panel 2: Decoder R² gap (orth - belief) for concept_logits
    ax = axes[0, 1]
    for tgt, color in [("concept_logits", "#333333"), ("hmm_next_token", "#4CAF50"),
                        ("logit_residual", "#9C27B0")]:
        gaps = []
        for layer in layer_indices:
            lr = decoder_results.layers.get(layer)
            if lr is None:
                gaps.append(float("nan"))
                continue
            orth_s = lr.scores.get("orth", {}).get(tgt)
            belief_s = lr.scores.get("belief", {}).get(tgt)
            if orth_s is not None and belief_s is not None:
                gaps.append(orth_s.r2 - belief_s.r2)
            else:
                gaps.append(float("nan"))
        ax.plot(layer_indices, gaps, "o-", label=tgt, linewidth=1.5, markersize=3,
                color=color)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²(orth) − R²(belief)")
    ax.set_title("Orthogonal Predictive Advantage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Ablation KL cost
    ax = axes[1, 0]
    for cond, color in [("ablate_orth", "#FF5722"), ("ablate_belief", "#2196F3")]:
        layers_c = [r.layer for r in intervention_results.results if r.condition == cond]
        kl_vals = [r.kl_vs_original_mean for r in intervention_results.results if r.condition == cond]
        ax.plot(layers_c, kl_vals, "o-", color=color,
                label=f"{'Remove orth' if cond == 'ablate_orth' else 'Remove belief'}",
                linewidth=1.5, markersize=4)
    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("KL(original ‖ ablated)")
    ax.set_title("Causal Effect of Ablation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Variance decomposition
    ax = axes[1, 1]
    var_b = [decomposition.layers[l].var_belief for l in layer_indices]
    var_o = [decomposition.layers[l].var_orth for l in layer_indices]
    ax.bar(layer_indices, var_b, color="#2196F3", alpha=0.7, label="Belief")
    ax.bar(layer_indices, var_o, bottom=var_b, color="#FF5722", alpha=0.7, label="Orthogonal")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Variance")
    ax.set_title("Variance Decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Later-Layer Computation: Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
