#!/usr/bin/env python3
"""Generate intervention panel figures for the paper.

Four figures: (patching | steering) × (all_conds | two_cond)
Layout: 2 rows (Strata, Wing) × 3 columns (k = 1, 5, 10).
Y-axis: absolute KL to target after intervention (log scale, nats).
Gray dashed horizontal line: unmodified model baseline (baseline_mean).
Orange line: averaged random-direction control across principled conditions.

Usage (from submission root):
    python experiments/final_intervention_plot.py

Update RUNS below to point at your single_seq_interventions output directories.
Figures are written to OUT_DIR (default: ./figures/).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.weight': 'light',
})
sns.set_context('notebook')

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# Update these to point at your single_seq_interventions output directories:
RUNS: dict[str, Path] = {
    "Strata": ROOT / "results/single_seq_interventions_strata",
    "Wing":   ROOT / "results/single_seq_interventions_wing",
}

K_COLS  = [1, 5, 10]
OUT_DIR = ROOT / "figures"

PRINCIPLED = ["optimal", "past_consistent", "splice"]

# ── Colors ────────────────────────────────────────────────────────────────────

ALL_COLORS: dict[str, str] = {
    "optimal":         "#1f77b4",
    "round_trip":      "#8c564b",
    "past_consistent": "#2ca02c",
    "splice":          "#9467bd",
    "random":          "#d62728",
}
ALL_LABELS: dict[str, str] = {
    "optimal":         "optimal",
    "round_trip":      "round-trip",
    "past_consistent": "past-consistent",
    "splice":          "past-inconsistent",
    "random":          "random (belief)",
}

TWO_COLORS: dict[str, str] = {
    "past_consistent": "#000000",
    "splice":          "#1f77b4",
}
TWO_LABELS: dict[str, str] = {
    "past_consistent": "past-consistent",
    "splice":          "past-inconsistent",
}

CONTROL_COLOR  = "#ff7f0e"
BASELINE_COLOR = "gray"

# ── Data helpers ──────────────────────────────────────────────────────────────

def load(run_dir: Path) -> dict:
    with open(run_dir / "plot_data" / "aggregated_plot_data.json") as f:
        return json.load(f)


def layer_stats(
    metric: dict,
    cond: str,
    k: int,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    k_str = str(k)
    means, stderrs = [], []
    for l in layers:
        e = metric[cond][k_str].get(str(l), {})
        means.append(e.get("mean", float("nan")))
        stderrs.append(e.get("stderr", float("nan")))
    return np.array(means), np.array(stderrs)


def control_avg(
    ctrl: dict,
    k: int,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    k_str = str(k)
    m_rows, s_rows = [], []
    for cond in PRINCIPLED:
        if cond not in ctrl:
            continue
        ms, ss = [], []
        for l in layers:
            e = ctrl[cond][k_str].get(str(l), {})
            ms.append(e.get("mean", float("nan")))
            ss.append(e.get("stderr", float("nan")))
        m_rows.append(ms)
        s_rows.append(ss)
    if not m_rows:
        nan = np.full(len(layers), float("nan"))
        return nan, nan
    return np.nanmean(m_rows, axis=0), np.nanmean(s_rows, axis=0)


# ── Figure builder ────────────────────────────────────────────────────────────

def make_figure(
    datasets: dict[str, dict],
    intervention: str,
    variant: str,
) -> plt.Figure:
    model_names = list(datasets.keys())
    n_rows = len(model_names)
    n_cols = len(K_COLS)

    target_key = f"{intervention}_to_target"
    ctrl_key   = f"{intervention}_control_to_target"

    if variant == "two_cond":
        conds  = ["past_consistent", "splice"]
        colors = TWO_COLORS
        labels = TWO_LABELS
    else:
        conds = (
            ["optimal", "round_trip", "past_consistent", "splice", "random"]
            if intervention == "patching"
            else ["optimal", "past_consistent", "splice", "random"]
        )
        colors = ALL_COLORS
        labels = ALL_LABELS

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols + 1.5, 3.5 * n_rows + 1.5),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    legend_elements: OrderedDict = OrderedDict()

    for ri, model_name in enumerate(model_names):
        data   = datasets[model_name]
        m      = data["metrics"]
        bm     = float(data["baseline_mean"])
        layers = [int(l) for l in data["layer_indices"]]

        for ci, k in enumerate(K_COLS):
            ax = axes[ri, ci]

            if ri == 0:
                ax.set_title(f"k = {k}", fontsize=23, pad=18)

            # Baseline
            bl_label = "unmodified model"
            h = ax.axhline(bm, color=BASELINE_COLOR, ls="--", lw=2)
            if bl_label not in legend_elements:
                legend_elements[bl_label] = h

            # Condition lines
            for cond in conds:
                if cond not in m.get(target_key, {}):
                    continue
                means, stderrs = layer_stats(m[target_key], cond, k, layers)
                color = colors[cond]
                label = labels[cond]
                upper = means + stderrs
                lower = np.maximum(means - stderrs, 1e-7)
                ax.fill_between(layers, upper, lower, color=color, alpha=0.15, linewidth=0)
                (h,) = ax.plot(layers, means, color=color, lw=5)
                if label not in legend_elements:
                    legend_elements[label] = h

            # Control
            if ctrl_key in m:
                ctrl_means, ctrl_stderrs = control_avg(m[ctrl_key], k, layers)
                ctrl_upper = ctrl_means + ctrl_stderrs
                ctrl_lower = np.maximum(ctrl_means - ctrl_stderrs, 1e-7)
                ctrl_label = "random control"
                ax.fill_between(layers, ctrl_upper, ctrl_lower,
                                color=CONTROL_COLOR, alpha=0.15, linewidth=0)
                (h,) = ax.plot(layers, ctrl_means, color=CONTROL_COLOR, lw=5)
                if ctrl_label not in legend_elements:
                    legend_elements[ctrl_label] = h

            ax.set_yscale("log")
            ax.set_xlabel("Layer", fontsize=22, labelpad=14)
            ax.tick_params(axis="both", labelsize=22, length=3)
            ax.grid(True, alpha=0.2, linewidth=0.5)

        axes[ri, 0].set_ylabel(f"{model_name}\nKL [nats]", fontsize=22)

    handles = list(legend_elements.values())
    lbls    = list(legend_elements.keys())
    fig.legend(handles, lbls,
               loc="lower center", ncol=len(handles),
               fontsize=22, frameon=False,
               bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout(w_pad=0)
    return fig


# ── Save ──────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        try:
            dpi = 200 if ext == "png" else None
            fig.savefig(str(p), bbox_inches="tight", dpi=dpi)
            print(f"  wrote {p.name}")
        except Exception as exc:
            print(f"  FAILED {p.name}: {exc}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    datasets = {name: load(path) for name, path in RUNS.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for intervention in ("patching", "steering"):
        for variant in ("all", "two_cond"):
            print(f"Building {intervention}/{variant} ...")
            fig = make_figure(datasets, intervention, variant)
            stem = (f"interventions_{intervention}"
                    if variant == "two_cond"
                    else f"interventions_{intervention}_{variant}")
            save_fig(fig, OUT_DIR / stem)


if __name__ == "__main__":
    main()
