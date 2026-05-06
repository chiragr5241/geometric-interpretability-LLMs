#!/usr/bin/env python3
"""Generate Xavier-style intervention panel figures for the In-Context-Beliefs paper.

Four figures: (patching | steering) × (all_conds | xavier3)
Layout: 2 rows (Strata, Wing) × 3 columns (k = 1, 5, 10).
Y-axis: absolute KL to target after intervention (log scale, nats).
Gray dashed horizontal line: unmodified model baseline (baseline_mean).
Orange dashed line: averaged random-direction control across principled conditions.

Usage (from repo root):
    .venv/bin/python experiments/plot_xavier_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO  = Path(__file__).resolve().parent.parent
PAPER = REPO.parent / "In-Context-Beliefs-Paper"

RUNS: dict[str, Path] = {
    "Strata": REPO / "outputs/dani/20260504_215708_single_seq_interventions_strata",
    "Wing":   REPO / "outputs/dani/20260504_215702_single_seq_interventions_wing",
}

K_COLS     = [1, 5, 10]
OUT_DIR    = PAPER / "figures"
PRINCIPLED = ["optimal", "past_consistent", "splice"]

# ── Colors ────────────────────────────────────────────────────────────────────

# All-conditions variant (orange reserved for control line)
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

# Xavier variant: 3 lines only
XAVIER_COLORS: dict[str, str] = {
    "past_consistent": "#000000",
    "splice":          "#1f77b4",
}
XAVIER_LABELS: dict[str, str] = {
    "past_consistent": "past-consistent",
    "splice":          "past-inconsistent",
}

CONTROL_COLOR  = "#ff7f0e"   # orange — averaged random-direction control
BASELINE_COLOR = "gray"      # unmodified model KL

# ── Data helpers ──────────────────────────────────────────────────────────────

def load(run_dir: Path) -> dict:
    with open(run_dir / "plot_data" / "aggregated_plot_data.json") as f:
        return json.load(f)


def layer_stats(
    metric: dict,
    cond: str,
    k: int,
    layers: list[int],
) -> tuple[list[float], list[float]]:
    """Return (means, stderrs) over layers for a given condition/k."""
    k_str = str(k)
    means, stderrs = [], []
    for l in layers:
        e = metric[cond][k_str].get(str(l), {})
        means.append(e.get("mean", float("nan")))
        stderrs.append(e.get("stderr", float("nan")))
    return means, stderrs


def control_avg(
    ctrl: dict,
    k: int,
    layers: list[int],
) -> tuple[list[float], list[float]]:
    """Average control means and stderrs over all PRINCIPLED conditions."""
    k_str = str(k)
    m_rows: list[list[float]] = []
    s_rows: list[list[float]] = []
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
        nan = [float("nan")] * len(layers)
        return nan, nan
    return (
        list(np.nanmean(m_rows, axis=0)),
        list(np.nanmean(s_rows, axis=0)),
    )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Figure builder ────────────────────────────────────────────────────────────

def make_figure(
    datasets: dict[str, dict],
    intervention: str,   # "patching" | "steering"
    variant: str,        # "all" | "xavier"
) -> go.Figure:
    model_names = list(datasets.keys())
    n_rows = len(model_names)
    n_cols = len(K_COLS)

    target_key = f"{intervention}_to_target"
    ctrl_key   = f"{intervention}_control_to_target"

    if variant == "xavier":
        conds  = ["past_consistent", "splice"]
        colors = XAVIER_COLORS
        labels = XAVIER_LABELS
    else:
        conds = (
            ["optimal", "round_trip", "past_consistent", "splice", "random"]
            if intervention == "patching"
            else ["optimal", "past_consistent", "splice", "random"]
        )
        colors = ALL_COLORS
        labels = ALL_LABELS

    # Column headers in row 1 only; row 2 gets empty strings
    subplot_titles = [f"k = {k}" for k in K_COLS] + [""] * n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.07,
    )

    shown_in_legend: set[str] = set()

    for ri, model_name in enumerate(model_names):
        data   = datasets[model_name]
        m      = data["metrics"]
        bm     = float(data["baseline_mean"])
        layers = [int(l) for l in data["layer_indices"]]
        xs     = [str(l) for l in layers]

        for ci, k in enumerate(K_COLS):
            row, col = ri + 1, ci + 1

            # ── Gray dashed baseline ──────────────────────────────────────────
            bl_label = "unmodified model"
            show_bl = bl_label not in shown_in_legend
            if show_bl:
                shown_in_legend.add(bl_label)
            fig.add_trace(go.Scatter(
                x=xs,
                y=[bm] * len(xs),
                mode="lines",
                line=dict(color=BASELINE_COLOR, dash="dash", width=1.5),
                name=bl_label,
                legendgroup=bl_label,
                showlegend=show_bl,
            ), row=row, col=col)

            # ── Condition lines ───────────────────────────────────────────────
            for cond in conds:
                if cond not in m.get(target_key, {}):
                    continue
                means, stderrs = layer_stats(m[target_key], cond, k, layers)
                color = colors[cond]
                label = labels[cond]
                show  = label not in shown_in_legend
                if show:
                    shown_in_legend.add(label)

                upper = [mn + se for mn, se in zip(means, stderrs)]
                lower = [max(mn - se, 1e-7) for mn, se in zip(means, stderrs)]

                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=_rgba(color, 0.12),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=label,
                ), row=row, col=col)

                fig.add_trace(go.Scatter(
                    x=xs,
                    y=means,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,
                    legendgroup=label,
                    showlegend=show,
                ), row=row, col=col)

            # ── Averaged control (solid orange) ───────────────────────────────
            if ctrl_key in m:
                ctrl_means, ctrl_stderrs = control_avg(m[ctrl_key], k, layers)
                ctrl_upper = [mn + se for mn, se in zip(ctrl_means, ctrl_stderrs)]
                ctrl_lower = [max(mn - se, 1e-7) for mn, se in zip(ctrl_means, ctrl_stderrs)]
                ctrl_label = "random control"
                show_ctrl  = ctrl_label not in shown_in_legend
                if show_ctrl:
                    shown_in_legend.add(ctrl_label)

                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1],
                    y=ctrl_upper + ctrl_lower[::-1],
                    fill="toself",
                    fillcolor=_rgba(CONTROL_COLOR, 0.12),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=ctrl_label,
                ), row=row, col=col)

                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ctrl_means,
                    mode="lines",
                    line=dict(color=CONTROL_COLOR, width=2),
                    name=ctrl_label,
                    legendgroup=ctrl_label,
                    showlegend=show_ctrl,
                ), row=row, col=col)

        # ── Row label: model name as y-axis title of leftmost subplot ─────────
        fig.update_yaxes(
            type="log",
            title_text=f"<b>{model_name}</b><br>KL [nats]",
            row=ri + 1, col=1,
        )
        for ci in range(2, n_cols + 1):
            fig.update_yaxes(type="log", title_text="", row=ri + 1, col=ci)

    # X-axis labels
    for ri in range(1, n_rows + 1):
        for ci in range(1, n_cols + 1):
            fig.update_xaxes(title_text="Layer", row=ri, col=ci)

    v_label = (
        "all conditions"
        if variant == "all"
        else "past-consistent / past-inconsistent / control"
    )
    fig.update_layout(
        height=380 * n_rows,
        width=320 * n_cols + 180,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=80, b=130, l=90, r=40),
    )
    return fig


# ── Save ──────────────────────────────────────────────────────────────────────

def save_fig(fig: go.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        try:
            fig.write_image(str(p), scale=2 if ext == "png" else 1)
            print(f"  wrote {p.name}")
        except Exception as exc:
            print(f"  FAILED {p.name}: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    datasets = {name: load(path) for name, path in RUNS.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for intervention in ("patching", "steering"):
        for variant in ("all", "xavier"):
            print(f"Building {intervention}/{variant} ...")
            fig = make_figure(datasets, intervention, variant)
            save_fig(fig, OUT_DIR / f"interventions_{intervention}_{variant}")


if __name__ == "__main__":
    main()
