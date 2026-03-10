from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from probes import ProbeResult


def _to_barycentric(beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N, 3) belief/probability arrays to 2D barycentric (x, y) coordinates."""
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2) * beliefs[:, 2]
    return x, y


def _belief_colors(beliefs: np.ndarray) -> list[str]:
    """Color each point by its belief state: RGB = (b0, b1, b2)."""
    clipped = np.clip(beliefs, 0.0, 1.0)
    return [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for r, g, b in clipped
    ]


def _simplex_outline_traces() -> list[go.BaseTraceType]:
    sqrt3 = np.sqrt(3)
    outline = go.Scatter(
        x=[0, 1, 0.5, 0],
        y=[0, 0, sqrt3 / 2, 0],
        mode="lines",
        line=dict(color="black", width=1),
        showlegend=False,
        hoverinfo="skip",
    )
    labels = go.Scatter(
        x=[-0.08, 1.08, 0.5],
        y=[-0.05, -0.05, sqrt3 / 2 + 0.06],
        mode="text",
        text=["S0", "S1", "S2"],
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(size=9),
    )
    return [outline, labels]


def _simplex_grid_traces() -> list[go.BaseTraceType]:
    """Dashed gray isolines at b0, b1, b2 = 1/3 and 2/3."""
    sqrt3 = np.sqrt(3)
    grid_color = "rgba(150,150,150,0.4)"
    traces = []
    for t in (1 / 3, 2 / 3):
        # b0 = t  →  from (1-t, 0) to (0.5*(1-t), (√3/2)*(1-t))
        traces.append(go.Scatter(
            x=[1 - t, 0.5 * (1 - t)], y=[0, (sqrt3 / 2) * (1 - t)],
            mode="lines", line=dict(color=grid_color, width=0.8, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        # b1 = t  →  from (t, 0) to (t + 0.5*(1-t), (√3/2)*(1-t))
        traces.append(go.Scatter(
            x=[t, t + 0.5 * (1 - t)], y=[0, (sqrt3 / 2) * (1 - t)],
            mode="lines", line=dict(color=grid_color, width=0.8, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        # b2 = t  →  from (0.5*t, (√3/2)*t) to (1 - 0.5*t, (√3/2)*t)
        traces.append(go.Scatter(
            x=[0.5 * t, 1 - 0.5 * t], y=[(sqrt3 / 2) * t, (sqrt3 / 2) * t],
            mode="lines", line=dict(color=grid_color, width=0.8, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def belief_simplex_traces(
    beliefs: np.ndarray,
    colors: list[str] | None = None,
    name: str = "",
    opacity: float = 0.6,
    size: int = 3,
) -> list[go.BaseTraceType]:
    """
    Plotly traces for a set of points plotted on the 2D simplex.

    beliefs: (N, 3) — raw values passed directly; barycentric projection is the only transform.
    colors:  per-point color strings. Defaults to coloring by belief state (RGB = b).
    """
    if colors is None:
        colors = _belief_colors(beliefs)
    x, y = _to_barycentric(beliefs)
    scatter = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color=colors, size=size, opacity=opacity),
        name=name,
        showlegend=bool(name),
        hoverinfo="skip",
    )
    return _simplex_outline_traces() + _simplex_grid_traces() + [scatter]


def plot_belief_grid(
    optimal_beliefs: list[np.ndarray],
    full_probe_results: dict[int, list[ProbeResult]],
    post_probe_results: dict[int, list[ProbeResult]],
    layer_indices: list[int],
    kl_thresholds: list[int],
    output_path: Path,
) -> None:
    """
    Build and save the belief geometry grid.

    Layout:
      Rows: [Optimal] + one row per layer in layer_indices
      Cols: [Full sequence probe] [Post-threshold probe]

    optimal_beliefs:    list of (L+1, 3) arrays, one per sequence.
    full_probe_results: {layer_idx: [ProbeResult, ...]} one per sequence.
    post_probe_results: same structure, trained on post-threshold activations.
    kl_thresholds:      list of t* values, one per sequence.
    output_path:        saved as <output_path>.html
    """
    n_layers = len(layer_indices)
    n_rows = 1 + n_layers
    n_cols = 2
    sqrt3 = np.sqrt(3)

    if len(kl_thresholds) == 1:
        threshold_label = f"t*={kl_thresholds[0]}"
    else:
        mean_t = float(np.mean(kl_thresholds))
        threshold_label = f"t*≈{mean_t:.1f}"

    subplot_titles = []
    for row_label in ["Optimal"] + [f"Layer {l}" for l in layer_indices]:
        subplot_titles += [
            f"{row_label} — full sequence",
            f"{row_label} — post-threshold ({threshold_label})",
        ]

    vertical_spacing = 0.2 / max(1, n_rows - 1)
    height_per_row = 250

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=vertical_spacing,
    )

    def _add(row: int, col: int, beliefs: np.ndarray, gt_beliefs: np.ndarray) -> None:
        colors = _belief_colors(gt_beliefs)
        for trace in belief_simplex_traces(beliefs, colors=colors):
            fig.add_trace(trace, row=row, col=col)

    # ── Row 1: optimal belief states ──────────────────────────────────────────
    # Col 1: all positions
    full_optimal = np.concatenate(optimal_beliefs, axis=0)
    _add(1, 1, full_optimal, full_optimal)

    # Col 2: post-threshold positions only (different t* per sequence)
    post_optimal = np.concatenate(
        [b[t:] for b, t in zip(optimal_beliefs, kl_thresholds)], axis=0
    )
    _add(1, 2, post_optimal, post_optimal)

    # ── Rows 2+: probe-projected beliefs ──────────────────────────────────────
    for row_offset, layer in enumerate(layer_indices):
        row = row_offset + 2
        for col, probe_dict in enumerate([full_probe_results, post_probe_results], start=1):
            results = probe_dict.get(layer, [])
            if not results:
                continue
            computed_all = np.concatenate(
                [pr.computed_belief_states for pr in results], axis=0
            )
            gt_all = np.concatenate([pr.gt_belief_states for pr in results], axis=0)
            _add(row, col, computed_all, gt_all)

    # ── Axis formatting ───────────────────────────────────────────────────────
    fig.update_xaxes(range=[-0.15, 1.15], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.1, sqrt3 / 2 + 0.1], showticklabels=False, showgrid=False, zeroline=False)

    fig.update_layout(
        height=height_per_row * n_rows,
        width=750,
        title_text="Belief geometry: probe projections vs. optimal<br>"
                   "<sup>Color = ground-truth belief state (RGB = S0, S1, S2)</sup>",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    output_path = Path(output_path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
