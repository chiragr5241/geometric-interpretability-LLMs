#!/usr/bin/env python3
"""
Probe subspace similarity — cross-layer analysis (Step 2 of SPAR-28).

Computes 28×28 layer-vs-layer subspace similarity matrices using two data sources:

  --pooled-dir   Directory from train_encoder_decoder. Loads one probe per layer
                 from probes/pooled/layer_X/, computes a single 28×28 matrix.

  --per-seq-dir  Directory from probes_cross_sequence_alignment with save_probes=True.
                 For each sequence, gathers 28 probes (one per layer) and computes
                 a 28×28 matrix. Averages across all sequences → mean ± std heatmaps.

Both analyses use pairwise_cosine_sim_matrix and pairwise_principal_angles from
src/metrics/probe_metrics.py. No model loading; runs in seconds.

Usage:
    python experiments/probe_subspace_similarity.py \\
        --pooled-dir outputs/dani/20260330_161347_train_encoder_decoder \\
        --per-seq-dir outputs/dani/20260402_222933_probes_cross_sequence_alignment
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_utils import setup_logging
from metrics.probe_metrics import pairwise_cosine_sim_matrix, pairwise_principal_angles
from probes import ProbeResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_output_dir(name: str, output_user: str = "dani") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "outputs" / output_user / f"{timestamp}_{name}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    return out_dir


def _heatmap(
    mat: np.ndarray,
    title: str,
    path: Path,
    colorscale: str,
    zrange: tuple[float, float] | None = None,
    axis_labels: list[str] | None = None,
) -> None:
    kwargs: dict = dict(colorscale=colorscale, showscale=True)
    if zrange is not None:
        kwargs["zmin"], kwargs["zmax"] = zrange
    if axis_labels is not None:
        kwargs["x"] = axis_labels
        kwargs["y"] = axis_labels
    fig = go.Figure(go.Heatmap(z=mat, **kwargs))
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Layer",
        width=700,
        height=650,
        margin=dict(t=80, b=60, l=60, r=20),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def _heatmap_with_std(
    mean_mat: np.ndarray,
    std_mat: np.ndarray,
    title: str,
    path: Path,
    colorscale: str,
    zrange: tuple[float, float] | None = None,
    axis_labels: list[str] | None = None,
) -> None:
    kwargs: dict = dict(colorscale=colorscale, showscale=True)
    if zrange is not None:
        kwargs["zmin"], kwargs["zmax"] = zrange
    if axis_labels is not None:
        kwargs["x"] = axis_labels
        kwargs["y"] = axis_labels

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Mean", "Std"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Heatmap(z=mean_mat, **kwargs), row=1, col=1)
    std_kwargs = dict(colorscale="Greys", showscale=True)
    if axis_labels is not None:
        std_kwargs["x"] = axis_labels
        std_kwargs["y"] = axis_labels
    fig.add_trace(go.Heatmap(z=std_mat, **std_kwargs), row=1, col=2)
    fig.update_layout(
        title=title,
        width=1200,
        height=600,
        margin=dict(t=80, b=60, l=60, r=20),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


# ── Pooled analysis ────────────────────────────────────────────────────────────

def run_pooled_analysis(
    pooled_dir: Path,
    out_dir: Path,
    logger,
) -> None:
    probe_root = pooled_dir / "probes" / "pooled"
    layer_dirs = sorted(probe_root.glob("layer_*"), key=lambda p: int(p.name.split("_")[1]))
    if not layer_dirs:
        logger.warning(f"No pooled probes found in {probe_root}; skipping pooled analysis.")
        return

    layer_indices = [int(p.name.split("_")[1]) for p in layer_dirs]
    logger.info(f"Pooled: loading {len(layer_indices)} probes from {probe_root}")

    probes: list[ProbeResult] = []
    for ld in layer_dirs:
        probes.append(ProbeResult.load(ld))

    logger.info("Pooled: computing cosine similarity matrix ...")
    cos_sim = pairwise_cosine_sim_matrix(probes)

    logger.info("Pooled: computing principal angles matrix ...")
    angles_deg, mean_angles, mean_angles_2 = pairwise_principal_angles(probes)

    axis_labels = [str(l) for l in layer_indices]
    fig_dir = out_dir / "figures"

    _heatmap(
        cos_sim,
        title="Layer-vs-layer cosine similarity (pooled probes)<br>"
              "<sup>Entry [i,j]: mean |diag| cosine sim between weight columns</sup>",
        path=fig_dir / "pooled_cosine_sim",
        colorscale="Blues",
        zrange=(0.0, 1.0),
        axis_labels=axis_labels,
    )
    _heatmap(
        mean_angles,
        title="Layer-vs-layer mean principal angle (pooled probes)<br>"
              "<sup>Entry [i,j]: mean of θ₁,θ₂,θ₃ in degrees</sup>",
        path=fig_dir / "pooled_principal_angles",
        colorscale="Viridis",
        axis_labels=axis_labels,
    )
    _heatmap(
        mean_angles_2,
        title="Layer-vs-layer mean principal angle θ₁,θ₂ (pooled probes)<br>"
              "<sup>Entry [i,j]: mean of first 2 principal angles in degrees</sup>",
        path=fig_dir / "pooled_principal_angles_2",
        colorscale="Viridis",
        axis_labels=axis_labels,
    )

    result = {
        "layer_indices": layer_indices,
        "cosine_sim": cos_sim.tolist(),
        "mean_principal_angles_deg": mean_angles.tolist(),
        "mean_principal_angles_2_deg": mean_angles_2.tolist(),
    }
    with open(out_dir / "pooled_results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Pooled: done. Figures saved to {fig_dir}")


# ── Per-sequence analysis ──────────────────────────────────────────────────────

def run_per_seq_analysis(
    per_seq_dir: Path,
    out_dir: Path,
    logger,
    variant: str = "full",
) -> None:
    probe_root = per_seq_dir / "probes"

    pattern = re.compile(r"^layer_(\d+)_seq_(\d+)_" + re.escape(variant) + r"$")
    matches = [
        (int(m.group(1)), int(m.group(2)), p)
        for p in probe_root.iterdir()
        if (m := pattern.match(p.name))
    ]
    if not matches:
        logger.warning(f"No per-sequence probes ({variant}) found in {probe_root}; skipping.")
        return

    layer_indices = sorted({layer for layer, _, _ in matches})
    seq_indices = sorted({seq for _, seq, _ in matches})
    logger.info(
        f"Per-seq ({variant}): found {len(layer_indices)} layers x {len(seq_indices)} sequences"
    )

    probe_map: dict[tuple[int, int], Path] = {
        (layer, seq): p for layer, seq, p in matches
    }

    cos_sim_mats: list[np.ndarray] = []
    angle_mats: list[np.ndarray] = []
    angle_2_mats: list[np.ndarray] = []

    for seq in seq_indices:
        logger.info(f"  Sequence {seq}: loading probes across {len(layer_indices)} layers ...")
        probes: list[ProbeResult] = []
        for layer in layer_indices:
            key = (layer, seq)
            if key not in probe_map:
                logger.warning(f"  Missing probe for layer={layer} seq={seq}; skipping sequence.")
                probes = []
                break
            probes.append(ProbeResult.load(probe_map[key]))

        if not probes:
            continue

        cos_sim = pairwise_cosine_sim_matrix(probes)
        _, mean_angles, mean_angles_2 = pairwise_principal_angles(probes)

        cos_sim_mats.append(cos_sim)
        angle_mats.append(mean_angles)
        angle_2_mats.append(mean_angles_2)

    if not cos_sim_mats:
        logger.warning("Per-seq: no complete sequences found; skipping.")
        return

    cos_sim_mean = np.mean(cos_sim_mats, axis=0)
    cos_sim_std = np.std(cos_sim_mats, axis=0)
    angle_mean = np.mean(angle_mats, axis=0)
    angle_std = np.std(angle_mats, axis=0)
    angle_2_mean = np.mean(angle_2_mats, axis=0)
    angle_2_std = np.std(angle_2_mats, axis=0)

    axis_labels = [str(l) for l in layer_indices]
    fig_dir = out_dir / "figures"

    _heatmap_with_std(
        cos_sim_mean,
        cos_sim_std,
        title=f"Layer-vs-layer cosine similarity — {variant} probes, per-sequence mean±std<br>"
              f"<sup>Entry [i,j]: mean |diag| cosine sim across {len(cos_sim_mats)} sequences</sup>",
        path=fig_dir / f"perseq_{variant}_cosine_sim",
        colorscale="Blues",
        zrange=(0.0, 1.0),
        axis_labels=axis_labels,
    )
    _heatmap_with_std(
        angle_mean,
        angle_std,
        title=f"Layer-vs-layer mean principal angle — {variant} probes, per-sequence mean±std<br>"
              f"<sup>Entry [i,j]: mean of θ₁,θ₂,θ₃ across {len(angle_mats)} sequences</sup>",
        path=fig_dir / f"perseq_{variant}_principal_angles",
        colorscale="Viridis",
        axis_labels=axis_labels,
    )
    _heatmap_with_std(
        angle_2_mean,
        angle_2_std,
        title=f"Layer-vs-layer mean principal angle θ₁,θ₂ — {variant} probes, per-sequence mean±std<br>"
              f"<sup>Entry [i,j]: mean of first 2 principal angles across {len(angle_2_mats)} sequences</sup>",
        path=fig_dir / f"perseq_{variant}_principal_angles_2",
        colorscale="Viridis",
        axis_labels=axis_labels,
    )

    result = {
        "layer_indices": layer_indices,
        "n_sequences": len(cos_sim_mats),
        "cosine_sim_mean": cos_sim_mean.tolist(),
        "cosine_sim_std": cos_sim_std.tolist(),
        "mean_principal_angles_deg_mean": angle_mean.tolist(),
        "mean_principal_angles_deg_std": angle_std.tolist(),
        "mean_principal_angles_2_deg_mean": angle_2_mean.tolist(),
        "mean_principal_angles_2_deg_std": angle_2_std.tolist(),
        "per_sequence_cosine_sim": [m.tolist() for m in cos_sim_mats],
        "per_sequence_principal_angles_deg": [m.tolist() for m in angle_mats],
    }
    with open(out_dir / f"perseq_{variant}_results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Per-seq ({variant}): done. Figures saved to {fig_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-layer probe subspace similarity")
    parser.add_argument(
        "--pooled-dir",
        type=str,
        default=None,
        help="Path to train_encoder_decoder output dir (contains probes/pooled/layer_X/)",
    )
    parser.add_argument(
        "--per-seq-dir",
        type=str,
        default=None,
        help="Path to probes_cross_sequence_alignment output dir (contains probes/layer_X_seq_Y_{full,post}/)",
    )
    parser.add_argument(
        "--output-user",
        type=str,
        default="dani",
        help="Subdirectory under outputs/ for results",
    )
    args = parser.parse_args()

    if not args.pooled_dir and not args.per_seq_dir:
        parser.error("At least one of --pooled-dir or --per-seq-dir must be provided.")

    out_dir = _setup_output_dir("probe_subspace_similarity", output_user=args.output_user)
    logger = setup_logging(out_dir, name="probe_subspace_similarity")
    logger.info(f"Output dir: {out_dir}")

    if args.pooled_dir:
        logger.info(f"=== Pooled analysis: {args.pooled_dir} ===")
        run_pooled_analysis(Path(args.pooled_dir), out_dir, logger)

    if args.per_seq_dir:
        for variant in ("full", "post"):
            logger.info(f"=== Per-sequence analysis ({variant}): {args.per_seq_dir} ===")
            run_per_seq_analysis(Path(args.per_seq_dir), out_dir, logger, variant=variant)

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
