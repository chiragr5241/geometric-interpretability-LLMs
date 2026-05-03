#!/usr/bin/env python3
"""Regenerate train_single_seq_encoder_decoder plots from saved JSON artifacts.

This is a one-off recovery script for runs where Plotly/Kaleido could only
write HTML during the original experiment. It reads plot_data/*.json plus the
per-layer probe metadata JSONs, then calls the same plotting helpers used by
train_single_seq_encoder_decoder.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encoder_decoder_utils import (  # noqa: E402
    decoder_loss_curves,
    encoder_mse_curves,
    layer_line_plot,
    simplex_scatter,
)
from plot_titles import format_hmm_process, with_hmm_subtitle  # noqa: E402


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _seq_indices(plot_data_dir: Path) -> list[int]:
    seqs: list[int] = []
    for path in plot_data_dir.glob("seq_*_layer_metrics.json"):
        name = path.name.removeprefix("seq_").removesuffix("_layer_metrics.json")
        if name.isdigit():
            seqs.append(int(name))
    return sorted(seqs)


def _hmm_metadata(out_dir: Path) -> tuple[str, dict[str, Any], str]:
    with open(out_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    hmm = config["hmm"]
    process_name = str(hmm["process_name"])
    process_params = dict(hmm.get("process_params") or {})
    return process_name, process_params, format_hmm_process(process_name, process_params)


def _plot_sequence(
    out_dir: Path,
    seq_i: int,
    process_name: str,
    process_params: dict[str, Any],
    hmm_subtitle: str,
) -> int:
    plot_data_dir = out_dir / "plot_data"
    seq_dir = out_dir / f"seq_{seq_i}"
    fig_dir = seq_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    layer_payload = _read_json(plot_data_dir / f"seq_{seq_i}_layer_metrics.json")
    layer_indices = [int(layer) for layer in layer_payload["layer_indices"]]
    metrics = layer_payload["metrics"]

    decoder_payload = _read_json(plot_data_dir / f"seq_{seq_i}_decoder_loss_curves.json")
    decoder_results = {
        int(layer): SimpleNamespace(
            train_loss_curve=curves["train_loss_curve"],
            eval_loss_curve=curves["eval_loss_curve"],
        )
        for layer, curves in decoder_payload["layers"].items()
    }

    probe_results = {}
    for layer in layer_indices:
        meta_path = seq_dir / f"layer_{layer}" / "probe_metadata.json"
        probe_results[layer] = SimpleNamespace(
            train_mse_curve=_read_json(meta_path)["train_mse_curve"]
        )

    for simplex_path in sorted(plot_data_dir.glob(f"seq_{seq_i}_simplex_layer_*.json")):
        payload = _read_json(simplex_path)
        layer = int(payload["layer"])
        simplex_scatter(
            np.asarray(payload["predicted_beliefs"], dtype=np.float32),
            np.asarray(payload["gt_beliefs"], dtype=np.float32),
            layer,
            fig_dir / f"simplex_layer_{layer}",
            title=with_hmm_subtitle(
                f"Encoder predictions — Layer {layer} (eval, N={len(payload['predicted_beliefs'])})",
                process_name,
                process_params,
            ),
        )

    decoder_loss_curves(
        decoder_results,
        layer_indices,
        fig_dir / "decoder_loss_curves",
        subtitle=hmm_subtitle,
    )
    encoder_mse_curves(
        probe_results,
        layer_indices,
        fig_dir / "encoder_mse_curves",
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        train_vals=[metrics[str(layer)]["train_mse"] for layer in layer_indices],
        eval_vals=[metrics[str(layer)]["eval_enc_mse"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="MSE",
        title=f"Encoder MSE by layer — Seq {seq_i}",
        path=fig_dir / "encoder_mse_per_layer",
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_enc_r2"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="R²",
        title=f"Encoder R² by layer — Seq {seq_i}",
        path=fig_dir / "encoder_r2_per_layer",
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_dec_loss"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="MSE",
        title=f"Decoder reconstruction loss by layer — Seq {seq_i}",
        path=fig_dir / "decoder_recon_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_dec_norm_loss"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="MSE / mean_norm²",
        title=f"Normalised decoder reconstruction loss by layer — Seq {seq_i}",
        path=fig_dir / "decoder_recon_norm_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_roundtrip_loss"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="Round-trip MSE",
        title=f"Round-trip loss by layer — Seq {seq_i}",
        path=fig_dir / "roundtrip_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    return 12


def _plot_aggregate(out_dir: Path, hmm_subtitle: str) -> int:
    payload = _read_json(out_dir / "plot_data" / "aggregate_layer_metrics.json")
    layer_indices = [int(layer) for layer in payload["layer_indices"]]
    metrics = payload["metrics"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_enc_r2"]["mean"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="R² (mean ± std)",
        title="Encoder R² by layer — aggregated across sequences",
        path=fig_dir / "encoder_r2_per_layer",
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_dec_loss"]["mean"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="Reconstruction MSE",
        title="Decoder reconstruction loss by layer — aggregated",
        path=fig_dir / "decoder_recon_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_dec_norm_loss"]["mean"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="MSE / mean_norm² (mean across sequences)",
        title="Normalised decoder reconstruction loss by layer — aggregated",
        path=fig_dir / "decoder_recon_norm_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    layer_line_plot(
        eval_vals=[metrics[str(layer)]["eval_roundtrip_loss"]["mean"] for layer in layer_indices],
        layer_indices=layer_indices,
        y_title="Round-trip MSE",
        title="Round-trip loss by layer — aggregated",
        path=fig_dir / "roundtrip_loss_per_layer",
        log_y=True,
        subtitle=hmm_subtitle,
    )
    return 4


def regenerate(out_dir: Path) -> int:
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")
    plot_data_dir = out_dir / "plot_data"
    if not plot_data_dir.exists():
        raise FileNotFoundError(f"Missing plot_data directory: {plot_data_dir}")

    process_name, process_params, hmm_subtitle = _hmm_metadata(out_dir)
    n_plots = 0
    for seq_i in _seq_indices(plot_data_dir):
        n_plots += _plot_sequence(out_dir, seq_i, process_name, process_params, hmm_subtitle)

    aggregate_path = plot_data_dir / "aggregate_layer_metrics.json"
    if aggregate_path.exists():
        n_plots += _plot_aggregate(out_dir, hmm_subtitle)

    return n_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate single-sequence encoder-decoder figures from JSON artifacts."
    )
    parser.add_argument("output_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for out_dir in args.output_dirs:
        n_plots = regenerate(out_dir.resolve())
        print(f"{out_dir}: regenerated {n_plots} plots")


if __name__ == "__main__":
    main()
