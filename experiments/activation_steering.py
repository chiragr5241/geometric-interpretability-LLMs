#!/usr/bin/env python3
"""SPAR-17 - Belief steering via additive interventions.

Usage:
    python experiments/activation_steering.py experiments/configs/activation_steering.yaml
    python experiments/activation_steering.py experiments/configs/activation_steering.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decoder import Decoder, DecoderResult
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import (
    build_emission_matrix,
    get_device,
    load_model,
    resolve_hmm_token_ids,
    setup_logging,
)
from hmm.hmm import Mess3HMM
from probes import Probe, ProbeResult


DRY_RUN_LAYERS = [0, 2, 6, 10, 17, 25]
DRY_RUN_SEQ_LEN = 100
DRY_RUN_N_SEQ = 5
DRY_RUN_N_DONORS = 2
DRY_RUN_N_RANDOM = 2
DRY_RUN_K_VALUES = [1, 5, 10]

_PAST_COLORS = {
    1: ("#1f77b4", "rgba(31,119,180,0.12)"),
    5: ("#9467bd", "rgba(148,103,189,0.12)"),
    10: ("#8c564b", "rgba(140,86,75,0.12)"),
    25: ("#e377c2", "rgba(227,119,194,0.12)"),
    50: ("#17becf", "rgba(23,190,207,0.12)"),
    100: ("#bcbd22", "rgba(188,189,34,0.12)"),
}
_OTHER_COLORS: dict[str, tuple[str, str]] = {
    "garbage_valid": ("#2ca02c", "rgba(44,160,44,0.12)"),
    "garbage_random": ("#d62728", "rgba(214,39,40,0.12)"),
}
_PROPAGATION_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_PAST_VARIANT_LABELS = {
    "past_consistent_shared_prefix": "shared-prefix",
    "past_consistent": "different-prefix",
}
_PAST_VARIANT_DASH = {
    "past_consistent_shared_prefix": "solid",
    "past_consistent": "dash",
}
_PAST_VARIANT_ORDER = {
    "past_consistent_shared_prefix": 0,
    "past_consistent": 1,
}


@dataclass
class ActivationSteeringConfig(ExperimentConfig):
    encoder_decoder_dir: str
    layer_indices: list[int]
    seq_length: int
    n_sequences: int
    batch_size: int
    n_donors: int
    n_random_samples: int
    k_values: list[int]
    pooled_probes: bool
    vocab_mapping: dict[str, int]
    n_ctx_override: int | None = None
    propagation_max_pairs: int = 8
    propagation_downstream_steps: int = 2


@dataclass
class SteeringSpec:
    seq_idx: int
    condition: str
    k: int | None
    sub_idx: int
    donor_idx: int | None
    positions: list[int]
    source_beliefs: np.ndarray
    target_beliefs: np.ndarray
    final_target_belief: np.ndarray


def _hmm_step(belief: np.ndarray, token_idx: int, t_3d: np.ndarray) -> np.ndarray:
    out = t_3d[token_idx] @ belief
    return (out / (out.sum() + 1e-10)).astype(np.float32)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * np.log(np.clip(p, 1e-10, None) / np.clip(q, 1e-10, None))).sum(axis=-1)


def _past_series_key(condition: str, k: int) -> str:
    if condition == "past_consistent":
        return f"past_consistent_k{k}"
    if condition == "past_consistent_shared_prefix":
        return f"past_consistent_shared_prefix_k{k}"
    raise ValueError(f"Unsupported past-consistent condition: {condition}")


def _parse_past_series_key(series_key: str) -> tuple[str, int] | None:
    for prefix, condition in (
        ("past_consistent_shared_prefix_k", "past_consistent_shared_prefix"),
        ("past_consistent_k", "past_consistent"),
    ):
        if series_key.startswith(prefix):
            return condition, int(series_key.removeprefix(prefix))
    return None


def _past_series_keys(k_values: list[int]) -> list[str]:
    out: list[str] = []
    for k in k_values:
        out.append(_past_series_key("past_consistent_shared_prefix", k))
        out.append(_past_series_key("past_consistent", k))
    return out


def _hex_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _series_key(condition: str, k: int | None) -> str:
    if condition in _PAST_VARIANT_LABELS:
        if k is None:
            raise ValueError(f"{condition} series requires k")
        return _past_series_key(condition, k)
    return condition


def _series_label(series_key: str) -> str:
    parsed = _parse_past_series_key(series_key)
    if parsed is not None:
        condition, k = parsed
        return f"past-consistent {_PAST_VARIANT_LABELS[condition]} K={k}"
    return series_key.replace("_", "-")


def _series_sort_key(series_key: str) -> tuple[int, int, int]:
    parsed = _parse_past_series_key(series_key)
    if parsed is not None:
        condition, k = parsed
        return (0, k, _PAST_VARIANT_ORDER[condition])
    if series_key == "garbage_valid":
        return (1, 0, 0)
    if series_key == "garbage_random":
        return (2, 0, 0)
    return (3, 0, 0)


def _series_style(series_key: str) -> tuple[str, str, str]:
    parsed = _parse_past_series_key(series_key)
    if parsed is not None:
        condition, k = parsed
        line_color, _ = _PAST_COLORS.get(k, ("#1f77b4", "rgba(31,119,180,0.12)"))
        alpha = 0.14 if condition == "past_consistent_shared_prefix" else 0.07
        return line_color, _hex_rgba(line_color, alpha), _PAST_VARIANT_DASH[condition]

    line_color, fill_color = _OTHER_COLORS.get(series_key, ("#7f7f7f", "rgba(127,127,127,0.12)"))
    return line_color, fill_color, "solid"


def _compute_hybrid_beliefs(
    source_beliefs: np.ndarray,
    donor_tokens: np.ndarray,
    k: int,
    t_3d: np.ndarray,
) -> np.ndarray:
    seq_len = donor_tokens.shape[0]
    belief = source_beliefs[seq_len - k].astype(np.float32)
    out: list[np.ndarray] = []
    for token_idx in donor_tokens[seq_len - k:]:
        belief = _hmm_step(belief, int(token_idx), t_3d)
        out.append(belief)
    return np.stack(out, axis=0).astype(np.float32)


def _build_specs(
    seq_beliefs: list[np.ndarray],
    seq_tokens: list[np.ndarray],
    hmm: Mess3HMM,
    t_3d: np.ndarray,
    k_values: list[int],
    n_donors: int,
    n_random_samples: int,
    rng: np.random.Generator,
) -> list[SteeringSpec]:
    n_sequences = len(seq_beliefs)
    seq_len = seq_tokens[0].shape[0]
    specs: list[SteeringSpec] = []

    for seq_idx in range(n_sequences):
        others = [other for other in range(n_sequences) if other != seq_idx]
        donor_count = min(n_donors, len(others))
        donors = rng.choice(others, size=donor_count, replace=False).tolist()

        for k in k_values:
            positions = list(range(seq_len - k, seq_len))
            source_slice = seq_beliefs[seq_idx][seq_len - k + 1 : seq_len + 1].astype(np.float32)
            for sub_idx, donor_idx in enumerate(donors):
                target_slice = _compute_hybrid_beliefs(
                    seq_beliefs[seq_idx],
                    seq_tokens[donor_idx],
                    k,
                    t_3d,
                )
                specs.append(
                    SteeringSpec(
                        seq_idx=seq_idx,
                        condition="past_consistent",
                        k=k,
                        sub_idx=sub_idx,
                        donor_idx=int(donor_idx),
                        positions=positions,
                        source_beliefs=source_slice,
                        target_beliefs=target_slice,
                        final_target_belief=target_slice[-1],
                    )
                )

            shared_prefix_belief = seq_beliefs[seq_idx][seq_len - k].astype(np.float32)
            for sub_idx in range(donor_count):
                _, target_slice = hmm.sample_continuation(shared_prefix_belief, k, rng)
                specs.append(
                    SteeringSpec(
                        seq_idx=seq_idx,
                        condition="past_consistent_shared_prefix",
                        k=k,
                        sub_idx=sub_idx,
                        donor_idx=None,
                        positions=positions,
                        source_beliefs=source_slice,
                        target_beliefs=target_slice,
                        final_target_belief=target_slice[-1],
                    )
                )

        final_source = seq_beliefs[seq_idx][seq_len : seq_len + 1].astype(np.float32)
        for sub_idx, donor_idx in enumerate(donors):
            final_target = seq_beliefs[donor_idx][seq_len].astype(np.float32)
            specs.append(
                SteeringSpec(
                    seq_idx=seq_idx,
                    condition="garbage_valid",
                    k=None,
                    sub_idx=sub_idx,
                    donor_idx=int(donor_idx),
                    positions=[seq_len - 1],
                    source_beliefs=final_source,
                    target_beliefs=final_target[None, :],
                    final_target_belief=final_target,
                )
            )

        for sub_idx in range(n_random_samples):
            final_target = rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float32)
            specs.append(
                SteeringSpec(
                    seq_idx=seq_idx,
                    condition="garbage_random",
                    k=None,
                    sub_idx=sub_idx,
                    donor_idx=None,
                    positions=[seq_len - 1],
                    source_beliefs=final_source,
                    target_beliefs=final_target[None, :],
                    final_target_belief=final_target,
                )
            )

    return specs


def _agg_records(records: list[tuple[int, int, float]]) -> dict:
    per_seq: dict[int, list[float]] = {}
    for seq_idx, _sub_idx, value in records:
        per_seq.setdefault(seq_idx, []).append(value)
    seq_means = [float(np.mean(values)) for values in per_seq.values()]
    n = len(seq_means)
    return {
        "seq_means": seq_means,
        "mean": float(np.mean(seq_means)),
        "std": float(np.std(seq_means)),
        "stderr": float(np.std(seq_means) / max(np.sqrt(n), 1.0)),
        "n_seqs": n,
    }


def _aggregate(
    kl_to_target_data: dict[str, dict[int, list[tuple[int, int, float]]]],
    kl_to_factual_data: dict[str, dict[int, list[tuple[int, int, float]]]],
    unsteered_to_target_data: dict[str, list[tuple[int, int, float]]],
    baseline_kl: list[float],
) -> tuple[dict[str, dict[int, dict]], dict[str, dict[int, dict]], dict[str, dict], dict]:
    agg_to_target = {
        series_key: {
            layer: _agg_records(records)
            for layer, records in by_layer.items()
        }
        for series_key, by_layer in kl_to_target_data.items()
    }
    agg_to_factual = {
        series_key: {
            layer: _agg_records(records)
            for layer, records in by_layer.items()
        }
        for series_key, by_layer in kl_to_factual_data.items()
    }
    agg_unsteered_to_target = {
        series_key: _agg_records(records)
        for series_key, records in unsteered_to_target_data.items()
    }
    n = len(baseline_kl)
    baseline = {
        "seq_means": [float(value) for value in baseline_kl],
        "mean": float(np.mean(baseline_kl)),
        "std": float(np.std(baseline_kl)),
        "stderr": float(np.std(baseline_kl) / max(np.sqrt(n), 1.0)),
        "n_seqs": n,
    }
    return agg_to_target, agg_to_factual, agg_unsteered_to_target, baseline


def _compute_steering_flat(
    specs: list[SteeringSpec],
    decoder: Decoder,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    target_flat = torch.from_numpy(np.concatenate([spec.target_beliefs for spec in specs], axis=0)).float().to(device)
    source_flat = torch.from_numpy(np.concatenate([spec.source_beliefs for spec in specs], axis=0)).float().to(device)
    with torch.no_grad():
        return decoder(target_flat).to(dtype=model_dtype) - decoder(source_flat).to(dtype=model_dtype)


def _steering_norm_records(
    specs: list[SteeringSpec],
    steering_flat: torch.Tensor,
) -> list[tuple[str, int, int, float]]:
    norms = torch.linalg.vector_norm(steering_flat.float(), dim=-1).cpu().numpy()
    records: list[tuple[str, int, int, float]] = []
    offset = 0
    for spec in specs:
        n_positions = len(spec.positions)
        series_key = _series_key(spec.condition, spec.k)
        for value in norms[offset : offset + n_positions]:
            records.append((series_key, spec.seq_idx, spec.sub_idx, float(value)))
        offset += n_positions
    return records


def _run_batch(
    model,
    llm_tokens: list[torch.Tensor],
    specs: list[SteeringSpec],
    decoder: Decoder,
    layer: int,
    final_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
    steering_flat: torch.Tensor | None = None,
) -> np.ndarray:
    tokens_batch = torch.cat([llm_tokens[spec.seq_idx] for spec in specs], dim=0).to(device)
    seq_len = int(tokens_batch.shape[1])
    batch_size = len(specs)
    if steering_flat is None:
        steering_flat = _compute_steering_flat(specs, decoder, device, model_dtype)

    delta = torch.zeros(
        (batch_size, seq_len, steering_flat.shape[-1]),
        dtype=model_dtype,
        device=device,
    )
    offset = 0
    for batch_idx, spec in enumerate(specs):
        n_positions = len(spec.positions)
        position_tensor = torch.tensor(spec.positions, dtype=torch.long, device=device)
        delta[batch_idx, position_tensor, :] += steering_flat[offset : offset + n_positions]
        offset += n_positions

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        return value + delta

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    probs_all = F.softmax(logits[:, final_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    return (probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


def _sorted_series_keys(agg: dict[str, dict[int, dict]]) -> list[str]:
    return sorted(agg.keys(), key=_series_sort_key)


def _plot_main_result(
    agg: dict[str, dict[int, dict]],
    k_values: list[int],
    layer_indices: list[int],
    baseline: dict,
    path: Path,
) -> None:
    layers_str = [str(layer) for layer in layer_indices]
    subplot_titles = [f"K={k}" for k in k_values] + ["Controls"]
    n_panels = len(subplot_titles)
    n_cols = min(3, n_panels)
    n_rows = ceil(n_panels / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
    )

    def add_series(
        series_key: str,
        row: int,
        col: int,
        showlegend: bool,
        legend_name: str | None = None,
    ) -> None:
        line_color, fill_color, dash = _series_style(series_key)
        means = [agg[series_key][layer]["mean"] for layer in layer_indices]
        stderrs = [agg[series_key][layer]["stderr"] for layer in layer_indices]
        upper = [mean + stderr for mean, stderr in zip(means, stderrs)]
        lower = [mean - stderr for mean, stderr in zip(means, stderrs)]

        fig.add_trace(
            go.Scatter(
                x=layers_str + layers_str[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                mode="lines",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=layers_str,
                y=means,
                name=legend_name or _series_label(series_key),
                mode="lines+markers",
                line=dict(color=line_color, width=2, dash=dash),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    for panel_idx, k in enumerate(k_values):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        fig.add_trace(
            go.Scatter(
                x=layers_str,
                y=[baseline["mean"]] * len(layer_indices),
                name="Unsteered baseline",
                mode="lines",
                line=dict(color="black", dash="dash", width=1.5),
                showlegend=(panel_idx == 0),
            ),
            row=row,
            col=col,
        )
        add_series(
            _past_series_key("past_consistent_shared_prefix", k),
            row,
            col,
            showlegend=(panel_idx == 0),
            legend_name="Past-consistent shared prefix",
        )
        add_series(
            _past_series_key("past_consistent", k),
            row,
            col,
            showlegend=(panel_idx == 0),
            legend_name="Past-consistent different prefix",
        )
        fig.update_yaxes(type="log", row=row, col=col)

    controls_idx = len(k_values)
    controls_row = controls_idx // n_cols + 1
    controls_col = controls_idx % n_cols + 1
    fig.add_trace(
        go.Scatter(
            x=layers_str,
            y=[baseline["mean"]] * len(layer_indices),
            name="Unsteered baseline",
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=False,
        ),
        row=controls_row,
        col=controls_col,
    )
    for series_key in ("garbage_valid", "garbage_random"):
        add_series(series_key, controls_row, controls_col, showlegend=(controls_idx == 0))
    fig.update_yaxes(type="log", row=controls_row, col=controls_col)

    for panel_idx in range(n_panels):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        fig.update_xaxes(title_text="Layer", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="KL [nats]", row=row, col=col)

    fig.update_layout(
        title=(
            "Activation steering: KL vs layer"
            "<br><sup>Per-K shared vs different prefix, plus controls; mean +/- stderr</sup>"
        ),
        height=max(420, 360 * n_rows + 80),
        width=min(420 * n_cols + 140, 1450),
        margin=dict(t=85, b=60, l=75, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_k_sweep(
    agg: dict[str, dict[int, dict]],
    k_values: list[int],
    layer_indices: list[int],
    path: Path,
) -> None:
    n_layers = len(layer_indices)
    n_cols = min(4, n_layers)
    n_rows = ceil(n_layers / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {layer}" for layer in layer_indices],
    )
    x_vals = [str(k) for k in k_values]

    for idx, layer in enumerate(layer_indices):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_legend = idx == 0
        for condition, color in (
            ("past_consistent_shared_prefix", "#1f77b4"),
            ("past_consistent", "#ff7f0e"),
        ):
            means = [agg[_past_series_key(condition, k)][layer]["mean"] for k in k_values]
            errs = [agg[_past_series_key(condition, k)][layer]["stderr"] for k in k_values]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=means,
                    error_y=dict(type="data", array=errs, visible=True),
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=_PAST_VARIANT_DASH[condition]),
                    name=_PAST_VARIANT_LABELS[condition],
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(type="log", row=row, col=col)

    fig.update_layout(
        title=(
            "Past-consistent steering: KL vs K"
            "<br><sup>All layers; shared-prefix (solid) vs different-prefix (dashed)</sup>"
        ),
        height=max(420, 240 * n_rows + 100),
        width=min(320 * n_cols + 120, 1500),
        margin=dict(t=85, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_steering_norm_vs_layer(
    agg: dict[str, dict[int, dict]],
    k_values: list[int],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(layer) for layer in layer_indices]
    fig = go.Figure()

    for k in k_values:
        for condition in ("past_consistent_shared_prefix", "past_consistent"):
            series_key = _past_series_key(condition, k)
            line_color, fill_color, dash = _series_style(series_key)
            means = [agg[series_key][layer]["mean"] for layer in layer_indices]
            stderrs = [agg[series_key][layer]["stderr"] for layer in layer_indices]
            upper = [mean + stderr for mean, stderr in zip(means, stderrs)]
            lower = [max(mean - stderr, 0.0) for mean, stderr in zip(means, stderrs)]

            fig.add_trace(
                go.Scatter(
                    x=layers_str + layers_str[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=layers_str,
                    y=means,
                    name=_series_label(series_key),
                    mode="lines+markers",
                    line=dict(color=line_color, width=2, dash=dash),
                )
            )

    fig.update_layout(
        title=(
            "Average steering-vector norm vs layer"
            "<br><sup>Shared-prefix (solid) vs different-prefix (dashed)</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Average vector norm",
        height=520,
        width=900,
        margin=dict(t=85, b=60, l=75, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_heatmaps(
    agg: dict[str, dict[int, dict]],
    k_values: list[int],
    layer_indices: list[int],
    fig_dir: Path,
) -> None:
    for k in k_values:
        series_keys = [
            _past_series_key("past_consistent_shared_prefix", k),
            _past_series_key("past_consistent", k),
        ]
        z_arrays: dict[str, np.ndarray] = {}
        n_seqs = 1
        for series_key in series_keys:
            n_seqs = max(n_seqs, agg[series_key][layer_indices[0]]["n_seqs"])
            z_by_layer: list[list[float]] = []
            for layer in layer_indices:
                row = agg[series_key][layer]["seq_means"]
                if len(row) < n_seqs:
                    row = row + [float("nan")] * (n_seqs - len(row))
                z_by_layer.append(row[:n_seqs])
            z_arrays[series_key] = np.log10(np.clip(np.array(z_by_layer, dtype=float).T, 1e-10, None))

        zmin = min(float(np.nanmin(z_arr)) for z_arr in z_arrays.values())
        zmax = max(float(np.nanmax(z_arr)) for z_arr in z_arrays.values())
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[_series_label(series_key) for series_key in series_keys],
        )
        for col, series_key in enumerate(series_keys, start=1):
            fig.add_trace(
                go.Heatmap(
                    z=z_arrays[series_key],
                    x=[str(layer) for layer in layer_indices],
                    y=[f"Seq {idx}" for idx in range(n_seqs)],
                    colorscale="Viridis",
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title="log10(KL)") if col == 2 else None,
                    showscale=(col == 2),
                ),
                row=1,
                col=col,
            )
        fig.update_layout(
            title=f"KL heatmap comparison - K={k}",
            xaxis_title="Layer",
            xaxis2_title="Layer",
            yaxis_title="Sequence",
            height=520,
            width=1200,
            margin=dict(t=70, b=60, l=70, r=40),
        )
        fig.write_image(str(fig_dir / f"heatmap_past_consistent_k{k}.png"))

    for series_key in _sorted_series_keys(agg):
        if _parse_past_series_key(series_key) is not None:
            continue
        n_seqs = max(agg[series_key][layer_indices[0]]["n_seqs"], 1)
        z_by_layer: list[list[float]] = []
        for layer in layer_indices:
            row = agg[series_key][layer]["seq_means"]
            if len(row) < n_seqs:
                row = row + [float("nan")] * (n_seqs - len(row))
            z_by_layer.append(row[:n_seqs])
        z_arr = np.log10(np.clip(np.array(z_by_layer, dtype=float).T, 1e-10, None))
        fig = go.Figure(
            go.Heatmap(
                z=z_arr,
                x=[str(layer) for layer in layer_indices],
                y=[f"Seq {idx}" for idx in range(n_seqs)],
                colorscale="Viridis",
                colorbar=dict(title="log10(KL)"),
            )
        )
        fig.update_layout(
            title=f"KL heatmap - {_series_label(series_key)}",
            xaxis_title="Layer",
            yaxis_title="Sequence",
            height=520,
            width=760,
            margin=dict(t=70, b=60, l=70, r=40),
        )
        fig.write_image(str(fig_dir / f"heatmap_{series_key}.png"))


def _plot_past_variant_overlay(
    agg: dict[str, dict[int, dict]],
    k_values: list[int],
    layer_indices: list[int],
    path: Path,
) -> None:
    n_cols = min(2, len(k_values))
    n_rows = ceil(len(k_values) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"K={k}" for k in k_values],
    )
    layers_str = [str(layer) for layer in layer_indices]

    for idx, k in enumerate(k_values):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_legend = idx == 0
        for condition in ("past_consistent_shared_prefix", "past_consistent"):
            series_key = _past_series_key(condition, k)
            line_color, fill_color, dash = _series_style(series_key)
            means = [agg[series_key][layer]["mean"] for layer in layer_indices]
            stderrs = [agg[series_key][layer]["stderr"] for layer in layer_indices]
            upper = [mean + stderr for mean, stderr in zip(means, stderrs)]
            lower = [mean - stderr for mean, stderr in zip(means, stderrs)]
            fig.add_trace(
                go.Scatter(
                    x=layers_str + layers_str[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    mode="lines",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=layers_str,
                    y=means,
                    mode="lines+markers",
                    line=dict(color=line_color, width=2, dash=dash),
                    name=_PAST_VARIANT_LABELS[condition],
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(type="log", row=row, col=col)

    fig.update_layout(
        title=(
            "Shared-prefix vs different-prefix steering"
            "<br><sup>KL(P_steered || P_opt(eta_target)) by layer for each K</sup>"
        ),
        height=max(420, 320 * n_rows + 80),
        width=1100,
        margin=dict(t=85, b=60, l=75, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_to_factual(
    agg_to_factual: dict[str, dict[int, dict]],
    baseline: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(layer) for layer in layer_indices]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=layers_str,
            y=[baseline["mean"]] * len(layer_indices),
            name="Unsteered baseline",
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
        )
    )

    for series_key in _sorted_series_keys(agg_to_factual):
        line_color, _, dash = _series_style(series_key)
        means = [agg_to_factual[series_key][layer]["mean"] for layer in layer_indices]
        errs = [agg_to_factual[series_key][layer]["stderr"] for layer in layer_indices]
        fig.add_trace(
            go.Scatter(
                x=layers_str,
                y=means,
                error_y=dict(type="data", array=errs, visible=True),
                name=_series_label(series_key),
                mode="lines+markers",
                line=dict(color=line_color, width=2, dash=dash),
            )
        )

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=(
            "KL to factual target after steering"
            "<br><sup>KL(P_steered || P_opt(eta_factual))</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=460,
        width=900,
        margin=dict(t=85, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_crossing(
    series_key: str,
    agg_to_target: dict[str, dict[int, dict]],
    agg_to_factual: dict[str, dict[int, dict]],
    agg_unsteered_to_target: dict[str, dict],
    baseline: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    n_layers = len(layer_indices)
    n_cols = min(6, n_layers)
    n_rows = ceil(n_layers / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {layer}" for layer in layer_indices],
        shared_yaxes=False,
        vertical_spacing=0.15 if n_rows > 1 else 0.1,
        horizontal_spacing=0.08,
    )

    uf_mean = baseline["mean"]
    uf_err = baseline["stderr"]
    ut_mean = agg_unsteered_to_target[series_key]["mean"]
    ut_err = agg_unsteered_to_target[series_key]["stderr"]

    for idx, layer in enumerate(layer_indices):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_legend = idx == 0
        pf_mean = agg_to_factual[series_key][layer]["mean"]
        pf_err = agg_to_factual[series_key][layer]["stderr"]
        pt_mean = agg_to_target[series_key][layer]["mean"]
        pt_err = agg_to_target[series_key][layer]["stderr"]

        fig.add_trace(
            go.Scatter(
                x=["Unsteered", "Steered"],
                y=[uf_mean, pf_mean],
                error_y=dict(type="data", array=[uf_err, pf_err], visible=True),
                name="KL to factual opt.",
                showlegend=show_legend,
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=["Unsteered", "Steered"],
                y=[ut_mean, pt_mean],
                error_y=dict(type="data", array=[ut_err, pt_err], visible=True),
                name="KL to target opt.",
                showlegend=show_legend,
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=6),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            f"Crossing plot: {_series_label(series_key)}"
            "<br><sup>Blue = KL to factual optimal | Orange = KL to target optimal</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 220, 1400),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_crossing_pair(
    k: int,
    agg_to_target: dict[str, dict[int, dict]],
    agg_to_factual: dict[str, dict[int, dict]],
    agg_unsteered_to_target: dict[str, dict],
    baseline: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    n_layers = len(layer_indices)
    n_cols = min(6, n_layers)
    n_rows = ceil(n_layers / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {layer}" for layer in layer_indices],
        shared_yaxes=False,
        vertical_spacing=0.15 if n_rows > 1 else 0.1,
        horizontal_spacing=0.08,
    )

    uf_mean = baseline["mean"]
    uf_err = baseline["stderr"]

    for idx, layer in enumerate(layer_indices):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_legend = idx == 0
        for metric_name, color in (("factual", "#1f77b4"), ("target", "#ff7f0e")):
            for condition in ("past_consistent_shared_prefix", "past_consistent"):
                series_key = _past_series_key(condition, k)
                ut_mean = agg_unsteered_to_target[series_key]["mean"]
                ut_err = agg_unsteered_to_target[series_key]["stderr"]
                steered_mean = (
                    agg_to_factual[series_key][layer]["mean"]
                    if metric_name == "factual"
                    else agg_to_target[series_key][layer]["mean"]
                )
                steered_err = (
                    agg_to_factual[series_key][layer]["stderr"]
                    if metric_name == "factual"
                    else agg_to_target[series_key][layer]["stderr"]
                )
                fig.add_trace(
                    go.Scatter(
                        x=["Unsteered", "Steered"],
                        y=[uf_mean, steered_mean] if metric_name == "factual" else [ut_mean, steered_mean],
                        error_y=dict(
                            type="data",
                            array=[uf_err, steered_err] if metric_name == "factual" else [ut_err, steered_err],
                            visible=True,
                        ),
                        name=f"{metric_name} | {_PAST_VARIANT_LABELS[condition]}",
                        showlegend=show_legend,
                        mode="lines+markers",
                        line=dict(color=color, width=2, dash=_PAST_VARIANT_DASH[condition]),
                        marker=dict(size=6),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=(
            f"Crossing plot: K={k}"
            "<br><sup>Blue = KL to factual optimal | Orange = KL to target optimal</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 260, 1450),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_causal_shift(
    series_key: str,
    agg_to_target: dict[str, dict[int, dict]],
    agg_to_factual: dict[str, dict[int, dict]],
    agg_unsteered_to_target: dict[str, dict],
    baseline: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(layer) for layer in layer_indices]
    uf_mean = baseline["mean"]
    uf_err = baseline["stderr"]
    ut_mean = agg_unsteered_to_target[series_key]["mean"]
    ut_err = agg_unsteered_to_target[series_key]["stderr"]

    delta_target = [ut_mean - agg_to_target[series_key][layer]["mean"] for layer in layer_indices]
    delta_factual = [agg_to_factual[series_key][layer]["mean"] - uf_mean for layer in layer_indices]
    err_target = [
        float(np.sqrt(ut_err**2 + agg_to_target[series_key][layer]["stderr"]**2))
        for layer in layer_indices
    ]
    err_factual = [
        float(np.sqrt(uf_err**2 + agg_to_factual[series_key][layer]["stderr"]**2))
        for layer in layer_indices
    ]

    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=layers_str,
            y=delta_factual,
            error_y=dict(type="data", array=err_factual, visible=True),
            name="Away from factual",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers_str,
            y=delta_target,
            error_y=dict(type="data", array=err_target, visible=True),
            name="Toward target",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6, symbol="square"),
        )
    )
    fig.update_layout(
        title=(
            f"Causal shift per layer: {_series_label(series_key)}"
            "<br><sup>Positive means steering moved output away from factual and toward target</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Delta KL [nats]",
        height=440,
        width=760,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_causal_shift_pair(
    k: int,
    agg_to_target: dict[str, dict[int, dict]],
    agg_to_factual: dict[str, dict[int, dict]],
    agg_unsteered_to_target: dict[str, dict],
    baseline: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(layer) for layer in layer_indices]
    uf_mean = baseline["mean"]
    uf_err = baseline["stderr"]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for metric_name, color, marker_symbol in (
        ("factual", "#1f77b4", "circle"),
        ("target", "#ff7f0e", "square"),
    ):
        for condition in ("past_consistent_shared_prefix", "past_consistent"):
            series_key = _past_series_key(condition, k)
            ut_mean = agg_unsteered_to_target[series_key]["mean"]
            ut_err = agg_unsteered_to_target[series_key]["stderr"]
            values = (
                [agg_to_factual[series_key][layer]["mean"] - uf_mean for layer in layer_indices]
                if metric_name == "factual"
                else [ut_mean - agg_to_target[series_key][layer]["mean"] for layer in layer_indices]
            )
            errs = (
                [
                    float(np.sqrt(uf_err**2 + agg_to_factual[series_key][layer]["stderr"]**2))
                    for layer in layer_indices
                ]
                if metric_name == "factual"
                else [
                    float(np.sqrt(ut_err**2 + agg_to_target[series_key][layer]["stderr"]**2))
                    for layer in layer_indices
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=layers_str,
                    y=values,
                    error_y=dict(type="data", array=errs, visible=True),
                    name=f"{metric_name} | {_PAST_VARIANT_LABELS[condition]}",
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=_PAST_VARIANT_DASH[condition]),
                    marker=dict(size=6, symbol=marker_symbol),
                )
            )

    fig.update_layout(
        title=(
            f"Causal shift per layer: K={k}"
            "<br><sup>Positive means steering moved output away from factual and toward target</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Delta KL [nats]",
        height=440,
        width=960,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _to_barycentric(beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3.0) / 2.0) * beliefs[:, 2]
    return x, y


def _simplex_outline(fig: go.Figure, row: int, col: int) -> None:
    sqrt3 = np.sqrt(3.0)
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0, 0.5, 0.0],
            y=[0.0, 0.0, sqrt3 / 2.0, 0.0],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _run_cached_batch(
    model,
    llm_tokens: list[torch.Tensor],
    specs: list[SteeringSpec],
    decoder: Decoder,
    layer: int,
    final_pos: int,
    cache_layers: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[int, np.ndarray]:
    tokens_batch = torch.cat([llm_tokens[spec.seq_idx] for spec in specs], dim=0).to(device)
    seq_len = int(tokens_batch.shape[1])
    batch_size = len(specs)
    steering_flat = _compute_steering_flat(specs, decoder, device, model_dtype)

    delta = torch.zeros(
        (batch_size, seq_len, steering_flat.shape[-1]),
        dtype=model_dtype,
        device=device,
    )
    offset = 0
    for batch_idx, spec in enumerate(specs):
        n_positions = len(spec.positions)
        position_tensor = torch.tensor(spec.positions, dtype=torch.long, device=device)
        delta[batch_idx, position_tensor, :] += steering_flat[offset : offset + n_positions]
        offset += n_positions

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        return value + delta

    hook_names = [f"blocks.{cache_layer}.hook_resid_post" for cache_layer in cache_layers]
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
            _, cache = model.run_with_cache(
                tokens_batch,
                names_filter=hook_names,
                return_type=None,
            )
    out = {
        cache_layer: cache[f"blocks.{cache_layer}.hook_resid_post"][:, final_pos, :].float().cpu().numpy()
        for cache_layer in cache_layers
    }
    del cache
    return out


def _run_clean_cached_batch(
    model,
    llm_tokens: list[torch.Tensor],
    seq_indices: list[int],
    final_pos: int,
    cache_layers: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    tokens_batch = torch.cat([llm_tokens[seq_idx] for seq_idx in seq_indices], dim=0).to(device)
    hook_names = [f"blocks.{cache_layer}.hook_resid_post" for cache_layer in cache_layers]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens_batch,
            names_filter=hook_names,
            return_type=None,
        )
    out = {
        cache_layer: cache[f"blocks.{cache_layer}.hook_resid_post"][:, final_pos, :].float().cpu().numpy()
        for cache_layer in cache_layers
    }
    del cache
    return out


def _encode_beliefs(probe: Probe, acts: np.ndarray) -> np.ndarray:
    device = next(probe.parameters()).device
    with torch.no_grad():
        beliefs = probe(torch.from_numpy(acts).float().to(device)).cpu().numpy()
    beliefs = np.clip(beliefs, 0.0, None)
    return beliefs / (beliefs.sum(axis=-1, keepdims=True) + 1e-10)


def _collect_propagation_data(
    model,
    llm_tokens: list[torch.Tensor],
    specs_k1: list[SteeringSpec],
    layer_indices: list[int],
    decoders: dict[int, Decoder],
    encoders: dict[int, Probe],
    final_pos: int,
    config: ActivationSteeringConfig,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[int, dict[str, object]]:
    selected_specs = specs_k1[: config.propagation_max_pairs]
    if not selected_specs:
        return {}

    layer_to_data: dict[int, dict[str, object]] = {}
    for layer_idx, layer in enumerate(layer_indices):
        readout_layers = layer_indices[layer_idx : layer_idx + config.propagation_downstream_steps + 1]
        clean_cache = _run_clean_cached_batch(
            model,
            llm_tokens,
            [spec.seq_idx for spec in selected_specs],
            final_pos,
            readout_layers,
            device,
        )
        steered_cache = _run_cached_batch(
            model,
            llm_tokens,
            selected_specs,
            decoders[layer],
            layer,
            final_pos,
            readout_layers,
            device,
            model_dtype,
        )
        factual = {
            readout_layer: _encode_beliefs(encoders[readout_layer], clean_cache[readout_layer])
            for readout_layer in readout_layers
        }
        steered = {
            readout_layer: _encode_beliefs(encoders[readout_layer], steered_cache[readout_layer])
            for readout_layer in readout_layers
        }
        target = np.stack([spec.final_target_belief for spec in selected_specs], axis=0)
        layer_to_data[layer] = {
            "readout_layers": readout_layers,
            "factual": factual,
            "steered": steered,
            "target": target,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return layer_to_data


def _plot_propagation(
    propagation_data: dict[int, dict[str, object]],
    layer_indices: list[int],
    path: Path,
) -> None:
    selected_layers = [layer for layer in layer_indices if layer in propagation_data]
    if not selected_layers:
        return

    n_layers = len(selected_layers)
    n_cols = min(4, n_layers)
    n_rows = ceil(n_layers / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Intervene L{layer}" for layer in selected_layers],
    )
    sqrt3 = np.sqrt(3.0)

    for idx, layer in enumerate(selected_layers):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        layer_data = propagation_data[layer]
        readout_layers = layer_data["readout_layers"]
        factual = layer_data["factual"]
        steered = layer_data["steered"]
        target = layer_data["target"]
        _simplex_outline(fig, row, col)

        tx, ty = _to_barycentric(target)
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="markers",
                marker=dict(color="black", symbol="x", size=7),
                name="target",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        for readout_idx, readout_layer in enumerate(readout_layers):
            factual_xy = _to_barycentric(factual[readout_layer])
            steered_xy = _to_barycentric(steered[readout_layer])
            line_x: list[float | None] = []
            line_y: list[float | None] = []
            for start_x, start_y, end_x, end_y in zip(
                factual_xy[0],
                factual_xy[1],
                steered_xy[0],
                steered_xy[1],
            ):
                line_x.extend([start_x, end_x, None])
                line_y.extend([start_y, end_y, None])
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color=_PROPAGATION_COLORS[readout_idx % len(_PROPAGATION_COLORS)], width=1.5),
                    name=f"readout L{readout_layer}",
                    showlegend=(idx == 0),
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=steered_xy[0],
                    y=steered_xy[1],
                    mode="markers",
                    marker=dict(
                        color=_PROPAGATION_COLORS[readout_idx % len(_PROPAGATION_COLORS)],
                        size=5,
                    ),
                    name=f"steered L{readout_layer}",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(range=[-0.1, 1.1], showticklabels=False, row=row, col=col)
        fig.update_yaxes(
            range=[-0.08, sqrt3 / 2.0 + 0.08],
            showticklabels=False,
            scaleanchor=f"x{idx + 1}" if idx > 0 else "x",
            scaleratio=1,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            "Belief propagation after K=1 past-consistent steering"
            "<br><sup>Arrows go from encoded factual beliefs to encoded steered beliefs; x marks target</sup>"
        ),
        height=max(360, 300 * n_rows + 80),
        width=min(320 * n_cols + 120, 1400),
        margin=dict(t=85, b=40, l=40, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation steering experiment")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-user",
        type=str,
        default=None,
        help="Override output_user from the config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            f"Quick test: layers {DRY_RUN_LAYERS}, "
            f"seq_length={DRY_RUN_SEQ_LEN}, n_sequences={DRY_RUN_N_SEQ}"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config, ActivationSteeringConfig)
    apply_runtime_overrides(config, output_user=args.output_user)
    if not config.pooled_probes:
        raise ValueError("Activation steering currently expects pooled probes/decoders")

    if args.dry_run:
        config.layer_indices = [layer for layer in DRY_RUN_LAYERS if layer in config.layer_indices]
        config.seq_length = DRY_RUN_SEQ_LEN
        config.n_sequences = DRY_RUN_N_SEQ
        config.n_donors = min(config.n_donors, DRY_RUN_N_DONORS)
        config.n_random_samples = min(config.n_random_samples, DRY_RUN_N_RANDOM)
        config.k_values = [k for k in config.k_values if k in DRY_RUN_K_VALUES and k < config.seq_length]
        config.experiment_name = f"{config.experiment_name}_dry_run"

    if not config.k_values:
        raise ValueError("Config must include at least one valid k value")
    if any(k <= 0 or k > config.seq_length for k in config.k_values):
        raise ValueError("All k_values must satisfy 1 <= k <= seq_length")

    device = get_device()
    rng = np.random.default_rng(seed=42)

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="act_steer")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir      : {out_dir}")
    logger.info(f"Device          : {device}")
    logger.info(f"Dry run         : {args.dry_run}")
    logger.info(f"N sequences     : {config.n_sequences}")
    logger.info(f"Seq length      : {config.seq_length}")
    logger.info(f"Batch size      : {config.batch_size}")
    logger.info(f"K values        : {config.k_values}")
    logger.info(f"N donors        : {config.n_donors}")
    logger.info(f"N random        : {config.n_random_samples}")
    logger.info(f"Layers          : {config.layer_indices}")
    logger.info(f"Encoder-dec dir : {config.encoder_decoder_dir}")

    enc_dec_dir = Path(config.encoder_decoder_dir)
    n_sequences = config.n_sequences
    seq_len = config.seq_length
    final_pos = seq_len - 1
    n_vocab = len(config.vocab_mapping)
    idx_to_token = {value: key for key, value in config.vocab_mapping.items()}

    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype

    hmm = Mess3HMM()
    hmm_params = config.hmm.process_params
    hmm.create_hmm(hmm_params["x"], hmm_params["alpha"])
    t_3d = hmm.T_3d_matrix.cpu().numpy()
    emit = build_emission_matrix(hmm)
    logger.info(f"Mess3 HMM: x={hmm_params['x']}, alpha={hmm_params['alpha']}")

    _, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_vocab, logger)

    decoder_base = enc_dec_dir / "decoders" / "pooled"
    probe_base = enc_dec_dir / "probes" / "pooled"
    decoders: dict[int, Decoder] = {}
    encoders: dict[int, Probe] = {}
    for layer in config.layer_indices:
        decoder_result = DecoderResult.load(decoder_base / f"layer_{layer}")
        probe_result = ProbeResult.load(probe_base / f"layer_{layer}")
        decoders[layer] = decoder_result.decoder.to(device)
        decoders[layer].eval()
        encoders[layer] = probe_result.probe.to(device)
        encoders[layer].eval()
    logger.info(f"Loaded {len(decoders)} decoders and encoders")

    logger.info("Phase 1: generating sequences and clean forward passes ...")
    seq_beliefs: list[np.ndarray] = []
    seq_tokens: list[np.ndarray] = []
    llm_tokens: list[torch.Tensor] = []
    p_unsteered: list[np.ndarray] = []
    p_opt_factual: list[np.ndarray] = []
    baseline_kl: list[float] = []

    for batch_start in range(0, n_sequences, config.batch_size):
        batch_n = min(config.batch_size, n_sequences - batch_start)
        token_batch, _, _ = hmm.generate_dataset(batch_n, seq_len, return_states=True)
        belief_batch = hmm.compute_belief_state(token_batch)

        llm_batch_list: list[torch.Tensor] = []
        for batch_idx in range(batch_n):
            tokens = token_batch[batch_idx].cpu().numpy()
            beliefs = belief_batch[batch_idx].cpu().numpy()
            seq_tokens.append(tokens)
            seq_beliefs.append(beliefs)
            p_opt_factual.append((beliefs[seq_len] @ emit.T).astype(np.float32))

            text = " ".join(idx_to_token[int(token)] for token in tokens)
            llm_token = model.to_tokens(text, prepend_bos=False, truncate=False)
            if llm_token.shape[1] != seq_len:
                raise ValueError(f"Expected {seq_len} tokens, got {llm_token.shape[1]}")
            llm_tokens.append(llm_token.cpu())
            llm_batch_list.append(llm_token)

        llm_batch = torch.cat(llm_batch_list, dim=0).to(device)
        with torch.no_grad():
            logits_clean = model.run_with_hooks(llm_batch, fwd_hooks=[], return_type="logits")

        probs_all = F.softmax(logits_clean[:, final_pos, :].float(), dim=-1)
        probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
        p_clean = probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)

        for batch_idx in range(batch_n):
            seq_idx = batch_start + batch_idx
            p_unsteered.append(p_clean[batch_idx].astype(np.float32))
            baseline_kl.append(float(_kl(p_clean[batch_idx : batch_idx + 1], p_opt_factual[seq_idx][None])[0]))

        del logits_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info(f"  Sequences {batch_start + 1}-{batch_start + batch_n}/{n_sequences} done")

    logger.info(
        f"Unsteered KL at pos {final_pos}: mean={np.mean(baseline_kl):.4f} std={np.std(baseline_kl):.4f}"
    )

    logger.info("Building steering specs ...")
    all_specs = _build_specs(
        seq_beliefs,
        seq_tokens,
        hmm,
        t_3d,
        config.k_values,
        config.n_donors,
        config.n_random_samples,
        rng,
    )
    per_sequence = len(all_specs) // max(n_sequences, 1)
    logger.info(f"  {len(all_specs)} total specs ({per_sequence} per sequence)")

    past_series_keys = _past_series_keys(config.k_values)
    series_keys = past_series_keys + [
        "garbage_valid",
        "garbage_random",
    ]
    kl_to_target_data: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        series_key: {layer: [] for layer in config.layer_indices}
        for series_key in series_keys
    }
    kl_to_factual_data: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        series_key: {layer: [] for layer in config.layer_indices}
        for series_key in series_keys
    }
    steering_norm_data: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        series_key: {layer: [] for layer in config.layer_indices}
        for series_key in past_series_keys
    }
    unsteered_to_target_data: dict[str, list[tuple[int, int, float]]] = {series_key: [] for series_key in series_keys}
    raw_records: list[dict[str, object]] = []

    logger.info("Pre-computing unsteered KL to steering targets ...")
    for spec in all_specs:
        series_key = _series_key(spec.condition, spec.k)
        target_prob = spec.final_target_belief @ emit.T
        kl_unsteered_to_target = float(_kl(p_unsteered[spec.seq_idx][None], target_prob[None])[0])
        unsteered_to_target_data[series_key].append((spec.seq_idx, spec.sub_idx, kl_unsteered_to_target))

    logger.info("Phase 2: intervention loop ...")
    n_batches = (len(all_specs) + config.batch_size - 1) // config.batch_size
    for layer_idx, layer in enumerate(config.layer_indices):
        logger.info(f"  Layer {layer} ({layer_idx + 1}/{len(config.layer_indices)}) ...")
        decoder = decoders[layer]

        for batch_idx in range(n_batches):
            start = batch_idx * config.batch_size
            specs_batch = all_specs[start : start + config.batch_size]
            steering_flat = _compute_steering_flat(specs_batch, decoder, device, model_dtype)
            p_steered = _run_batch(
                model,
                llm_tokens,
                specs_batch,
                decoder,
                layer,
                final_pos,
                mid_tok_ids,
                device,
                model_dtype,
                steering_flat=steering_flat,
            )
            norm_records = _steering_norm_records(specs_batch, steering_flat)

            target_probs = np.stack([spec.final_target_belief for spec in specs_batch], axis=0) @ emit.T
            factual_probs = np.stack([p_opt_factual[spec.seq_idx] for spec in specs_batch], axis=0)
            kl_to_target = _kl(p_steered, target_probs)
            kl_to_factual = _kl(p_steered, factual_probs)

            for series_key, seq_idx, sub_idx, value in norm_records:
                if series_key not in steering_norm_data:
                    continue
                steering_norm_data[series_key][layer].append((seq_idx, sub_idx, value))

            for item_idx, spec in enumerate(specs_batch):
                series_key = _series_key(spec.condition, spec.k)
                kl_target_value = float(kl_to_target[item_idx])
                kl_factual_value = float(kl_to_factual[item_idx])
                kl_to_target_data[series_key][layer].append((spec.seq_idx, spec.sub_idx, kl_target_value))
                kl_to_factual_data[series_key][layer].append((spec.seq_idx, spec.sub_idx, kl_factual_value))

                record = {
                    "layer": layer,
                    "condition": spec.condition,
                    "series_key": series_key,
                    "k": spec.k,
                    "seq_idx": spec.seq_idx,
                    "sub_idx": spec.sub_idx,
                    "donor_idx": spec.donor_idx,
                    "positions": spec.positions,
                    "kl_to_target": kl_target_value,
                    "kl_to_factual": kl_factual_value,
                    "p_steered": [float(value) for value in p_steered[item_idx]],
                    "eta_target_final": [float(value) for value in spec.final_target_belief],
                }
                raw_records.append(record)

        layer_means = {
            series_key: np.mean([value for _, _, value in kl_to_target_data[series_key][layer]])
            for series_key in series_keys
        }
        layer_norm_means = {
            series_key: np.mean([value for _, _, value in steering_norm_data[series_key][layer]])
            for series_key in steering_norm_data
        }
        past_summary = "  ".join(
            (
                f"K={k}:shared={layer_means[_past_series_key('past_consistent_shared_prefix', k)]:.4f}"
                f"/diff={layer_means[_past_series_key('past_consistent', k)]:.4f}"
            )
            for k in config.k_values
        )
        past_norm_summary = "  ".join(
            (
                f"K={k}:shared={layer_norm_means[_past_series_key('past_consistent_shared_prefix', k)]:.4f}"
                f"/diff={layer_norm_means[_past_series_key('past_consistent', k)]:.4f}"
            )
            for k in config.k_values
        )
        logger.info(
            f"    {past_summary}  "
            f"garbage-valid={layer_means['garbage_valid']:.4f}  "
            f"garbage-random={layer_means['garbage_random']:.4f}"
        )
        logger.info(f"    steering norms  {past_norm_summary}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info("Aggregating ...")
    agg_to_target, agg_to_factual, agg_unsteered_to_target, baseline = _aggregate(
        kl_to_target_data,
        kl_to_factual_data,
        unsteered_to_target_data,
        baseline_kl,
    )
    agg_steering_norm = {
        series_key: {
            layer: _agg_records(records)
            for layer, records in by_layer.items()
        }
        for series_key, by_layer in steering_norm_data.items()
    }

    logger.info("Collecting propagation plot data ...")
    specs_k1 = [
        spec
        for spec in all_specs
        if spec.condition == "past_consistent" and spec.k == 1 and spec.sub_idx == 0
    ]
    propagation_data = _collect_propagation_data(
        model,
        llm_tokens,
        specs_k1,
        config.layer_indices,
        decoders,
        encoders,
        final_pos,
        config,
        device,
        model_dtype,
    )

    metrics = {
        "unsteered_kl_per_seq": [float(value) for value in baseline_kl],
        "unsteered_kl_mean": baseline["mean"],
        "unsteered_kl_std": baseline["std"],
        "baseline": baseline,
        "agg_unsteered_to_target": agg_unsteered_to_target,
        "conditions_to_target": {
            series_key: {str(layer): values for layer, values in by_layer.items()}
            for series_key, by_layer in agg_to_target.items()
        },
        "conditions_to_factual": {
            series_key: {str(layer): values for layer, values in by_layer.items()}
            for series_key, by_layer in agg_to_factual.items()
        },
        "steering_norms": {
            series_key: {str(layer): values for layer, values in by_layer.items()}
            for series_key, by_layer in agg_steering_norm.items()
        },
        "past_consistent_steering_norms": {
            series_key: {str(layer): values for layer, values in by_layer.items()}
            for series_key, by_layer in agg_steering_norm.items()
        },
        "raw_records": raw_records,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    logger.info("Generating plots ...")
    _plot_main_result(agg_to_target, config.k_values, config.layer_indices, baseline, fig_dir / "kl_vs_layer")
    _plot_k_sweep(agg_to_target, config.k_values, config.layer_indices, fig_dir / "kl_vs_k")
    _plot_past_variant_overlay(
        agg_to_target,
        config.k_values,
        config.layer_indices,
        fig_dir / "past_consistent_overlay",
    )
    _plot_steering_norm_vs_layer(
        agg_steering_norm,
        config.k_values,
        config.layer_indices,
        fig_dir / "steering_norm_vs_layer",
    )
    _plot_heatmaps(agg_to_target, config.k_values, config.layer_indices, fig_dir)
    _plot_to_factual(agg_to_factual, baseline, config.layer_indices, fig_dir / "kl_to_factual")
    for k in config.k_values:
        _plot_crossing_pair(
            k,
            agg_to_target,
            agg_to_factual,
            agg_unsteered_to_target,
            baseline,
            config.layer_indices,
            fig_dir / f"crossing_past_consistent_k{k}",
        )
        _plot_causal_shift_pair(
            k,
            agg_to_target,
            agg_to_factual,
            agg_unsteered_to_target,
            baseline,
            config.layer_indices,
            fig_dir / f"causal_shift_past_consistent_k{k}",
        )
    for series_key in ("garbage_valid", "garbage_random"):
        _plot_crossing(
            series_key,
            agg_to_target,
            agg_to_factual,
            agg_unsteered_to_target,
            baseline,
            config.layer_indices,
            fig_dir / f"crossing_{series_key}",
        )
        _plot_causal_shift(
            series_key,
            agg_to_target,
            agg_to_factual,
            agg_unsteered_to_target,
            baseline,
            config.layer_indices,
            fig_dir / f"causal_shift_{series_key}",
        )
    _plot_propagation(propagation_data, config.layer_indices, fig_dir / "belief_propagation_k1")

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
