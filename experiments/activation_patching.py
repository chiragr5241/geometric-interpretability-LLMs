#!/usr/bin/env python3
"""SPAR-15 — Activation patching: causal test of belief geometry.

For each sequence, patches the residual stream at the last token position
(t = patch_position) with the decoder output D_L(η_target), then measures
KL divergence across four conditions:

  factual        — η_target = true belief after full prefix
  counterfactual — η_target = belief if last token were different (×2)
  garbage_valid  — η_target = belief from a donor sequence (×n_garbage_valid)
  garbage_random — η_target = uniform sample on the 2-simplex (×n_garbage_random)

For the counterfactual condition, additionally tracks:
  - KL(P_unpatched || P_opt(η_cf))         pre-patch KL to counterfactual target
  - KL(P_patched_cf || P_opt(η_factual))   post-patch KL to factual target

These quantities power three new plots:
  - crossing_plot   : before/after KL for each layer (small multiples)
  - causal_shift    : how much patching shifts KL per layer (summary line)
  - heatmap_shift_* : per-sequence shift (layer × sequence)

Usage:
    python experiments/activation_patching.py experiments/configs/activation_patching.yaml
    python experiments/activation_patching.py experiments/configs/activation_patching.yaml --dry-run
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


CONDITIONS = ["factual", "counterfactual", "garbage_valid", "garbage_random"]

DRY_RUN_LAYERS  = [0, 2, 6, 10, 17, 25]
DRY_RUN_SEQ_LEN = 100
DRY_RUN_N_SEQ   = 5

_COLOR_FACTUAL = "#1f77b4"
_COLOR_CF      = "#ff7f0e"
_COLORS: dict[str, tuple[str, str]] = {
    "factual":        (_COLOR_FACTUAL, "rgba(31,119,180,0.12)"),
    "counterfactual": (_COLOR_CF,      "rgba(255,127,14,0.12)"),
    "garbage_valid":  ("#2ca02c",      "rgba(44,160,44,0.12)"),
    "garbage_random": ("#d62728",      "rgba(214,39,40,0.12)"),
}


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ActivationPatchingConfig(ExperimentConfig):
    encoder_decoder_dir: str
    layer_indices: list[int]
    seq_length: int
    n_sequences: int
    batch_size: int
    patch_position: int
    n_garbage_valid: int
    n_garbage_random: int
    pooled_probes: bool
    vocab_mapping: dict[str, int]
    n_ctx_override: int | None = None


# ── Patch spec ───────────────────────────────────────────────────────────────

@dataclass
class PatchSpec:
    seq_idx: int
    condition: str
    sub_idx: int
    target_belief: np.ndarray   # (n_states,) float32, sums to 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hmm_step(belief: np.ndarray, token_idx: int, T_3d: np.ndarray) -> np.ndarray:
    """One HMM transition+emission step: normalize(T_3d[token] @ belief)."""
    out = T_3d[token_idx] @ belief
    return (out / (out.sum() + 1e-10)).astype(np.float32)


def _kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """KL(P || Q) per row, both (..., n). Returns (...,)."""
    return (P * np.log(np.clip(P, 1e-10, None) / np.clip(Q, 1e-10, None))).sum(axis=-1)


def _build_specs(
    seq_beliefs: list[np.ndarray],
    seq_tokens: list[np.ndarray],
    T_3d: np.ndarray,
    n_garbage_valid: int,
    n_garbage_random: int,
    rng: np.random.Generator,
) -> list[PatchSpec]:
    N = len(seq_beliefs)
    L = seq_tokens[0].shape[0]
    specs: list[PatchSpec] = []

    for i in range(N):
        eta_L = seq_beliefs[i][L].astype(np.float32)
        specs.append(PatchSpec(i, "factual", 0, eta_L))

        last_tok = int(seq_tokens[i][L - 1])
        alts = [z for z in range(T_3d.shape[0]) if z != last_tok]
        for sub_i, z in enumerate(alts):
            eta_c = _hmm_step(seq_beliefs[i][L - 1], z, T_3d)
            specs.append(PatchSpec(i, "counterfactual", sub_i, eta_c))

        others = [j for j in range(N) if j != i]
        donors = rng.choice(others, size=min(n_garbage_valid, len(others)), replace=False)
        for sub_i, donor in enumerate(donors):
            specs.append(PatchSpec(i, "garbage_valid", sub_i, seq_beliefs[donor][L].astype(np.float32)))

        for sub_i in range(n_garbage_random):
            specs.append(PatchSpec(i, "garbage_random", sub_i, rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float32)))

    return specs


def _run_batch(
    model,
    llm_tokens: list[torch.Tensor],
    specs: list[PatchSpec],
    decoder: Decoder,
    layer: int,
    patch_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Run one hooked forward pass. Returns P_patched (B, n_hmm_tokens)."""
    tokens_batch = torch.cat([llm_tokens[s.seq_idx] for s in specs], dim=0).to(device)
    eta_batch = torch.from_numpy(np.stack([s.target_belief for s in specs])).float().to(device)

    with torch.no_grad():
        target_acts = decoder(eta_batch).to(dtype=model_dtype)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, patch_pos, :] = target_acts
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    probs_all = F.softmax(logits[:, patch_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    P_patched = probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)
    return P_patched.astype(np.float32)


def _agg_records(records: list[tuple[int, int, float]]) -> dict:
    """Average sub-conditions within each sequence, then compute stats across sequences."""
    per_seq: dict[int, list[float]] = {}
    for seq_idx, _sub_idx, val in records:
        per_seq.setdefault(seq_idx, []).append(val)
    seq_means = [float(np.mean(v)) for v in per_seq.values()]
    n = len(seq_means)
    return {
        "seq_means": seq_means,
        "mean": float(np.mean(seq_means)),
        "std": float(np.std(seq_means)),
        "stderr": float(np.std(seq_means) / max(np.sqrt(n), 1)),
        "n_seqs": n,
    }


def _aggregate(
    kl_data: dict[str, dict[int, list[tuple[int, int, float]]]],
    kl_patched_cf_to_factual: dict[int, list[tuple[int, int, float]]],
    kl_unpatched_to_cf_records: list[tuple[int, int, float]],
    unpatched_kl: list[float],
) -> tuple[dict, dict, dict, dict]:
    agg: dict[str, dict[int, dict]] = {}
    for cond, by_layer in kl_data.items():
        agg[cond] = {}
        for layer, records in by_layer.items():
            agg[cond][layer] = _agg_records(records)

    agg_cf_to_factual: dict[int, dict] = {
        layer: _agg_records(recs) for layer, recs in kl_patched_cf_to_factual.items()
    }

    agg_unpatched_to_cf = _agg_records(kl_unpatched_to_cf_records)

    n = len(unpatched_kl)
    agg_unpatched_to_factual = {
        "seq_means": list(unpatched_kl),
        "mean": float(np.mean(unpatched_kl)),
        "std": float(np.std(unpatched_kl)),
        "stderr": float(np.std(unpatched_kl) / max(np.sqrt(n), 1)),
        "n_seqs": n,
    }

    return agg, agg_cf_to_factual, agg_unpatched_to_cf, agg_unpatched_to_factual


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_main_result(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    unpatched_mean: float,
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers_str,
        y=[unpatched_mean] * len(layer_indices),
        name="Unpatched (baseline)",
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
    ))

    for cond in CONDITIONS:
        line_color, fill_color = _COLORS[cond]
        means   = [agg[cond][l]["mean"]   for l in layer_indices]
        stderrs = [agg[cond][l]["stderr"] for l in layer_indices]
        upper = [m + e for m, e in zip(means, stderrs)]
        lower = [m - e for m, e in zip(means, stderrs)]

        fig.add_trace(go.Scatter(
            x=layers_str + layers_str[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=means,
            name=cond.replace("_", "-"),
            mode="lines+markers",
            line=dict(color=line_color, width=2),
        ))

    fig.update_yaxes(type="log")
    fig.update_layout(
        title="Activation patching: KL vs layer by condition<br>"
              "<sup>KL(P_patched || P_bayesian_optimal(η_target)) — log scale — mean ± stderr</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=500, width=820,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_delta_kl(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]
    delta = [
        agg["counterfactual"][l]["mean"] - agg["factual"][l]["mean"]
        for l in layer_indices
    ]
    fig = go.Figure(go.Scatter(
        x=layers_str, y=delta,
        mode="lines+markers",
        line=dict(color=_COLOR_CF, width=2),
        marker=dict(size=6),
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title="ΔKL = KL_counterfactual − KL_factual per layer<br>"
              "<sup>Difference in reconstruction error between counterfactual and factual conditions</sup>",
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats]",
        height=420, width=720,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_heatmaps(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    fig_dir: Path,
) -> None:
    for cond in CONDITIONS:
        n_seqs = max(agg[cond][layer_indices[0]]["n_seqs"], 1)
        z_by_layer = []
        for layer in layer_indices:
            row = agg[cond][layer]["seq_means"]
            if len(row) < n_seqs:
                row = row + [float("nan")] * (n_seqs - len(row))
            z_by_layer.append(row[:n_seqs])
        # z_by_layer: (n_layers, n_seqs) → transpose to (n_seqs, n_layers) for seq × layer layout
        z_arr = np.log10(np.clip(np.array(z_by_layer, dtype=float).T, 1e-10, None))

        fig = go.Figure(go.Heatmap(
            z=z_arr,
            x=[str(l) for l in layer_indices],
            y=[f"Seq {i}" for i in range(n_seqs)],
            colorscale="Viridis",
            colorbar=dict(title="log₁₀(KL)"),
        ))
        fig.update_layout(
            title=f"KL heatmap — {cond.replace('_', '-')} (log₁₀ scale)",
            xaxis_title="Layer",
            yaxis_title="Sequence",
            height=500, width=700,
            margin=dict(t=70, b=60, l=70, r=40),
        )
        fig.write_image(str(fig_dir / f"heatmap_{cond}.png"))


def _plot_crossing(
    agg: dict[str, dict[int, dict]],
    agg_cf_to_factual: dict[int, dict],
    agg_unpatched_to_cf: dict,
    agg_unpatched_to_factual: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    """Small-multiples crossing plot: KL before/after patching, one panel per layer.

    Blue line: KL(P || P_opt(η_factual)) — should go UP after patching with η_cf.
    Orange line: KL(P || P_opt(η_cf))    — should go DOWN after patching with η_cf.
    Crossing of the two lines = the model causally adopted the injected belief.
    """
    n_layers = len(layer_indices)
    n_cols = min(6, n_layers)
    n_rows = ceil(n_layers / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layer_indices],
        shared_yaxes=False,
        vertical_spacing=0.15 if n_rows > 1 else 0.1,
        horizontal_spacing=0.08,
    )

    uf_mean  = agg_unpatched_to_factual["mean"]
    uf_err   = agg_unpatched_to_factual["stderr"]
    ucf_mean = agg_unpatched_to_cf["mean"]
    ucf_err  = agg_unpatched_to_cf["stderr"]

    for i, layer in enumerate(layer_indices):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = (i == 0)

        pf_mean  = agg_cf_to_factual[layer]["mean"]
        pf_err   = agg_cf_to_factual[layer]["stderr"]
        pcf_mean = agg["counterfactual"][layer]["mean"]
        pcf_err  = agg["counterfactual"][layer]["stderr"]

        fig.add_trace(go.Scatter(
            x=["Unpatched", "Patched"],
            y=[uf_mean, pf_mean],
            error_y=dict(type="data", array=[uf_err, pf_err], visible=True),
            name="KL to factual opt.",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color=_COLOR_FACTUAL, width=2),
            marker=dict(size=6),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=["Unpatched", "Patched"],
            y=[ucf_mean, pcf_mean],
            error_y=dict(type="data", array=[ucf_err, pcf_err], visible=True),
            name="KL to cf opt.",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color=_COLOR_CF, width=2),
            marker=dict(size=6),
        ), row=row, col=col)

    fig.update_layout(
        title=(
            "Crossing plot: KL before / after counterfactual patching<br>"
            "<sup>Blue = KL to factual optimal | Orange = KL to cf optimal — crossing = causal use</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 220, 1400),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_causal_shift(
    agg: dict[str, dict[int, dict]],
    agg_cf_to_factual: dict[int, dict],
    agg_unpatched_to_cf: dict,
    agg_unpatched_to_factual: dict,
    layer_indices: list[int],
    path: Path,
) -> None:
    """Summary line plot: how much counterfactual patching shifts KL per layer.

    shift_to_cf     = KL_unpatched→cf    − KL_patched→cf     (positive = moved toward cf)
    shift_from_fact = KL_patched→factual − KL_unpatched→factual (positive = moved away from factual)
    If patching is causal, both curves should be positive and similar in magnitude.
    """
    layers_str = [str(l) for l in layer_indices]
    ucf_mean = agg_unpatched_to_cf["mean"]
    ucf_err  = agg_unpatched_to_cf["stderr"]
    uf_mean  = agg_unpatched_to_factual["mean"]
    uf_err   = agg_unpatched_to_factual["stderr"]

    # Signed change in each KL: positive = KL increased, negative = KL decreased.
    # After a successful causal patch: KL→factual goes UP, KL→cf goes DOWN.
    delta_cf   = [agg["counterfactual"][l]["mean"] - ucf_mean for l in layer_indices]
    delta_fact = [agg_cf_to_factual[l]["mean"] - uf_mean      for l in layer_indices]

    err_cf = [
        float(np.sqrt(ucf_err**2 + agg["counterfactual"][l]["stderr"]**2))
        for l in layer_indices
    ]
    err_fact = [
        float(np.sqrt(uf_err**2 + agg_cf_to_factual[l]["stderr"]**2))
        for l in layer_indices
    ]

    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.add_trace(go.Scatter(
        x=layers_str, y=delta_fact,
        error_y=dict(type="data", array=err_fact, visible=True),
        name="ΔKL to factual optimal (↑ = moved away from factual)",
        mode="lines+markers",
        line=dict(color=_COLOR_FACTUAL, width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=layers_str, y=delta_cf,
        error_y=dict(type="data", array=err_cf, visible=True),
        name="ΔKL to cf optimal (↓ = moved toward cf)",
        mode="lines+markers",
        line=dict(color=_COLOR_CF, width=2),
        marker=dict(size=6, symbol="square"),
    ))

    fig.update_layout(
        title=(
            "Causal shift per layer: counterfactual patching<br>"
            "<sup>ΔKL = KL_patched − KL_unpatched: factual ↑, cf ↓ = causal use</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats]",
        height=440, width=760,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_shift_heatmaps(
    kl_data: dict[str, dict[int, list[tuple[int, int, float]]]],
    kl_patched_cf_to_factual: dict[int, list[tuple[int, int, float]]],
    kl_unpatched_to_cf_records: list[tuple[int, int, float]],
    unpatched_kl: list[float],
    layer_indices: list[int],
    fig_dir: Path,
) -> None:
    """Layer × sequence heatmaps for both causal shift directions."""
    ucf_per_seq: dict[int, list[float]] = {}
    for seq_idx, _sub_idx, val in kl_unpatched_to_cf_records:
        ucf_per_seq.setdefault(seq_idx, []).append(val)

    all_seq_ids = sorted(ucf_per_seq.keys())
    ucf_seq_means = {i: float(np.mean(ucf_per_seq[i])) for i in all_seq_ids}

    z_to_cf: list[list[float]] = []
    z_from_fact: list[list[float]] = []

    for layer in layer_indices:
        pcf_per_seq: dict[int, list[float]] = {}
        for seq_idx, _sub_idx, val in kl_data["counterfactual"][layer]:
            pcf_per_seq.setdefault(seq_idx, []).append(val)

        pfact_per_seq: dict[int, list[float]] = {}
        for seq_idx, _sub_idx, val in kl_patched_cf_to_factual[layer]:
            pfact_per_seq.setdefault(seq_idx, []).append(val)

        row_to_cf: list[float] = []
        row_from_fact: list[float] = []
        for seq_i in all_seq_ids:
            ucf_m = ucf_seq_means.get(seq_i, float("nan"))
            pcf_m = float(np.mean(pcf_per_seq.get(seq_i, [float("nan")])))
            uf_m  = unpatched_kl[seq_i] if seq_i < len(unpatched_kl) else float("nan")
            pf_m  = float(np.mean(pfact_per_seq.get(seq_i, [float("nan")])))
            row_to_cf.append(pcf_m - ucf_m)
            row_from_fact.append(pf_m - uf_m)

        z_to_cf.append(row_to_cf)
        z_from_fact.append(row_from_fact)

    for z_list, title, fname in [
        (
            z_to_cf,
            "ΔKL toward cf — KL_patched→cf − KL_unpatched→cf",
            "heatmap_shift_to_cf",
        ),
        (
            z_from_fact,
            "ΔKL from factual — KL_patched→factual − KL_unpatched→factual",
            "heatmap_shift_from_factual",
        ),
    ]:
        # z_list: (n_layers, n_seqs) → transpose to (n_seqs, n_layers) for seq × layer layout
        z_arr = np.array(z_list, dtype=float).T
        zmax = float(np.nanmax(np.abs(z_arr))) if not np.all(np.isnan(z_arr)) else 1.0

        fig = go.Figure(go.Heatmap(
            z=z_arr,
            x=[str(l) for l in layer_indices],
            y=[f"Seq {i}" for i in all_seq_ids],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="ΔKL [nats]"),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Sequence",
            height=500, width=700,
            margin=dict(t=70, b=60, l=70, r=40),
        )
        fig.write_image(str(fig_dir / f"{fname}.png"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching experiment")
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

    config = load_config(args.config, ActivationPatchingConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.layer_indices  = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.seq_length     = DRY_RUN_SEQ_LEN
        config.n_sequences    = DRY_RUN_N_SEQ
        config.patch_position = DRY_RUN_SEQ_LEN - 1
        config.experiment_name = config.experiment_name + "_dry_run"

    device = get_device()
    rng = np.random.default_rng(seed=42)

    out_dir = setup_output_dir(config)
    logger  = setup_logging(out_dir, name="act_patch")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir      : {out_dir}")
    logger.info(f"Device          : {device}")
    logger.info(f"Dry run         : {args.dry_run}")
    logger.info(f"N sequences     : {config.n_sequences}")
    logger.info(f"Seq length      : {config.seq_length}")
    logger.info(f"Patch position  : {config.patch_position}")
    logger.info(f"Batch size      : {config.batch_size}")
    logger.info(f"Layers          : {config.layer_indices}")
    logger.info(f"Encoder-dec dir : {config.encoder_decoder_dir}")

    enc_dec_dir = Path(config.encoder_decoder_dir)
    N = config.n_sequences
    L = config.seq_length
    patch_pos = config.patch_position
    n_vocab = len(config.vocab_mapping)
    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype

    # ── HMM ──────────────────────────────────────────────────────────────────
    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    T_3d = hmm.T_3d_matrix.cpu().numpy()
    emit = build_emission_matrix(hmm)   # (n_tokens, n_states)
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_vocab, logger)

    # ── Load decoders ─────────────────────────────────────────────────────────
    decoder_base = enc_dec_dir / "decoders" / "pooled"
    decoders: dict[int, Decoder] = {}
    for layer in config.layer_indices:
        dr = DecoderResult.load(decoder_base / f"layer_{layer}")
        decoders[layer] = dr.decoder.to(device)
        decoders[layer].eval()
    logger.info(f"Loaded {len(decoders)} decoders from {decoder_base}")

    # ── Phase 1: Generate sequences + clean forward passes ───────────────────
    logger.info("Phase 1: generating sequences and clean forward passes ...")
    seq_beliefs:   list[np.ndarray]  = []   # each (L+1, n_states)
    seq_tokens:    list[np.ndarray]  = []   # each (L,) HMM token indices
    llm_tokens:    list[torch.Tensor] = []  # each (1, L) on CPU
    P_unpatched:   list[np.ndarray]  = []   # each (n_hmm_tokens,)
    P_opt_factual: list[np.ndarray]  = []   # each (n_hmm_tokens,)
    unpatched_kl:  list[float]       = []

    for batch_start in range(0, N, config.batch_size):
        B_clean = min(config.batch_size, N - batch_start)

        tok_batch, _, _ = hmm.generate_dataset(B_clean, L, return_states=True)
        bel_batch = hmm.compute_belief_state(tok_batch)

        llm_batch_list: list[torch.Tensor] = []
        for b in range(B_clean):
            toks = tok_batch[b].cpu().numpy()
            bels = bel_batch[b].cpu().numpy()
            seq_tokens.append(toks)
            seq_beliefs.append(bels)
            P_opt_factual.append((bels[L] @ emit.T).astype(np.float32))

            text = " ".join(idx_to_token[int(t)] for t in toks)
            llm_tok = model.to_tokens(text, prepend_bos=False, truncate=False)
            assert llm_tok.shape[1] == L, f"Expected {L} tokens, got {llm_tok.shape[1]}"
            llm_tokens.append(llm_tok.cpu())
            llm_batch_list.append(llm_tok)

        llm_batch = torch.cat(llm_batch_list, dim=0).to(device)
        with torch.no_grad():
            logits_clean = model.run_with_hooks(llm_batch, fwd_hooks=[], return_type="logits")

        probs_all = F.softmax(logits_clean[:, patch_pos, :].float(), dim=-1)
        probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
        P_clean = probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)

        for b in range(B_clean):
            seq_idx = batch_start + b
            P_unpatched.append(P_clean[b].astype(np.float32))
            kl = float(_kl(P_clean[b : b + 1], P_opt_factual[seq_idx][None])[0])
            unpatched_kl.append(kl)

        del logits_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info(f"  Sequences {batch_start + 1}–{batch_start + B_clean}/{N} done")

    unpatched_mean = float(np.mean(unpatched_kl))
    logger.info(f"Unpatched KL at pos {patch_pos}: mean={unpatched_mean:.4f} std={np.std(unpatched_kl):.4f}")

    # ── Build patch specs ─────────────────────────────────────────────────────
    logger.info("Building patch specs ...")
    all_specs = _build_specs(
        seq_beliefs, seq_tokens, T_3d,
        config.n_garbage_valid, config.n_garbage_random, rng,
    )
    logger.info(f"  {len(all_specs)} total specs ({len(all_specs) // N} per sequence)")

    # ── Pre-compute KL(P_unpatched || P_opt(η_cf)) — layer-independent ───────
    logger.info("Pre-computing unpatched KL to counterfactual targets ...")
    kl_unpatched_to_cf_records: list[tuple[int, int, float]] = []
    for spec in all_specs:
        if spec.condition == "counterfactual":
            P_opt_cf = spec.target_belief @ emit.T          # (n_hmm_tokens,)
            kl = float(_kl(P_unpatched[spec.seq_idx][None], P_opt_cf[None])[0])
            kl_unpatched_to_cf_records.append((spec.seq_idx, spec.sub_idx, kl))

    # ── Phase 2: Intervention loop (per layer, batched) ───────────────────────
    logger.info("Phase 2: intervention loop ...")

    kl_data: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        cond: {layer: [] for layer in config.layer_indices}
        for cond in CONDITIONS
    }
    kl_patched_cf_to_factual: dict[int, list[tuple[int, int, float]]] = {
        layer: [] for layer in config.layer_indices
    }

    n_batches = (len(all_specs) + config.batch_size - 1) // config.batch_size

    for layer_i, layer in enumerate(config.layer_indices):
        logger.info(f"  Layer {layer} ({layer_i + 1}/{len(config.layer_indices)}) ...")
        decoder = decoders[layer]

        for batch_i in range(n_batches):
            start = batch_i * config.batch_size
            specs_batch = all_specs[start : start + config.batch_size]

            P_pat = _run_batch(
                model, llm_tokens, specs_batch, decoder,
                layer, patch_pos, mid_tok_ids, device, model_dtype,
            )   # (B, n_hmm_tokens)

            etas = np.stack([s.target_belief for s in specs_batch])
            kl_to_target = _kl(P_pat, etas @ emit.T)       # (B,)

            for j, spec in enumerate(specs_batch):
                kl_data[spec.condition][layer].append(
                    (spec.seq_idx, spec.sub_idx, float(kl_to_target[j]))
                )
                if spec.condition == "counterfactual":
                    kl_to_factual = float(
                        _kl(P_pat[j : j + 1], P_opt_factual[spec.seq_idx][None])[0]
                    )
                    kl_patched_cf_to_factual[layer].append(
                        (spec.seq_idx, spec.sub_idx, kl_to_factual)
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        factual_mean  = np.mean([kl for _, _, kl in kl_data["factual"][layer]])
        counter_mean  = np.mean([kl for _, _, kl in kl_data["counterfactual"][layer]])
        logger.info(
            f"    factual={factual_mean:.4f}  counterfactual={counter_mean:.4f}  "
            f"Δ={counter_mean - factual_mean:+.4f}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("Aggregating ...")
    agg, agg_cf_to_factual, agg_unpatched_to_cf, agg_unpatched_to_factual = _aggregate(
        kl_data, kl_patched_cf_to_factual, kl_unpatched_to_cf_records, unpatched_kl,
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "unpatched_kl_per_seq": [float(v) for v in unpatched_kl],
        "unpatched_kl_mean": unpatched_mean,
        "unpatched_kl_std": float(np.std(unpatched_kl)),
        "agg_unpatched_to_factual": agg_unpatched_to_factual,
        "agg_unpatched_to_cf": agg_unpatched_to_cf,
        "agg_cf_to_factual": {str(l): v for l, v in agg_cf_to_factual.items()},
        "conditions": {
            cond: {str(layer): v for layer, v in by_layer.items()}
            for cond, by_layer in agg.items()
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    _plot_main_result(agg, config.layer_indices, unpatched_mean, fig_dir / "kl_vs_layer")
    _plot_delta_kl(agg, config.layer_indices, fig_dir / "delta_kl_causal")
    _plot_heatmaps(agg, config.layer_indices, fig_dir)
    _plot_crossing(
        agg, agg_cf_to_factual, agg_unpatched_to_cf, agg_unpatched_to_factual,
        config.layer_indices, fig_dir / "crossing_plot",
    )
    _plot_causal_shift(
        agg, agg_cf_to_factual, agg_unpatched_to_cf, agg_unpatched_to_factual,
        config.layer_indices, fig_dir / "causal_shift",
    )
    _plot_shift_heatmaps(
        kl_data, kl_patched_cf_to_factual, kl_unpatched_to_cf_records,
        unpatched_kl, config.layer_indices, fig_dir,
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
