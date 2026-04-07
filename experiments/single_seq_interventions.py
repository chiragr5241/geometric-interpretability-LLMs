#!/usr/bin/env python3
"""SPAR-29 Phase 3 — Unified patching, steering, and belief-subspace ablation.

Loads per-sequence encoder-decoder weights from a Phase 1+2 output directory,
then runs four classes of intervention via TransformerLens hooks at every target
layer, measuring KL divergence at the last evaluation position.

Patching (replace act with decoder(target_belief)):
    optimal       — target = Bayesian-optimal belief (ground truth)
    round_trip    — target = encoder(activation), i.e. round-trip projection
    past_consistent — target = belief from a k-token HMM continuation sharing the prefix
    random        — target = uniform simplex sample

Steering (add decoder(target) − decoder(encoder(act))):
    optimal, past_consistent, random

Belief-subspace ablation (remove the belief-relevant component at all positions):
    belief        — act → act − P @ act, P = W(WᵀW)⁻¹Wᵀ (belief subspace projector)
    random        — same with n_random_ablation_draws random 3-D subspaces (control)

Alignment (no BOS):
    cache position t  ──▶  beliefs[t+1]
    eval window: act positions [eval_act_start, L), beliefs [eval_act_start+1, L+1)
    k-position interventions target positions [L−k, …, L−1]; KL measured at L−1.

Usage:
    python experiments/single_seq_interventions.py \\
        experiments/configs/single_seq_interventions.yaml
    python experiments/single_seq_interventions.py \\
        experiments/configs/single_seq_interventions.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

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

DRY_RUN_N_SEQ = 2
DRY_RUN_N_EVAL_POSITIONS = 3  # last N positions of eval window used for k-interventions
DRY_RUN_LAYERS = [0, 2, 10, 17, 27]
DRY_RUN_K_VALUES = [1, 3]

PATCH_CONDITIONS = ["optimal", "round_trip", "past_consistent", "random"]
STEER_CONDITIONS = ["optimal", "past_consistent", "random"]

_PATCH_COLORS: dict[str, str] = {
    "optimal": "#1f77b4",
    "round_trip": "#ff7f0e",
    "past_consistent": "#2ca02c",
    "random": "#d62728",
}
_STEER_COLORS: dict[str, str] = {
    "optimal": "#1f77b4",
    "past_consistent": "#2ca02c",
    "random": "#d62728",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SingleSeqInterventionsConfig(ExperimentConfig):
    training_dir: str = ""
    layer_indices: list[int] = field(default_factory=list)
    vocab_mapping: dict[str, int] = field(default_factory=dict)
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    n_random_ablation_draws: int = 5
    n_past_consistent_draws: int = 3
    n_random_patch_draws: int = 3
    batch_size: int = 8
    n_ctx_override: int | None = None
    post_convergence_start: int = 30
    train_eval_split: float = 0.7
    random_seed: int = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p ∥ q) per row. Both (..., n). Returns (...,)."""
    return (p * np.log(np.clip(p, 1e-10, None) / np.clip(q, 1e-10, None))).sum(axis=-1)


def _eval_split_idx(L: int, P: int, train_eval_split: float) -> int:
    n_post_conv = L - P + 1
    return int(n_post_conv * train_eval_split)


def _orthogonal_projector(W: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto col-span of W: P = W (WᵀW)⁻¹ Wᵀ.  W: (d, k)."""
    WtW = W.T @ W
    try:
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        WtW_inv = np.linalg.pinv(WtW)
    return W @ WtW_inv @ W.T


def _random_projector(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random k-dim orthogonal projector in R^d via QR decomposition."""
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q @ Q.T


def _agg_records(records: list[tuple[int, int, float]]) -> dict:
    per_seq: dict[int, list[float]] = {}
    for seq_idx, _sub_idx, val in records:
        per_seq.setdefault(seq_idx, []).append(val)
    seq_means = [float(np.mean(vs)) for vs in per_seq.values()]
    n = len(seq_means)
    return {
        "seq_means": seq_means,
        "mean": float(np.mean(seq_means)),
        "std": float(np.std(seq_means)),
        "stderr": float(np.std(seq_means) / max(np.sqrt(n), 1.0)),
        "n_seqs": n,
    }


# ── Hook-based forward passes ─────────────────────────────────────────────────

def _run_patched(
    model,
    llm_tokens: torch.Tensor,
    layer: int,
    positions: list[int],
    target_acts: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Run a batched patched forward pass. Returns P_patched (B, n_hmm).

    target_acts: (B, len(positions), d_model) — target activation per position.
    llm_tokens:  (1, L) — single sequence (repeated B times along batch dim).
    """
    B = target_acts.shape[0]
    tokens_batch = llm_tokens.expand(B, -1).to(device)
    pos_tensor = torch.tensor(positions, dtype=torch.long)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, pos_tensor, :] = target_acts.to(device=value.device, dtype=value.dtype)
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    probs_all = F.softmax(logits[:, measure_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    return (probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


def _run_steered(
    model,
    llm_tokens: torch.Tensor,
    layer: int,
    positions: list[int],
    steering_deltas: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Run a batched steered forward pass (additive intervention). Returns P_steered (B, n_hmm).

    steering_deltas: (B, len(positions), d_model).
    """
    B = steering_deltas.shape[0]
    tokens_batch = llm_tokens.expand(B, -1).to(device)
    pos_tensor = torch.tensor(positions, dtype=torch.long)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, pos_tensor, :] = value[:, pos_tensor, :] + steering_deltas.to(device=value.device, dtype=value.dtype)
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    probs_all = F.softmax(logits[:, measure_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    return (probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


def _run_ablated(
    model,
    llm_tokens: torch.Tensor,
    layer: int,
    projectors: list[np.ndarray],
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Run belief-subspace ablation at ALL positions. Returns P_ablated (B, n_hmm).

    projectors: list of B projector matrices (d, d).  For each b, hook does:
        act[b, :, :] -= P[b] @ act[b, :, :]
    """
    B = len(projectors)
    tokens_batch = llm_tokens.expand(B, -1).to(device)
    P_batch = torch.from_numpy(np.stack(projectors, axis=0)).float().to(device)  # (B, d, d)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # value: (B, L, d)
        v = value.float()
        # P_batch @ v[b].T → (B, d, L); transpose to (B, L, d)
        proj = torch.bmm(P_batch, v.transpose(1, 2)).transpose(1, 2)
        value = (v - proj).to(device=value.device, dtype=value.dtype)
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    probs_all = F.softmax(logits[:, measure_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    return (probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


# ── Target belief computation ─────────────────────────────────────────────────

def _optimal_targets(seq_beliefs: np.ndarray, positions: list[int]) -> np.ndarray:
    """Bayesian-optimal beliefs at each position: belief[t+1] for act at t.
    Returns (len(positions), n_states).
    """
    return np.stack([seq_beliefs[t + 1] for t in positions], axis=0).astype(np.float32)


def _round_trip_targets(
    probe: Probe,
    clean_acts: np.ndarray,
    decoder: Decoder,
    device: torch.device,
) -> np.ndarray:
    """decoder(encoder(act)) at each position. clean_acts: (k, d_model). Returns (k, d_model)."""
    probe = probe.to(device)
    decoder = decoder.to(device)
    with torch.no_grad():
        acts_t = torch.from_numpy(clean_acts).float().to(device)
        encoded = probe(acts_t)      # (k, n_states)
        decoded = decoder(encoded)   # (k, d_model)
    return decoded.float().cpu().numpy()


def _past_consistent_targets(
    seq_beliefs: np.ndarray,
    positions: list[int],
    hmm: Mess3HMM,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a k-token HMM continuation from the belief at positions[0].

    Alignment: `sample_continuation(beliefs[L-k], k, rng)` returns beliefs[j] =
    belief after step j, which is the target for act at position L-k+j (= positions[j]).
    positions[j] + 1 = L-k+j+1 → new_beliefs[j]. ✓

    Returns (len(positions), n_states).
    """
    k = len(positions)
    # belief at positions[0] = beliefs after tokens[0:positions[0]+1]
    # but we need the belief BEFORE seeing tokens at these positions, i.e. beliefs[positions[0]]
    initial_belief = seq_beliefs[positions[0]].astype(np.float32)
    _, new_beliefs = hmm.sample_continuation(initial_belief, k, rng)
    # new_beliefs[j] = belief after seeing one more sampled token, starting from initial_belief
    # target for act at positions[j] = new_beliefs[j] (maps to "beliefs[positions[j]+1]" in the new path)
    return new_beliefs.astype(np.float32)


def _random_targets(positions: list[int], rng: np.random.Generator, n_states: int = 3) -> np.ndarray:
    """Uniform simplex samples for each position. Returns (len(positions), n_states)."""
    return rng.dirichlet(np.ones(n_states), size=len(positions)).astype(np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_kl_vs_layer(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    k: int,
    baseline_mean: float,
    title: str,
    colors: dict[str, str],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers_str,
        y=[baseline_mean] * len(layer_indices),
        name="Unintervened (baseline)",
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
    ))

    for cond, color in colors.items():
        if cond not in agg:
            continue
        means = [agg[cond][l]["mean"] for l in layer_indices]
        stderrs = [agg[cond][l]["stderr"] for l in layer_indices]
        upper = [m + e for m, e in zip(means, stderrs)]
        lower = [m - e for m, e in zip(means, stderrs)]
        rgba = color.lstrip("#")
        r, g, b = int(rgba[0:2], 16), int(rgba[2:4], 16), int(rgba[4:6], 16)
        fill_color = f"rgba({r},{g},{b},0.12)"

        fig.add_trace(go.Scatter(
            x=layers_str + layers_str[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=means,
            name=cond.replace("_", "-"),
            mode="lines+markers",
            line=dict(color=color, width=2),
        ))

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"{title} (k={k})<br><sup>KL(P_intervened ∥ P_opt(η_target)) — log scale — mean ± stderr</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=500, width=820,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_heatmap(
    agg_by_layer: dict[int, dict],
    layer_indices: list[int],
    title: str,
    path: Path,
) -> None:
    n_seqs = agg_by_layer[layer_indices[0]]["n_seqs"]
    rows_by_layer = []
    for l in layer_indices:
        row = agg_by_layer[l]["seq_means"]
        if len(row) < n_seqs:
            row = row + [float("nan")] * (n_seqs - len(row))
        rows_by_layer.append(row[:n_seqs])
    z = np.log10(np.clip(np.array(rows_by_layer, dtype=float).T, 1e-10, None))
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(l) for l in layer_indices],
        y=[f"Seq {i}" for i in range(z.shape[0])],
        colorscale="Viridis",
        colorbar=dict(title="log₁₀(KL)"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Sequence",
        height=500, width=700,
        margin=dict(t=70, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_ablation_curve(
    agg_belief: dict[int, dict],
    agg_random: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    for key, agg, color, name in [
        ("belief", agg_belief, "#d62728", "Belief-subspace ablation"),
        ("random", agg_random, "#7f7f7f", "Random-subspace ablation (control)"),
    ]:
        means = [agg[l]["mean"] for l in layer_indices]
        stderrs = [agg[l]["stderr"] for l in layer_indices]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=means,
            error_y=dict(type="data", array=stderrs, visible=True),
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Belief-subspace ablation: KL increase vs random control<br>"
              "<sup>KL(P_ablated ∥ P_opt) — mean ± stderr across sequences</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=460, width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_roundtrip_comparison(
    agg_patch: dict[str, dict[int, dict]],
    layer_indices: list[int],
    baseline_mean: float,
    k: int,
    path: Path,
) -> None:
    """H2A vs H2B: round-trip vs optimal vs baseline, patching only."""
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers_str,
        y=[baseline_mean] * len(layer_indices),
        name="Baseline (unpatched)",
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
    ))
    for cond, color in [("optimal", "#1f77b4"), ("round_trip", "#ff7f0e")]:
        means = [agg_patch[cond][l]["mean"] for l in layer_indices]
        stderrs = [agg_patch[cond][l]["stderr"] for l in layer_indices]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=means,
            error_y=dict(type="data", array=stderrs, visible=True),
            name=cond.replace("_", "-"),
            mode="lines+markers",
            line=dict(color=color, width=2),
        ))

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"H2A vs H2B: round-trip vs optimal patching (k={k})<br>"
              "<sup>round-trip ≈ optimal → H2A (subspace matters); ≈ baseline → H2B (clean beliefs matter)</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=480, width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_crossing(
    agg_to_opt: dict[int, dict],
    agg_to_clean: dict[int, dict],
    baseline_to_opt: float,
    layer_indices: list[int],
    k: int,
    condition: str,
    title: str,
    path: Path,
) -> None:
    """Small-multiples crossing plot: before/after for a single condition.

    Blue: KL(P ∥ P_opt) — should go DOWN after intervention.
    Orange: KL(P ∥ P_unintervened) — should go UP (output actually changed).
    Crossing of the two = causal effect on model beliefs.
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

    for i, layer in enumerate(layer_indices):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0

        pt_opt_mean = agg_to_opt[layer]["mean"]
        pt_opt_err = agg_to_opt[layer]["stderr"]
        pt_clean_mean = agg_to_clean[layer]["mean"]
        pt_clean_err = agg_to_clean[layer]["stderr"]

        fig.add_trace(go.Scatter(
            x=["Baseline", "Intervened"],
            y=[baseline_to_opt, pt_opt_mean],
            error_y=dict(type="data", array=[0.0, pt_opt_err], visible=True),
            name="KL to optimal",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=["Baseline", "Intervened"],
            y=[0.0, pt_clean_mean],
            error_y=dict(type="data", array=[0.0, pt_clean_err], visible=True),
            name="KL to unintervened",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
        ), row=row, col=col)

    fig.update_layout(
        title=(
            f"{title} — crossing ({condition.replace('_', '-')}, k={k})<br>"
            "<sup>Blue = KL to optimal (↓ = causal) | Orange = KL to unintervened (↑ = output changed)</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 220, 1400),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_causal_shift(
    agg_to_opt: dict[str, dict[int, dict]],
    baseline_to_opt: float,
    layer_indices: list[int],
    k: int,
    title: str,
    colors: dict[str, str],
    path: Path,
) -> None:
    """Summary: ΔKL per layer — how much each condition moves output toward optimal.

    ΔKL = KL_baseline_to_opt − KL_intervened_to_opt (positive = moved toward optimal).
    """
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for cond, color in colors.items():
        if cond not in agg_to_opt:
            continue
        delta = [baseline_to_opt - agg_to_opt[cond][l]["mean"] for l in layer_indices]
        err = [agg_to_opt[cond][l]["stderr"] for l in layer_indices]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=delta,
            error_y=dict(type="data", array=err, visible=True),
            name=cond.replace("_", "-"),
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=(
            f"{title} — causal shift (k={k})<br>"
            "<sup>ΔKL = KL_baseline − KL_intervened toward optimal: positive = causal</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats] (↑ = toward optimal)",
        height=460,
        width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_ablation_kl_to_output(
    agg_belief: dict[int, dict],
    agg_random: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    """KL(P_ablated ∥ P_clean) for both ablation conditions.

    Shows how much ablation perturbs the output distribution (independent of optimality).
    """
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    for agg, color, name in [
        (agg_belief, "#d62728", "Belief-subspace ablation"),
        (agg_random, "#7f7f7f", "Random-subspace ablation (control)"),
    ]:
        means = [agg[l]["mean"] for l in layer_indices]
        stderrs = [agg[l]["stderr"] for l in layer_indices]
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=means,
            error_y=dict(type="data", array=stderrs, visible=True),
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=(
            "Ablation: KL to unintervened output<br>"
            "<sup>KL(P_ablated ∥ P_clean) — output perturbation per condition</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=460,
        width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Single-sequence interventions (SPAR-29 Phase 3)")
    parser.add_argument("config", type=str)
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help=f"Quick test: {DRY_RUN_N_SEQ} seqs, {DRY_RUN_N_EVAL_POSITIONS} eval positions")
    args = parser.parse_args()

    config = load_config(args.config, SingleSeqInterventionsConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    dry_run = args.dry_run
    if dry_run:
        config.layer_indices = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.k_values = DRY_RUN_K_VALUES
        config.experiment_name = f"{config.experiment_name}_dry_run"

    rng = np.random.default_rng(config.random_seed)
    device = get_device()
    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="interventions")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    training_dir = Path(config.training_dir)
    if not training_dir.is_absolute():
        training_dir = project_root / training_dir
    npz_path = training_dir / "hmm_data.npz"

    logger.info(f"Output dir      : {out_dir}")
    logger.info(f"Training dir    : {training_dir}")
    logger.info(f"Device          : {device}")
    logger.info(f"Dry run         : {dry_run}")
    logger.info(f"Layers          : {config.layer_indices}")
    logger.info(f"k values        : {config.k_values}")

    arr = np.load(npz_path)
    all_tokens = arr["tokens"]    # (N, L) int64
    all_beliefs = arr["beliefs"]  # (N, L+1, n_states) float32

    N_total = all_tokens.shape[0]
    N = min(DRY_RUN_N_SEQ, N_total) if dry_run else N_total
    L = all_tokens.shape[1]
    n_states = all_beliefs.shape[2]

    P = config.post_convergence_start
    split_idx = _eval_split_idx(L, P, config.train_eval_split)
    eval_act_start = (P - 1) + split_idx
    # eval window: act positions [eval_act_start, L), beliefs [eval_act_start+1, L+1)
    # measure KL at position L-1 (the last eval act position)
    measure_pos = L - 1

    n_eval = L - eval_act_start
    if dry_run:
        # In dry-run the k_values are reduced (DRY_RUN_K_VALUES) so the
        # interventions naturally land within the last DRY_RUN_N_EVAL_POSITIONS
        # positions of the eval window.
        pass

    max_k = max(config.k_values)
    assert max_k <= n_eval, (
        f"max k={max_k} exceeds eval window size={n_eval}. "
        f"Reduce k_values or increase seq_length/train_eval_split."
    )

    logger.info(f"Seq length      : {L}")
    logger.info(f"Post-conv start : {P}")
    logger.info(f"Eval act start  : {eval_act_start}")
    logger.info(f"Measure pos     : {measure_pos}")
    logger.info(f"N sequences     : {N}")

    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype

    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    emit = build_emission_matrix(hmm)  # (n_tokens, n_states)

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_vocab = len(config.vocab_mapping)
    _first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_vocab, logger)

    # ── Data structures for results ───────────────────────────────────────────
    # patching[cond][k][layer] = list of (seq_idx, sub_idx, kl)
    patch_kl: dict[str, dict[int, dict[int, list[tuple[int, int, float]]]]] = {
        cond: {k: {l: [] for l in config.layer_indices} for k in config.k_values}
        for cond in PATCH_CONDITIONS
    }
    # steering[cond][k][layer] = list of (seq_idx, sub_idx, kl)
    steer_kl: dict[str, dict[int, dict[int, list[tuple[int, int, float]]]]] = {
        cond: {k: {l: [] for l in config.layer_indices} for k in config.k_values}
        for cond in STEER_CONDITIONS
    }
    # ablation[cond][layer] = list of (seq_idx, sub_idx, kl)
    ablation_kl: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        "belief": {l: [] for l in config.layer_indices},
        "random": {l: [] for l in config.layer_indices},
    }
    ablation_kl_to_clean: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        "belief": {l: [] for l in config.layer_indices},
        "random": {l: [] for l in config.layer_indices},
    }
    patch_kl_to_clean: dict[str, dict[int, dict[int, list[tuple[int, int, float]]]]] = {
        cond: {k: {l: [] for l in config.layer_indices} for k in config.k_values}
        for cond in PATCH_CONDITIONS
    }
    steer_kl_to_clean: dict[str, dict[int, dict[int, list[tuple[int, int, float]]]]] = {
        cond: {k: {l: [] for l in config.layer_indices} for k in config.k_values}
        for cond in STEER_CONDITIONS
    }
    baseline_kl: list[float] = []

    # ── Process sequences ─────────────────────────────────────────────────────
    for seq_i in range(N):
        logger.info(f"=== Sequence {seq_i}/{N-1} ===")
        seq_tokens = all_tokens[seq_i]    # (L,)
        seq_beliefs = all_beliefs[seq_i]  # (L+1, n_states)

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L

        # Load encoder + decoder weights for this sequence
        seq_dir = training_dir / f"seq_{seq_i}"
        encoders: dict[int, Probe] = {}
        decoders: dict[int, Decoder] = {}
        for layer in config.layer_indices:
            layer_dir = seq_dir / f"layer_{layer}"
            encoders[layer] = ProbeResult.load_weights_only(layer_dir).probe.to(device)
            decoders[layer] = DecoderResult.load(layer_dir).decoder.to(device)
            encoders[layer].eval()
            decoders[layer].eval()

        # ── Clean forward pass: cache all target layers ───────────────────────
        hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                llm_tokens, names_filter=hook_names, return_type=None,
            )
        clean_acts_per_layer: dict[int, np.ndarray] = {
            layer: clean_cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            for layer in config.layer_indices
        }
        del clean_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        # Baseline KL at measure_pos
        with torch.no_grad():
            logits_clean = model.run_with_hooks(llm_tokens, fwd_hooks=[], return_type="logits")
        probs_all = F.softmax(logits_clean[0, measure_pos, :].float(), dim=-1)
        probs_hmm = probs_all[mid_tok_ids].cpu().numpy()
        P_clean = probs_hmm / (probs_hmm.sum() + 1e-10)
        eta_baseline = seq_beliefs[measure_pos + 1].astype(np.float32)
        P_opt_baseline = (eta_baseline @ emit.T).astype(np.float32)
        baseline_kl.append(float(_kl(P_clean[None], P_opt_baseline[None])[0]))
        del logits_clean
        logger.info(f"  Baseline KL: {baseline_kl[-1]:.4f}")

        # ── Per-layer intervention loop ───────────────────────────────────────
        for layer in config.layer_indices:
            probe = encoders[layer]
            decoder = decoders[layer]

            # ─ Ablation ─────────────────────────────────────────────────────
            W = probe.W.detach().float().cpu().numpy()  # (d_model, n_states)
            P_belief = _orthogonal_projector(W).astype(np.float32)
            random_projs = [
                _random_projector(W.shape[0], W.shape[1], rng)
                for _ in range(config.n_random_ablation_draws)
            ]
            all_projs = [P_belief] + random_projs  # belief first, then randoms

            P_abl = _run_ablated(
                model, llm_tokens, layer, all_projs,
                measure_pos, mid_tok_ids, device, model_dtype,
            )  # (1 + n_random_draws, n_hmm)

            eta_opt = seq_beliefs[measure_pos + 1].astype(np.float32)
            P_opt = (eta_opt @ emit.T).astype(np.float32)

            kl_abl = _kl(P_abl, P_opt[None])
            ablation_kl["belief"][layer].append((seq_i, 0, float(kl_abl[0])))
            for draw_i, kl_v in enumerate(kl_abl[1:]):
                ablation_kl["random"][layer].append((seq_i, draw_i, float(kl_v)))

            kl_abl_to_clean = _kl(P_abl, P_clean[None])
            ablation_kl_to_clean["belief"][layer].append((seq_i, 0, float(kl_abl_to_clean[0])))
            for draw_i, kl_v in enumerate(kl_abl_to_clean[1:]):
                ablation_kl_to_clean["random"][layer].append((seq_i, draw_i, float(kl_v)))

            # ─ Per-k patching and steering ──────────────────────────────────
            for k in config.k_values:
                positions = list(range(L - k, L))
                clean_acts_k = clean_acts_per_layer[layer][positions, :]  # (k, d_model)

                # Pre-compute optimal + past-consistent + random target beliefs (k, n_states)
                eta_optimal = _optimal_targets(seq_beliefs, positions)
                eta_past_consistent_list = [
                    _past_consistent_targets(seq_beliefs, positions, hmm, rng)
                    for _ in range(config.n_past_consistent_draws)
                ]
                eta_random_list = [
                    _random_targets(positions, rng, n_states)
                    for _ in range(config.n_random_patch_draws)
                ]

                # Round-trip: decoder(encoder(act)) for each of the k positions
                round_trip_acts = _round_trip_targets(probe, clean_acts_k, decoder, device)  # (k, d_model)

                # Patching: target_acts (B, k, d_model)
                # Batch order: optimal, round_trip, n_past, n_random
                with torch.no_grad():
                    dec_optimal = decoder(torch.from_numpy(eta_optimal).float().to(device)).cpu()
                    dec_past = [
                        decoder(torch.from_numpy(eta_pc).float().to(device)).cpu()
                        for eta_pc in eta_past_consistent_list
                    ]
                    dec_random = [
                        decoder(torch.from_numpy(eta_r).float().to(device)).cpu()
                        for eta_r in eta_random_list
                    ]

                patch_target_acts = torch.stack(
                    [dec_optimal]
                    + [torch.from_numpy(round_trip_acts)]
                    + dec_past
                    + dec_random,
                    dim=0,
                )  # (1+1+n_past+n_rand, k, d_model)

                P_patch = _run_patched(
                    model, llm_tokens, layer, positions,
                    patch_target_acts, measure_pos, mid_tok_ids, device, model_dtype,
                )  # (batch, n_hmm)

                # KL for each patched output — target is the belief at the LAST position
                eta_last = seq_beliefs[measure_pos + 1].astype(np.float32)
                P_opt_last = (eta_last @ emit.T).astype(np.float32)
                kl_patch = _kl(P_patch, P_opt_last[None])

                b_idx = 0
                patch_kl["optimal"][k][layer].append((seq_i, 0, float(kl_patch[b_idx]))); b_idx += 1
                patch_kl["round_trip"][k][layer].append((seq_i, 0, float(kl_patch[b_idx]))); b_idx += 1
                for draw_i in range(config.n_past_consistent_draws):
                    patch_kl["past_consistent"][k][layer].append((seq_i, draw_i, float(kl_patch[b_idx]))); b_idx += 1
                for draw_i in range(config.n_random_patch_draws):
                    patch_kl["random"][k][layer].append((seq_i, draw_i, float(kl_patch[b_idx]))); b_idx += 1

                kl_patch_to_clean = _kl(P_patch, P_clean[None])
                b_idx = 0
                patch_kl_to_clean["optimal"][k][layer].append((seq_i, 0, float(kl_patch_to_clean[b_idx]))); b_idx += 1
                patch_kl_to_clean["round_trip"][k][layer].append((seq_i, 0, float(kl_patch_to_clean[b_idx]))); b_idx += 1
                for draw_i in range(config.n_past_consistent_draws):
                    patch_kl_to_clean["past_consistent"][k][layer].append((seq_i, draw_i, float(kl_patch_to_clean[b_idx]))); b_idx += 1
                for draw_i in range(config.n_random_patch_draws):
                    patch_kl_to_clean["random"][k][layer].append((seq_i, draw_i, float(kl_patch_to_clean[b_idx]))); b_idx += 1

                # Steering: delta = decoder(target) − decoder(encoder(act))
                with torch.no_grad():
                    source_beliefs = probe(torch.from_numpy(clean_acts_k).float().to(device))  # (k, n_states)
                    dec_source = decoder(source_beliefs)  # (k, d_model)

                    steer_optimal = (dec_optimal.to(device) - dec_source).unsqueeze(0).cpu()
                    steer_past = [
                        (dp.to(device) - dec_source).unsqueeze(0).cpu()
                        for dp in dec_past
                    ]
                    steer_random = [
                        (dr.to(device) - dec_source).unsqueeze(0).cpu()
                        for dr in dec_random
                    ]

                steer_deltas = torch.cat(
                    [steer_optimal] + steer_past + steer_random, dim=0
                )  # (1+n_past+n_rand, k, d_model)

                P_steer = _run_steered(
                    model, llm_tokens, layer, positions,
                    steer_deltas, measure_pos, mid_tok_ids, device, model_dtype,
                )

                kl_steer = _kl(P_steer, P_opt_last[None])

                s_idx = 0
                steer_kl["optimal"][k][layer].append((seq_i, 0, float(kl_steer[s_idx]))); s_idx += 1
                for draw_i in range(config.n_past_consistent_draws):
                    steer_kl["past_consistent"][k][layer].append((seq_i, draw_i, float(kl_steer[s_idx]))); s_idx += 1
                for draw_i in range(config.n_random_patch_draws):
                    steer_kl["random"][k][layer].append((seq_i, draw_i, float(kl_steer[s_idx]))); s_idx += 1

                kl_steer_to_clean = _kl(P_steer, P_clean[None])
                s_idx = 0
                steer_kl_to_clean["optimal"][k][layer].append((seq_i, 0, float(kl_steer_to_clean[s_idx]))); s_idx += 1
                for draw_i in range(config.n_past_consistent_draws):
                    steer_kl_to_clean["past_consistent"][k][layer].append((seq_i, draw_i, float(kl_steer_to_clean[s_idx]))); s_idx += 1
                for draw_i in range(config.n_random_patch_draws):
                    steer_kl_to_clean["random"][k][layer].append((seq_i, draw_i, float(kl_steer_to_clean[s_idx]))); s_idx += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        logger.info(f"  Seq {seq_i} done.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("Aggregating results ...")

    n_bl = len(baseline_kl)
    agg_baseline = {
        "mean": float(np.mean(baseline_kl)),
        "std": float(np.std(baseline_kl)),
        "stderr": float(np.std(baseline_kl) / max(np.sqrt(n_bl), 1.0)),
        "per_seq": [float(v) for v in baseline_kl],
    }

    agg_patch: dict[str, dict[int, dict[int, dict]]] = {
        cond: {
            k: {l: _agg_records(patch_kl[cond][k][l]) for l in config.layer_indices}
            for k in config.k_values
        }
        for cond in PATCH_CONDITIONS
    }
    agg_steer: dict[str, dict[int, dict[int, dict]]] = {
        cond: {
            k: {l: _agg_records(steer_kl[cond][k][l]) for l in config.layer_indices}
            for k in config.k_values
        }
        for cond in STEER_CONDITIONS
    }
    agg_ablation = {
        "belief": {l: _agg_records(ablation_kl["belief"][l]) for l in config.layer_indices},
        "random": {l: _agg_records(ablation_kl["random"][l]) for l in config.layer_indices},
    }
    agg_patch_to_clean: dict[str, dict[int, dict[int, dict]]] = {
        cond: {
            k: {l: _agg_records(patch_kl_to_clean[cond][k][l]) for l in config.layer_indices}
            for k in config.k_values
        }
        for cond in PATCH_CONDITIONS
    }
    agg_steer_to_clean: dict[str, dict[int, dict[int, dict]]] = {
        cond: {
            k: {l: _agg_records(steer_kl_to_clean[cond][k][l]) for l in config.layer_indices}
            for k in config.k_values
        }
        for cond in STEER_CONDITIONS
    }
    agg_ablation_to_clean = {
        "belief": {l: _agg_records(ablation_kl_to_clean["belief"][l]) for l in config.layer_indices},
        "random": {l: _agg_records(ablation_kl_to_clean["random"][l]) for l in config.layer_indices},
    }

    metrics = {
        "baseline": agg_baseline,
        "patching": {
            cond: {str(k): {str(l): v for l, v in by_layer.items()} for k, by_layer in by_k.items()}
            for cond, by_k in agg_patch.items()
        },
        "steering": {
            cond: {str(k): {str(l): v for l, v in by_layer.items()} for k, by_layer in by_k.items()}
            for cond, by_k in agg_steer.items()
        },
        "ablation": {
            cond: {str(l): v for l, v in by_layer.items()}
            for cond, by_layer in agg_ablation.items()
        },
        "patching_to_clean": {
            cond: {str(k): {str(l): v for l, v in by_layer.items()} for k, by_layer in by_k.items()}
            for cond, by_k in agg_patch_to_clean.items()
        },
        "steering_to_clean": {
            cond: {str(k): {str(l): v for l, v in by_layer.items()} for k, by_layer in by_k.items()}
            for cond, by_k in agg_steer_to_clean.items()
        },
        "ablation_to_clean": {
            cond: {str(l): v for l, v in by_layer.items()}
            for cond, by_layer in agg_ablation_to_clean.items()
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    baseline_mean = agg_baseline["mean"]

    # Patching: KL vs layer, one plot per k
    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_patch[cond][k] for cond in PATCH_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation patching: KL vs layer by condition",
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_kl_vs_layer_k{k}",
        )
        _plot_crossing(
            agg_to_opt=agg_patch["optimal"][k],
            agg_to_clean=agg_patch_to_clean["optimal"][k],
            baseline_to_opt=baseline_mean,
            layer_indices=config.layer_indices,
            k=k,
            condition="optimal",
            title="Activation patching",
            path=fig_dir / f"patching_crossing_optimal_k{k}",
        )
        _plot_causal_shift(
            agg_to_opt={cond: agg_patch[cond][k] for cond in PATCH_CONDITIONS},
            baseline_to_opt=baseline_mean,
            layer_indices=config.layer_indices,
            k=k,
            title="Activation patching",
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_causal_shift_k{k}",
        )

    # Steering: KL vs layer, one plot per k
    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_steer[cond][k] for cond in STEER_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation steering: KL vs layer by condition",
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_kl_vs_layer_k{k}",
        )
        _plot_crossing(
            agg_to_opt=agg_steer["optimal"][k],
            agg_to_clean=agg_steer_to_clean["optimal"][k],
            baseline_to_opt=baseline_mean,
            layer_indices=config.layer_indices,
            k=k,
            condition="optimal",
            title="Activation steering",
            path=fig_dir / f"steering_crossing_optimal_k{k}",
        )
        _plot_causal_shift(
            agg_to_opt={cond: agg_steer[cond][k] for cond in STEER_CONDITIONS},
            baseline_to_opt=baseline_mean,
            layer_indices=config.layer_indices,
            k=k,
            title="Activation steering",
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_causal_shift_k{k}",
        )

    # Heatmaps (largest k, optimal condition)
    k_max = max(config.k_values)
    _plot_heatmap(
        agg_patch["optimal"][k_max],
        config.layer_indices,
        title=f"Patching KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        path=fig_dir / f"heatmap_patching_optimal_k{k_max}",
    )
    _plot_heatmap(
        agg_steer["optimal"][k_max],
        config.layer_indices,
        title=f"Steering KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        path=fig_dir / f"heatmap_steering_optimal_k{k_max}",
    )

    # Ablation causal importance curve (KL to optimal)
    _plot_ablation_curve(
        agg_ablation["belief"], agg_ablation["random"],
        config.layer_indices,
        path=fig_dir / "ablation_causal_importance",
    )
    # Ablation KL to unintervened output (output perturbation)
    _plot_ablation_kl_to_output(
        agg_ablation_to_clean["belief"], agg_ablation_to_clean["random"],
        config.layer_indices,
        path=fig_dir / "ablation_kl_to_output",
    )

    # Round-trip comparison (H2A vs H2B), largest k
    _plot_roundtrip_comparison(
        {cond: agg_patch[cond][k_max] for cond in ["optimal", "round_trip"]},
        config.layer_indices,
        baseline_mean=baseline_mean,
        k=k_max,
        path=fig_dir / f"roundtrip_h2a_vs_h2b_k{k_max}",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
