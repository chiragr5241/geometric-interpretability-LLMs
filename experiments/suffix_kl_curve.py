#!/usr/bin/env python3
"""SPAR-27/32 — Suffix-length KL curve: theoretical prediction quality vs. context depth.

For each suffix length K = 1..k_max, a "forgetful" predictor that only has access
to the last K tokens is compared to the Bayesian-optimal predictor with full history.
The cost of limited context is KL(p_full || p_suffix), plotted as a function of K.

Supports two config modes:
  - Single process: set ``hmm.process_name`` / ``hmm.process_params`` as before.
  - Multi-process: set ``processes`` to a list of dicts with keys
    ``process_name``, ``process_params``, and optional ``label``.
    All processes are overlaid on the same pair of figures.

This is a pure HMM computation — no model inference needed.

Usage:
    python experiments/suffix_kl_curve.py experiments/configs/suffix_kl_curve_mess3.yaml
    python experiments/suffix_kl_curve.py experiments/configs/suffix_kl_curve_fern.yaml
    python experiments/suffix_kl_curve.py experiments/configs/suffix_kl_curve_comparison.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import build_emission_matrix, get_device, setup_logging
from hmm import build_process_hmm

# Tableau-10 colours (line, semi-transparent fill)
_COLORS: list[tuple[str, str]] = [
    ("#1f77b4", "rgba(31,119,180,0.15)"),
    ("#ff7f0e", "rgba(255,127,14,0.15)"),
    ("#2ca02c", "rgba(44,160,44,0.15)"),
    ("#d62728", "rgba(214,39,40,0.15)"),
    ("#9467bd", "rgba(148,103,189,0.15)"),
    ("#8c564b", "rgba(140,86,75,0.15)"),
    ("#e377c2", "rgba(227,119,194,0.15)"),
    ("#7f7f7f", "rgba(127,127,127,0.15)"),
]


@dataclass
class SuffixKLCurveConfig(ExperimentConfig):
    n_sequences: int
    seq_length: int
    k_max: int
    n_positions_per_seq: int
    model_empirical_kl: float | None = field(default=None)
    random_seed: int = field(default=42)
    # Multi-process mode: list of dicts with keys process_name, process_params, label (opt).
    # If set, the hmm field is ignored.
    processes: list[dict] | None = field(default=None)


def _kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """KL(P || Q) per row, both shape (..., n_tokens). Returns (...,)."""
    return (P * np.log(np.clip(P, 1e-300, None) / np.clip(Q, 1e-300, None))).sum(axis=-1)


def _belief_update_f64(T_3d: np.ndarray, tokens_batch: np.ndarray) -> np.ndarray:
    """Run Bayesian filtering in float64 from a uniform prior.

    Parameters
    ----------
    T_3d : np.ndarray
        Transition tensor, shape (n_tokens, n_states, n_states), float64.
    tokens_batch : np.ndarray
        Integer token indices, shape (M, K).

    Returns
    -------
    np.ndarray
        Beliefs after all K observations, shape (M, n_states), float64.
    """
    M, K = tokens_batch.shape
    n_states = T_3d.shape[1]
    belief = np.full((M, n_states), 1.0 / n_states, dtype=np.float64)
    for k in range(K):
        tok = tokens_batch[:, k]          # (M,)
        belief = np.einsum("mij,mj->mi", T_3d[tok], belief)
        belief /= belief.sum(axis=-1, keepdims=True) + 1e-30
    return belief


def compute_suffix_kl(
    hmm: Mess3HMM,
    tokens: np.ndarray,
    full_beliefs: np.ndarray,
    emit: np.ndarray,
    k_max: int,
    n_positions_per_seq: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """Compute mean KL(p_full || p_suffix) for each suffix length K = 1..k_max.

    Parameters
    ----------
    tokens : np.ndarray
        Shape (N, L) — integer HMM token indices.
    full_beliefs : np.ndarray
        Shape (N, L+1, n_states) — full-history beliefs from compute_belief_state.
        Index t corresponds to the belief after observing tokens[:, t-1] (t=0 is prior).
    emit : np.ndarray
        Shape (n_tokens, n_states) — emission matrix from build_emission_matrix.
    k_max : int
        Maximum suffix length to evaluate.
    n_positions_per_seq : int
        Number of positions to sample per sequence per K.
    rng : np.random.Generator

    Returns
    -------
    dict mapping K -> {"mean": float, "stderr": float, "n_pairs": int}
    """
    N, L = tokens.shape
    results: dict[int, dict] = {}

    # Use float64 throughout to avoid float32 precision floor (~4e-9).
    T_3d_f64  = hmm.T_3d_matrix.cpu().numpy().astype(np.float64)
    emit_f64  = emit.astype(np.float64)
    beliefs64 = full_beliefs.astype(np.float64)

    for K in range(1, k_max + 1):
        # Valid positions: need at least K tokens before and including position t.
        # tokens[:, t-K:t] is the suffix; full_beliefs[:, t] is the belief after t tokens.
        # t ranges from K-1 to L-1 (0-indexed), giving (L - K + 1) valid positions.
        valid_positions = np.arange(K - 1, L)
        n_sample = min(n_positions_per_seq, len(valid_positions))

        # For each sequence, sample n_sample positions uniformly without replacement.
        all_suffixes: list[np.ndarray] = []
        all_full_p:   list[np.ndarray] = []

        for i in range(N):
            chosen = rng.choice(valid_positions, size=n_sample, replace=False)
            for t in chosen:
                all_suffixes.append(tokens[i, t - K + 1 : t + 1])
                b_full = beliefs64[i, t + 1]      # (n_states,) after t+1 tokens
                all_full_p.append(b_full @ emit_f64.T)

        suffixes_arr = np.stack(all_suffixes)   # (M, K)
        p_full_arr   = np.stack(all_full_p)     # (M, n_tokens)
        M = suffixes_arr.shape[0]

        suffix_beliefs = _belief_update_f64(T_3d_f64, suffixes_arr)  # (M, n_states), float64
        p_suffix = suffix_beliefs @ emit_f64.T  # (M, n_tokens)

        # Renormalise (should already sum to ~1, but guard against numerics).
        p_full_arr = p_full_arr / (p_full_arr.sum(axis=-1, keepdims=True) + 1e-30)
        p_suffix   = p_suffix   / (p_suffix.sum(axis=-1, keepdims=True) + 1e-30)

        kl_fwd = _kl(p_full_arr, p_suffix)    # KL(p_full || p_suffix), shape (M,)
        kl_rev = _kl(p_suffix, p_full_arr)    # KL(p_suffix || p_full), shape (M,)

        results[K] = {
            "forward": {
                "mean":   float(np.mean(kl_fwd)),
                "stderr": float(np.std(kl_fwd) / np.sqrt(M)),
            },
            "reverse": {
                "mean":   float(np.mean(kl_rev)),
                "stderr": float(np.std(kl_rev) / np.sqrt(M)),
            },
            "n_pairs": M,
        }

    return results


def _kl_trace(
    ks: list[int],
    stats: list[dict],
    name: str,
    color: str,
    fill_color: str,
) -> list:
    means = [s["mean"]   for s in stats]
    errs  = [s["stderr"] for s in stats]
    upper = [m + e for m, e in zip(means, errs)]
    lower = [m - e for m, e in zip(means, errs)]
    return [
        go.Scatter(
            x=ks + ks[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
        ),
        go.Scatter(
            x=ks,
            y=means,
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ),
    ]


def _save_plot(
    fig: go.Figure,
    title: str,
    subtitle: str,
    model_empirical_kl: float | None,
    path: Path,
) -> None:
    if model_empirical_kl is not None:
        fig.add_hline(
            y=model_empirical_kl,
            line=dict(color="#d62728", dash="dash", width=1.5),
            annotation_text=f"Model empirical KL = {model_empirical_kl:.4f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        xaxis_title="Suffix length K",
        yaxis=dict(title="KL [nats]", type="log"),
        height=480,
        width=760,
        margin=dict(t=90, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def plot_suffix_kl_curves(
    all_results: list[tuple[str, dict[int, dict]]],
    model_empirical_kl: float | None,
    fig_dir: Path,
) -> None:
    """Overlay suffix-KL curves for one or more processes on a pair of figures."""
    fig_fwd = go.Figure()
    fig_rev = go.Figure()

    for i, (label, results) in enumerate(all_results):
        color, fill_color = _COLORS[i % len(_COLORS)]
        ks = sorted(results.keys())
        fwd_stats = [results[k]["forward"] for k in ks]
        rev_stats  = [results[k]["reverse"]  for k in ks]
        for trace in _kl_trace(ks, fwd_stats, label, color, fill_color):
            fig_fwd.add_trace(trace)
        for trace in _kl_trace(ks, rev_stats, label, color, fill_color):
            fig_rev.add_trace(trace)

    multi = len(all_results) > 1
    base_title = "Suffix-length KL curve" + (" comparison" if multi else "")
    _save_plot(
        fig_fwd,
        base_title,
        "KL(p_full ‖ p_suffix) — mean ± 1 stderr across sampled (sequence, position) pairs",
        model_empirical_kl,
        fig_dir / "suffix_kl_forward",
    )
    _save_plot(
        fig_rev,
        base_title,
        "KL(p_suffix ‖ p_full) — mean ± 1 stderr across sampled (sequence, position) pairs",
        model_empirical_kl,
        fig_dir / "suffix_kl_reverse",
    )


def _make_label(process_name: str, process_params: dict) -> str:
    parts = ", ".join(f"{k}={v}" for k, v in process_params.items())
    return f"{process_name} ({parts})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Suffix-length KL curve experiment")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-user", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config, SuffixKLCurveConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    out_dir = setup_output_dir(config)
    logger  = setup_logging(out_dir, name="suffix_kl")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir         : {out_dir}")
    logger.info(f"N sequences        : {config.n_sequences}")
    logger.info(f"Seq length         : {config.seq_length}")
    logger.info(f"K max              : {config.k_max}")
    logger.info(f"Positions per seq  : {config.n_positions_per_seq}")
    logger.info(f"Model empirical KL : {config.model_empirical_kl}")

    torch.manual_seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)

    # ── Build the list of process specs ──────────────────────────────────────
    if config.processes:
        specs = config.processes
        logger.info(f"Multi-process mode: {len(specs)} processes")
    else:
        p = config.hmm
        specs = [{"process_name": p.process_name, "process_params": p.process_params}]

    # ── Run each process ──────────────────────────────────────────────────────
    all_results: list[tuple[str, dict[int, dict]]] = []
    all_metrics: dict = {}

    for spec in specs:
        pname   = spec["process_name"]
        pparams = spec["process_params"]
        label   = spec.get("label") or _make_label(pname, pparams)

        logger.info(f"── {label} ──")
        hmm  = build_process_hmm(pname, pparams)
        emit = build_emission_matrix(hmm)

        logger.info(f"  Generating {config.n_sequences} sequences of length {config.seq_length} ...")
        tokens_torch, _, _ = hmm.generate_dataset(config.n_sequences, config.seq_length)
        full_beliefs_torch  = hmm.compute_belief_state(tokens_torch)
        tokens       = tokens_torch.cpu().numpy()
        full_beliefs = full_beliefs_torch.cpu().numpy()
        logger.info("  Sequences and beliefs computed.")

        logger.info(f"  Computing suffix KL for K = 1..{config.k_max} ...")
        results = compute_suffix_kl(
            hmm, tokens, full_beliefs, emit,
            k_max=config.k_max,
            n_positions_per_seq=config.n_positions_per_seq,
            rng=rng,
        )
        for K in sorted(results):
            r = results[K]
            logger.info(
                f"    K={K:2d}  fwd={r['forward']['mean']:.3e}  "
                f"rev={r['reverse']['mean']:.3e}  n={r['n_pairs']}"
            )

        all_results.append((label, results))
        all_metrics[label] = {str(K): v for K, v in results.items()}

    # ── Save metrics ──────────────────────────────────────────────────────────
    if config.model_empirical_kl is not None:
        all_metrics["model_empirical_kl"] = config.model_empirical_kl
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_suffix_kl_curves(all_results, config.model_empirical_kl, fig_dir)
    logger.info("Saved suffix_kl_forward.png and suffix_kl_reverse.png")
    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
