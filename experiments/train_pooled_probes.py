#!/usr/bin/env python3
"""
Experiment 3 — Train pooled cross-sequence probes.

Pools post-convergence activations from n_train_sequences sequences, trains
one probe per layer on the concatenated data, then evaluates on n_test_sequences
held-out sequences.  Probe weights are the main deliverable consumed by SPAR-12.

The train/eval split is sequence-level: the probe never sees any token from the
eval sequences during training, giving an unambiguous measure of generalisation.

Usage:
    python experiments/train_pooled_probes.py experiments/configs/train_pooled_probes.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import (
    build_emission_matrix,
    compute_optimal_probs,
    get_device,
    get_model_probs,
    get_model_probs_projected,
    load_model,
    resolve_hmm_token_ids,
    setup_logging,
)
from hmm.hmm import Mess3HMM
from metrics.probe_metrics import (
    find_kl_threshold,
    pairwise_cosine_sim_matrix,
    pairwise_principal_angles,
)
from probes import Probe, ProbeInput, ProbeResult, train_probes_batched


@dataclass
class TrainPooledProbesConfig(ExperimentConfig):
    layer_indices: list[int]
    seq_length: int
    kl_params: dict[str, float]
    vocab_mapping: dict[str, int]
    n_train_sequences: int
    n_test_sequences: int
    max_probes_per_batch: int = 200
    n_ctx_override: int | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _evaluate_probe(
    probe: Probe,
    activations: np.ndarray,
    beliefs: np.ndarray,
) -> tuple[float, float]:
    probe_device = next(probe.parameters()).device
    probe.eval()
    with torch.no_grad():
        pred = probe(torch.from_numpy(activations).float().to(probe_device)).cpu().numpy()
    mse = float(np.mean((pred - beliefs) ** 2))
    ss_res = float(np.sum((pred - beliefs) ** 2))
    ss_tot = float(np.sum((beliefs - beliefs.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    return mse, r2


def _metric_heatmap(
    z: np.ndarray,
    layer_indices: list[int],
    n_train: int,
    title: str,
    colorbar_title: str,
    path: Path,
    colorscale: str = "Viridis",
    zrange: tuple[float, float] | None = None,
) -> None:
    n_layers, n_seqs = z.shape
    x_labels = [f"T{i}" for i in range(n_train)] + [
        f"E{i}" for i in range(n_seqs - n_train)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(range(n_seqs)),
            y=list(range(n_layers)),
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, thickness=14, len=0.8),
            zmin=zrange[0] if zrange else None,
            zmax=zrange[1] if zrange else None,
        )
    )

    fig.add_shape(
        type="rect",
        x0=n_train - 0.5,
        x1=n_seqs - 0.5,
        y0=-0.5,
        y1=n_layers - 0.5,
        xref="x",
        yref="y",
        fillcolor="rgba(255,80,80,0.10)",
        line_width=0,
    )
    fig.add_shape(
        type="line",
        x0=n_train - 0.5,
        x1=n_train - 0.5,
        y0=-0.5,
        y1=n_layers - 0.5,
        xref="x",
        yref="y",
        line=dict(color="red", width=2.5, dash="dash"),
    )
    fig.add_annotation(
        x=n_train - 0.35,
        y=n_layers - 0.5,
        xref="x",
        yref="y",
        text="eval →",
        showarrow=False,
        font=dict(color="red", size=11),
        yanchor="top",
        xanchor="left",
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_seqs)),
            ticktext=x_labels,
            title="Sequence (T=train, E=eval)",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_layers)),
            ticktext=[str(l) for l in layer_indices],
            title="Layer",
        ),
        height=440,
        width=920,
        margin=dict(t=80, b=70, l=70, r=110),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def _summary_line_plot(
    train_means: list[float],
    eval_means: list[float],
    train_stds: list[float],
    eval_stds: list[float],
    layer_indices: list[int],
    y_title: str,
    title: str,
    path: Path,
) -> None:
    layers = [str(l) for l in layer_indices]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=train_means,
            error_y=dict(type="data", array=train_stds, visible=True),
            name="Train",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=eval_means,
            error_y=dict(type="data", array=eval_stds, visible=True),
            name="Eval",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=y_title,
        height=420,
        width=720,
        margin=dict(t=70, b=60, l=70, r=40),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train pooled probes")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-user",
        type=str,
        default=None,
        help="Override output_user from the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config, TrainPooledProbesConfig)
    apply_runtime_overrides(config, output_user=args.output_user)
    N_train: int = config.n_train_sequences
    N_test: int = config.n_test_sequences
    N_total: int = N_train + N_test
    device = get_device()

    out_dir = setup_output_dir(config)
    (out_dir / "probes" / "pooled").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, name="pooled_probes")

    logger.info(f"Output dir      : {out_dir}")
    logger.info(f"Device          : {device}")
    logger.info(f"Train sequences : {N_train}")
    logger.info(f"Eval sequences  : {N_test}")
    logger.info(f"Layers          : {config.layer_indices}")
    logger.info(f"Config          : {config}")

    seq_seeds: list[int] = [
        int(torch.randint(2**31, (1,)).item()) for _ in range(N_total)
    ]

    # ── Model + HMM ────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    n_ctx = model.cfg.n_ctx
    if config.seq_length >= n_ctx:
        raise ValueError(
            f"seq_length={config.seq_length} >= model n_ctx={n_ctx}. "
            f"Set seq_length < {n_ctx} (accounting for BOS token)."
        )

    hmm = Mess3HMM()
    p = config.hmm.process_params
    if "x" in p and "alpha" in p:
        hmm.create_hmm(p["x"], p["alpha"])
        logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    emit = build_emission_matrix(hmm)

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(
        model, idx_to_token, n_hmm_tokens, logger
    )

    use_projected: bool = bool(config.kl_params.get("include_junk", False))
    logger.info(
        f"KL variant: {'projected (include_junk=True)' if use_projected else 'standard'}"
    )

    L = config.seq_length
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Phase 1: Forward passes ────────────────────────────────────────────────
    all_acts: list[dict[int, np.ndarray]] = []
    all_beliefs: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []
    all_optimal_probs: list[np.ndarray] = []
    all_model_probs: list[np.ndarray] = []
    all_kl_thresholds: list[int] = []
    all_kl_crossed: list[bool] = []

    for seq_idx in range(N_total):
        split_label = "train" if seq_idx < N_train else "eval"
        seq_seed = seq_seeds[seq_idx]
        torch.manual_seed(seq_seed)
        logger.info(
            f"Sequence {seq_idx + 1}/{N_total} [{split_label}]: seed={seq_seed}"
        )

        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
        seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy()

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=True, truncate=False)
        assert llm_tokens.shape[1] == L + 1, (
            f"Expected {L + 1} LLM tokens, got {llm_tokens.shape[1]}."
        )

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type="logits",
            )

        model_probs = (
            get_model_probs_projected(logits, first_tok_id, mid_tok_ids, L)
            if use_projected
            else get_model_probs(logits, first_tok_id, mid_tok_ids, L)
        )
        optimal_probs = compute_optimal_probs(seq_beliefs, emit)

        kl_t, kl_crossed = find_kl_threshold(
            model_probs, optimal_probs, **config.kl_params, logger=logger
        )
        if kl_crossed:
            logger.info(f"  KL t* = {kl_t}/{L}")
        else:
            logger.warning(f"  KL t* = {kl_t}/{L} (argmin fallback, threshold not crossed)")

        seq_acts = {
            layer: cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            for layer in config.layer_indices
        }

        del logits, cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_acts.append(seq_acts)
        all_beliefs.append(seq_beliefs)
        all_tokens.append(seq_tokens)
        all_optimal_probs.append(optimal_probs)
        all_model_probs.append(model_probs)
        all_kl_thresholds.append(kl_t)
        all_kl_crossed.append(kl_crossed)

    with open(out_dir / "kl_thresholds.json", "w") as f:
        json.dump(
            {
                "thresholds": all_kl_thresholds,
                "crossed": all_kl_crossed,
                "mean": float(np.mean(all_kl_thresholds)),
                "n_train": N_train,
                "n_test": N_test,
                "seq_seeds": seq_seeds,
            },
            f,
            indent=2,
        )

    # ── Phase 2a: Train pooled probe ───────────────────────────────────────────
    logger.info(
        f"Phase 2a: pooling post-convergence data from {N_train} train sequences ..."
    )
    pooled_inputs: dict[int, ProbeInput] = {}
    for layer in config.layer_indices:
        acts_list, beliefs_list, tokens_list, opt_list, model_list = [], [], [], [], []
        for seq_idx in range(N_train):
            post_start = max(all_kl_thresholds[seq_idx], 1)
            acts_list.append(all_acts[seq_idx][layer][post_start:])
            beliefs_list.append(all_beliefs[seq_idx][post_start:])
            tokens_list.append(all_tokens[seq_idx][post_start - 1:])
            opt_list.append(all_optimal_probs[seq_idx][post_start - 1:])
            model_list.append(all_model_probs[seq_idx][post_start - 1:])
        pooled_inputs[layer] = ProbeInput(
            np.concatenate(acts_list, axis=0),
            np.concatenate(beliefs_list, axis=0),
            np.concatenate(tokens_list, axis=0),
            np.concatenate(opt_list, axis=0),
            np.concatenate(model_list, axis=0),
        )
        n_pts = pooled_inputs[layer].activations.shape[0]
        logger.info(f"  Layer {layer}: {n_pts} pooled training points")

    logger.info("Training pooled probes ...")
    pooled_results: dict[int, ProbeResult] = train_probes_batched(
        pooled_inputs, max_probes_per_batch=config.max_probes_per_batch
    )

    # ── Phase 2b: Train per-sequence probes (for subspace comparison) ──────────
    logger.info(
        f"Phase 2b: training per-sequence probes "
        f"({N_total} seqs × {len(config.layer_indices)} layers) ..."
    )
    per_seq_inputs: dict[tuple[int, int], ProbeInput] = {}
    for layer in config.layer_indices:
        for seq_idx in range(N_total):
            post_start = max(all_kl_thresholds[seq_idx], 1)
            per_seq_inputs[(layer, seq_idx)] = ProbeInput(
                all_acts[seq_idx][layer][post_start:],
                all_beliefs[seq_idx][post_start:],
                all_tokens[seq_idx][post_start - 1:],
                all_optimal_probs[seq_idx][post_start - 1:],
                all_model_probs[seq_idx][post_start - 1:],
            )

    per_seq_results_flat: dict[tuple[int, int], ProbeResult] = train_probes_batched(
        per_seq_inputs, max_probes_per_batch=config.max_probes_per_batch
    )
    per_seq_results: dict[int, list[ProbeResult]] = {
        layer: [per_seq_results_flat[(layer, i)] for i in range(N_total)]
        for layer in config.layer_indices
    }

    # ── Phase 3: Evaluate ──────────────────────────────────────────────────────
    logger.info("Phase 3: evaluating pooled probe on all sequences ...")

    n_layers = len(config.layer_indices)
    mse_mat = np.zeros((n_layers, N_total))
    r2_mat = np.zeros((n_layers, N_total))
    cosine_sim_mat = np.zeros((n_layers, N_total))
    theta1_mat = np.zeros((n_layers, N_total))
    theta_mean2_mat = np.zeros((n_layers, N_total))

    metrics: dict[str, dict] = {}

    for layer_idx, layer in enumerate(config.layer_indices):
        pooled_pr = pooled_results[layer]

        for seq_idx in range(N_total):
            post_start = max(all_kl_thresholds[seq_idx], 1)
            acts = all_acts[seq_idx][layer][post_start:]
            beliefs = all_beliefs[seq_idx][post_start:]
            mse, r2 = _evaluate_probe(pooled_pr.probe, acts, beliefs)
            mse_mat[layer_idx, seq_idx] = mse
            r2_mat[layer_idx, seq_idx] = r2

        # Cosine sim and principal angles: pooled probe (row 0) vs each per-seq probe
        all_probes_this_layer: list[ProbeResult] = [pooled_pr] + per_seq_results[layer]
        cos_sim_full = pairwise_cosine_sim_matrix(all_probes_this_layer)
        angles_full, _, _ = pairwise_principal_angles(all_probes_this_layer)

        cosine_sim_mat[layer_idx] = cos_sim_full[0, 1:]
        theta1_mat[layer_idx] = angles_full[0, 1:, 0]
        theta_mean2_mat[layer_idx] = angles_full[0, 1:, :2].mean(axis=-1)

        logger.info(
            f"Layer {layer}: "
            f"pooled_internal_mse={pooled_pr.test_mse:.4f}  "
            f"cross_mse train={mse_mat[layer_idx, :N_train].mean():.4f}"
            f"±{mse_mat[layer_idx, :N_train].std():.4f} "
            f"eval={mse_mat[layer_idx, N_train:].mean():.4f}"
            f"±{mse_mat[layer_idx, N_train:].std():.4f}  "
            f"R² train={r2_mat[layer_idx, :N_train].mean():.3f} "
            f"eval={r2_mat[layer_idx, N_train:].mean():.3f}  "
            f"cos_sim train={cosine_sim_mat[layer_idx, :N_train].mean():.3f} "
            f"eval={cosine_sim_mat[layer_idx, N_train:].mean():.3f}  "
            f"θ₁ train={theta1_mat[layer_idx, :N_train].mean():.1f}° "
            f"eval={theta1_mat[layer_idx, N_train:].mean():.1f}°"
        )

        metrics[str(layer)] = {
            "pooled_probe_internal_test_mse": float(pooled_pr.test_mse),
            "train": {
                "mse_per_seq": [float(mse_mat[layer_idx, i]) for i in range(N_train)],
                "r2_per_seq": [float(r2_mat[layer_idx, i]) for i in range(N_train)],
                "cosine_sim_per_seq": [
                    float(cosine_sim_mat[layer_idx, i]) for i in range(N_train)
                ],
                "theta1_deg_per_seq": [
                    float(theta1_mat[layer_idx, i]) for i in range(N_train)
                ],
                "theta_mean2_deg_per_seq": [
                    float(theta_mean2_mat[layer_idx, i]) for i in range(N_train)
                ],
                "mse_mean": float(mse_mat[layer_idx, :N_train].mean()),
                "mse_std": float(mse_mat[layer_idx, :N_train].std()),
                "r2_mean": float(r2_mat[layer_idx, :N_train].mean()),
                "r2_std": float(r2_mat[layer_idx, :N_train].std()),
                "cosine_sim_mean": float(cosine_sim_mat[layer_idx, :N_train].mean()),
                "theta1_mean_deg": float(theta1_mat[layer_idx, :N_train].mean()),
                "theta_mean2_mean_deg": float(theta_mean2_mat[layer_idx, :N_train].mean()),
            },
            "eval": {
                "mse_per_seq": [
                    float(mse_mat[layer_idx, i]) for i in range(N_train, N_total)
                ],
                "r2_per_seq": [
                    float(r2_mat[layer_idx, i]) for i in range(N_train, N_total)
                ],
                "cosine_sim_per_seq": [
                    float(cosine_sim_mat[layer_idx, i]) for i in range(N_train, N_total)
                ],
                "theta1_deg_per_seq": [
                    float(theta1_mat[layer_idx, i]) for i in range(N_train, N_total)
                ],
                "theta_mean2_deg_per_seq": [
                    float(theta_mean2_mat[layer_idx, i]) for i in range(N_train, N_total)
                ],
                "mse_mean": float(mse_mat[layer_idx, N_train:].mean()),
                "mse_std": float(mse_mat[layer_idx, N_train:].std()),
                "r2_mean": float(r2_mat[layer_idx, N_train:].mean()),
                "r2_std": float(r2_mat[layer_idx, N_train:].std()),
                "cosine_sim_mean": float(cosine_sim_mat[layer_idx, N_train:].mean()),
                "theta1_mean_deg": float(theta1_mat[layer_idx, N_train:].mean()),
                "theta_mean2_mean_deg": float(theta_mean2_mat[layer_idx, N_train:].mean()),
            },
        }

    # ── Save probe weights ─────────────────────────────────────────────────────
    logger.info("Saving pooled probe weights ...")
    for layer in config.layer_indices:
        pooled_results[layer].save(out_dir / "probes" / "pooled" / f"layer_{layer}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    fig_dir = out_dir / "figures"

    _metric_heatmap(
        mse_mat,
        config.layer_indices,
        N_train,
        title="Pooled probe MSE — per layer × sequence",
        colorbar_title="MSE",
        path=fig_dir / "mse_heatmap",
        colorscale="Viridis_r",
    )

    _metric_heatmap(
        r2_mat,
        config.layer_indices,
        N_train,
        title="Pooled probe R² — per layer × sequence",
        colorbar_title="R²",
        path=fig_dir / "r2_heatmap",
        colorscale="Blues",
        zrange=(0.0, 1.0),
    )

    _metric_heatmap(
        cosine_sim_mat,
        config.layer_indices,
        N_train,
        title="Cosine similarity: pooled probe vs per-seq probe — per layer × sequence",
        colorbar_title="cos sim",
        path=fig_dir / "cosine_sim_heatmap",
        colorscale="Blues",
        zrange=(0.0, 1.0),
    )

    _metric_heatmap(
        theta1_mat,
        config.layer_indices,
        N_train,
        title="First principal angle θ₁: pooled probe vs per-seq probe — per layer × sequence",
        colorbar_title="θ₁ (°)",
        path=fig_dir / "theta1_heatmap",
        colorscale="Viridis",
    )

    _metric_heatmap(
        theta_mean2_mat,
        config.layer_indices,
        N_train,
        title="Mean principal angle mean(θ₁,θ₂): pooled probe vs per-seq probe — per layer × sequence",
        colorbar_title="mean(θ₁,θ₂) (°)",
        path=fig_dir / "theta_mean2_heatmap",
        colorscale="Viridis",
    )

    _summary_line_plot(
        [metrics[str(l)]["train"]["mse_mean"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["mse_mean"] for l in config.layer_indices],
        [metrics[str(l)]["train"]["mse_std"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["mse_std"] for l in config.layer_indices],
        config.layer_indices,
        y_title="MSE",
        title="Pooled probe cross-sequence MSE by layer",
        path=fig_dir / "mse_per_layer",
    )

    _summary_line_plot(
        [metrics[str(l)]["train"]["r2_mean"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["r2_mean"] for l in config.layer_indices],
        [metrics[str(l)]["train"]["r2_std"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["r2_std"] for l in config.layer_indices],
        config.layer_indices,
        y_title="R²",
        title="Pooled probe R² by layer",
        path=fig_dir / "r2_per_layer",
    )

    _summary_line_plot(
        [metrics[str(l)]["train"]["cosine_sim_mean"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["cosine_sim_mean"] for l in config.layer_indices],
        [0.0] * n_layers,
        [0.0] * n_layers,
        config.layer_indices,
        y_title="Cosine similarity",
        title="Cosine similarity: pooled probe vs per-seq probes by layer",
        path=fig_dir / "cosine_sim_per_layer",
    )

    _summary_line_plot(
        [metrics[str(l)]["train"]["theta1_mean_deg"] for l in config.layer_indices],
        [metrics[str(l)]["eval"]["theta1_mean_deg"] for l in config.layer_indices],
        [0.0] * n_layers,
        [0.0] * n_layers,
        config.layer_indices,
        y_title="θ₁ (°)",
        title="First principal angle θ₁: pooled probe vs per-seq probes by layer",
        path=fig_dir / "theta1_per_layer",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
