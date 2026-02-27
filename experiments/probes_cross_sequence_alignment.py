#!/usr/bin/env python3
"""
Experiment 2 — Cross-sequence subspace alignment.

Trains one probe per sequence (both on full-sequence and post-KL-threshold
activations) then evaluates each probe on every other sequence's activations,
producing M×M cross-MSE and subspace similarity matrices per layer.

Usage:
    python experiments/probes_cross_sequence_alignment.py experiments/configs/probes_cross_sequence_alignment.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import dataclass

from experiment import ExperimentConfig, load_config, setup_output_dir


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
from metrics.probe_metrics import cross_mse_matrix, find_kl_threshold, pairwise_cosine_sim_matrix
from probes import ProbeInput, ProbeResult, train_probes_batched
from visualization import plot_belief_grid

@dataclass
class ProbesCrossSequenceAlignmentConfig(ExperimentConfig):
    layer_indices: list[int]
    seq_length: int
    kl_params: dict[str, float]
    vocab_mapping: dict[str, int]
    n_sequences: int
    max_probes_per_batch: int = 200
    n_ctx_override: int | None = None
    save_probes: bool = False

# ── Plotting helpers ───────────────────────────────────────────────────────────

def _heatmap_grid(
    mats_per_layer: dict[int, np.ndarray],
    layer_indices: list[int],
    title: str,
    path: Path,
    colorscale: str = "Viridis",
    zrange: tuple[float, float] | None = None,
) -> None:
    n_cols = 4
    n_layers = len(layer_indices)
    n_rows = (n_layers + n_cols - 1) // n_cols

    subplot_titles = [f"Layer {l}" for l in layer_indices]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=max(0.04, 0.15 / max(n_rows - 1, 1)),
    )

    last_idx = len(layer_indices) - 1
    for idx, layer in enumerate(layer_indices):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        mat = mats_per_layer[layer]
        is_last = idx == last_idx
        kwargs: dict = dict(
            colorscale=colorscale,
            showscale=is_last,
        )
        if zrange is not None:
            kwargs["zmin"], kwargs["zmax"] = zrange
        if is_last:
            kwargs["colorbar"] = dict(len=0.35, lenmode="fraction", thickness=12, x=1.01)
        fig.add_trace(go.Heatmap(z=mat, **kwargs), row=row, col=col)

    cell_px = 180
    fig.update_layout(
        title=title,
        height=cell_px * n_rows + 80,
        width=cell_px * n_cols + 40,
        margin=dict(t=80, b=20, l=20, r=60),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def plot_mse_ratio(
    results: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers = [str(l) for l in layer_indices]
    full_ratios = [results[l]["full_ratio"] for l in layer_indices]
    post_ratios = [results[l]["post_ratio"] for l in layer_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=full_ratios, name="Full probes", mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=post_ratios, name="Post-threshold probes", mode="lines+markers",
    ))
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="gray",
        annotation_text="1.0 (off-diag = diag)", annotation_position="top right",
    )
    fig.update_layout(
        title=(
            "Off-diagonal / diagonal cross-MSE ratio per layer<br>"
            "<sup>Ratio ≈ 1 → probe generalises across sequences (consistent subspace)</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Off-diag / diag MSE ratio",
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def plot_diag_vs_offdiag(
    results: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers = [str(l) for l in layer_indices]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Full probes", "Post-threshold probes"])
    for col, prefix in enumerate(["full", "post"], start=1):
        diag = [results[l][f"{prefix}_diag_mean"] for l in layer_indices]
        off = [results[l][f"{prefix}_off_diag_mean"] for l in layer_indices]
        fig.add_trace(go.Scatter(x=layers, y=diag, name="Diagonal (own seq)", mode="lines+markers"), row=1, col=col)
        fig.add_trace(go.Scatter(x=layers, y=off, name="Off-diagonal (cross seq)", mode="lines+markers"), row=1, col=col)

    fig.update_layout(
        title="Diagonal vs. off-diagonal cross-MSE per layer",
        yaxis_title="MSE",
        showlegend=True,
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1], ProbesCrossSequenceAlignmentConfig)
    M: int = config.n_sequences
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="exp2")

    logger.info(f"Output dir  : {out_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"N sequences : {M}")
    logger.info(f"Config      : {config}")

    # ── Sequence seeds — captured before model loading, which resets torch RNG ─
    seq_seeds: list[int] = [int(torch.randint(2**31, (1,)).item()) for _ in range(M)]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    n_ctx = model.cfg.n_ctx
    if config.seq_length is not None and config.seq_length >= n_ctx:
        raise ValueError(
            f"seq_length={config.seq_length} >= model n_ctx={n_ctx}. "
            f"Set seq_length < {n_ctx} (accounting for BOS token)."
        )

    # ── HMM ───────────────────────────────────────────────────────────────────
    hmm = Mess3HMM()
    p = config.hmm.process_params
    if "x" in p and "alpha" in p:
        hmm.create_hmm(p["x"], p["alpha"])
        logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    emit = build_emission_matrix(hmm)   # (n_tokens, n_states)

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(
        model, idx_to_token, n_hmm_tokens, logger
    )

    use_projected: bool = bool(config.kl_params.get("include_junk", False))
    logger.info(f"KL variant  : {'projected (include_junk=True)' if use_projected else 'standard (include_junk=False)'} — KL(optimal ‖ model)")

    L = config.seq_length
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Phase 1: Forward passes ────────────────────────────────────────────────
    # all_acts[seq_idx][layer] = (L+1, d_model) float32 numpy array
    all_acts: list[dict[int, np.ndarray]] = []
    all_beliefs: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []
    all_optimal_probs: list[np.ndarray] = []
    all_model_probs: list[np.ndarray] = []
    all_kl_thresholds: list[int] = []
    all_kl_crossed: list[bool] = []
    all_seq_seeds: list[int] = []

    for seq_idx in range(M):
        seq_seed = seq_seeds[seq_idx]
        torch.manual_seed(seq_seed)
        logger.info(f"Sequence {seq_idx + 1}/{M}: seed={seq_seed}, running forward pass ...")

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

        if use_projected:
            model_probs = get_model_probs_projected(logits, first_tok_id, mid_tok_ids, L)
        else:
            model_probs = get_model_probs(logits, first_tok_id, mid_tok_ids, L)
        optimal_probs = compute_optimal_probs(seq_beliefs, emit)

        kl_t, kl_crossed = find_kl_threshold(
            model_probs, optimal_probs, **config.kl_params, logger=logger,
        )
        if kl_crossed:
            logger.info(f"  KL threshold t* = {kl_t} / {L}  (fraction={config.kl_params['fraction']}, smooth_window={config.kl_params['smooth_window']}, min_position={config.kl_params.get('min_position', 0)})")
        else:
            logger.warning(f"  KL threshold t* = {kl_t} / {L}  (fallback argmin; fraction threshold never crossed, min_position={config.kl_params.get('min_position', 0)})")

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
        all_seq_seeds.append(seq_seed)

    with open(out_dir / "kl_thresholds.json", "w") as f:
        json.dump(
            {
                "thresholds": all_kl_thresholds,
                "crossed": all_kl_crossed,
                "mean": float(np.mean(all_kl_thresholds)),
                "seq_length": L,
                "fraction": config.kl_params["fraction"],
                "smooth_window": config.kl_params["smooth_window"],
                "seq_seeds": all_seq_seeds,
            },
            f,
            indent=2,
        )

    # ── Phase 2: Train probes ─────────────────────────────────────────────────
    total_probes = len(config.layer_indices) * M * 2
    logger.info(f"Phase 2: training {total_probes} probes (batched) ...")

    probe_inputs: dict[tuple[int, int, str], ProbeInput] = {}
    for layer in config.layer_indices:
        for seq_idx in range(M):
            acts = all_acts[seq_idx][layer]
            beliefs = all_beliefs[seq_idx]
            tokens = all_tokens[seq_idx]
            optimal_preds = all_optimal_probs[seq_idx]
            model_preds = all_model_probs[seq_idx]
            kl_t = all_kl_thresholds[seq_idx]
            post_start = max(kl_t, 1)

            probe_inputs[(layer, seq_idx, "full")] = ProbeInput(
                acts, beliefs, tokens, optimal_preds, model_preds,
            )
            probe_inputs[(layer, seq_idx, "post")] = ProbeInput(
                acts[post_start:], beliefs[post_start:],
                tokens[post_start - 1:], optimal_preds[post_start - 1:],
                model_preds[post_start - 1:],
            )

    probe_results = train_probes_batched(probe_inputs, max_probes_per_batch=config.max_probes_per_batch)

    full_probes_done: dict[int, list[ProbeResult]] = {l: [] for l in config.layer_indices}
    post_probes_done: dict[int, list[ProbeResult]] = {l: [] for l in config.layer_indices}
    for layer in config.layer_indices:
        for seq_idx in range(M):
            full_pr = probe_results[(layer, seq_idx, "full")]
            full_pr.kl_threshold = all_kl_thresholds[seq_idx]
            full_probes_done[layer].append(full_pr)

            post_pr = probe_results[(layer, seq_idx, "post")]
            post_pr.kl_threshold = all_kl_thresholds[seq_idx]
            post_probes_done[layer].append(post_pr)

    # Activations now live inside ProbeResults; free the raw cache.
    del all_acts

    # ── Phase 3: Cross-MSE and subspace similarity matrices ───────────────────
    results: dict[int, dict] = {}

    for layer in config.layer_indices:
        full_prs = full_probes_done[layer]
        post_prs = post_probes_done[layer]

        full_cross = cross_mse_matrix(
            full_prs,
            [pr.activations[pr.test_split_idx:] for pr in full_prs],
            [pr.gt_belief_states[pr.test_split_idx:] for pr in full_prs],
        )

        post_cross = cross_mse_matrix(
            post_prs,
            [pr.activations[pr.test_split_idx:] for pr in post_prs],
            [pr.gt_belief_states[pr.test_split_idx:] for pr in post_prs],
        )
        full_sim = pairwise_cosine_sim_matrix(full_prs)
        post_sim = pairwise_cosine_sim_matrix(post_prs)

        mask = ~np.eye(M, dtype=bool)
        full_diag = float(np.diag(full_cross).mean())
        full_off = float(full_cross[mask].mean())
        post_diag = float(np.diag(post_cross).mean())
        post_off = float(post_cross[mask].mean())

        results[layer] = {
            "full_cross_mse": full_cross,
            "post_cross_mse": post_cross,
            "full_similarity": full_sim,
            "post_similarity": post_sim,
            "full_diag_mean": full_diag,
            "full_off_diag_mean": full_off,
            "full_ratio": full_off / (full_diag + 1e-10),
            "post_diag_mean": post_diag,
            "post_off_diag_mean": post_off,
            "post_ratio": post_off / (post_diag + 1e-10),
        }

        logger.info(
            f"Layer {layer}: "
            f"full ratio={results[layer]['full_ratio']:.3f} "
            f"(diag={full_diag:.4f}, off={full_off:.4f})  |  "
            f"post ratio={results[layer]['post_ratio']:.3f} "
            f"(diag={post_diag:.4f}, off={post_off:.4f})"
        )

    # ── Save probes ───────────────────────────────────────────────────────────
    if config.save_probes:
        logger.info("Saving ProbeResults ...")
        for layer in config.layer_indices:
            for seq_idx, (full_pr, post_pr) in enumerate(
                zip(full_probes_done[layer], post_probes_done[layer])
            ):
                full_pr.save(out_dir / "probes" / f"layer_{layer}_seq_{seq_idx}_full")
                post_pr.save(out_dir / "probes" / f"layer_{layer}_seq_{seq_idx}_post")
    else:
        logger.info("Skipping ProbeResults save (save_probes=False)")

    # ── Save scalar results ───────────────────────────────────────────────────
    serialisable = {
        str(l): {
            "full_diag_mean": results[l]["full_diag_mean"],
            "full_off_diag_mean": results[l]["full_off_diag_mean"],
            "full_ratio": results[l]["full_ratio"],
            "post_diag_mean": results[l]["post_diag_mean"],
            "post_off_diag_mean": results[l]["post_off_diag_mean"],
            "post_ratio": results[l]["post_ratio"],
        }
        for l in config.layer_indices
    }
    with open(out_dir / "compare_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # ── Save matrices ─────────────────────────────────────────────────────────
    np.savez(
        out_dir / "matrices.npz",
        **{f"layer_{l}_full_cross_mse": results[l]["full_cross_mse"] for l in config.layer_indices},
        **{f"layer_{l}_post_cross_mse": results[l]["post_cross_mse"] for l in config.layer_indices},
        **{f"layer_{l}_full_similarity": results[l]["full_similarity"] for l in config.layer_indices},
        **{f"layer_{l}_post_similarity": results[l]["post_similarity"] for l in config.layer_indices},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    fig_dir = out_dir / "figures"

    _heatmap_grid(
        {l: results[l]["full_cross_mse"] for l in config.layer_indices},
        config.layer_indices,
        title="Cross-MSE matrix (full-sequence probes) — entry [i,j]: probe i on sequence j",
        path=fig_dir / "cross_mse_full",
        colorscale="Viridis",
    )

    _heatmap_grid(
        {l: results[l]["post_cross_mse"] for l in config.layer_indices},
        config.layer_indices,
        title="Cross-MSE matrix (post-threshold probes) — entry [i,j]: probe i on sequence j",
        path=fig_dir / "cross_mse_post",
        colorscale="Viridis",
    )

    _heatmap_grid(
        {l: results[l]["full_similarity"] for l in config.layer_indices},
        config.layer_indices,
        title="Subspace similarity (full-sequence probes) — mean |diag cosine sim| per probe pair",
        path=fig_dir / "subspace_sim_full",
        colorscale="Blues",
        zrange=(0.0, 1.0),
    )

    _heatmap_grid(
        {l: results[l]["post_similarity"] for l in config.layer_indices},
        config.layer_indices,
        title="Subspace similarity (post-threshold probes) — mean |diag cosine sim| per probe pair",
        path=fig_dir / "subspace_sim_post",
        colorscale="Blues",
        zrange=(0.0, 1.0),
    )

    plot_mse_ratio(results, config.layer_indices, fig_dir / "mse_ratio_per_layer")
    plot_diag_vs_offdiag(results, config.layer_indices, fig_dir / "diag_vs_offdiag_per_layer")

    # Belief geometry: show every 4th layer + last to keep plot size manageable
    plot_layers = sorted(set(config.layer_indices[::4] + [config.layer_indices[-1]]))
    plot_belief_grid(
        optimal_beliefs=all_beliefs,
        full_probe_results={l: full_probes_done[l] for l in plot_layers},
        post_probe_results={l: post_probes_done[l] for l in plot_layers},
        layer_indices=plot_layers,
        kl_thresholds=all_kl_thresholds,
        output_path=fig_dir / "belief_geometry",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
