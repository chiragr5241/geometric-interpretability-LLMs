#!/usr/bin/env python3
"""
Experiment 1 — Transient period vs. post-threshold probes.

Usage:
    python experiments/probes_full_seq_vs_kl_threshold.py experiments/configs/probes_full_seq_vs_kl_threshold.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import dataclass

from experiment import ExperimentConfig, load_config, setup_output_dir


@dataclass
class ProbesFullSeqVsKLThresholdConfig(ExperimentConfig):
    layer_indices: list[int]
    seq_length: int
    kl_params: dict[str, float]
    vocab_mapping: dict[str, int]
    n_runs: int = 1
    max_probes_per_batch: int = 200
    n_ctx_override: int | None = None
    save_probes: bool = False

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
from metrics.probe_metrics import compare_probes, compute_kl, find_kl_threshold
from probes import ProbeInput, ProbeResult, train_probes_batched
from visualization import plot_belief_grid


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_kl_over_sequence(
    model_probs: np.ndarray,
    optimal_probs: np.ndarray,
    kl_t: int,
    fraction: float,
    smooth_window: int,
    path: Path,
    min_position: int = 0,
    include_junk: bool = False,
) -> None:
    import plotly.graph_objects as go

    kl_raw, kl_smooth = compute_kl(model_probs, optimal_probs, smooth_window, include_junk=include_junk)

    kl_min, kl_max = kl_smooth.min(), kl_smooth.max()
    threshold_val = kl_min + fraction * (kl_max - kl_min)
    kl_search = kl_smooth[min_position:]
    kl_argmin = int(np.argmin(kl_search)) + min_position

    positions = np.arange(len(kl_raw))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions, y=kl_raw,
        mode="lines",
        name="KL (raw)",
        line=dict(color="lightblue", width=1),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=positions, y=kl_smooth,
        mode="lines",
        name=f"KL (smoothed, w={smooth_window})",
        line=dict(color="royalblue", width=2),
    ))
    fig.add_hline(
        y=threshold_val,
        line=dict(color="red", dash="dash", width=1.5),
        annotation_text=f"threshold (fraction={fraction})",
        annotation_position="top right",
    )
    if min_position > 0:
        fig.add_vline(
            x=min_position,
            line=dict(color="green", dash="dash", width=1.5),
            annotation_text=f"min_position={min_position}",
            annotation_position="top right",
        )
    fig.add_vline(
        x=kl_argmin,
        line=dict(color="purple", dash="dash", width=1.5),
        annotation_text=f"argmin KL={kl_argmin}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=kl_t,
        line=dict(color="orange", dash="dot", width=1.5),
        annotation_text=f"t*={kl_t}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="KL(optimal ‖ model) over sequence (run 0) with threshold",
        xaxis_title="Sequence position",
        yaxis_title="KL(optimal ‖ model)",
        legend=dict(x=0.7, y=0.95),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def plot_mse_comparison_with_errorbars(
    full_mse_per_layer: dict[int, list[float]],
    post_mse_per_layer: dict[int, list[float]],
    layer_indices: list[int],
    n_runs: int,
    path: Path,
) -> None:
    import plotly.graph_objects as go

    layers = [str(l) for l in layer_indices]
    full_means = [float(np.mean(full_mse_per_layer[l])) for l in layer_indices]
    full_stds = [float(np.std(full_mse_per_layer[l])) for l in layer_indices]
    post_means = [float(np.mean(post_mse_per_layer[l])) for l in layer_indices]
    post_stds = [float(np.std(post_mse_per_layer[l])) for l in layer_indices]

    error_bar = dict(type="data", visible=True, thickness=1, width=3)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=full_means,
        error_y=dict(**error_bar, array=full_stds),
        name="Full probe",
        width=0.35,
    ))
    fig.add_trace(go.Bar(
        x=layers, y=post_means,
        error_y=dict(**error_bar, array=post_stds),
        name="Post-threshold probe",
        width=0.35,
    ))
    title = f"Test MSE: full vs. post-threshold probes per layer (n={n_runs} runs, ±1 std)"
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Test MSE",
        barmode="group",
        bargroupgap=0.3,
        width=1600,
        height=500,
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def plot_column_cosine_similarity(
    compare_results: dict[int, dict],
    layer_indices: list[int],
    path: Path,
    state_labels: list[str] | None = None,
    title: str = "Column cosine similarity: full vs. post-threshold probe weight vectors",
) -> None:
    """
    Per-layer heatmaps of cosine similarity between per-component probe weight vectors,
    plus a final panel showing the mean across layers.

    Rows    = belief components of the full-sequence probe (probe A)
    Columns = belief components of the post-threshold probe (probe B)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_layers = len(layer_indices)
    n_panels = n_layers + 1   # one per layer + mean
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    sample_mat = compare_results[layer_indices[0]]["column_cosine_sim"]
    n_states = sample_mat.shape[0]
    labels = state_labels or [str(i) for i in range(n_states)]

    titles = [f"Layer {l}" for l in layer_indices] + ["Mean across layers"]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=max(0.06, 0.18 / n_rows),
    )

    all_mats = np.stack(
        [compare_results[l]["column_cosine_sim"] for l in layer_indices], axis=0
    )   # (n_layers, n_states, n_states)
    mean_mat = all_mats.mean(axis=0)

    def _add_heatmap(mat: np.ndarray, row: int, col: int) -> None:
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmin=-1.0,
                zmax=1.0,
                showscale=False,
                text=[[f"{v:.2f}" for v in row_vals] for row_vals in mat],
                texttemplate="%{text}",
            ),
            row=row,
            col=col,
        )

    for idx, layer in enumerate(layer_indices):
        panel_row = idx // n_cols + 1
        panel_col = idx % n_cols + 1
        _add_heatmap(compare_results[layer]["column_cosine_sim"], panel_row, panel_col)

    mean_row = n_layers // n_cols + 1
    mean_col = n_layers % n_cols + 1
    _add_heatmap(mean_mat, mean_row, mean_col)

    bottom_row_indices = set(
        range((n_rows - 1) * n_cols + 1, n_rows * n_cols + 1)
    )
    for ax_idx in range(1, n_panels + 1):
        x_key = "xaxis" if ax_idx == 1 else f"xaxis{ax_idx}"
        y_key = "yaxis" if ax_idx == 1 else f"yaxis{ax_idx}"
        if ax_idx in bottom_row_indices:
            fig.layout[x_key].title = dict(text="post-threshold probe", font=dict(size=10))
        if ax_idx % n_cols == 1:
            fig.layout[y_key].title = dict(text="full probe", font=dict(size=10))

    cell_px = 200
    fig.update_layout(
        title=title,
        height=cell_px * n_rows,
        width=cell_px * n_cols,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1], ProbesFullSeqVsKLThresholdConfig)
    n_runs = config.n_runs
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="exp1")

    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Device     : {device}")
    logger.info(f"n_runs     : {n_runs}")
    logger.info(f"Config     : {config}")

    # ── Sequence seeds — captured before model loading, which resets torch RNG ─
    seq_seeds: list[int] = [int(torch.randint(2**31, (1,)).item()) for _ in range(n_runs)]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    n_ctx = model.cfg.n_ctx
    if config.seq_length >= n_ctx:
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
    # all_acts[run_idx][layer] = (L+1, d_model) float32 numpy array
    all_acts: list[dict[int, np.ndarray]] = []
    all_beliefs: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []
    all_optimal_probs: list[np.ndarray] = []
    all_model_probs: list[np.ndarray] = []
    all_kl_thresholds: list[int] = []
    all_kl_crossed: list[bool] = []

    for run_idx in range(n_runs):
        seq_seed = seq_seeds[run_idx]
        torch.manual_seed(seq_seed)
        logger.info(f"Run {run_idx + 1}/{n_runs}: seed={seq_seed}, generating sequence of length {L} ...")

        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
        seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy()

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=True, truncate=False)
        assert llm_tokens.shape[1] == L + 1, (
            f"Expected {L+1} LLM tokens, got {llm_tokens.shape[1]}. "
            "Check that every HMM token maps to a single LLM token."
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
            logger.info(f"  KL threshold t* = {kl_t} / {L}")
        else:
            logger.warning(f"  KL threshold t* = {kl_t} / {L}  (fallback argmin; threshold never crossed)")

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
                "seq_length": L,
                "seq_seeds": seq_seeds,
            },
            f,
            indent=2,
        )

    # ── Phase 2: Train probes ─────────────────────────────────────────────────
    # All probes share d_model and n_states, so layers can be freely mixed.
    # train_probes_batched chunks internally by max_probes_per_batch to keep
    # the padded activation tensor (N, max_len, d_model) within memory budget.
    total_probes = len(config.layer_indices) * n_runs * 2
    logger.info(
        f"Phase 2: training {total_probes} probes "
        f"(max_probes_per_batch={config.max_probes_per_batch}) ..."
    )

    all_probe_inputs: dict[tuple[int, int, str], ProbeInput] = {}
    for layer in config.layer_indices:
        for run_idx in range(n_runs):
            acts = all_acts[run_idx][layer]
            kl_t = all_kl_thresholds[run_idx]
            post_start = max(kl_t, 1)
            all_probe_inputs[(run_idx, layer, "full")] = ProbeInput(
                acts, all_beliefs[run_idx], all_tokens[run_idx],
                all_optimal_probs[run_idx], all_model_probs[run_idx],
            )
            all_probe_inputs[(run_idx, layer, "post")] = ProbeInput(
                acts[post_start:], all_beliefs[run_idx][post_start:],
                all_tokens[run_idx][post_start - 1:],
                all_optimal_probs[run_idx][post_start - 1:],
                all_model_probs[run_idx][post_start - 1:],
            )

    all_probe_results = train_probes_batched(
        all_probe_inputs, max_probes_per_batch=config.max_probes_per_batch
    )

    full_probes: dict[int, list[ProbeResult]] = {l: [] for l in config.layer_indices}
    post_probes: dict[int, list[ProbeResult]] = {l: [] for l in config.layer_indices}
    for layer in config.layer_indices:
        for run_idx in range(n_runs):
            full_pr = all_probe_results[(run_idx, layer, "full")]
            full_pr.kl_threshold = all_kl_thresholds[run_idx]
            full_probes[layer].append(full_pr)

            post_pr = all_probe_results[(run_idx, layer, "post")]
            post_pr.kl_threshold = all_kl_thresholds[run_idx]
            post_probes[layer].append(post_pr)

        mean_full = float(np.mean([pr.test_mse for pr in full_probes[layer]]))
        mean_post = float(np.mean([pr.test_mse for pr in post_probes[layer]]))
        logger.info(
            f"Layer {layer}: mean full MSE={mean_full:.4f}, mean post MSE={mean_post:.4f}"
        )

    # ── Phase 3: Aggregate ─────────────────────────────────────────────────────
    full_mse_per_layer: dict[int, list[float]] = {
        l: [pr.test_mse for pr in full_probes[l]] for l in config.layer_indices
    }
    post_mse_per_layer: dict[int, list[float]] = {
        l: [pr.test_mse for pr in post_probes[l]] for l in config.layer_indices
    }

    # ── Save ProbeResults ─────────────────────────────────────────────────────
    if config.save_probes:
        logger.info("Saving ProbeResults ...")
        for layer in config.layer_indices:
            for run_idx in range(n_runs):
                full_probes[layer][run_idx].save(out_dir / "probes" / f"layer_{layer}_run_{run_idx}_full")
                post_probes[layer][run_idx].save(out_dir / "probes" / f"layer_{layer}_run_{run_idx}_post")
    else:
        logger.info("Skipping ProbeResults save (save_probes=False)")

    mse_serialisable = {
        str(l): {
            "full_mse_runs": full_mse_per_layer[l],
            "full_mse_mean": float(np.mean(full_mse_per_layer[l])),
            "full_mse_std": float(np.std(full_mse_per_layer[l])),
            "post_mse_runs": post_mse_per_layer[l],
            "post_mse_mean": float(np.mean(post_mse_per_layer[l])),
            "post_mse_std": float(np.std(post_mse_per_layer[l])),
        }
        for l in config.layer_indices
    }
    with open(out_dir / "mse_results.json", "w") as f:
        json.dump(mse_serialisable, f, indent=2)

    # ── Cosine similarity: per-run full vs post, averaged across runs ─────────
    compare_results: dict[int, dict] = {}
    for layer in config.layer_indices:
        per_run = [
            compare_probes(full_probes[layer][r], post_probes[layer][r])
            for r in range(n_runs)
        ]
        mean_cos_sim = np.mean(
            [cmp["column_cosine_sim"] for cmp in per_run], axis=0
        )
        compare_results[layer] = {
            **per_run[0],  # single-run extras (cross_mse etc.) from run 0 when n_runs==1
            "column_cosine_sim": mean_cos_sim,
        }
        logger.info(
            f"Layer {layer}: mean diagonal cosine sim"
            f"{'  (avg over runs)' if n_runs > 1 else ''}="
            f"{np.diag(mean_cos_sim).mean():.4f}"
        )

    serialisable: dict = {
        str(l): {"column_cosine_sim": compare_results[l]["column_cosine_sim"].tolist()}
        for l in config.layer_indices
    }
    if n_runs == 1:
        for l in config.layer_indices:
            serialisable[str(l)].update({
                "test_mse_a": float(compare_results[l]["test_mse_a"]),
                "test_mse_b": float(compare_results[l]["test_mse_b"]),
                "cross_mse_ab": float(compare_results[l]["cross_mse_ab"]),
                "cross_mse_ba": float(compare_results[l]["cross_mse_ba"]),
            })
    with open(out_dir / "compare_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")

    plot_mse_comparison_with_errorbars(
        full_mse_per_layer,
        post_mse_per_layer,
        config.layer_indices,
        n_runs,
        out_dir / "figures" / "mse_comparison.png",
    )

    plot_kl_over_sequence(
        all_model_probs[0],
        all_optimal_probs[0],
        all_kl_thresholds[0],
        fraction=config.kl_params["fraction"],
        smooth_window=config.kl_params["smooth_window"],
        path=out_dir / "figures" / "kl_over_sequence.png",
        min_position=config.kl_params.get("min_position", 0),
        include_junk=use_projected,
    )

    plot_column_cosine_similarity(
        compare_results,
        config.layer_indices,
        out_dir / "figures" / "column_cosine_similarity.png",
        state_labels=list(config.vocab_mapping.keys()),
        title=(
            "Column cosine similarity: full vs. post-threshold probe weight vectors"
            + (f" (averaged over {n_runs} runs)" if n_runs > 1 else "")
        ),
    )

    # Belief geometry using run 0 as representative
    plot_layers = config.layer_indices if n_runs == 1 else sorted(
        set(config.layer_indices[::4] + [config.layer_indices[-1]])
    )
    plot_belief_grid(
        optimal_beliefs=[all_beliefs[0]],
        full_probe_results={l: [full_probes[l][0]] for l in plot_layers},
        post_probe_results={l: [post_probes[l][0]] for l in plot_layers},
        layer_indices=plot_layers,
        kl_thresholds=[all_kl_thresholds[0]],
        output_path=out_dir / "figures" / "belief_geometry",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
