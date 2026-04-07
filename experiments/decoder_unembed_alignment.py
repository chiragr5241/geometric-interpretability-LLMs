#!/usr/bin/env python3
"""Decoder-unembedding alignment analysis (SPAR-30).

For each trained decoder D ∈ R^(d_model × n_states) at each layer, measures
how aligned its columns are with the model's unembedding directions for the
HMM output tokens.

Concretely, for each layer:
  - Routing matrix  R = W_U_abc @ D ∈ R^(n_vocab × n_states)
  - Gain-adjusted   R_γ = (W_U_abc ⊙ γ) @ D  (LayerNorm gain scaling)
  - Cosine similarity matrix between W_U rows and D columns
  - Singular values of R (rank / alignment strength)

Ground-truth control: R² of a linear map from ground-truth belief states to
log P(token | belief).  High R² means a linear routing would be sufficient;
combined with high alignment it suggests the decoder writes to logit-relevant
directions.

Alignment convention (prepend_bos=False):
  cache position t  →  belief_states[t]   (simplexity: belief after token t)

Usage:
    python experiments/decoder_unembed_alignment.py \\
        experiments/configs/decoder_unembed_alignment.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_hmm_sequences
from decoder import DecoderInput, DecoderResult
from decoder import train_batched as train_decoders_batched
from experiment import ExperimentConfig, HMMConfig, apply_runtime_overrides, setup_output_dir
from experiment_utils import get_concept_token_ids, get_device, setup_logging


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DecoderUnembedAlignmentConfig(ExperimentConfig):
    layer_indices: list[int] = field(default_factory=list)
    seq_length: int = 2000
    n_sequences: int = 10
    post_convergence_start: int = 30
    random_seed: int = 42
    n_ctx_override: int | None = None
    decoder_params: dict[str, Any] = field(default_factory=dict)
    max_decoders_per_batch: int = 4
    sweeps: list[dict[str, Any]] = field(default_factory=list)
    default_vocab_tokens: dict[int, list[str]] = field(
        default_factory=lambda: {2: ["A", "B"], 3: ["A", "B", "C"]}
    )
    simplex_layers: list[int] = field(default_factory=list)


def load_alignment_config(path: str) -> DecoderUnembedAlignmentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    hmm_raw = raw.pop("hmm", {"process_name": "sweep", "process_params": {}})
    sweeps_raw = raw.pop("sweeps", [])
    default_vocab_raw = raw.pop("default_vocab_tokens", {2: ["A", "B"], 3: ["A", "B", "C"]})
    default_vocab = {int(k): v for k, v in default_vocab_raw.items()}
    return DecoderUnembedAlignmentConfig(
        hmm=HMMConfig(**hmm_raw),
        sweeps=sweeps_raw,
        default_vocab_tokens=default_vocab,
        **raw,
    )


# ── Sweep helpers ─────────────────────────────────────────────────────────────

def _expand_param_grid(param_grid: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = sorted(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[param_grid[k] for k in keys])]


def _make_label(process_name: str, params: dict[str, float]) -> str:
    parts = [process_name] + [f"{k}{params[k]}" for k in sorted(params)]
    return "_".join(parts)


def _resolve_vocab(hmm_obj: Any, entry: dict[str, Any], defaults: dict[int, list[str]]) -> list[str]:
    if "vocab_tokens" in entry and entry["vocab_tokens"] is not None:
        return entry["vocab_tokens"]
    return defaults.get(hmm_obj.vocab_size, [chr(65 + i) for i in range(hmm_obj.vocab_size)])


# ── Data collection ───────────────────────────────────────────────────────────

def collect_decoder_inputs(
    process_name: str,
    process_params: dict[str, float],
    vocab_tokens: list[str],
    model: Any,
    layer_indices: list[int],
    seq_length: int,
    n_sequences: int,
    post_convergence_start: int,
    random_seed: int,
    logger: logging.Logger,
) -> tuple[dict[int, DecoderInput], np.ndarray, np.ndarray]:
    """Run LLM forward passes and pool post-convergence positions.

    Returns
    -------
    decoder_inputs : dict[layer → DecoderInput]
    beliefs_pooled : (N, n_states)  — ground-truth beliefs (post-convergence)
    obs_probs_pooled : (N, n_vocab) — P(next token | belief), same rows
    """
    P = post_convergence_start
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]

    hmm_data = generate_hmm_sequences(
        process_name=process_name,
        process_params=process_params,
        n_sequences=n_sequences,
        seq_length=seq_length,
        random_seed=random_seed,
    )
    tokens = hmm_data.tokens          # (n_seq, seq_length) int
    belief_states = hmm_data.belief_states  # (n_seq, seq_length, n_states)
    obs_probs = hmm_data.obs_probs    # (n_seq, seq_length, n_vocab)

    acts_all: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}
    beliefs_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []

    letter_set = set(vocab_tokens)

    for seq_idx in range(n_sequences):
        logger.info(f"    Forward pass {seq_idx + 1}/{n_sequences}")
        symbols = [vocab_tokens[t] for t in tokens[seq_idx]]
        prompt = symbols[0] + " " + " ".join(symbols[1:])
        input_ids = model.to_tokens(prompt, prepend_bos=False)

        str_tokens = model.to_str_tokens(prompt, prepend_bos=False)
        positions = [i for i, tok in enumerate(str_tokens) if tok.strip() in letter_set]
        n_use = min(len(positions), seq_length)
        if len(positions) != seq_length:
            logger.warning(
                f"      Seq {seq_idx}: expected {seq_length} letter positions, got {len(positions)}"
            )
        positions = positions[:n_use]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_names,
                return_type=None,
            )

        for layer in layer_indices:
            acts = cache[f"blocks.{layer}.hook_resid_post"][0]  # (seq_len, d_model)
            acts_at_positions = acts[positions].float().cpu().numpy()
            acts_all[layer].append(acts_at_positions[P:])
        del cache

        beliefs_all.append(belief_states[seq_idx, P:n_use])
        obs_all.append(obs_probs[seq_idx, P:n_use])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beliefs_pooled = np.concatenate(beliefs_all, axis=0)      # (N, n_states)
    obs_probs_pooled = np.concatenate(obs_all, axis=0)         # (N, n_vocab)

    decoder_inputs: dict[int, DecoderInput] = {}
    for layer in layer_indices:
        acts_pooled = np.concatenate(acts_all[layer], axis=0)  # (N, d_model)
        decoder_inputs[layer] = DecoderInput(
            activations=acts_pooled,
            belief_states=beliefs_pooled,
        )

    return decoder_inputs, beliefs_pooled, obs_probs_pooled


# ── Alignment metrics ─────────────────────────────────────────────────────────

def compute_alignment_metrics(
    decoder_result: DecoderResult,
    W_U_abc: np.ndarray,
    W_U_abc_scaled: np.ndarray,
) -> dict[str, Any]:
    """Compute alignment between one decoder and the unembedding matrix.

    Parameters
    ----------
    decoder_result : DecoderResult
    W_U_abc : (n_vocab, d_model) — unembedding rows for HMM tokens (raw)
    W_U_abc_scaled : (n_vocab, d_model) — same, multiplied element-wise by ln_final gain γ

    Returns
    -------
    dict with routing matrices, cosine similarities, singular values, norms.
    """
    D = decoder_result.decoder.W.detach().float().cpu().numpy()  # (d_model, n_states)

    routing = W_U_abc @ D             # (n_vocab, n_states)
    routing_scaled = W_U_abc_scaled @ D

    # Cosine similarities: cosim[v, j] = cos(W_U_abc[v], D[:, j])
    W_U_norms = np.linalg.norm(W_U_abc, axis=1, keepdims=True) + 1e-10   # (n_vocab, 1)
    D_norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-10            # (1, n_states)
    cosim = (W_U_abc @ D) / (W_U_norms * D_norms)                         # (n_vocab, n_states)

    W_U_s_norms = np.linalg.norm(W_U_abc_scaled, axis=1, keepdims=True) + 1e-10
    cosim_scaled = (W_U_abc_scaled @ D) / (W_U_s_norms * D_norms)

    _, svs, _ = np.linalg.svd(routing, full_matrices=False)
    _, svs_scaled, _ = np.linalg.svd(routing_scaled, full_matrices=False)

    frob = float(np.linalg.norm(routing, "fro"))
    frob_scaled = float(np.linalg.norm(routing_scaled, "fro"))

    # Normalised Frobenius: ||R||_F / (||W_U||_F * ||D||_F)
    denom = (float(np.linalg.norm(W_U_abc, "fro")) * float(np.linalg.norm(D, "fro"))) + 1e-10
    denom_s = (float(np.linalg.norm(W_U_abc_scaled, "fro")) * float(np.linalg.norm(D, "fro"))) + 1e-10

    return {
        "routing": routing.tolist(),
        "routing_scaled": routing_scaled.tolist(),
        "cosim": cosim.tolist(),
        "cosim_scaled": cosim_scaled.tolist(),
        "singular_values": svs.tolist(),
        "singular_values_scaled": svs_scaled.tolist(),
        "top_sv": float(svs[0]),
        "top_sv_scaled": float(svs_scaled[0]),
        "frob_norm": frob,
        "frob_norm_scaled": frob_scaled,
        "normalised_frob": frob / denom,
        "normalised_frob_scaled": frob_scaled / denom_s,
        "mean_abs_cosim": float(np.abs(cosim).mean()),
        "mean_abs_cosim_scaled": float(np.abs(cosim_scaled).mean()),
        "max_abs_cosim": float(np.abs(cosim).max()),
        "max_abs_cosim_scaled": float(np.abs(cosim_scaled).max()),
    }


def compute_gt_r2(beliefs: np.ndarray, obs_probs: np.ndarray) -> dict[str, Any]:
    """R² of a linear map from belief states to log observation probabilities.

    This is Xavier's ground-truth control: how well can we predict
    log P(token | belief) linearly from the belief vector?
    High R² is a ceiling on how much linear alignment is meaningful.
    """
    log_obs = np.log(obs_probs + 1e-10)  # (N, n_vocab)
    reg = LinearRegression().fit(beliefs, log_obs)
    log_obs_pred = reg.predict(beliefs)
    gt_r2_overall = float(r2_score(log_obs, log_obs_pred, multioutput="uniform_average"))
    gt_r2_per_token = [
        float(r2_score(log_obs[:, v], log_obs_pred[:, v]))
        for v in range(log_obs.shape[1])
    ]
    return {"r2_overall": gt_r2_overall, "r2_per_token": gt_r2_per_token}


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _line_color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


def plot_metric_vs_layer(
    metrics_by_label: dict[str, dict[int, dict[str, Any]]],
    metric_key: str,
    y_title: str,
    title: str,
    path: Path,
) -> None:
    fig = go.Figure()
    for idx, (label, by_layer) in enumerate(metrics_by_label.items()):
        layers = sorted(by_layer.keys())
        vals = [by_layer[l][metric_key] for l in layers]
        fig.add_trace(go.Scatter(
            x=layers, y=vals, mode="lines+markers",
            name=label, line=dict(color=_line_color(idx), width=2),
        ))
    fig.update_layout(
        title=title, xaxis_title="Layer", yaxis_title=y_title,
        legend=dict(orientation="v"), height=480, width=760,
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_routing_heatmap_grid(
    by_layer: dict[int, dict[str, Any]],
    simplex_layers: list[int],
    label: str,
    path: Path,
    key: str = "routing",
    title_suffix: str = "",
) -> None:
    layers_to_plot = [l for l in simplex_layers if l in by_layer]
    if not layers_to_plot:
        return
    n_cols = min(len(layers_to_plot), 3)
    n_rows = (len(layers_to_plot) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layers_to_plot],
    )
    for i, layer in enumerate(layers_to_plot):
        matrix = np.array(by_layer[layer][key])
        row, col = divmod(i, n_cols)
        fig.add_trace(
            go.Heatmap(
                z=matrix.tolist(),
                colorscale="RdBu", zmid=0,
                showscale=(i == 0),
            ),
            row=row + 1, col=col + 1,
        )
    fig.update_layout(
        title=f"{label} — {title_suffix}",
        height=280 * n_rows + 80,
        width=340 * n_cols,
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_cosim_heatmap_grid(
    by_layer: dict[int, dict[str, Any]],
    simplex_layers: list[int],
    label: str,
    path: Path,
    key: str = "cosim",
    vocab_tokens: list[str] | None = None,
) -> None:
    layers_to_plot = [l for l in simplex_layers if l in by_layer]
    if not layers_to_plot:
        return
    n_cols = min(len(layers_to_plot), 3)
    n_rows = (len(layers_to_plot) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layers_to_plot],
    )
    for i, layer in enumerate(layers_to_plot):
        matrix = np.array(by_layer[layer][key])  # (n_vocab, n_states)
        n_states = matrix.shape[1]
        row, col = divmod(i, n_cols)
        fig.add_trace(
            go.Heatmap(
                z=matrix.tolist(),
                x=[f"state {j}" for j in range(n_states)],
                y=vocab_tokens or [f"tok {v}" for v in range(matrix.shape[0])],
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                showscale=(i == 0),
            ),
            row=row + 1, col=col + 1,
        )
    fig.update_layout(
        title=f"{label} — cosine similarity (W_U rows vs D columns)",
        height=300 * n_rows + 80,
        width=360 * n_cols,
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_singular_values_vs_layer(
    by_layer: dict[int, dict[str, Any]],
    label: str,
    path: Path,
) -> None:
    layers = sorted(by_layer.keys())
    top_sv = [by_layer[l]["top_sv"] for l in layers]
    top_sv_s = [by_layer[l]["top_sv_scaled"] for l in layers]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=top_sv, mode="lines+markers", name="raw W_U",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=top_sv_s, mode="lines+markers", name="γ-scaled W_U",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"{label} — top singular value of routing matrix vs layer",
        xaxis_title="Layer", yaxis_title="Top singular value",
        height=420, width=680,
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_gt_r2_bar(
    gt_r2_by_label: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    labels = list(gt_r2_by_label.keys())
    r2_vals = [gt_r2_by_label[l]["r2_overall"] for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=r2_vals, marker_color="#636EFA",
    ))
    fig.update_layout(
        title="Ground-truth R² (beliefs → log P(token | belief))",
        xaxis_title="Process config", yaxis_title="R²",
        xaxis_tickangle=-30, height=420, width=max(500, 100 * len(labels) + 200),
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_alignment_vs_gt_r2(
    gt_r2_by_label: dict[str, dict[str, Any]],
    metrics_by_label: dict[str, dict[int, dict[str, Any]]],
    layer_indices: list[int],
    path: Path,
) -> None:
    """Scatter: ground-truth R² vs best-layer top singular value."""
    fig = go.Figure()
    for idx, label in enumerate(gt_r2_by_label):
        gt_r2 = gt_r2_by_label[label]["r2_overall"]
        by_layer = metrics_by_label.get(label, {})
        if not by_layer:
            continue
        best_sv = max(by_layer[l]["top_sv"] for l in by_layer)
        fig.add_trace(go.Scatter(
            x=[gt_r2], y=[best_sv], mode="markers+text",
            name=label, text=[label],
            textposition="top center",
            marker=dict(color=_line_color(idx), size=12),
        ))
    fig.update_layout(
        title="Ground-truth R² vs best-layer routing alignment (top SV)",
        xaxis_title="Ground-truth R² (beliefs → log P)",
        yaxis_title="Best-layer top singular value",
        showlegend=False, height=480, width=600,
    )
    fig.write_image(str(path.with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--output-user", type=str, default=None)
    args = parser.parse_args()

    config = load_alignment_config(args.config)
    apply_runtime_overrides(config, output_user=args.output_user)

    device = get_device()
    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="decoder_align")
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Device     : {device}")

    from experiment_utils import load_model
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    W_U = model.unembed.W_U.detach().float().cpu().numpy()    # (d_model, vocab_size)
    gamma = model.ln_final.w.detach().float().cpu().numpy()   # (d_model,)

    all_metrics_by_label: dict[str, dict[int, dict[str, Any]]] = {}
    all_gt_r2_by_label: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []

    for entry in config.sweeps:
        process_name: str = entry["process_name"]
        param_grid: dict[str, list[float]] = entry.get("param_grid", {})
        combos = _expand_param_grid(param_grid) if param_grid else [{}]

        for params in combos:
            label = _make_label(process_name, params)
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Config: {label}")
            logger.info(f"{'=' * 60}")

            config_dir = out_dir / "configs" / label
            (config_dir / "figures").mkdir(parents=True, exist_ok=True)

            from data_generation import generate_hmm_sequences as _gen
            hmm_obj = _gen(
                process_name=process_name,
                process_params=params,
                n_sequences=1,
                seq_length=2,
                random_seed=config.random_seed,
            ).hmm
            vocab_tokens = _resolve_vocab(hmm_obj, entry, config.default_vocab_tokens)
            logger.info(
                f"  HMM: vocab_size={hmm_obj.vocab_size}, n_states={hmm_obj.num_states}, "
                f"vocab_tokens={vocab_tokens}"
            )

            concept_to_id = get_concept_token_ids(model, vocab_tokens)
            hmm_token_ids = [concept_to_id[t] for t in vocab_tokens]
            logger.info(f"  LLM token IDs: {dict(zip(vocab_tokens, hmm_token_ids))}")

            W_U_abc = W_U.T[hmm_token_ids, :]              # (n_vocab, d_model)
            W_U_abc_scaled = W_U_abc * gamma[None, :]      # element-wise γ scaling

            # ── Data collection ─────────────────────────────────────────────
            logger.info("  Collecting activations ...")
            decoder_inputs, beliefs_pooled, obs_probs_pooled = collect_decoder_inputs(
                process_name=process_name,
                process_params=params,
                vocab_tokens=vocab_tokens,
                model=model,
                layer_indices=config.layer_indices,
                seq_length=config.seq_length,
                n_sequences=config.n_sequences,
                post_convergence_start=config.post_convergence_start,
                random_seed=config.random_seed,
                logger=logger,
            )
            logger.info(f"  Pooled {beliefs_pooled.shape[0]} post-convergence positions")

            # ── Train decoders ───────────────────────────────────────────────
            dec_p = config.decoder_params
            logger.info(
                f"  Training decoders (lr={dec_p.get('lr', 1e-3)}, "
                f"max_epochs={dec_p.get('max_epochs', 10000)}, "
                f"patience={dec_p.get('patience', 200)}) ..."
            )
            decoder_results: dict[int, DecoderResult] = train_decoders_batched(
                decoder_inputs,
                lr=dec_p.get("lr", 1e-3),
                max_epochs=dec_p.get("max_epochs", 10000),
                patience=dec_p.get("patience", 200),
                min_relative_improvement=dec_p.get("min_relative_improvement", 1e-3),
                max_decoders_per_batch=config.max_decoders_per_batch,
            )
            logger.info("  Decoders trained.")

            # ── Ground-truth R² ──────────────────────────────────────────────
            logger.info("  Computing ground-truth R² ...")
            gt_r2 = compute_gt_r2(beliefs_pooled, obs_probs_pooled)
            logger.info(f"  GT R² overall: {gt_r2['r2_overall']:.4f}")
            all_gt_r2_by_label[label] = gt_r2

            # ── Alignment metrics per layer ──────────────────────────────────
            logger.info("  Computing alignment metrics ...")
            by_layer: dict[int, dict[str, Any]] = {}
            for layer in config.layer_indices:
                m = compute_alignment_metrics(
                    decoder_results[layer], W_U_abc, W_U_abc_scaled,
                )
                by_layer[layer] = m
                logger.info(
                    f"    Layer {layer:2d}: top_sv={m['top_sv']:.4f}  "
                    f"top_sv_scaled={m['top_sv_scaled']:.4f}  "
                    f"mean_|cosim|={m['mean_abs_cosim']:.4f}"
                )
            all_metrics_by_label[label] = by_layer

            # ── Save per-config metrics ──────────────────────────────────────
            per_config_metrics = {
                "process_name": process_name,
                "process_params": params,
                "vocab_tokens": vocab_tokens,
                "gt_r2": gt_r2,
                "alignment_by_layer": {
                    str(l): {
                        k: v for k, v in m.items()
                        if k not in ("routing", "routing_scaled", "cosim", "cosim_scaled")
                    }
                    for l, m in by_layer.items()
                },
            }
            with open(config_dir / "metrics.json", "w") as f:
                json.dump(per_config_metrics, f, indent=2)

            # ── Per-config figures ───────────────────────────────────────────
            fig_dir = config_dir / "figures"

            plot_singular_values_vs_layer(
                by_layer, label, fig_dir / "singular_values_vs_layer"
            )

            plot_routing_heatmap_grid(
                by_layer, config.simplex_layers, label,
                fig_dir / "routing_heatmap",
                key="routing", title_suffix="routing matrix W_U @ D",
            )
            plot_routing_heatmap_grid(
                by_layer, config.simplex_layers, label,
                fig_dir / "routing_heatmap_scaled",
                key="routing_scaled", title_suffix="γ-scaled routing matrix",
            )
            plot_cosim_heatmap_grid(
                by_layer, config.simplex_layers, label,
                fig_dir / "cosim_heatmap",
                key="cosim", vocab_tokens=vocab_tokens,
            )

            summary.append({
                "label": label,
                "process_name": process_name,
                "process_params": params,
                "gt_r2_overall": gt_r2["r2_overall"],
                "best_layer_top_sv": max(by_layer[l]["top_sv"] for l in by_layer),
                "best_layer_top_sv_scaled": max(by_layer[l]["top_sv_scaled"] for l in by_layer),
                "best_layer_mean_cosim": max(by_layer[l]["mean_abs_cosim"] for l in by_layer),
            })

    # ── Cross-config figures ─────────────────────────────────────────────────
    logger.info("\nGenerating cross-config figures ...")
    fig_dir = out_dir / "figures"

    plot_metric_vs_layer(
        all_metrics_by_label, "top_sv",
        "Top singular value", "Routing matrix top SV vs layer",
        fig_dir / "top_sv_vs_layer",
    )
    plot_metric_vs_layer(
        all_metrics_by_label, "top_sv_scaled",
        "Top singular value (γ-scaled)", "γ-scaled routing matrix top SV vs layer",
        fig_dir / "top_sv_scaled_vs_layer",
    )
    plot_metric_vs_layer(
        all_metrics_by_label, "mean_abs_cosim",
        "Mean |cosine similarity|", "Mean |cosine sim| (W_U rows vs D cols) vs layer",
        fig_dir / "mean_cosim_vs_layer",
    )
    plot_metric_vs_layer(
        all_metrics_by_label, "normalised_frob",
        "Normalised Frobenius norm", "Normalised ||W_U @ D||_F vs layer",
        fig_dir / "normalised_frob_vs_layer",
    )

    plot_gt_r2_bar(all_gt_r2_by_label, fig_dir / "gt_r2_bar")
    plot_alignment_vs_gt_r2(
        all_gt_r2_by_label, all_metrics_by_label,
        config.layer_indices, fig_dir / "alignment_vs_gt_r2",
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"model_name": config.model_name, "configs": summary},
            f, indent=2,
        )

    logger.info(f"\nDone. Outputs in {out_dir}")
    logger.info("Summary:")
    for s in summary:
        logger.info(
            f"  {s['label']}: gt_R²={s['gt_r2_overall']:.4f}  "
            f"best_top_sv={s['best_layer_top_sv']:.4f}  "
            f"best_mean_cosim={s['best_layer_mean_cosim']:.4f}"
        )


if __name__ == "__main__":
    main()
