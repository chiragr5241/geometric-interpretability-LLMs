#!/usr/bin/env python3
"""
Experiment 1 — Transient period vs. post-threshold probes.

Usage:
    python experiments/experiment_1_transient.py experiments/configs/experiment_1_transient.yaml
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, load_config, setup_output_dir
from hmm.hmm import Mess3HMM
from metrics.probe_metrics import compare_probes, find_kl_threshold
from probes import ProbeResult, train_probe
from visualization import plot_belief_grid


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp1")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device, logger: logging.Logger):
    import transformers
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        from huggingface_hub.constants import HF_HUB_CACHE
        transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE
    from transformer_lens import HookedTransformer

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    logger.info(f"Loading '{model_name}' on {device} with dtype={dtype} ...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=str(device),
    )
    model.eval()
    elapsed = time.time() - t0
    logger.info(
        f"Model loaded in {elapsed:.1f}s  |  "
        f"layers={model.cfg.n_layers}  d_model={model.cfg.d_model}"
    )
    return model


# ── HMM helpers ───────────────────────────────────────────────────────────────

def build_emission_matrix(hmm: Mess3HMM) -> np.ndarray:
    """
    emit[token_idx, state_idx] = P(token | state)
    Derived by marginalising T[token, from_state, to_state] over next states.
    Shape: (n_tokens, n_states)
    """
    T = hmm.T_3d_matrix.cpu().numpy()   # (n_tokens, n_states, n_states)
    return T.sum(axis=-1)               # (n_tokens, n_states)


def compute_optimal_probs(beliefs: np.ndarray, emit: np.ndarray) -> np.ndarray:
    """
    Bayesian-optimal next-token probabilities.

    beliefs: (L+1, n_states)  — b_0 .. b_L
    emit:    (n_tokens, n_states)
    Returns: (L, n_tokens)    — P(t_j | b_j) for j = 0 .. L-1
    """
    return beliefs[:-1] @ emit.T   # (L, n_tokens)


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def resolve_hmm_token_ids(
    model,
    idx_to_token: dict[int, str],
    n_tokens: int,
    logger: logging.Logger,
) -> tuple[int, list[int]]:
    """
    Return (first_tok_id, mid_tok_ids).

    first_tok_id: LLM token ID for the very first HMM token (no leading space).
    mid_tok_ids:  LLM token IDs for [' A', ' B', ' C'] (space-prefixed, used at
                  positions >= 1 in the HMM sequence).

    Asserts that each HMM token maps to exactly one LLM token.
    """
    ordered = [idx_to_token[i] for i in range(n_tokens)]

    sample_text = " ".join(ordered)
    mid_ids = model.to_tokens(sample_text, prepend_bos=False)[0].tolist()
    assert len(mid_ids) == n_tokens, (
        f"HMM tokens {ordered!r} tokenise to {len(mid_ids)} LLM tokens "
        f"(expected {n_tokens}): {model.to_str_tokens(sample_text, prepend_bos=False)}"
    )

    first_text = ordered[0]
    first_ids = model.to_tokens(first_text, prepend_bos=False)[0].tolist()
    assert len(first_ids) == 1, (
        f"First HMM token '{first_text}' tokenises to >1 LLM tokens: {first_ids}"
    )

    logger.info(
        f"HMM vocab -> LLM token IDs: "
        f"first={first_ids[0]} ({first_text!r}), "
        f"mid={dict(zip(ordered, mid_ids))}"
    )
    return first_ids[0], mid_ids


def get_model_probs(
    logits: torch.Tensor,
    first_tok_id: int,
    mid_tok_ids: list[int],
    seq_len: int,
) -> np.ndarray:
    """
    Extract renormalised next-token probs over the HMM vocabulary.

    logits shape: (1, llm_seq_len, vocab_size)
    Returns:      (seq_len, n_hmm_tokens) — for predicting t_0 .. t_{L-1}

    logits[:, 0, :] predicts position 1 = t_0 (first HMM token, no leading space).
    logits[:, j, :] for j >= 1 predicts space-prefixed tokens.
    """
    n = len(mid_tok_ids)
    with torch.no_grad():
        probs_full = F.softmax(logits[0, :seq_len, :].float(), dim=-1)   # (seq_len, vocab_size)

    out = np.empty((seq_len, n), dtype=np.float32)

    # Position 0: use the no-space token ID for the first HMM token
    p0 = probs_full[0, mid_tok_ids].cpu().numpy().copy()
    if first_tok_id not in mid_tok_ids:
        p0[0] = probs_full[0, first_tok_id].item()
    out[0] = p0 / (p0.sum() + 1e-10)

    # Positions 1..L-1: space-prefixed tokens
    if seq_len > 1:
        mid = probs_full[1:seq_len, :][:, mid_tok_ids].cpu().numpy()
        out[1:] = mid / (mid.sum(axis=-1, keepdims=True) + 1e-10)

    return out


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_mse_comparison(
    compare_results: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    import plotly.graph_objects as go

    layers = [str(l) for l in layer_indices]
    full_mses = [compare_results[l]["test_mse_a"] for l in layer_indices]
    post_mses = [compare_results[l]["test_mse_b"] for l in layer_indices]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=layers, y=full_mses, name="Full probe"))
    fig.add_trace(go.Bar(x=layers, y=post_mses, name="Post-threshold probe"))
    fig.update_layout(
        title="Test MSE: full vs. post-threshold probes per layer",
        xaxis_title="Layer",
        yaxis_title="Test MSE",
        barmode="group",
    )
    fig.write_image(str(Path(path).with_suffix(".png")))


def plot_column_cosine_similarity(
    compare_results: dict[int, dict],
    layer_indices: list[int],
    path: Path,
    state_labels: list[str] | None = None,
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
        title="Column cosine similarity: full vs. post-threshold probe weight vectors",
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

    load_dotenv()
    config: ExperimentConfig = load_config(sys.argv[1])
    device = _get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir)

    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Device     : {device}")
    logger.info(f"Config     : {config}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger)

    # ── HMM ───────────────────────────────────────────────────────────────────
    hmm = Mess3HMM()
    p = config.hmm.process_params
    if "x" in p and "alpha" in p:
        hmm.create_hmm(p["x"], p["alpha"])
        logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    emit = build_emission_matrix(hmm)   # (n_tokens, n_states)

    # ── Resolve LLM token IDs for A / B / C ───────────────────────────────────
    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(
        model, idx_to_token, n_hmm_tokens, logger
    )

    # ── Generate one sequence ─────────────────────────────────────────────────
    L = config.seq_length
    logger.info(f"Generating one sequence of length {L} ...")
    tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
    beliefs_batch = hmm.compute_belief_state(tokens_batch)   # (1, L+1, 3)

    seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()   # (L,)
    seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy() # (L+1, 3)

    # ── Forward pass ──────────────────────────────────────────────────────────
    text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
    llm_tokens = model.to_tokens(text, prepend_bos=True)   # (1, L+1)
    assert llm_tokens.shape[1] == L + 1, (
        f"Expected {L+1} LLM tokens, got {llm_tokens.shape[1]}. "
        "Check that every HMM token maps to a single LLM token."
    )

    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]
    logger.info("Running forward pass ...")
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            llm_tokens,
            names_filter=hook_names,
            return_type="logits",
        )

    # ── KL threshold ──────────────────────────────────────────────────────────
    model_probs = get_model_probs(logits, first_tok_id, mid_tok_ids, L)   # (L, n_tokens)
    optimal_probs = compute_optimal_probs(seq_beliefs, emit)               # (L, n_tokens)

    kl_t: int = find_kl_threshold(model_probs, optimal_probs, **config.kl_params)
    logger.info(f"KL threshold t* = {kl_t} / {L}")

    with open(out_dir / "kl_threshold.json", "w") as f:
        json.dump({"kl_threshold": kl_t, "seq_length": L}, f, indent=2)

    # ── Per-layer probes ───────────────────────────────────────────────────────
    # Residual at LLM position j encodes b_j (belief after seeing t_{j-1}).
    # Full probe trains on positions 0..L; post-threshold on kl_t..L.
    post_start = max(kl_t, 1)   # guard against t*=0

    full_probes: dict[int, ProbeResult] = {}
    post_probes: dict[int, ProbeResult] = {}
    compare_results: dict[int, dict] = {}

    for layer in config.layer_indices:
        acts: np.ndarray = cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
        # shape: (L+1, d_model)

        logger.info(f"Layer {layer}: training full probe ...")
        full_pr = train_probe(
            activations=acts,
            gt_belief_states=seq_beliefs,
            tokens=seq_tokens,
            gt_next_token_preds=optimal_probs,
            computed_next_token_preds=model_probs,
        )
        full_pr.kl_threshold = kl_t

        logger.info(f"Layer {layer}: training post-threshold probe (from t*={kl_t}) ...")
        post_pr = train_probe(
            activations=acts[post_start:],
            gt_belief_states=seq_beliefs[post_start:],
            tokens=seq_tokens[post_start - 1:],
            gt_next_token_preds=optimal_probs[post_start - 1:],
            computed_next_token_preds=model_probs[post_start - 1:],
        )
        post_pr.kl_threshold = kl_t

        full_probes[layer] = full_pr
        post_probes[layer] = post_pr
        compare_results[layer] = compare_probes(full_pr, post_pr)

        cos_sim = compare_results[layer]["column_cosine_sim"]
        logger.info(
            f"Layer {layer}: full test MSE={full_pr.test_mse:.4f}, "
            f"post test MSE={post_pr.test_mse:.4f}, "
            f"mean diagonal cosine sim={np.diag(cos_sim).mean():.4f}"
        )

    del logits, cache

    # ── Save ProbeResults ─────────────────────────────────────────────────────
    logger.info("Saving ProbeResults ...")
    for layer in config.layer_indices:
        full_probes[layer].save(out_dir / "probes" / f"layer_{layer}_full")
        post_probes[layer].save(out_dir / "probes" / f"layer_{layer}_post")

    serialisable = {
        str(l): {
            "test_mse_a": float(compare_results[l]["test_mse_a"]),
            "test_mse_b": float(compare_results[l]["test_mse_b"]),
            "cross_mse_ab": float(compare_results[l]["cross_mse_ab"]),
            "cross_mse_ba": float(compare_results[l]["cross_mse_ba"]),
            "column_cosine_sim": compare_results[l]["column_cosine_sim"].tolist(),
        }
        for l in config.layer_indices
    }
    with open(out_dir / "compare_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")

    plot_belief_grid(
        optimal_beliefs=[seq_beliefs],
        full_probe_results={l: [full_probes[l]] for l in config.layer_indices},
        post_probe_results={l: [post_probes[l]] for l in config.layer_indices},
        layer_indices=config.layer_indices,
        kl_thresholds=[kl_t],
        output_path=out_dir / "figures" / "belief_geometry",
    )
    plot_mse_comparison(compare_results, config.layer_indices, out_dir / "figures" / "mse_comparison.png")
    plot_column_cosine_similarity(
        compare_results,
        config.layer_indices,
        out_dir / "figures" / "column_cosine_similarity.png",
        state_labels=list(config.vocab_mapping.keys()),
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
