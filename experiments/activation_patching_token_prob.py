#!/usr/bin/env python3
"""SPAR — Activation patching: token probability control experiment.

Tests whether the factual-vs-counterfactual KL gap from SPAR-15 is driven
by token probability rather than the factual/counterfactual distinction itself.

Design: for each base sequence (prefix of L-1 tokens), generate 3 variant
sequences by appending each possible final token (A, B, C). For each variant
× each of 3 target beliefs we get 9 patch specs per base sequence.

Labels:
  is_factual  — variant_token == belief_token (sequence ends in the token
                whose belief is being patched)
  is_likely   — belief_token is the highest-probability next token given the
                prefix belief

Sanity check: for a fixed belief, KL should be identical across all 3 variant
sequences (different final tokens) since patching is a wholesale replacement
and causal masking isolates positions 0..(L-2) from L-1.

Main result: {factual, cf} × {likely, unlikely} KL vs layer.
Prediction: factual+likely ≈ cf+likely, factual+unlikely ≈ cf+unlikely.

Usage:
    python experiments/activation_patching_token_prob.py \\
        experiments/configs/activation_patching_token_prob.yaml
    python experiments/activation_patching_token_prob.py \\
        experiments/configs/activation_patching_token_prob.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
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


DRY_RUN_LAYERS  = [0, 2, 6, 10, 17, 25]
DRY_RUN_SEQ_LEN = 100
DRY_RUN_N_SEQ   = 5

_BINS = ["factual_likely", "factual_unlikely", "cf_likely", "cf_unlikely"]

_BIN_COLORS: dict[str, tuple[str, str]] = {
    "factual_likely":   ("#1f77b4", "rgba(31,119,180,0.12)"),
    "factual_unlikely": ("#aec7e8", "rgba(174,199,232,0.10)"),
    "cf_likely":        ("#ff7f0e", "rgba(255,187,120,0.18)"),   # lighter fill than the line
    "cf_unlikely":      ("#ffbb78", "rgba(255,204,153,0.12)"),
}

_BIN_LABELS: dict[str, str] = {
    "factual_likely":   "factual + likely",
    "factual_unlikely": "factual + unlikely",
    "cf_likely":        "cf + likely",
    "cf_unlikely":      "cf + unlikely",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ActivationPatchingTokenProbConfig(ExperimentConfig):
    encoder_decoder_dir: str
    layer_indices: list[int]
    seq_length: int
    n_sequences: int
    batch_size: int
    patch_position: int
    vocab_mapping: dict[str, int]
    n_ctx_override: int | None = None


# ── Patch spec ────────────────────────────────────────────────────────────────

@dataclass
class PatchSpec:
    base_idx: int
    variant_token_idx: int   # token at position L-1 in this variant sequence
    belief_token_idx: int    # token whose Bayesian belief is patched in
    is_factual: bool         # variant_token_idx == belief_token_idx
    is_likely: bool          # belief_token_idx has highest P(token | prefix_belief)
    token_prob: float        # P(belief_token | prefix_belief) via emission matrix
    belief_shift_l2: float   # ||η_target - prefix_belief||₂
    target_belief: np.ndarray  # (n_states,) float32


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hmm_step(belief: np.ndarray, token_idx: int, T_3d: np.ndarray) -> np.ndarray:
    out = T_3d[token_idx] @ belief
    return (out / (out.sum() + 1e-10)).astype(np.float32)


def _kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return (P * np.log(np.clip(P, 1e-10, None) / np.clip(Q, 1e-10, None))).sum(axis=-1)


def _bin_key(is_factual: bool, is_likely: bool) -> str:
    return ("factual" if is_factual else "cf") + "_" + ("likely" if is_likely else "unlikely")


def _build_specs(
    prefix_beliefs: list[np.ndarray],
    T_3d: np.ndarray,
    emit: np.ndarray,
    n_tokens: int,
) -> list[PatchSpec]:
    specs: list[PatchSpec] = []
    for base_idx, prefix_bel in enumerate(prefix_beliefs):
        tok_probs = prefix_bel @ emit.T          # (n_tokens,)
        likely_idx = int(np.argmax(tok_probs))
        targets = [_hmm_step(prefix_bel, z, T_3d) for z in range(n_tokens)]
        for vt in range(n_tokens):
            for bt in range(n_tokens):
                eta = targets[bt]
                specs.append(PatchSpec(
                    base_idx=base_idx,
                    variant_token_idx=vt,
                    belief_token_idx=bt,
                    is_factual=(vt == bt),
                    is_likely=(bt == likely_idx),
                    token_prob=float(tok_probs[bt]),
                    belief_shift_l2=float(np.linalg.norm(eta - prefix_bel)),
                    target_belief=eta,
                ))
    return specs


def _run_batch(
    model,
    llm_variants: list[list[torch.Tensor]],
    specs: list[PatchSpec],
    decoder: Decoder,
    layer: int,
    patch_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    tokens_batch = torch.cat(
        [llm_variants[s.base_idx][s.variant_token_idx] for s in specs], dim=0
    ).to(device)
    eta_batch = torch.from_numpy(
        np.stack([s.target_belief for s in specs])
    ).float().to(device)

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
    P = probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)
    return P.astype(np.float32)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate_bins(
    records: list[dict],
    layer_indices: list[int],
) -> dict[str, dict[int, dict]]:
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        key = _bin_key(rec["is_factual"], rec["is_likely"])
        raw[rec["layer"]][rec["base_idx"]][key].append(rec["kl"])

    agg: dict[str, dict[int, dict]] = {b: {} for b in _BINS}
    for layer in layer_indices:
        per_seq: dict[str, list[float]] = {b: [] for b in _BINS}
        for base_kls in raw[layer].values():
            for b in _BINS:
                kls = base_kls.get(b, [])
                if kls:
                    per_seq[b].append(float(np.mean(kls)))
        for b in _BINS:
            vals = per_seq[b]
            n = len(vals)
            agg[b][layer] = {
                "mean":     float(np.mean(vals)) if vals else float("nan"),
                "stderr":   float(np.std(vals) / max(np.sqrt(n), 1)) if vals else float("nan"),
                "n":        n,
                "seq_means": vals,
            }
    return agg


def _sanity_stats(
    records: list[dict],
    layer_indices: list[int],
) -> dict[int, dict]:
    """For each (base_idx, belief_token_idx), std of KL across 3 variant sequences."""
    raw: dict = defaultdict(lambda: defaultdict(list))
    for rec in records:
        raw[rec["layer"]][(rec["base_idx"], rec["belief_token_idx"])].append(rec["kl"])

    stats: dict[int, dict] = {}
    for layer in layer_indices:
        stds = [float(np.std(kls)) for kls in raw[layer].values() if len(kls) >= 2]
        stats[layer] = {
            "mean_within_std": float(np.mean(stds)) if stds else float("nan"),
            "max_within_std":  float(np.max(stds))  if stds else float("nan"),
        }
    return stats


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLOR_LIKELY        = "#1f77b4"
_COLOR_UNLIKELY      = "#ff7f0e"
_FILL_LIKELY         = "rgba(31,119,180,0.13)"
_FILL_UNLIKELY       = "rgba(255,127,14,0.13)"

_PANELS: list[tuple[str, str, str]] = [
    ("Factual",        "factual_likely", "factual_unlikely"),
    ("Counterfactual", "cf_likely",      "cf_unlikely"),
]


def _plot_kl_vs_layer(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    unpatched_mean: float,
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Factual patches", "Counterfactual patches"],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    for col_i, (_title, likely_bin, unlikely_bin) in enumerate(_PANELS):
        col = col_i + 1
        show_legend = (col_i == 0)

        for bin_key, fill_color in [(likely_bin, _FILL_LIKELY), (unlikely_bin, _FILL_UNLIKELY)]:
            means   = [agg[bin_key][l]["mean"]   for l in layer_indices]
            stderrs = [agg[bin_key][l]["stderr"] for l in layer_indices]
            upper = [m + e for m, e in zip(means, stderrs)]
            lower = [m - e for m, e in zip(means, stderrs)]
            fig.add_trace(go.Scatter(
                x=layers_str + layers_str[::-1],
                y=upper + lower[::-1],
                fill="toself", fillcolor=fill_color,
                line=dict(width=0), showlegend=False,
                hoverinfo="skip", mode="lines",
            ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=layers_str,
            y=[unpatched_mean] * len(layer_indices),
            name="Unpatched",
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=show_legend,
        ), row=1, col=col)

        for bin_key, label, color in [
            (likely_bin,   "likely",   _COLOR_LIKELY),
            (unlikely_bin, "unlikely", _COLOR_UNLIKELY),
        ]:
            means = [agg[bin_key][l]["mean"] for l in layer_indices]
            fig.add_trace(go.Scatter(
                x=layers_str, y=means,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                showlegend=show_legend,
            ), row=1, col=col)

    fig.update_yaxes(type="log")
    fig.update_xaxes(title_text="Layer")
    fig.update_yaxes(title_text="KL [nats]", col=1)
    fig.update_layout(
        title=(
            "Activation patching: KL vs layer — factual vs counterfactual patches<br>"
            "<sup>KL(P_patched ‖ P_opt(η_target)) — log scale — mean ± stderr. "
            "If panels match, the factual/cf gap is fully explained by token probability.</sup>"
        ),
        height=460, width=1000,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_sanity_check(
    sanity: dict[int, dict],
    layer_indices: list[int],
    path: Path,
) -> None:
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers_str,
        y=[sanity[l]["mean_within_std"] for l in layer_indices],
        name="Mean within-group std",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=layers_str,
        y=[sanity[l]["max_within_std"] for l in layer_indices],
        name="Max within-group std",
        mode="lines+markers",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title=(
            "Sanity check: KL invariance across variant sequences for fixed belief<br>"
            "<sup>Std of KL across 3 variants (different actual final token) with the same patched belief. "
            "Should be ≈ 0 if the actual token at the patch position is irrelevant.</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="Std of KL [nats]",
        height=420, width=720,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _add_trend_line(
    fig,
    x_vals: list[float],
    y_vals: list[float],
    row: int,
    col: int,
) -> None:
    if len(x_vals) < 5:
        return
    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_vals, dtype=float)
    valid = (y_arr > 0) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 5:
        return
    try:
        coeffs = np.polyfit(x_arr[valid], np.log(y_arr[valid]), deg=2)
        x_fit = np.linspace(x_arr[valid].min(), x_arr[valid].max(), 120)
        y_fit = np.exp(np.polyval(coeffs, x_fit))
        fig.add_trace(go.Scatter(
            x=x_fit.tolist(), y=y_fit.tolist(),
            mode="lines",
            line=dict(color="rgba(0,0,0,0.30)", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ), row=row, col=col)
    except Exception:
        pass


def _plot_scatter(
    records: list[dict],
    layer_indices: list[int],
    x_key: str,
    x_label: str,
    title_suffix: str,
    path: Path,
    n_cols: int = 6,
) -> None:
    n_rows = (len(layer_indices) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in layer_indices],
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    by_layer: dict[int, dict[str, list]] = {
        l: {"xf": [], "yf": [], "xc": [], "yc": []} for l in layer_indices
    }
    for rec in records:
        if rec["layer"] not in by_layer:
            continue
        d = by_layer[rec["layer"]]
        if rec["is_factual"]:
            d["xf"].append(rec[x_key])
            d["yf"].append(rec["kl"])
        else:
            d["xc"].append(rec[x_key])
            d["yc"].append(rec["kl"])

    for i, layer in enumerate(layer_indices):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = (i == 0)
        d = by_layer[layer]

        fig.add_trace(go.Scatter(
            x=d["xc"], y=d["yc"],
            name="counterfactual",
            mode="markers",
            marker=dict(color="#ff7f0e", size=7, opacity=0.55),
            showlegend=show_legend,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=d["xf"], y=d["yf"],
            name="factual",
            mode="markers",
            marker=dict(color="#1f77b4", size=7, opacity=0.55),
            showlegend=show_legend,
        ), row=row, col=col)

        _add_trend_line(fig, d["xf"] + d["xc"], d["yf"] + d["yc"], row, col)

    fig.update_yaxes(type="log")

    for i, layer in enumerate(layer_indices):
        row = i // n_cols + 1
        col = i % n_cols + 1
        if row == n_rows:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="KL [nats]", row=row, col=col)

    fig.update_layout(
        title=(
            f"KL vs {title_suffix}<br>"
            "<sup>Blue = factual | Orange = counterfactual | Black curve = combined trend. "
            "Prediction: same curve regardless of color.</sup>"
        ),
        height=220 * n_rows + 120,
        width=220 * n_cols + 180,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching token probability experiment")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            f"Quick test: layers {DRY_RUN_LAYERS}, "
            f"seq_length={DRY_RUN_SEQ_LEN}, n_sequences={DRY_RUN_N_SEQ}"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config, ActivationPatchingTokenProbConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.layer_indices   = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.seq_length      = DRY_RUN_SEQ_LEN
        config.n_sequences     = DRY_RUN_N_SEQ
        config.patch_position  = DRY_RUN_SEQ_LEN - 1
        config.experiment_name = config.experiment_name + "_dry_run"

    device = get_device()
    out_dir = setup_output_dir(config)
    logger  = setup_logging(out_dir, name="act_patch_tp")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir    : {out_dir}")
    logger.info(f"Device        : {device}")
    logger.info(f"Dry run       : {args.dry_run}")
    logger.info(f"N sequences   : {config.n_sequences}")
    logger.info(f"Seq length    : {config.seq_length}")
    logger.info(f"Patch pos     : {config.patch_position}")
    logger.info(f"Batch size    : {config.batch_size}")
    logger.info(f"Layers        : {config.layer_indices}")
    logger.info(f"Enc-dec dir   : {config.encoder_decoder_dir}")

    enc_dec_dir = Path(config.encoder_decoder_dir)
    N         = config.n_sequences
    L         = config.seq_length
    patch_pos = config.patch_position
    n_vocab   = len(config.vocab_mapping)
    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    prefix_len = L - 1    # prefix has L-1 tokens; variants append one more to reach L

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype

    # ── HMM ───────────────────────────────────────────────────────────────────
    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    T_3d = hmm.T_3d_matrix.cpu().numpy()
    emit = build_emission_matrix(hmm)   # (n_tokens, n_states)
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_vocab, logger)

    # ── Decoders ──────────────────────────────────────────────────────────────
    decoder_base = enc_dec_dir / "decoders" / "pooled"
    decoders: dict[int, Decoder] = {}
    for layer in config.layer_indices:
        dr = DecoderResult.load(decoder_base / f"layer_{layer}")
        decoders[layer] = dr.decoder.to(device)
        decoders[layer].eval()
    logger.info(f"Loaded {len(decoders)} decoders from {decoder_base}")

    # ── Phase 1: generate prefix sequences + variant clean forward passes ─────
    logger.info("Phase 1: generating prefix sequences and variant forward passes ...")

    prefix_beliefs: list[np.ndarray]       = []   # (n_states,) per base sequence
    llm_variants:   list[list[torch.Tensor]] = []  # [base_idx][z] → (1, L) CPU tensor
    unpatched_kl_list: list[float]          = []   # KL(P_unpatched || P_opt(η_z)) per variant

    for batch_start in range(0, N, config.batch_size):
        B = min(config.batch_size, N - batch_start)

        prefix_toks_batch, _, _ = hmm.generate_dataset(B, prefix_len, return_states=True)
        prefix_bels_batch = hmm.compute_belief_state(prefix_toks_batch)
        # prefix_bels_batch shape: (B, prefix_len+1, n_states) = (B, L, n_states)
        # prefix_bels_batch[b, prefix_len] = belief after all L-1 prefix tokens

        batch_variants: list[list[torch.Tensor]] = []
        flat_variants:  list[torch.Tensor]        = []

        for b in range(B):
            prefix_toks_np = prefix_toks_batch[b].cpu().numpy()
            prefix_bel = prefix_bels_batch[b, prefix_len].cpu().numpy().astype(np.float32)
            prefix_beliefs.append(prefix_bel)

            prefix_text = " ".join(idx_to_token[int(t)] for t in prefix_toks_np)
            variants_for_b: list[torch.Tensor] = []
            for z in range(n_vocab):
                variant_text = prefix_text + " " + idx_to_token[z]
                tok = model.to_tokens(variant_text, prepend_bos=False, truncate=False).cpu()
                assert tok.shape[1] == L, f"Expected {L} LLM tokens, got {tok.shape[1]}"
                variants_for_b.append(tok)
                flat_variants.append(tok)
            batch_variants.append(variants_for_b)
            llm_variants.append(variants_for_b)

        # Run clean forward passes in sub-batches of config.batch_size
        P_clean_flat: list[np.ndarray] = []
        for sub_start in range(0, len(flat_variants), config.batch_size):
            sub_batch = torch.cat(
                flat_variants[sub_start : sub_start + config.batch_size], dim=0
            ).to(device)
            with torch.no_grad():
                logits_sub = model.run_with_hooks(sub_batch, fwd_hooks=[], return_type="logits")
            probs_all = F.softmax(logits_sub[:, patch_pos, :].float(), dim=-1)
            probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
            P_sub = (probs_hmm / (probs_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)
            P_clean_flat.extend([P_sub[i] for i in range(len(sub_batch))])
            del logits_sub
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        for b in range(B):
            base_idx   = batch_start + b
            prefix_bel = prefix_beliefs[base_idx]
            for z in range(n_vocab):
                p_clean_z = P_clean_flat[b * n_vocab + z]
                eta_z     = _hmm_step(prefix_bel, z, T_3d)
                p_opt_z   = (eta_z @ emit.T).astype(np.float32)
                unpatched_kl_list.append(float(_kl(p_clean_z[None], p_opt_z[None])[0]))

        logger.info(f"  Sequences {batch_start + 1}–{batch_start + B}/{N} done")

    unpatched_mean = float(np.mean(unpatched_kl_list))
    logger.info(f"Unpatched KL mean (all 3N variants): {unpatched_mean:.4f}")

    # ── Build patch specs ─────────────────────────────────────────────────────
    logger.info("Building patch specs ...")
    all_specs = _build_specs(prefix_beliefs, T_3d, emit, n_vocab)
    logger.info(f"  {len(all_specs)} total specs ({n_vocab ** 2} per base sequence)")

    # ── Phase 2: intervention loop (per layer, batched) ───────────────────────
    logger.info("Phase 2: intervention loop ...")
    all_records: list[dict] = []
    n_batches = (len(all_specs) + config.batch_size - 1) // config.batch_size

    for layer_i, layer in enumerate(config.layer_indices):
        logger.info(f"  Layer {layer} ({layer_i + 1}/{len(config.layer_indices)}) ...")
        decoder = decoders[layer]

        for batch_i in range(n_batches):
            start       = batch_i * config.batch_size
            specs_batch = all_specs[start : start + config.batch_size]

            P_pat = _run_batch(
                model, llm_variants, specs_batch, decoder,
                layer, patch_pos, mid_tok_ids, device, model_dtype,
            )

            etas     = np.stack([s.target_belief for s in specs_batch])
            kl_vals  = _kl(P_pat, etas @ emit.T)

            for j, spec in enumerate(specs_batch):
                all_records.append({
                    "layer":             layer,
                    "base_idx":          spec.base_idx,
                    "variant_token_idx": spec.variant_token_idx,
                    "belief_token_idx":  spec.belief_token_idx,
                    "is_factual":        spec.is_factual,
                    "is_likely":         spec.is_likely,
                    "token_prob":        spec.token_prob,
                    "belief_shift_l2":   spec.belief_shift_l2,
                    "kl":                float(kl_vals[j]),
                })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        layer_recs = [r for r in all_records if r["layer"] == layer]
        fact_mean  = float(np.mean([r["kl"] for r in layer_recs if r["is_factual"]]))
        cf_mean    = float(np.mean([r["kl"] for r in layer_recs if not r["is_factual"]]))
        logger.info(
            f"    factual={fact_mean:.4f}  cf={cf_mean:.4f}  Δ={cf_mean - fact_mean:+.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("Saving records ...")
    with open(out_dir / "records.json", "w") as f:
        json.dump(all_records, f)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("Aggregating ...")
    agg    = _aggregate_bins(all_records, config.layer_indices)
    sanity = _sanity_stats(all_records, config.layer_indices)

    metrics = {
        "unpatched_kl_mean": unpatched_mean,
        "agg":    {b: {str(l): v for l, v in by_layer.items()} for b, by_layer in agg.items()},
        "sanity": {str(l): v for l, v in sanity.items()},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    _plot_kl_vs_layer(agg, config.layer_indices, unpatched_mean, fig_dir / "kl_vs_layer")
    _plot_sanity_check(sanity, config.layer_indices, fig_dir / "sanity_check")
    _plot_scatter(
        all_records, config.layer_indices,
        x_key="token_prob",
        x_label="P(token | prefix belief)",
        title_suffix="token probability",
        path=fig_dir / "scatter_kl_vs_token_prob",
    )
    _plot_scatter(
        all_records, config.layer_indices,
        x_key="belief_shift_l2",
        x_label="‖η_target − prefix belief‖₂",
        title_suffix="belief shift ‖η_target − prefix belief‖₂",
        path=fig_dir / "scatter_kl_vs_belief_shift",
    )
    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
