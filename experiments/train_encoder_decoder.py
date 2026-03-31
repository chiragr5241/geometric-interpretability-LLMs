#!/usr/bin/env python3
"""Train encoder-decoder pairs for activation patching (SPAR-14 / SPAR-21).

Trains a linear encoder (belief-state probe) and an affine decoder
(activation reconstructor) at each target layer.  Both are trained on
pooled activations from HMM sequences and evaluated on held-out data.

Supports two data-collection modes controlled by ``balance`` in the config:

  balance: false  (default)
      Pool all post-convergence positions from N_train sequences.  Evaluate
      on N_eval held-out sequences.

  balance: true
      Generate a large pool of sequences, bin every position by token
      probability p_t (surprise / ambiguous / expected), subsample to
      equal-size bins, then stratified 80/20 train/eval split.  Produces
      extra diagnostics: bin histogram, p_t fidelity grids, natural
      (unbalanced) eval, and optional baseline comparison.

No BOS token is used (prepend_bos=False).  Alignment:
    cache position t  ->  beliefs[t+1]

Usage:
    python experiments/train_encoder_decoder.py experiments/configs/train_encoder_decoder.yaml
    python experiments/train_encoder_decoder.py experiments/configs/train_encoder_decoder.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decoder import DecoderInput, DecoderResult
from decoder import train_batched as train_decoders_batched
from encoder_decoder_utils import (
    binned_mean,
    decoder_loss_curves,
    encoder_mse_curves,
    evaluate_decoder,
    evaluate_encoder,
    evaluate_roundtrip,
    fidelity_arrays,
    layer_line_plot,
    make_grid_figure,
    pearson,
    r2,
    simplex_scatter,
)
from experiment import ExperimentConfig, apply_runtime_overrides, load_config, setup_output_dir
from experiment_utils import build_emission_matrix, get_device, load_model, setup_logging
from hmm.hmm import Mess3HMM
from probes import ProbeInput, ProbeResult, train_probes_batched

DRY_RUN_LAYERS = [0, 2, 6, 10, 17, 25]
DRY_RUN_SEQ_LEN = 100
DRY_RUN_TRANSIENT = 15

RowSpec = tuple[int, int, float, int]


@dataclass
class TrainEncoderDecoderConfig(ExperimentConfig):
    layer_indices: list[int] = field(default_factory=list)
    seq_length: int = 400
    post_convergence_start: int = 100
    vocab_mapping: dict[str, int] = field(default_factory=dict)
    encoder_params: dict[str, Any] = field(default_factory=dict)
    decoder_params: dict[str, Any] = field(default_factory=dict)
    simplex_layers: list[int] = field(default_factory=list)
    max_probes_per_batch: int = 22
    max_decoders_per_batch: int = 4
    n_ctx_override: int | None = None
    random_seed: int = 42

    n_train_sequences: int = 15
    n_eval_sequences: int = 5

    balance: bool = False
    balance_target_size: int | None = None
    n_pool_sequences: int | None = None
    train_frac: float = 0.8
    n_natural_eval_sequences: int = 25
    n_bins_scatter: int = 20
    baseline_encoder_decoder_dir: str | None = None


def _bin_label(p_t: float) -> int:
    if p_t < 0.3:
        return 0
    if p_t < 0.6:
        return 1
    return 2


def _pass1_collect(
    n_seq: int,
    L: int,
    hmm: Mess3HMM,
    emit_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int, float]]]]:
    bin_rows: list[list[tuple[int, int, float]]] = [[], [], []]
    tok_list: list[np.ndarray] = []
    bel_list: list[np.ndarray] = []
    for seq_idx in range(n_seq):
        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens = tokens_batch[0].cpu().numpy().astype(np.int64)
        seq_beliefs = beliefs_batch[0].cpu().numpy().astype(np.float32)
        tok_list.append(seq_tokens)
        bel_list.append(seq_beliefs)
        for t in range(L):
            b_pref = seq_beliefs[t]
            tok = int(seq_tokens[t])
            p_t = float((b_pref * emit_np[tok]).sum())
            bin_rows[_bin_label(p_t)].append((seq_idx, t, p_t))
    return np.stack(tok_list, axis=0), np.stack(bel_list, axis=0), bin_rows


def _balance_and_stratify(
    bin_rows: list[list[tuple[int, int, float]]],
    per_bin_target: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[list[RowSpec], list[RowSpec], int]:
    counts = [len(bin_rows[b]) for b in range(3)]
    m_raw = min(counts)
    m = min(m_raw, per_bin_target) if per_bin_target > 0 else m_raw
    m = max(m, 0)
    balanced: list[RowSpec] = []
    for b in range(3):
        rows = list(bin_rows[b])
        rng.shuffle(rows)
        take = min(m, len(rows))
        for seq_idx, t, p_t in rows[:take]:
            balanced.append((seq_idx, t, p_t, b))
    train_specs: list[RowSpec] = []
    eval_specs: list[RowSpec] = []
    for b in range(3):
        rb = [row for row in balanced if row[3] == b]
        rng.shuffle(rb)
        n_tr = int(len(rb) * train_frac)
        train_specs.extend(rb[:n_tr])
        eval_specs.extend(rb[n_tr:])
    return train_specs, eval_specs, m


def _fill_train_and_eval_rows(
    train_specs: list[RowSpec],
    eval_specs: list[RowSpec],
    all_tokens: np.ndarray,
    all_beliefs: np.ndarray,
    model: Any,
    layer_indices: list[int],
    hook_names: list[str],
    idx_to_token: dict[int, str],
    logger: logging.Logger,
) -> tuple[
    tuple[dict[int, np.ndarray], np.ndarray, np.ndarray],
    tuple[dict[int, np.ndarray], np.ndarray, np.ndarray],
]:
    by_seq: dict[int, list[tuple[str, int, RowSpec]]] = defaultdict(list)
    for k, spec in enumerate(train_specs):
        by_seq[spec[0]].append(("train", k, spec))
    for k, spec in enumerate(eval_specs):
        by_seq[spec[0]].append(("eval", k, spec))

    n_tr = len(train_specs)
    n_ev = len(eval_specs)
    d_model = model.cfg.d_model
    n_states = all_beliefs.shape[-1]
    train_acts: dict[int, np.ndarray] = {
        layer: np.zeros((n_tr, d_model), dtype=np.float32) for layer in layer_indices
    }
    eval_acts: dict[int, np.ndarray] = {
        layer: np.zeros((n_ev, d_model), dtype=np.float32) for layer in layer_indices
    }
    train_beliefs = np.zeros((n_tr, n_states), dtype=np.float32)
    eval_beliefs = np.zeros((n_ev, n_states), dtype=np.float32)
    train_p_t = np.zeros((n_tr,), dtype=np.float64)
    eval_p_t = np.zeros((n_ev,), dtype=np.float64)
    for seq_idx in sorted(by_seq.keys()):
        seq_tokens = all_tokens[seq_idx]
        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == all_tokens.shape[1]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type=None,
            )
        for pool, k, spec in by_seq[seq_idx]:
            _s, t, p_t, _b = spec
            if pool == "train":
                for layer in layer_indices:
                    train_acts[layer][k] = (
                        cache[f"blocks.{layer}.hook_resid_post"][0, t].float().cpu().numpy()
                    )
                train_beliefs[k] = all_beliefs[seq_idx, t + 1]
                train_p_t[k] = p_t
            else:
                for layer in layer_indices:
                    eval_acts[layer][k] = (
                        cache[f"blocks.{layer}.hook_resid_post"][0, t].float().cpu().numpy()
                    )
                eval_beliefs[k] = all_beliefs[seq_idx, t + 1]
                eval_p_t[k] = p_t
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"    Forward seq {seq_idx + 1}/{all_tokens.shape[0]} (train+eval rows)")

    return (
        (train_acts, train_beliefs, train_p_t),
        (eval_acts, eval_beliefs, eval_p_t),
    )


def _natural_eval_tensors(
    n_seq: int,
    L: int,
    P: int,
    hmm: Mess3HMM,
    model: Any,
    layer_indices: list[int],
    hook_names: list[str],
    idx_to_token: dict[int, str],
    emit_np: np.ndarray,
    logger: logging.Logger,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    d_model = model.cfg.d_model
    n_states = hmm.num_states
    n_usable = L - P
    acts: dict[int, np.ndarray] = {
        layer: np.zeros((n_seq * n_usable, d_model), dtype=np.float32)
        for layer in layer_indices
    }
    beliefs_true = np.zeros((n_seq * n_usable, n_states), dtype=np.float32)
    p_t_flat = np.zeros((n_seq * n_usable,), dtype=np.float64)
    row = 0
    for seq_idx in range(n_seq):
        logger.info(f"  Natural eval sequence {seq_idx + 1}/{n_seq}")
        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens = tokens_batch[0].cpu().numpy().astype(np.int64)
        seq_beliefs = beliefs_batch[0].cpu().numpy().astype(np.float32)
        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L
        with torch.no_grad():
            _, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type=None,
            )
        for t in range(P, L):
            b_pref = seq_beliefs[t]
            tok = int(seq_tokens[t])
            p_t = float((b_pref * emit_np[tok]).sum())
            for layer in layer_indices:
                acts[layer][row] = (
                    cache[f"blocks.{layer}.hook_resid_post"][0, t].float().cpu().numpy()
                )
            beliefs_true[row] = seq_beliefs[t + 1]
            p_t_flat[row] = p_t
            row += 1
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return acts, beliefs_true, p_t_flat


def _eval_rows_by_sequence(eval_specs: list[RowSpec]) -> dict[int, list[int]]:
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k, spec in enumerate(eval_specs):
        seq_idx, t, _p_t, _b = spec
        buckets[seq_idx].append((t, k))
    out: dict[int, list[int]] = {}
    for seq_idx, pairs in buckets.items():
        pairs.sort(key=lambda x: x[0])
        out[seq_idx] = [row_k for _t, row_k in pairs]
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _collect_unbalanced(
    config: TrainEncoderDecoderConfig,
    model: Any,
    hmm: Mess3HMM,
    idx_to_token: dict[int, str],
    hook_names: list[str],
    logger: logging.Logger,
) -> tuple[
    dict[int, ProbeInput],
    dict[int, DecoderInput],
    list[dict[int, np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
]:
    N_train = config.n_train_sequences
    N_eval = config.n_eval_sequences
    N_total = N_train + N_eval
    L = config.seq_length
    P = config.post_convergence_start
    n_vocab = len(config.vocab_mapping)

    logger.info("Phase 1: forward passes (no BOS) ...")
    all_acts: list[dict[int, np.ndarray]] = []
    all_beliefs: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []

    for seq_idx in range(N_total):
        label = "train" if seq_idx < N_train else "eval"
        logger.info(f"  Sequence {seq_idx + 1}/{N_total} [{label}]")

        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        beliefs_batch = hmm.compute_belief_state(tokens_batch)
        seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
        seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy()

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L

        with torch.no_grad():
            _, cache = model.run_with_cache(
                llm_tokens,
                names_filter=hook_names,
                return_type=None,
            )

        seq_acts = {
            layer: cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu().numpy()
            for layer in config.layer_indices
        }
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_acts.append(seq_acts)
        all_beliefs.append(seq_beliefs)
        all_tokens.append(seq_tokens)

    logger.info("Phase 2: pooling post-convergence training data ...")
    n_pts_per_seq = L - P + 1
    logger.info(f"  {n_pts_per_seq} positions/seq x {N_train} seqs = {N_train * n_pts_per_seq} total")

    pooled_encoder_inputs: dict[int, ProbeInput] = {}
    pooled_decoder_inputs: dict[int, DecoderInput] = {}

    for layer in config.layer_indices:
        acts_list, beliefs_list, tokens_list = [], [], []
        for seq_idx in range(N_train):
            acts_list.append(all_acts[seq_idx][layer][P - 1:])
            beliefs_list.append(all_beliefs[seq_idx][P:])
            tokens_list.append(all_tokens[seq_idx][P - 1:])

        acts_pooled = np.concatenate(acts_list, axis=0)
        beliefs_pooled = np.concatenate(beliefs_list, axis=0)
        tokens_pooled = np.concatenate(tokens_list, axis=0)
        n_total = acts_pooled.shape[0]
        logger.info(f"  Layer {layer}: {n_total} pooled points")

        pooled_encoder_inputs[layer] = ProbeInput(
            activations=acts_pooled,
            gt_belief_states=beliefs_pooled,
            tokens=tokens_pooled,
            gt_next_token_preds=np.zeros((n_total, n_vocab), dtype=np.float32),
            computed_next_token_preds=np.zeros((n_total, n_vocab), dtype=np.float32),
        )
        pooled_decoder_inputs[layer] = DecoderInput(
            activations=acts_pooled,
            belief_states=beliefs_pooled,
        )

    return pooled_encoder_inputs, pooled_decoder_inputs, all_acts, all_beliefs, all_tokens


def _collect_balanced(
    config: TrainEncoderDecoderConfig,
    model: Any,
    hmm: Mess3HMM,
    emit_np: np.ndarray,
    idx_to_token: dict[int, str],
    hook_names: list[str],
    rng: np.random.Generator,
    out_dir: Path,
    logger: logging.Logger,
) -> tuple[
    dict[int, ProbeInput],
    dict[int, DecoderInput],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[RowSpec],
]:
    L = config.seq_length
    n_vocab = len(config.vocab_mapping)

    unbalanced_size = config.n_train_sequences * (L - config.post_convergence_start + 1)
    target_size = config.balance_target_size or int(math.ceil(unbalanced_size * 1.5))
    per_bin_target = target_size // 3

    n_pool = config.n_pool_sequences
    if n_pool is None:
        surprise_frac = 0.10
        safety = 1.5
        total_positions_needed = per_bin_target / (config.train_frac * surprise_frac) * safety
        n_pool = max(20, int(math.ceil(total_positions_needed / L)))
    logger.info(f"Balance target_size={target_size} (per_bin={per_bin_target}), pool sequences={n_pool}")

    logger.info("Pass 1: HMM sequences + p_t bins (no LLM) ...")
    all_tokens, all_beliefs, bin_rows = _pass1_collect(n_pool, L, hmm, emit_np)
    counts_before = [len(bin_rows[b]) for b in range(3)]
    logger.info(f"  Bin counts (surprise / ambiguous / expected): {counts_before}")

    train_specs, eval_specs, m_bal = _balance_and_stratify(
        bin_rows, per_bin_target, config.train_frac, rng,
    )
    counts_after = [sum(1 for row in train_specs + eval_specs if row[3] == b) for b in range(3)]
    logger.info(f"  Balanced per-bin cap used: {m_bal}, rows per bin after: {counts_after}")
    logger.info(f"  Stratified train rows: {len(train_specs)}, balanced eval: {len(eval_specs)}")

    if m_bal == 0:
        raise RuntimeError("No positions in smallest bin; increase n_pool_sequences.")
    if per_bin_target > 0 and m_bal < per_bin_target:
        logger.warning(
            f"Smallest bin has only {m_bal} positions (target per_bin={per_bin_target}). "
            "Consider increasing n_pool_sequences."
        )

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Bar(x=["surprise", "ambiguous", "expected"], y=counts_before, name="Before balance")
    )
    fig_hist.add_trace(
        go.Bar(x=["surprise", "ambiguous", "expected"], y=[m_bal, m_bal, m_bal], name="After balance")
    )
    fig_hist.update_layout(
        title="p_t bin counts: full pool vs balanced subsample",
        barmode="group", height=420, width=560,
    )
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_hist.write_image(str(out_dir / "figures" / "bin_histogram.png"))

    with open(out_dir / "pool_stats.json", "w") as f:
        json.dump({
            "balance_target_size": target_size,
            "per_bin_target": per_bin_target,
            "counts_before_balance": counts_before,
            "min_bin_cap": m_bal,
            "counts_after_balance_per_bin": counts_after,
            "n_train_rows": len(train_specs),
            "n_balanced_eval_rows": len(eval_specs),
        }, f, indent=2)

    logger.info("Pass 2: LLM forwards for train + balanced eval rows ...")
    (train_acts, train_beliefs, train_p_t), (eval_acts, eval_beliefs, eval_p_t) = (
        _fill_train_and_eval_rows(
            train_specs, eval_specs, all_tokens, all_beliefs, model,
            config.layer_indices, hook_names, idx_to_token, logger,
        )
    )

    n_tr = train_beliefs.shape[0]
    dummy_pred = np.zeros((n_tr, n_vocab), dtype=np.float32)
    pooled_encoder_inputs: dict[int, ProbeInput] = {}
    pooled_decoder_inputs: dict[int, DecoderInput] = {}
    for layer in config.layer_indices:
        pooled_encoder_inputs[layer] = ProbeInput(
            activations=train_acts[layer],
            gt_belief_states=train_beliefs,
            tokens=np.zeros((n_tr,), dtype=np.int64),
            gt_next_token_preds=dummy_pred,
            computed_next_token_preds=dummy_pred,
        )
        pooled_decoder_inputs[layer] = DecoderInput(
            activations=train_acts[layer],
            belief_states=train_beliefs,
        )

    return pooled_encoder_inputs, pooled_decoder_inputs, eval_acts, eval_beliefs, eval_p_t, eval_specs


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train encoder-decoder pairs")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-user", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, TrainEncoderDecoderConfig)
    apply_runtime_overrides(config, output_user=args.output_user)

    if args.dry_run:
        config.layer_indices = [l for l in DRY_RUN_LAYERS if l in config.layer_indices]
        config.seq_length = DRY_RUN_SEQ_LEN
        config.post_convergence_start = DRY_RUN_TRANSIENT
        config.experiment_name = f"{config.experiment_name}_dry_run"
        config.encoder_params = {**config.encoder_params, "epochs": 80}
        config.decoder_params = {
            **config.decoder_params,
            "max_epochs": 600,
            "patience": 60,
        }
        if config.balance:
            config.n_pool_sequences = 8
            config.balance_target_size = 90
            config.n_natural_eval_sequences = 3
        else:
            config.n_train_sequences = 3
            config.n_eval_sequences = 2

    rng = np.random.default_rng(config.random_seed)
    device = get_device()
    out_dir = setup_output_dir(config)
    (out_dir / "probes" / "pooled").mkdir(parents=True, exist_ok=True)
    (out_dir / "decoders" / "pooled").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, name="enc_dec")

    L = config.seq_length
    P = config.post_convergence_start
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Device     : {device}")
    logger.info(f"Balanced   : {config.balance}")
    logger.info(f"Dry run    : {args.dry_run}")
    logger.info(f"Layers     : {config.layer_indices}")

    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    emit_np = build_emission_matrix(hmm)
    logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_vocab = len(config.vocab_mapping)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Data collection ─────────────────────────────────────────────────────
    eval_specs: list[RowSpec] | None = None
    eval_acts_balanced: dict[int, np.ndarray] | None = None
    eval_beliefs_balanced: np.ndarray | None = None
    eval_p_t_balanced: np.ndarray | None = None
    all_acts_unbalanced: list[dict[int, np.ndarray]] | None = None
    all_beliefs_unbalanced: list[np.ndarray] | None = None

    if config.balance:
        (
            pooled_encoder_inputs,
            pooled_decoder_inputs,
            eval_acts_balanced,
            eval_beliefs_balanced,
            eval_p_t_balanced,
            eval_specs,
        ) = _collect_balanced(
            config, model, hmm, emit_np, idx_to_token, hook_names, rng, out_dir, logger,
        )
    else:
        (
            pooled_encoder_inputs,
            pooled_decoder_inputs,
            all_acts_unbalanced,
            all_beliefs_unbalanced,
            _all_tokens,
        ) = _collect_unbalanced(
            config, model, hmm, idx_to_token, hook_names, logger,
        )

    # ── Train encoders ──────────────────────────────────────────────────────
    enc_p = config.encoder_params
    logger.info(
        f"Training encoders (lr={enc_p.get('lr', 1e-3)}, "
        f"epochs={enc_p.get('epochs', 1000)}) ..."
    )
    pooled_encoder_results: dict[int, ProbeResult] = train_probes_batched(
        pooled_encoder_inputs,
        lr=enc_p.get("lr", 1e-3),
        epochs=enc_p.get("epochs", 1000),
        max_probes_per_batch=config.max_probes_per_batch,
    )

    # ── Train decoders ──────────────────────────────────────────────────────
    dec_p = config.decoder_params
    logger.info(
        f"Training decoders (lr={dec_p.get('lr', 1e-3)}, "
        f"max_epochs={dec_p.get('max_epochs', 1000)}, "
        f"patience={dec_p.get('patience', 200)}, "
        f"min_relative_improvement={dec_p.get('min_relative_improvement', 1e-3)}) ..."
    )
    pooled_decoder_results: dict[int, DecoderResult] = train_decoders_batched(
        pooled_decoder_inputs,
        lr=dec_p.get("lr", 1e-3),
        max_epochs=dec_p.get("max_epochs", 1000),
        patience=dec_p.get("patience", 200),
        min_relative_improvement=dec_p.get("min_relative_improvement", 1e-3),
        max_decoders_per_batch=config.max_decoders_per_batch,
    )

    del pooled_encoder_inputs, pooled_decoder_inputs

    # ── Evaluate ────────────────────────────────────────────────────────────
    logger.info("Evaluating ...")
    per_layer_metrics: dict[str, dict[str, Any]] = {}

    for layer in config.layer_indices:
        pr = pooled_encoder_results[layer]
        dr = pooled_decoder_results[layer]
        split_idx = pr.test_split_idx
        train_pred = pr.computed_belief_states[:split_idx]
        train_gt = pr.gt_belief_states[:split_idx]
        test_pred = pr.computed_belief_states[split_idx:]
        test_gt = pr.gt_belief_states[split_idx:]

        enc_internal_train_mse = float(np.mean((train_pred - train_gt) ** 2))
        enc_internal_train_r2 = r2(train_pred, train_gt)
        enc_internal_test_mse = pr.test_mse
        enc_internal_test_r2 = r2(test_pred, test_gt)

        eval_enc_mses: list[float] = []
        eval_enc_r2s: list[float] = []
        eval_dec_losses: list[float] = []
        eval_dec_norm_losses: list[float] = []
        eval_rt_losses: list[float] = []

        if config.balance:
            assert eval_acts_balanced is not None
            assert eval_beliefs_balanced is not None
            assert eval_specs is not None
            row_by_seq = _eval_rows_by_sequence(eval_specs)
            for _seq_idx, ks in sorted(row_by_seq.items()):
                acts_e = eval_acts_balanced[layer][ks]
                beliefs_e = eval_beliefs_balanced[ks]
                enc_mse, enc_r2_val = evaluate_encoder(pr.probe, acts_e, beliefs_e)
                dec_loss, dec_norm = evaluate_decoder(dr.decoder, beliefs_e, acts_e)
                rt_loss = evaluate_roundtrip(pr.probe, dr.decoder, acts_e)
                eval_enc_mses.append(enc_mse)
                eval_enc_r2s.append(enc_r2_val)
                eval_dec_losses.append(dec_loss)
                eval_dec_norm_losses.append(dec_norm)
                eval_rt_losses.append(rt_loss)
        else:
            assert all_acts_unbalanced is not None
            assert all_beliefs_unbalanced is not None
            N_train = config.n_train_sequences
            N_total = N_train + config.n_eval_sequences
            for seq_idx in range(N_train, N_total):
                acts_e = all_acts_unbalanced[seq_idx][layer][P - 1:]
                beliefs_e = all_beliefs_unbalanced[seq_idx][P:]
                enc_mse, enc_r2_val = evaluate_encoder(pr.probe, acts_e, beliefs_e)
                dec_loss, dec_norm = evaluate_decoder(dr.decoder, beliefs_e, acts_e)
                rt_loss = evaluate_roundtrip(pr.probe, dr.decoder, acts_e)
                eval_enc_mses.append(enc_mse)
                eval_enc_r2s.append(enc_r2_val)
                eval_dec_losses.append(dec_loss)
                eval_dec_norm_losses.append(dec_norm)
                eval_rt_losses.append(rt_loss)

        logger.info(
            f"Layer {layer:2d}: "
            f"enc_r2 int={enc_internal_test_r2:.3f} eval={np.mean(eval_enc_r2s):.3f}  "
            f"dec_loss int={dr.eval_loss:.4e} eval={np.mean(eval_dec_losses):.4e}  "
            f"rt_eval={np.mean(eval_rt_losses):.4e}"
        )

        per_layer_metrics[str(layer)] = {
            "encoder": {
                "internal_train_mse": enc_internal_train_mse,
                "internal_train_r2": enc_internal_train_r2,
                "internal_test_mse": enc_internal_test_mse,
                "internal_test_r2": enc_internal_test_r2,
                "eval_mse_per_seq": [float(v) for v in eval_enc_mses],
                "eval_r2_per_seq": [float(v) for v in eval_enc_r2s],
                "eval_mse_mean": float(np.mean(eval_enc_mses)),
                "eval_r2_mean": float(np.mean(eval_enc_r2s)),
            },
            "decoder": {
                "internal_train_loss": dr.train_loss,
                "internal_eval_loss": dr.eval_loss,
                "internal_train_normalized_loss": dr.normalized_train_loss,
                "internal_eval_normalized_loss": dr.normalized_eval_loss,
                "eval_loss_per_seq": [float(v) for v in eval_dec_losses],
                "eval_normalized_loss_per_seq": [float(v) for v in eval_dec_norm_losses],
                "eval_loss_mean": float(np.mean(eval_dec_losses)),
                "eval_normalized_loss_mean": float(np.mean(eval_dec_norm_losses)),
            },
            "roundtrip": {
                "eval_loss_per_seq": [float(v) for v in eval_rt_losses],
                "eval_loss_mean": float(np.mean(eval_rt_losses)),
            },
        }

    # ── Save weights ────────────────────────────────────────────────────────
    logger.info("Saving encoder/decoder weights ...")
    for layer in config.layer_indices:
        pooled_encoder_results[layer].save(out_dir / "probes" / "pooled" / f"layer_{layer}")
        pooled_decoder_results[layer].save(out_dir / "decoders" / "pooled" / f"layer_{layer}")

    # ── Balance-specific: natural eval + fidelity ───────────────────────────
    metrics: dict[str, Any] = {**per_layer_metrics}

    if config.balance:
        assert eval_acts_balanced is not None
        assert eval_beliefs_balanced is not None
        assert eval_p_t_balanced is not None

        logger.info("Natural eval forward passes ...")
        nat_acts, nat_beliefs, nat_p_t = _natural_eval_tensors(
            config.n_natural_eval_sequences, L, P, hmm, model,
            config.layer_indices, hook_names, idx_to_token, emit_np, logger,
        )

        def eval_regime_arrays(
            acts_by_layer: dict[int, np.ndarray],
            beliefs_v: np.ndarray,
            p_t_v: np.ndarray,
        ) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
            per_layer: dict[str, Any] = {}
            scatter: dict[str, dict[int, np.ndarray]] = {
                "dec_norm": {}, "dec_raw": {}, "rt_norm": {}, "rt_raw": {},
            }
            for layer_idx in config.layer_indices:
                pr_l = pooled_encoder_results[layer_idx]
                dr_l = pooled_decoder_results[layer_idx]
                md, mdn, mrt, mrtn = fidelity_arrays(
                    acts_by_layer[layer_idx], beliefs_v,
                    pr_l.probe, dr_l.decoder, device,
                )
                scatter["dec_raw"][layer_idx] = md
                scatter["dec_norm"][layer_idx] = mdn
                scatter["rt_raw"][layer_idx] = mrt
                scatter["rt_norm"][layer_idx] = mrtn
                acts_t = torch.from_numpy(acts_by_layer[layer_idx]).float().to(device)
                with torch.no_grad():
                    pred_b = pr_l.probe(acts_t).cpu().numpy()
                per_layer[str(layer_idx)] = {
                    "pearson_dec_norm": pearson(p_t_v, mdn),
                    "pearson_rt_norm": pearson(p_t_v, mrtn),
                    "mean_mse_dec_norm": float(mdn.mean()),
                    "mean_mse_rt_norm": float(mrtn.mean()),
                    "enc_mse": float(np.mean((pred_b - beliefs_v) ** 2)),
                    "enc_r2": r2(pred_b, beliefs_v),
                }
            return per_layer, scatter

        bal_per, sc_bal = eval_regime_arrays(eval_acts_balanced, eval_beliefs_balanced, eval_p_t_balanced)
        nat_per, sc_nat = eval_regime_arrays(nat_acts, nat_beliefs, nat_p_t)
        metrics["balanced_eval"] = bal_per
        metrics["natural_eval"] = nat_per

        if config.baseline_encoder_decoder_dir:
            base = Path(config.baseline_encoder_decoder_dir)
            dec_b = base / "decoders" / "pooled"
            pr_b = base / "probes" / "pooled"
            metrics["baseline_natural_eval"] = {}
            for layer_idx in config.layer_indices:
                probe_b = ProbeResult.load(pr_b / f"layer_{layer_idx}").probe.to(device)
                decoder_b = DecoderResult.load(dec_b / f"layer_{layer_idx}").decoder.to(device)
                _md, mdn, _mrt, _mrtn = fidelity_arrays(
                    nat_acts[layer_idx], nat_beliefs, probe_b, decoder_b, device,
                )
                metrics["baseline_natural_eval"][str(layer_idx)] = {
                    "pearson_dec_norm": pearson(nat_p_t, mdn),
                    "mean_mse_dec_norm": float(mdn.mean()),
                }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    # ── Plots ───────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    decoder_loss_curves(pooled_decoder_results, config.layer_indices, fig_dir / "decoder_loss_curves")
    encoder_mse_curves(pooled_encoder_results, config.layer_indices, fig_dir / "encoder_mse_curves")

    layer_line_plot(
        train_vals=[metrics[str(l)]["encoder"]["internal_train_r2"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["encoder"]["eval_r2_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="R²",
        title="Encoder R² by layer",
        path=fig_dir / "encoder_r2_per_layer",
    )
    layer_line_plot(
        train_vals=[metrics[str(l)]["decoder"]["internal_train_loss"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["decoder"]["eval_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Reconstruction loss (MSE)",
        title="Decoder reconstruction loss by layer (log scale)",
        path=fig_dir / "decoder_recon_loss_per_layer",
        log_y=True,
    )
    layer_line_plot(
        train_vals=[metrics[str(l)]["decoder"]["internal_train_normalized_loss"] for l in config.layer_indices],
        eval_vals=[metrics[str(l)]["decoder"]["eval_normalized_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Normalized reconstruction loss",
        title="Decoder normalized reconstruction loss by layer (log scale)",
        path=fig_dir / "decoder_normalized_loss_per_layer",
        log_y=True,
    )
    layer_line_plot(
        eval_vals=[metrics[str(l)]["roundtrip"]["eval_loss_mean"] for l in config.layer_indices],
        layer_indices=config.layer_indices,
        y_title="Round-trip loss (MSE)",
        title="Round-trip loss ||act - D(E(act))||² by layer (eval)",
        path=fig_dir / "roundtrip_loss_per_layer",
        log_y=True,
    )

    if config.balance:
        assert eval_p_t_balanced is not None
        nb = config.n_bins_scatter
        for sc, p_ref, regime_key, title_suffix in [
            (sc_bal, eval_p_t_balanced, "balanced_eval", "balanced eval"),
            (sc_nat, nat_p_t, "natural_eval", "natural eval"),
        ]:
            specs_plot = [
                ("dec_unnorm", sc["dec_raw"], "Token probability p_t", "Decoder MSE  ||a - a_hat||²", f"Decoder-only MSE vs p_t ({title_suffix})"),
                ("dec_norm", sc["dec_norm"], "Token probability p_t", "Decoder MSE / ||a||²", f"Decoder-only normalized MSE vs p_t ({title_suffix})"),
                ("rt_unnorm", sc["rt_raw"], "Token probability p_t", "Round-trip MSE", f"Round-trip MSE vs p_t ({title_suffix})"),
                ("rt_norm", sc["rt_norm"], "Token probability p_t", "Round-trip MSE / ||a||²", f"Round-trip normalized MSE vs p_t ({title_suffix})"),
            ]
            subdir = fig_dir / regime_key
            subdir.mkdir(parents=True, exist_ok=True)
            for name, y_by_layer, xl, yl, plot_title in specs_plot:
                fig = make_grid_figure(config.layer_indices, p_ref, y_by_layer, xl, yl, plot_title, nb)
                fig.write_image(str(subdir / f"{name}.png"))

    for layer in config.simplex_layers:
        if layer not in config.layer_indices:
            continue
        pr = pooled_encoder_results[layer]
        enc_dev = next(pr.probe.parameters()).device
        if config.balance:
            assert eval_acts_balanced is not None
            assert eval_beliefs_balanced is not None
            with torch.no_grad():
                pred = pr.probe(torch.from_numpy(eval_acts_balanced[layer]).float().to(enc_dev)).cpu().numpy()
            n_ev = len(pred)
            simplex_scatter(
                pred, eval_beliefs_balanced, layer, fig_dir / f"simplex_layer_{layer}",
                title=f"Encoder predictions — Layer {layer} (eval, N={n_ev})",
            )
        else:
            assert all_acts_unbalanced is not None
            assert all_beliefs_unbalanced is not None
            all_pred, all_gt = [], []
            for seq_idx in range(config.n_train_sequences, config.n_train_sequences + config.n_eval_sequences):
                acts_e = all_acts_unbalanced[seq_idx][layer][P - 1:]
                beliefs_e = all_beliefs_unbalanced[seq_idx][P:]
                with torch.no_grad():
                    pred = pr.probe(torch.from_numpy(acts_e).float().to(enc_dev)).cpu().numpy()
                all_pred.append(pred)
                all_gt.append(beliefs_e)
            simplex_scatter(
                np.concatenate(all_pred, axis=0),
                np.concatenate(all_gt, axis=0),
                layer,
                fig_dir / f"simplex_layer_{layer}",
            )

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
