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

Metrics:
    KL-to-target  — KL(P_intervened ∥ P_opt(η_target)) per condition (primary metric).
    KL-to-factual — KL(P_intervened ∥ P_opt(η_factual)) (used for crossing plots).
    KL-to-clean   — KL(P_intervened ∥ P_clean) in 4-category space (A, B, C, junk).
    Ablation uses 4-category (A, B, C, junk) distributions to capture shifts in
    probability mass between HMM tokens and the rest of the vocabulary.

Architecture (batched cross-sequence):
    Phase A: clean forward passes — truncate to L_trunc = eval_act_start + max_k,
             run_with_cache, store P_clean + tail activations per layer.
    Phase B: pre-compute targets — ablation projectors, patch targets, steer deltas
             for each (seq, layer, k), stored in SequenceData objects.
    Phase C: batched interventions — outer loop over layers; ablation and
             combined patch+steer batched across all sequences.

Usage:
    python experiments/single_seq_interventions.py \\
        experiments/configs/single_seq_interventions.yaml
    python experiments/single_seq_interventions.py \\
        experiments/configs/single_seq_interventions.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
DRY_RUN_N_EVAL_POSITIONS = 3
DRY_RUN_LAYERS = [0, 2, 10, 17, 27]
DRY_RUN_K_VALUES = [1, 3]

PATCH_CONDITIONS = ["optimal", "round_trip", "past_consistent", "random"]
STEER_CONDITIONS = ["optimal", "past_consistent", "random"]

CHECKPOINT_FILENAME = "phase_c_checkpoint.json"

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
    max_intervention_batch: int | None = None
    max_variants_per_pass: int | None = None
    n_ctx_override: int | None = None
    post_convergence_start: int = 30
    train_eval_split: float = 0.7
    random_seed: int = 42


# ── Sequence data container ────────────────────────────────────────────────────

@dataclass
class SequenceData:
    seq_idx: int
    llm_tokens: torch.Tensor                              # (1, L_trunc)
    P_clean_3: np.ndarray                                 # (n_hmm,) normalised
    P_clean_4: np.ndarray                                 # (n_hmm+1,) with junk category
    P_opt: np.ndarray                                     # (n_hmm,) Bayesian-optimal
    baseline_kl: float
    clean_acts_tail: dict[int, np.ndarray]                # layer -> (max_k, d_model)
    ablation_projs: dict[int, list[np.ndarray]] = field(default_factory=dict)
    patch_targets: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    steer_deltas: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    patch_target_P_opts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    steer_target_P_opts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p ∥ q) per row. Both (..., n). Returns (...,)."""
    return (p * np.log(np.clip(p, 1e-10, None) / np.clip(q, 1e-10, None))).sum(axis=-1)


def _to_3cat(p: np.ndarray) -> np.ndarray:
    """Drop junk column and renormalise to 3 HMM categories."""
    p3 = p[..., :3].copy()
    return (p3 / (p3.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


def _eval_split_idx(L: int, P: int, train_eval_split: float) -> int:
    n_post_conv = L - P + 1
    return int(n_post_conv * train_eval_split)


def _orthonormal_basis(W: np.ndarray) -> np.ndarray:
    """Orthonormal basis for col-span of W via QR.  W: (d, k) -> Q: (d, k)."""
    Q, _ = np.linalg.qr(W)
    return Q.astype(np.float32)


def _random_basis(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal basis.  Returns Q: (d, k)."""
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _serialize_ckl(src: dict) -> dict:
    """Serialize cond->k->layer->list[tuple] to all-string-key JSON-safe dict."""
    return {
        cond: {
            str(k): {str(l): [list(t) for t in records] for l, records in by_layer.items()}
            for k, by_layer in by_k.items()
        }
        for cond, by_k in src.items()
    }


def _deserialize_ckl(
    raw: dict, conds: list[str], k_values: list[int], layer_indices: list[int]
) -> dict:
    """Restore cond->k->layer->list[tuple] from JSON checkpoint."""
    result: dict = {
        cond: {k: {l: [] for l in layer_indices} for k in k_values}
        for cond in conds
    }
    for cond in conds:
        if cond not in raw:
            continue
        for k in k_values:
            sk = str(k)
            if sk not in raw[cond]:
                continue
            for l in layer_indices:
                if str(l) not in raw[cond][sk]:
                    continue
                result[cond][k][l] = [tuple(t) for t in raw[cond][sk][str(l)]]
    return result


def _serialize_abl(src: dict) -> dict:
    """Serialize variant->layer->list[tuple] to all-string-key JSON-safe dict."""
    return {
        variant: {str(l): [list(t) for t in records] for l, records in by_layer.items()}
        for variant, by_layer in src.items()
    }


def _deserialize_abl(raw: dict, variants: list[str], layer_indices: list[int]) -> dict:
    """Restore variant->layer->list[tuple] from JSON checkpoint."""
    result: dict = {v: {l: [] for l in layer_indices} for v in variants}
    for v in variants:
        if v not in raw:
            continue
        for l in layer_indices:
            if str(l) not in raw[v]:
                continue
            result[v][l] = [tuple(t) for t in raw[v][str(l)]]
    return result


def _save_checkpoint(
    out_dir: Path,
    completed_layers: list[int],
    patch_kl_to_target: dict,
    patch_kl_to_factual: dict,
    patch_kl_to_clean: dict,
    baseline_kl_to_target_patch: dict,
    steer_kl_to_target: dict,
    steer_kl_to_factual: dict,
    steer_kl_to_clean: dict,
    baseline_kl_to_target_steer: dict,
    ablation_kl: dict,
    ablation_kl_to_clean: dict,
) -> None:
    """Atomically save Phase C progress after completing each layer."""
    chk = {
        "completed_layers": completed_layers,
        "patch_kl_to_target": _serialize_ckl(patch_kl_to_target),
        "patch_kl_to_factual": _serialize_ckl(patch_kl_to_factual),
        "patch_kl_to_clean": _serialize_ckl(patch_kl_to_clean),
        "baseline_kl_to_target_patch": _serialize_ckl(baseline_kl_to_target_patch),
        "steer_kl_to_target": _serialize_ckl(steer_kl_to_target),
        "steer_kl_to_factual": _serialize_ckl(steer_kl_to_factual),
        "steer_kl_to_clean": _serialize_ckl(steer_kl_to_clean),
        "baseline_kl_to_target_steer": _serialize_ckl(baseline_kl_to_target_steer),
        "ablation_kl": _serialize_abl(ablation_kl),
        "ablation_kl_to_clean": _serialize_abl(ablation_kl_to_clean),
    }
    tmp = out_dir / f"{CHECKPOINT_FILENAME}.tmp"
    with open(tmp, "w") as f:
        json.dump(chk, f)
    tmp.rename(out_dir / CHECKPOINT_FILENAME)


def _load_checkpoint(
    chk_path: Path,
    conds_patch: list[str],
    conds_steer: list[str],
    k_values: list[int],
    layer_indices: list[int],
) -> dict:
    """Load Phase C checkpoint and deserialize all result dicts."""
    with open(chk_path) as f:
        raw = json.load(f)
    return {
        "completed_layers": raw["completed_layers"],
        "patch_kl_to_target": _deserialize_ckl(raw["patch_kl_to_target"], conds_patch, k_values, layer_indices),
        "patch_kl_to_factual": _deserialize_ckl(raw["patch_kl_to_factual"], conds_patch, k_values, layer_indices),
        "patch_kl_to_clean": _deserialize_ckl(raw["patch_kl_to_clean"], conds_patch, k_values, layer_indices),
        "baseline_kl_to_target_patch": _deserialize_ckl(raw["baseline_kl_to_target_patch"], conds_patch, k_values, layer_indices),
        "steer_kl_to_target": _deserialize_ckl(raw["steer_kl_to_target"], conds_steer, k_values, layer_indices),
        "steer_kl_to_factual": _deserialize_ckl(raw["steer_kl_to_factual"], conds_steer, k_values, layer_indices),
        "steer_kl_to_clean": _deserialize_ckl(raw["steer_kl_to_clean"], conds_steer, k_values, layer_indices),
        "baseline_kl_to_target_steer": _deserialize_ckl(raw["baseline_kl_to_target_steer"], conds_steer, k_values, layer_indices),
        "ablation_kl": _deserialize_abl(raw["ablation_kl"], ["belief", "random"], layer_indices),
        "ablation_kl_to_clean": _deserialize_abl(raw["ablation_kl_to_clean"], ["belief", "random"], layer_indices),
    }


def _extract_4cat(
    logits: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
) -> np.ndarray:
    """Extract (B, 4) distribution: [P(A), P(B), P(C), P(junk)]."""
    probs_all = F.softmax(logits[:, measure_pos, :].float(), dim=-1)
    probs_hmm = probs_all[:, mid_tok_ids].cpu().numpy()
    probs_junk = np.clip(1.0 - probs_hmm.sum(axis=-1, keepdims=True), 1e-10, None)
    return np.concatenate([probs_hmm, probs_junk], axis=-1).astype(np.float32)


# ── Attention monkey-patch ─────────────────────────────────────────────────────

def _patch_attention_dtype(model) -> None:
    """Force TransformerLens attention softmax to stay in model dtype (fp16).

    By default F.softmax promotes fp16→float32, doubling attention-pattern VRAM.
    This patches each attention block to compute softmax in the model's dtype and
    removes the redundant NaN→zero mask (NaN can't arise when causal mask uses
    the model dtype's -inf).
    """
    from transformer_lens.components.abstract_attention import AbstractAttention

    target_dtype = model.cfg.dtype

    _orig_forward = AbstractAttention.forward

    def _patched_forward(self, *args, **kwargs):
        import torch.nn.functional as _F
        orig_softmax = _F.softmax
        def _fp16_softmax(input, dim=None, _stacklevel=3, dtype=None):
            return orig_softmax(input, dim=dim, dtype=dtype or target_dtype)
        _F.softmax = _fp16_softmax
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            _F.softmax = orig_softmax

    AbstractAttention.forward = _patched_forward


# ── Hook-based forward passes ─────────────────────────────────────────────────

def _run_ablated(
    model,
    tokens_batch: torch.Tensor,
    layer: int,
    bases: list[np.ndarray],
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Run belief-subspace ablation at ALL positions. Returns P_ablated (B, 4).

    tokens_batch: (B, L).
    bases: list of B orthonormal basis matrices Q, each (d, k).
    Projection is Q @ Q^T @ v, computed as Q @ (Q^T @ v) to avoid materialising (d, d).
    """
    Q_batch = torch.from_numpy(np.stack(bases, axis=0)).to(device=device, dtype=model_dtype)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        QtV = torch.bmm(Q_batch.transpose(1, 2), value.transpose(1, 2))
        proj = torch.bmm(Q_batch, QtV).transpose(1, 2)
        return value - proj

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch.to(device),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    return _extract_4cat(logits, measure_pos, mid_tok_ids)


def _run_combined(
    model,
    tokens_batch: torch.Tensor,
    layer: int,
    positions: list[int],
    n_patch: int,
    patch_targets: torch.Tensor,
    steer_deltas: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Merged patching + steering in a single forward pass. Returns P_out (B_total, 4)."""
    pos_tensor = torch.tensor(positions, dtype=torch.long)
    patch_t = patch_targets.to(device=device, dtype=model_dtype)
    steer_t = steer_deltas.to(device=device, dtype=model_dtype)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:n_patch, pos_tensor, :] = patch_t
        value[n_patch:, pos_tensor, :] = value[n_patch:, pos_tensor, :] + steer_t
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch.to(device),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    return _extract_4cat(logits, measure_pos, mid_tok_ids)


# ── Batch assembly helpers ────────────────────────────────────────────────────

def _chunk(items: list, max_size: int) -> list[list]:
    return [items[i:i + max_size] for i in range(0, len(items), max_size)]


def _build_ablation_batch(
    seq_data_list: list[SequenceData],
    layer: int,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    tokens_list: list[torch.Tensor] = []
    projs_list: list[np.ndarray] = []
    for sd in seq_data_list:
        n_projs = len(sd.ablation_projs[layer])
        tokens_list.append(sd.llm_tokens.expand(n_projs, -1))
        projs_list.extend(sd.ablation_projs[layer])
    return torch.cat(tokens_list, dim=0), projs_list


def _build_combined_batch(
    seq_data_list: list[SequenceData],
    layer: int,
    k: int,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    patch_tokens: list[torch.Tensor] = []
    steer_tokens: list[torch.Tensor] = []
    patch_tgts: list[torch.Tensor] = []
    steer_dlts: list[torch.Tensor] = []
    for sd in seq_data_list:
        n_p = sd.patch_targets[(layer, k)].shape[0]
        n_s = sd.steer_deltas[(layer, k)].shape[0]
        patch_tokens.append(sd.llm_tokens.expand(n_p, -1))
        steer_tokens.append(sd.llm_tokens.expand(n_s, -1))
        patch_tgts.append(sd.patch_targets[(layer, k)])
        steer_dlts.append(sd.steer_deltas[(layer, k)])
    n_patch = sum(t.shape[0] for t in patch_tgts)
    tokens = torch.cat(patch_tokens + steer_tokens, dim=0)
    return tokens, n_patch, torch.cat(patch_tgts, dim=0), torch.cat(steer_dlts, dim=0)


# ── Result scattering helpers ─────────────────────────────────────────────────

def _scatter_ablation_results(
    P_abl_4: np.ndarray,
    sub_batch: list[SequenceData],
    layer: int,
    ablation_kl: dict[str, dict[int, list[tuple[int, int, float]]]],
    ablation_kl_to_clean: dict[str, dict[int, list[tuple[int, int, float]]]],
) -> None:
    b_offset = 0
    for sd in sub_batch:
        n_projs = len(sd.ablation_projs[layer])
        P_slice_4 = P_abl_4[b_offset:b_offset + n_projs]
        P_slice_3 = _to_3cat(P_slice_4)

        kl_abl = _kl(P_slice_3, sd.P_opt[None])
        ablation_kl["belief"][layer].append((sd.seq_idx, 0, float(kl_abl[0])))
        for draw_i, kl_v in enumerate(kl_abl[1:]):
            ablation_kl["random"][layer].append((sd.seq_idx, draw_i, float(kl_v)))

        kl_abl_to_clean = _kl(P_slice_4, sd.P_clean_4[None])
        ablation_kl_to_clean["belief"][layer].append((sd.seq_idx, 0, float(kl_abl_to_clean[0])))
        for draw_i, kl_v in enumerate(kl_abl_to_clean[1:]):
            ablation_kl_to_clean["random"][layer].append((sd.seq_idx, draw_i, float(kl_v)))

        b_offset += n_projs


def _scatter_combined_results(
    P_out_4: np.ndarray,
    n_patch_total: int,
    sub_batch: list[SequenceData],
    layer: int,
    k: int,
    patch_kl_to_target: dict,
    patch_kl_to_factual: dict,
    steer_kl_to_target: dict,
    steer_kl_to_factual: dict,
    patch_kl_to_clean: dict,
    steer_kl_to_clean: dict,
    baseline_kl_to_target_patch: dict,
    baseline_kl_to_target_steer: dict,
    n_past_consistent_draws: int,
    n_random_patch_draws: int,
) -> None:
    P_patch_all_4 = P_out_4[:n_patch_total]
    P_steer_all_4 = P_out_4[n_patch_total:]
    P_patch_all_3 = _to_3cat(P_patch_all_4)
    P_steer_all_3 = _to_3cat(P_steer_all_4)

    p_offset, s_offset = 0, 0
    for sd in sub_batch:
        n_p = sd.patch_targets[(layer, k)].shape[0]
        n_s = sd.steer_deltas[(layer, k)].shape[0]
        P_patch_3 = P_patch_all_3[p_offset:p_offset + n_p]
        P_steer_3 = P_steer_all_3[s_offset:s_offset + n_s]
        P_patch_4 = P_patch_all_4[p_offset:p_offset + n_p]
        P_steer_4 = P_steer_all_4[s_offset:s_offset + n_s]

        patch_P_opts = sd.patch_target_P_opts[(layer, k)]
        steer_P_opts = sd.steer_target_P_opts[(layer, k)]

        kl_to_tgt = _kl(P_patch_3, patch_P_opts)
        kl_to_fac = _kl(P_patch_3, sd.P_opt[None])
        kl_to_cln = _kl(P_patch_4, sd.P_clean_4[None])
        bl_to_tgt = _kl(sd.P_clean_3[None].repeat(n_p, axis=0), patch_P_opts)

        b_idx = 0
        for cond, n_draws in [("optimal", 1), ("round_trip", 1),
                               ("past_consistent", n_past_consistent_draws),
                               ("random", n_random_patch_draws)]:
            for draw_i in range(n_draws):
                patch_kl_to_target[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_tgt[b_idx])))
                patch_kl_to_factual[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_fac[b_idx])))
                patch_kl_to_clean[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_cln[b_idx])))
                baseline_kl_to_target_patch[cond][k][layer].append((sd.seq_idx, draw_i, float(bl_to_tgt[b_idx])))
                b_idx += 1

        kl_s_tgt = _kl(P_steer_3, steer_P_opts)
        kl_s_fac = _kl(P_steer_3, sd.P_opt[None])
        kl_s_cln = _kl(P_steer_4, sd.P_clean_4[None])
        bl_s_tgt = _kl(sd.P_clean_3[None].repeat(n_s, axis=0), steer_P_opts)

        s_idx = 0
        for cond, n_draws in [("optimal", 1),
                               ("past_consistent", n_past_consistent_draws),
                               ("random", n_random_patch_draws)]:
            for draw_i in range(n_draws):
                steer_kl_to_target[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_tgt[s_idx])))
                steer_kl_to_factual[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_fac[s_idx])))
                steer_kl_to_clean[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_cln[s_idx])))
                baseline_kl_to_target_steer[cond][k][layer].append((sd.seq_idx, draw_i, float(bl_s_tgt[s_idx])))
                s_idx += 1

        p_offset += n_p
        s_offset += n_s


# ── Target belief computation ─────────────────────────────────────────────────

def _optimal_targets(seq_beliefs: np.ndarray, positions: list[int]) -> np.ndarray:
    return np.stack([seq_beliefs[t + 1] for t in positions], axis=0).astype(np.float32)


def _round_trip_targets(
    probe: Probe,
    clean_acts: np.ndarray,
    decoder: Decoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (decoded_acts (k, d_model), round_trip_beliefs (k, n_states))."""
    with torch.no_grad():
        acts_t = torch.from_numpy(clean_acts).float()
        encoded = probe(acts_t)
        decoded = decoder(encoded)
    return decoded.float().numpy(), encoded.float().numpy()


def _past_consistent_targets(
    seq_beliefs: np.ndarray,
    positions: list[int],
    hmm: Mess3HMM,
    rng: np.random.Generator,
) -> np.ndarray:
    k = len(positions)
    initial_belief = seq_beliefs[positions[0]].astype(np.float32)
    _, new_beliefs = hmm.sample_continuation(initial_belief, k, rng)
    return new_beliefs.astype(np.float32)


def _random_targets(positions: list[int], rng: np.random.Generator, n_states: int = 3) -> np.ndarray:
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
    title: str,
    subtitle: str,
    path: Path,
) -> None:
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

    fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
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
    agg_to_factual: dict[int, dict],
    agg_to_target: dict[int, dict],
    baseline_to_factual: float,
    agg_baseline_to_target: dict[int, dict],
    layer_indices: list[int],
    k: int,
    condition: str,
    title: str,
    path: Path,
) -> None:
    """Old-style crossing plot.

    Blue: KL(P ∥ P_opt_factual) — should go UP after patching with wrong belief.
    Orange: KL(P ∥ P_opt_target) — should go DOWN (model adopts injected belief).
    Crossing of the two = causal adoption of injected belief.
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

        pt_fac_mean = agg_to_factual[layer]["mean"]
        pt_fac_err = agg_to_factual[layer]["stderr"]
        pt_tgt_mean = agg_to_target[layer]["mean"]
        pt_tgt_err = agg_to_target[layer]["stderr"]
        bl_tgt_mean = agg_baseline_to_target[layer]["mean"]
        bl_tgt_err = agg_baseline_to_target[layer]["stderr"]

        fig.add_trace(go.Scatter(
            x=["Unpatched", "Patched"],
            y=[baseline_to_factual, pt_fac_mean],
            error_y=dict(type="data", array=[0.0, pt_fac_err], visible=True),
            name="KL to factual opt.",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=["Unpatched", "Patched"],
            y=[bl_tgt_mean, pt_tgt_mean],
            error_y=dict(type="data", array=[bl_tgt_err, pt_tgt_err], visible=True),
            name="KL to target opt.",
            showlegend=show_legend,
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
        ), row=row, col=col)

    fig.update_layout(
        title=(
            f"{title} — crossing ({condition.replace('_', '-')}, k={k})<br>"
            "<sup>Blue = KL to factual (↑ = moved away) | Orange = KL to target (↓ = adopted)</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 220, 1400),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path.with_suffix(".png")))


def _plot_causal_shift(
    agg_to_target: dict[str, dict[int, dict]],
    agg_baseline_to_target: dict[str, dict[int, dict]],
    layer_indices: list[int],
    k: int,
    title: str,
    colors: dict[str, str],
    path: Path,
) -> None:
    """ΔKL = KL_baseline_to_target − KL_intervened_to_target (positive = moved toward target)."""
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for cond, color in colors.items():
        if cond not in agg_to_target:
            continue
        delta = [
            agg_baseline_to_target[cond][l]["mean"] - agg_to_target[cond][l]["mean"]
            for l in layer_indices
        ]
        err = [agg_to_target[cond][l]["stderr"] for l in layer_indices]
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
            "<sup>ΔKL = KL_baseline − KL_intervened toward target: positive = causal</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats] (↑ = toward target)",
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
    parser.add_argument(
        "--resume", type=str, default=None, metavar="DIR",
        help="Resume an interrupted run from DIR (loads phase_c_checkpoint.json; re-runs Phase A+B, skips completed Phase C layers)",
    )
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

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        out_dir = resume_dir
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    else:
        out_dir = setup_output_dir(config)

    logger = setup_logging(out_dir, name="interventions")
    if resume_dir is not None:
        logger.info(f"=== RESUMING from {resume_dir} ===")
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
    logger.info(f"Max batch       : {config.max_intervention_batch if config.max_intervention_batch is not None else 'auto'}")

    arr = np.load(npz_path)
    all_tokens = arr["tokens"]
    all_beliefs = arr["beliefs"]

    N_total = all_tokens.shape[0]
    N = min(DRY_RUN_N_SEQ, N_total) if dry_run else N_total
    L = all_tokens.shape[1]
    n_states = all_beliefs.shape[2]

    P = config.post_convergence_start
    split_idx = _eval_split_idx(L, P, config.train_eval_split)
    eval_act_start = (P - 1) + split_idx

    max_k = max(config.k_values)
    L_trunc = eval_act_start + max_k
    measure_pos = L_trunc - 1

    n_eval = L - eval_act_start
    assert max_k <= n_eval, (
        f"max k={max_k} exceeds eval window size={n_eval}. "
        f"Reduce k_values or increase seq_length/train_eval_split."
    )
    assert L_trunc <= L, f"L_trunc={L_trunc} exceeds L={L}"

    logger.info(f"Seq length      : {L}")
    logger.info(f"Post-conv start : {P}")
    logger.info(f"Eval act start  : {eval_act_start}")
    logger.info(f"L_trunc         : {L_trunc}  (was {L})")
    logger.info(f"Measure pos     : {measure_pos}")
    logger.info(f"N sequences     : {N}")

    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)
    model_dtype: torch.dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        _patch_attention_dtype(model)
        logger.info("Patched attention softmax to stay in fp16")

    hmm = Mess3HMM()
    p = config.hmm.process_params
    hmm.create_hmm(p["x"], p["alpha"])
    emit = build_emission_matrix(hmm)

    idx_to_token = {v: k for k, v in config.vocab_mapping.items()}
    n_vocab = len(config.vocab_mapping)
    _first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, n_vocab, logger)

    # ── Data structures for results ───────────────────────────────────────────
    def _make_cond_k_layer(conds: list[str]) -> dict:
        return {
            cond: {k: {l: [] for l in config.layer_indices} for k in config.k_values}
            for cond in conds
        }

    patch_kl_to_target = _make_cond_k_layer(PATCH_CONDITIONS)
    patch_kl_to_factual = _make_cond_k_layer(PATCH_CONDITIONS)
    patch_kl_to_clean = _make_cond_k_layer(PATCH_CONDITIONS)
    baseline_kl_to_target_patch = _make_cond_k_layer(PATCH_CONDITIONS)

    steer_kl_to_target = _make_cond_k_layer(STEER_CONDITIONS)
    steer_kl_to_factual = _make_cond_k_layer(STEER_CONDITIONS)
    steer_kl_to_clean = _make_cond_k_layer(STEER_CONDITIONS)
    baseline_kl_to_target_steer = _make_cond_k_layer(STEER_CONDITIONS)

    ablation_kl: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        "belief": {l: [] for l in config.layer_indices},
        "random": {l: [] for l in config.layer_indices},
    }
    ablation_kl_to_clean: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        "belief": {l: [] for l in config.layer_indices},
        "random": {l: [] for l in config.layer_indices},
    }
    baseline_kl_list: list[float] = []

    hook_names = [f"blocks.{l}.hook_resid_post" for l in config.layer_indices]

    # ── Phase A: Clean forward passes ─────────────────────────────────────────
    logger.info("=== Phase A: Clean forward passes ===")
    all_seq_data: list[SequenceData] = []

    for seq_i in range(N):
        logger.info(f"  Seq {seq_i}/{N-1} — clean pass")
        seq_tokens = all_tokens[seq_i]
        seq_beliefs = all_beliefs[seq_i]

        text = " ".join(idx_to_token[int(t)] for t in seq_tokens[:L_trunc])
        llm_tokens = model.to_tokens(text, prepend_bos=False, truncate=False)
        assert llm_tokens.shape[1] == L_trunc, (
            f"Expected {L_trunc} tokens, got {llm_tokens.shape[1]}"
        )

        with torch.no_grad():
            logits, clean_cache = model.run_with_cache(
                llm_tokens, names_filter=hook_names, return_type="logits",
            )

        probs_all = F.softmax(logits[0, measure_pos, :].float(), dim=-1)
        probs_hmm = probs_all[mid_tok_ids].cpu().numpy()
        P_clean_3 = (probs_hmm / (probs_hmm.sum() + 1e-10)).astype(np.float32)
        probs_junk = max(float(1.0 - probs_hmm.sum()), 1e-10)
        P_clean_4 = np.append(probs_hmm, probs_junk).astype(np.float32)
        del logits

        eta_baseline = seq_beliefs[measure_pos + 1].astype(np.float32)
        P_opt = (eta_baseline @ emit.T).astype(np.float32)
        bkl = float(_kl(P_clean_3[None], P_opt[None])[0])
        baseline_kl_list.append(bkl)
        logger.info(f"    Baseline KL: {bkl:.4f}")

        clean_acts_tail: dict[int, np.ndarray] = {
            layer: clean_cache[f"blocks.{layer}.hook_resid_post"][0, -max_k:, :].float().cpu().numpy()
            for layer in config.layer_indices
        }
        del clean_cache

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        all_seq_data.append(SequenceData(
            seq_idx=seq_i,
            llm_tokens=llm_tokens.cpu(),
            P_clean_3=P_clean_3,
            P_clean_4=P_clean_4,
            P_opt=P_opt,
            baseline_kl=bkl,
            clean_acts_tail=clean_acts_tail,
        ))

    # Clear TransformerLens hook contexts: run_with_cache writes
    # hook.ctx["activation"] on every HookPoint; with clear_contexts=False
    # (the default) those GPU tensors persist on the model indefinitely.
    model.reset_hooks(clear_contexts=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Phase B: Pre-compute all targets ─────────────────────────────────────
    logger.info("=== Phase B: Pre-computing targets ===")

    n_abl_per_seq = 1 + config.n_random_ablation_draws
    n_patch_per_seq = 1 + 1 + config.n_past_consistent_draws + config.n_random_patch_draws
    n_steer_per_seq = 1 + config.n_past_consistent_draws + config.n_random_patch_draws
    n_combined_per_seq = n_patch_per_seq + n_steer_per_seq

    for sd in all_seq_data:
        seq_i = sd.seq_idx
        seq_beliefs = all_beliefs[seq_i]
        logger.info(f"  Seq {seq_i} — computing targets for {len(config.layer_indices)} layers")

        for layer in config.layer_indices:
            layer_dir = training_dir / f"seq_{seq_i}" / f"layer_{layer}"
            encoder = ProbeResult.load_weights_only(layer_dir).probe
            decoder = DecoderResult.load(layer_dir).decoder
            encoder.eval()
            decoder.eval()

            W = encoder.W.detach().float().numpy()
            Q_belief = _orthonormal_basis(W)
            random_bases = [
                _random_basis(W.shape[0], W.shape[1], rng)
                for _ in range(config.n_random_ablation_draws)
            ]
            sd.ablation_projs[layer] = [Q_belief] + random_bases

            for k in config.k_values:
                clean_acts_k = sd.clean_acts_tail[layer][-k:]
                positions = list(range(L_trunc - k, L_trunc))

                eta_optimal = _optimal_targets(seq_beliefs, positions)
                eta_past_list = [
                    _past_consistent_targets(seq_beliefs, positions, hmm, rng)
                    for _ in range(config.n_past_consistent_draws)
                ]
                eta_random_list = [
                    _random_targets(positions, rng, n_states)
                    for _ in range(config.n_random_patch_draws)
                ]

                round_trip_acts, round_trip_beliefs = _round_trip_targets(
                    encoder, clean_acts_k, decoder,
                )

                with torch.no_grad():
                    dec_optimal = decoder(torch.from_numpy(eta_optimal).float())
                    dec_past = [
                        decoder(torch.from_numpy(ep).float())
                        for ep in eta_past_list
                    ]
                    dec_random = [
                        decoder(torch.from_numpy(er).float())
                        for er in eta_random_list
                    ]

                sd.patch_targets[(layer, k)] = torch.stack(
                    [dec_optimal, torch.from_numpy(round_trip_acts)] + dec_past + dec_random,
                    dim=0,
                )

                measure_belief_idx = -1
                patch_P_opt_list = [
                    (eta_optimal[measure_belief_idx] @ emit.T).astype(np.float32),
                    (round_trip_beliefs[measure_belief_idx] @ emit.T).astype(np.float32),
                ]
                for ep in eta_past_list:
                    patch_P_opt_list.append((ep[measure_belief_idx] @ emit.T).astype(np.float32))
                for er in eta_random_list:
                    patch_P_opt_list.append((er[measure_belief_idx] @ emit.T).astype(np.float32))
                sd.patch_target_P_opts[(layer, k)] = np.stack(patch_P_opt_list, axis=0)

                with torch.no_grad():
                    source_beliefs = encoder(torch.from_numpy(clean_acts_k).float())
                    dec_source = decoder(source_beliefs)
                    steer_optimal = (dec_optimal - dec_source).unsqueeze(0)
                    steer_past = [
                        (dp - dec_source).unsqueeze(0)
                        for dp in dec_past
                    ]
                    steer_random = [
                        (dr - dec_source).unsqueeze(0)
                        for dr in dec_random
                    ]

                sd.steer_deltas[(layer, k)] = torch.cat(
                    [steer_optimal] + steer_past + steer_random, dim=0
                )

                steer_P_opt_list = [
                    (eta_optimal[measure_belief_idx] @ emit.T).astype(np.float32),
                ]
                for ep in eta_past_list:
                    steer_P_opt_list.append((ep[measure_belief_idx] @ emit.T).astype(np.float32))
                for er in eta_random_list:
                    steer_P_opt_list.append((er[measure_belief_idx] @ emit.T).astype(np.float32))
                sd.steer_target_P_opts[(layer, k)] = np.stack(steer_P_opt_list, axis=0)

            del encoder, decoder

    # ── Load checkpoint (resume) ──────────────────────────────────────────────
    completed_layers: list[int] = []
    if resume_dir is not None:
        chk_path = out_dir / CHECKPOINT_FILENAME
        if chk_path.exists():
            logger.info(f"Loading checkpoint: {chk_path}")
            chk = _load_checkpoint(
                chk_path, PATCH_CONDITIONS, STEER_CONDITIONS,
                config.k_values, config.layer_indices,
            )
            completed_layers = chk["completed_layers"]
            patch_kl_to_target = chk["patch_kl_to_target"]
            patch_kl_to_factual = chk["patch_kl_to_factual"]
            patch_kl_to_clean = chk["patch_kl_to_clean"]
            baseline_kl_to_target_patch = chk["baseline_kl_to_target_patch"]
            steer_kl_to_target = chk["steer_kl_to_target"]
            steer_kl_to_factual = chk["steer_kl_to_factual"]
            steer_kl_to_clean = chk["steer_kl_to_clean"]
            baseline_kl_to_target_steer = chk["baseline_kl_to_target_steer"]
            ablation_kl = chk["ablation_kl"]
            ablation_kl_to_clean = chk["ablation_kl_to_clean"]
            logger.info(f"  Restored {len(completed_layers)} completed layers: {completed_layers}")
        else:
            logger.warning(f"  No checkpoint found at {chk_path}; Phase C will run from scratch")

    # ── Phase C: Batched interventions ────────────────────────────────────────
    logger.info("=== Phase C: Batched interventions ===")

    if config.max_intervention_batch is not None:
        max_B = config.max_intervention_batch
    elif device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        dtype_bytes = 2 if model_dtype == torch.float16 else 4
        d = model.cfg.d_model
        L = L_trunc
        n_h = model.cfg.n_heads
        d_mlp = getattr(model.cfg, "d_mlp", 4 * d)
        resid_per_item = L * d * dtype_bytes * 2
        qkv_per_item = L * 3 * d * dtype_bytes
        attn_per_item = n_h * L * L * dtype_bytes * 2
        mlp_per_item = L * d_mlp * dtype_bytes * 2
        bytes_per_item = resid_per_item + qkv_per_item + attn_per_item + mlp_per_item
        max_B = max(16, int(free_bytes * 0.3 / bytes_per_item))
        logger.info(f"Auto batch size : {max_B} (free VRAM: {free_bytes / 1e9:.1f} GB, "
                    f"est {bytes_per_item / 1e6:.0f} MB/item)")
    else:
        max_B = 96
    abl_chunk_size = max(1, max_B // n_abl_per_seq)
    combined_chunk_size = max(1, max_B // n_combined_per_seq)

    n_abl_chunks = len(_chunk(all_seq_data, abl_chunk_size))
    n_combined_chunks_per_k = len(_chunk(all_seq_data, combined_chunk_size))
    n_k = len(config.k_values)
    passes_per_layer = n_abl_chunks + n_k * n_combined_chunks_per_k

    logger.info(
        f"  Batch sizing: abl_chunk={abl_chunk_size} seqs ({n_abl_per_seq}/seq, "
        f"{n_abl_chunks} pass{'es' if n_abl_chunks > 1 else ''}), "
        f"combined_chunk={combined_chunk_size} seqs ({n_combined_per_seq}/seq, "
        f"{n_combined_chunks_per_k} pass{'es' if n_combined_chunks_per_k > 1 else ''}/k × {n_k} k = "
        f"{n_k * n_combined_chunks_per_k}), "
        f"total={passes_per_layer} passes/layer"
    )

    n_layers = len(config.layer_indices)
    phase_c_start = time.perf_counter()
    layer_times: list[float] = []

    for layer_idx, layer in enumerate(config.layer_indices):
        if layer in completed_layers:
            logger.info(f"  Layer {layer} ({layer_idx + 1}/{n_layers}) — skipped (checkpoint)")
            continue
        layer_start = time.perf_counter()
        pass_count = 0
        logger.info(f"  Layer {layer} ({layer_idx + 1}/{n_layers})")

        def _run_abl_safe(batch: list[SequenceData]) -> None:
            tb, pr = _build_ablation_batch(batch, layer)
            try:
                P_abl = _run_ablated(
                    model, tb, layer, pr,
                    measure_pos, mid_tok_ids, device, model_dtype,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) <= 1:
                    raise
                mid = max(1, len(batch) // 2)
                logger.warning(f"    OOM on ablation ({len(batch)} seqs), splitting → {mid}+{len(batch)-mid}")
                _run_abl_safe(batch[:mid])
                _run_abl_safe(batch[mid:])
                return
            _scatter_ablation_results(
                P_abl, batch, layer, ablation_kl, ablation_kl_to_clean,
            )

        abl_chunks = _chunk(all_seq_data, abl_chunk_size)
        for sub_batch in abl_chunks:
            _run_abl_safe(sub_batch)
            pass_count += 1
        abl_elapsed = time.perf_counter() - layer_start
        logger.info(f"    ablation done ({pass_count}/{passes_per_layer} passes, {abl_elapsed:.1f}s)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for k_idx, k in enumerate(config.k_values):
            k_start = time.perf_counter()
            positions = list(range(L_trunc - k, L_trunc))

            def _run_comb_safe(batch: list[SequenceData], _k: int = k, _pos: list[int] = positions) -> None:
                tb, np_b, pt_b, sd_b = _build_combined_batch(batch, layer, _k)

                max_v = config.max_variants_per_pass
                if max_v is not None and tb.shape[0] > max_v:
                    # Variant-level chunking: split patch items then steer items into
                    # sub-passes of ≤ max_variants_per_pass, limiting attention-pattern
                    # peak memory to max_v × n_heads × seq_len² × dtype_bytes.
                    patch_tb = tb[:np_b]
                    steer_tb = tb[np_b:]
                    n_steer = tb.shape[0] - np_b
                    pieces: list[np.ndarray] = []
                    for start in range(0, np_b, max_v):
                        end = min(start + max_v, np_b)
                        pieces.append(_run_combined(
                            model, patch_tb[start:end], layer, _pos,
                            end - start, pt_b[start:end], sd_b[:0],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    for start in range(0, n_steer, max_v):
                        end = min(start + max_v, n_steer)
                        pieces.append(_run_combined(
                            model, steer_tb[start:end], layer, _pos,
                            0, pt_b[:0], sd_b[start:end],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    P_out = np.concatenate(pieces, axis=0)
                else:
                    try:
                        P_out = _run_combined(
                            model, tb, layer, _pos,
                            np_b, pt_b, sd_b,
                            measure_pos, mid_tok_ids, device, model_dtype,
                        )
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if len(batch) <= 1:
                            raise
                        mid = max(1, len(batch) // 2)
                        logger.warning(f"    OOM on combined ({len(batch)} seqs), splitting → {mid}+{len(batch)-mid}")
                        _run_comb_safe(batch[:mid], _k, _pos)
                        _run_comb_safe(batch[mid:], _k, _pos)
                        return

                _scatter_combined_results(
                    P_out, np_b, batch, layer, _k,
                    patch_kl_to_target, patch_kl_to_factual,
                    steer_kl_to_target, steer_kl_to_factual,
                    patch_kl_to_clean, steer_kl_to_clean,
                    baseline_kl_to_target_patch, baseline_kl_to_target_steer,
                    config.n_past_consistent_draws, config.n_random_patch_draws,
                )

            combined_chunks = _chunk(all_seq_data, combined_chunk_size)
            for sub_batch in combined_chunks:
                _run_comb_safe(sub_batch)
                pass_count += 1
            k_elapsed = time.perf_counter() - k_start
            logger.info(
                f"    k={k} done ({pass_count}/{passes_per_layer} passes, {k_elapsed:.1f}s)"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        completed_layers.append(layer)
        _save_checkpoint(
            out_dir, completed_layers,
            patch_kl_to_target, patch_kl_to_factual, patch_kl_to_clean,
            baseline_kl_to_target_patch,
            steer_kl_to_target, steer_kl_to_factual, steer_kl_to_clean,
            baseline_kl_to_target_steer,
            ablation_kl, ablation_kl_to_clean,
        )

        elapsed = time.perf_counter() - layer_start
        layer_times.append(elapsed)
        layers_done = len(layer_times)
        avg_layer = sum(layer_times) / layers_done
        layers_left = n_layers - (len(completed_layers))
        eta_s = avg_layer * layers_left
        logger.info(
            f"  Layer {layer} done in {elapsed:.1f}s  |  ETA: {eta_s / 60:.1f} min"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    total_elapsed = time.perf_counter() - phase_c_start
    logger.info(f"Phase C total: {total_elapsed / 60:.1f} min")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("Aggregating results ...")

    n_bl = len(baseline_kl_list)
    agg_baseline = {
        "mean": float(np.mean(baseline_kl_list)),
        "std": float(np.std(baseline_kl_list)),
        "stderr": float(np.std(baseline_kl_list) / max(np.sqrt(n_bl), 1.0)),
        "per_seq": [float(v) for v in baseline_kl_list],
    }

    def _agg_cond_k_layer(src: dict, conds: list[str]) -> dict:
        return {
            cond: {
                k: {l: _agg_records(src[cond][k][l]) for l in config.layer_indices}
                for k in config.k_values
            }
            for cond in conds
        }

    agg_patch_to_target = _agg_cond_k_layer(patch_kl_to_target, PATCH_CONDITIONS)
    agg_patch_to_factual = _agg_cond_k_layer(patch_kl_to_factual, PATCH_CONDITIONS)
    agg_patch_to_clean = _agg_cond_k_layer(patch_kl_to_clean, PATCH_CONDITIONS)
    agg_bl_to_target_patch = _agg_cond_k_layer(baseline_kl_to_target_patch, PATCH_CONDITIONS)
    agg_steer_to_target = _agg_cond_k_layer(steer_kl_to_target, STEER_CONDITIONS)
    agg_steer_to_factual = _agg_cond_k_layer(steer_kl_to_factual, STEER_CONDITIONS)
    agg_steer_to_clean = _agg_cond_k_layer(steer_kl_to_clean, STEER_CONDITIONS)
    agg_bl_to_target_steer = _agg_cond_k_layer(baseline_kl_to_target_steer, STEER_CONDITIONS)

    agg_ablation = {
        "belief": {l: _agg_records(ablation_kl["belief"][l]) for l in config.layer_indices},
        "random": {l: _agg_records(ablation_kl["random"][l]) for l in config.layer_indices},
    }
    agg_ablation_to_clean = {
        "belief": {l: _agg_records(ablation_kl_to_clean["belief"][l]) for l in config.layer_indices},
        "random": {l: _agg_records(ablation_kl_to_clean["random"][l]) for l in config.layer_indices},
    }

    def _to_json(d: dict, conds: list[str]) -> dict:
        return {
            cond: {str(k): {str(l): v for l, v in by_layer.items()} for k, by_layer in by_k.items()}
            for cond, by_k in d.items() if cond in conds
        }

    metrics = {
        "baseline": agg_baseline,
        "patching_to_target": _to_json(agg_patch_to_target, PATCH_CONDITIONS),
        "patching_to_factual": _to_json(agg_patch_to_factual, PATCH_CONDITIONS),
        "patching_to_clean": _to_json(agg_patch_to_clean, PATCH_CONDITIONS),
        "baseline_kl_to_target_patch": _to_json(agg_bl_to_target_patch, PATCH_CONDITIONS),
        "steering_to_target": _to_json(agg_steer_to_target, STEER_CONDITIONS),
        "steering_to_factual": _to_json(agg_steer_to_factual, STEER_CONDITIONS),
        "steering_to_clean": _to_json(agg_steer_to_clean, STEER_CONDITIONS),
        "baseline_kl_to_target_steer": _to_json(agg_bl_to_target_steer, STEER_CONDITIONS),
        "ablation": {
            cond: {str(l): v for l, v in by_layer.items()}
            for cond, by_layer in agg_ablation.items()
        },
        "ablation_to_clean_4cat": {
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

    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation patching: KL vs layer by condition",
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_kl_vs_layer_k{k}",
        )
        for cross_cond in ["past_consistent", "random"]:
            _plot_crossing(
                agg_to_factual=agg_patch_to_factual[cross_cond][k],
                agg_to_target=agg_patch_to_target[cross_cond][k],
                baseline_to_factual=baseline_mean,
                agg_baseline_to_target=agg_bl_to_target_patch[cross_cond][k],
                layer_indices=config.layer_indices,
                k=k,
                condition=cross_cond,
                title="Activation patching",
                path=fig_dir / f"patching_crossing_{cross_cond}_k{k}",
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_patch[cond][k] for cond in PATCH_CONDITIONS},
            layer_indices=config.layer_indices,
            k=k,
            title="Activation patching",
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_causal_shift_k{k}",
        )

    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation steering: KL vs layer by condition",
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_kl_vs_layer_k{k}",
        )
        for cross_cond in ["past_consistent", "random"]:
            _plot_crossing(
                agg_to_factual=agg_steer_to_factual[cross_cond][k],
                agg_to_target=agg_steer_to_target[cross_cond][k],
                baseline_to_factual=baseline_mean,
                agg_baseline_to_target=agg_bl_to_target_steer[cross_cond][k],
                layer_indices=config.layer_indices,
                k=k,
                condition=cross_cond,
                title="Activation steering",
                path=fig_dir / f"steering_crossing_{cross_cond}_k{k}",
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_steer[cond][k] for cond in STEER_CONDITIONS},
            layer_indices=config.layer_indices,
            k=k,
            title="Activation steering",
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_causal_shift_k{k}",
        )

    k_max = max(config.k_values)
    _plot_heatmap(
        agg_patch_to_target["optimal"][k_max],
        config.layer_indices,
        title=f"Patching KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        path=fig_dir / f"heatmap_patching_optimal_k{k_max}",
    )
    _plot_heatmap(
        agg_steer_to_target["optimal"][k_max],
        config.layer_indices,
        title=f"Steering KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        path=fig_dir / f"heatmap_steering_optimal_k{k_max}",
    )

    _plot_ablation_curve(
        agg_ablation["belief"], agg_ablation["random"],
        config.layer_indices,
        title="Belief-subspace ablation: KL to optimal",
        subtitle="KL(P_ablated ∥ P_opt) — 3-cat normalised — log scale",
        path=fig_dir / "ablation_causal_importance",
    )
    _plot_ablation_curve(
        agg_ablation_to_clean["belief"], agg_ablation_to_clean["random"],
        config.layer_indices,
        title="Belief-subspace ablation: output perturbation (4-category)",
        subtitle="KL(P_ablated ∥ P_clean) — 4-cat (A, B, C, junk) — log scale",
        path=fig_dir / "ablation_kl_to_output_4cat",
    )

    _plot_roundtrip_comparison(
        {cond: agg_patch_to_target[cond][k_max] for cond in ["optimal", "round_trip"]},
        config.layer_indices,
        baseline_mean=baseline_mean,
        k=k_max,
        path=fig_dir / f"roundtrip_h2a_vs_h2b_k{k_max}",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
