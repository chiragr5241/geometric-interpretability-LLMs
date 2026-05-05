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
    KL-to-target  — KL(P_opt(η_target) ∥ P_intervened) per condition (primary metric).
    KL-to-factual — KL(P_opt(η_factual) ∥ P_intervened) (used for crossing plots).
    KL-to-clean   — KL(P_clean ∥ P_intervened) in HMM-token + junk space.
    Ablation uses HMM-token + junk distributions to capture shifts in
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
from hmm import BeliefHMM, build_process_hmm
from plot_titles import format_hmm_process
from probes import Probe, ProbeResult

DRY_RUN_N_SEQ = 2
DRY_RUN_N_EVAL_POSITIONS = 3
DRY_RUN_LAYERS = [0, 17]
DRY_RUN_K_VALUES = [1]

PATCH_CONDITIONS = ["optimal", "round_trip", "past_consistent", "splice", "random"]
STEER_CONDITIONS = ["optimal", "past_consistent", "splice", "random"]

# Conditions whose targets are "principled" (i.e. real beliefs);
# the run_with_controls path scores the random-direction control against each.
PRINCIPLED_CONDITIONS = ["optimal", "past_consistent", "splice"]

CHECKPOINT_FILENAME = "phase_c_checkpoint.json"

_PATCH_COLORS: dict[str, str] = {
    "optimal": "#1f77b4",
    "round_trip": "#ff7f0e",
    "past_consistent": "#2ca02c",
    "splice": "#9467bd",
    "random": "#d62728",
}
_STEER_COLORS: dict[str, str] = {
    "optimal": "#1f77b4",
    "past_consistent": "#2ca02c",
    "splice": "#9467bd",
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
    n_splice_draws: int = 3
    run_with_controls: bool = True
    n_control_draws: int = 1
    max_intervention_batch: int | None = None
    max_variants_per_pass: int | None = None
    n_ctx_override: int | None = None
    post_convergence_start: int = 30
    train_eval_split: float = 0.7
    random_seed: int = 42
    seq_exclude: list[int] = field(default_factory=list)
    save_html: bool = False


# ── Sequence data container ────────────────────────────────────────────────────

@dataclass
class SequenceData:
    seq_idx: int
    llm_tokens: torch.Tensor                              # (1, L_trunc)
    P_clean_hmm: np.ndarray                               # (n_hmm,) normalised
    P_clean_projected: np.ndarray                         # (n_hmm+1,) with junk category
    P_opt: np.ndarray                                     # (n_hmm,) Bayesian-optimal
    baseline_kl: float
    clean_acts_tail: dict[int, np.ndarray]                # layer -> (max_k, d_model)
    ablation_projs: dict[int, list[np.ndarray]] = field(default_factory=dict)
    patch_targets: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    steer_deltas: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    patch_target_P_opts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    steer_target_P_opts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # Index of the first row in patch_target_P_opts / steer_target_P_opts for each
    # principled condition, so the random-direction control can be scored against them.
    patch_target_principled_offsets: dict[tuple[int, int], dict[str, list[int]]] = field(default_factory=dict)
    steer_target_principled_offsets: dict[tuple[int, int], dict[str, list[int]]] = field(default_factory=dict)
    # Random-direction "control" decoder outputs / steering deltas (independent of any cond).
    patch_controls: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    steer_controls: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p ∥ q) per row. Both (..., n). Returns (...,)."""
    return (p * np.log(np.clip(p, 1e-10, None) / np.clip(q, 1e-10, None))).sum(axis=-1)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_fig(fig: go.Figure, path: Path, save_html: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_html:
        fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")))
    except Exception:
        pass


def _to_hmm_simplex(p: np.ndarray, n_hmm_tokens: int) -> np.ndarray:
    """Drop junk mass and renormalise to the HMM-token simplex."""
    p_hmm = p[..., :n_hmm_tokens].copy()
    return (p_hmm / (p_hmm.sum(axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


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


def _agg_paired_diff(
    cond_records: list[tuple[int, int, float]],
    ctl_records: list[tuple[int, int, float]],
) -> dict:
    """Per-seq mean(cond) − mean(ctl), then aggregate across sequences."""
    cond_per_seq: dict[int, list[float]] = {}
    for seq_idx, _, val in cond_records:
        cond_per_seq.setdefault(seq_idx, []).append(val)
    ctl_per_seq: dict[int, list[float]] = {}
    for seq_idx, _, val in ctl_records:
        ctl_per_seq.setdefault(seq_idx, []).append(val)
    seq_diffs: list[float] = []
    for seq_idx, vs in cond_per_seq.items():
        if seq_idx not in ctl_per_seq:
            continue
        seq_diffs.append(float(np.mean(vs) - np.mean(ctl_per_seq[seq_idx])))
    n = len(seq_diffs)
    return {
        "seq_means": seq_diffs,
        "mean": float(np.mean(seq_diffs)) if n else float("nan"),
        "std": float(np.std(seq_diffs)) if n else float("nan"),
        "stderr": float(np.std(seq_diffs) / max(np.sqrt(n), 1.0)) if n else float("nan"),
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
    patch_control_kl_to_target: dict | None = None,
    patch_control_kl_to_factual: dict | None = None,
    steer_control_kl_to_target: dict | None = None,
    steer_control_kl_to_factual: dict | None = None,
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
    if patch_control_kl_to_target is not None:
        chk["patch_control_kl_to_target"] = _serialize_ckl(patch_control_kl_to_target)
        chk["patch_control_kl_to_factual"] = _serialize_ckl(patch_control_kl_to_factual)
        chk["steer_control_kl_to_target"] = _serialize_ckl(steer_control_kl_to_target)
        chk["steer_control_kl_to_factual"] = _serialize_ckl(steer_control_kl_to_factual)
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
    conds_principled: list[str] | None = None,
) -> dict:
    """Load Phase C checkpoint and deserialize all result dicts."""
    with open(chk_path) as f:
        raw = json.load(f)
    out = {
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
    if conds_principled is not None and "patch_control_kl_to_target" in raw:
        out["patch_control_kl_to_target"] = _deserialize_ckl(
            raw["patch_control_kl_to_target"], conds_principled, k_values, layer_indices
        )
        out["patch_control_kl_to_factual"] = _deserialize_ckl(
            raw["patch_control_kl_to_factual"], conds_principled, k_values, layer_indices
        )
        out["steer_control_kl_to_target"] = _deserialize_ckl(
            raw["steer_control_kl_to_target"], conds_principled, k_values, layer_indices
        )
        out["steer_control_kl_to_factual"] = _deserialize_ckl(
            raw["steer_control_kl_to_factual"], conds_principled, k_values, layer_indices
        )
    return out


def _extract_projected_probs(
    logits: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
) -> np.ndarray:
    """Extract (B, n_hmm_tokens + 1) probabilities with a final junk category."""
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
    """Run belief-subspace ablation at ALL positions. Returns projected probabilities.

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

    return _extract_projected_probs(logits, measure_pos, mid_tok_ids)


def _run_combined(
    model,
    tokens_batch: torch.Tensor,
    layer: int,
    positions: list[int],
    n_patch: int,
    n_steer: int,
    patch_targets: torch.Tensor,
    steer_deltas: torch.Tensor,
    measure_pos: int,
    mid_tok_ids: list[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> np.ndarray:
    """Merged patching + steering (+ optional controls) in a single forward pass.

    Layout of tokens_batch rows: [patch (n_patch) | steer (n_steer) | patch_control | steer_control].
    patch_targets includes both patch and patch_control rows in order.
    steer_deltas includes both steer and steer_control rows in order.
    """
    pos_tensor = torch.tensor(positions, dtype=torch.long)
    patch_t = patch_targets.to(device=device, dtype=model_dtype)
    steer_t = steer_deltas.to(device=device, dtype=model_dtype)
    n_patch_total = patch_t.shape[0]

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # Rows [0, n_patch): patch (replace).
        if n_patch > 0:
            value[:n_patch, pos_tensor, :] = patch_t[:n_patch]
        # Rows [n_patch, n_patch+n_steer): steer (add).
        if n_steer > 0:
            value[n_patch:n_patch + n_steer, pos_tensor, :] = (
                value[n_patch:n_patch + n_steer, pos_tensor, :] + steer_t[:n_steer]
            )
        # Rows [n_patch+n_steer, n_patch+n_steer+n_ctl_p): patch_control (replace).
        n_ctl_p = n_patch_total - n_patch
        if n_ctl_p > 0:
            value[n_patch + n_steer:n_patch + n_steer + n_ctl_p, pos_tensor, :] = patch_t[n_patch:]
        # Remaining rows: steer_control (add).
        n_ctl_s = steer_t.shape[0] - n_steer
        if n_ctl_s > 0:
            value[n_patch + n_steer + n_ctl_p:, pos_tensor, :] = (
                value[n_patch + n_steer + n_ctl_p:, pos_tensor, :] + steer_t[n_steer:]
            )
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens_batch.to(device),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            return_type="logits",
        )

    return _extract_projected_probs(logits, measure_pos, mid_tok_ids)


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
    run_with_controls: bool,
) -> tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
    """Build combined batch ordered as: [patch | steer | patch_control | steer_control].

    Returns (tokens, n_patch, n_steer, patch_tgts (incl ctl), steer_dlts (incl ctl)).
    Control items are appended to the patch/steer hook tensors so the same hook
    function injects them; downstream scatter splits at n_patch + n_steer + ctl_p.
    """
    patch_tokens: list[torch.Tensor] = []
    steer_tokens: list[torch.Tensor] = []
    patch_tgts: list[torch.Tensor] = []
    steer_dlts: list[torch.Tensor] = []
    ctl_patch_tokens: list[torch.Tensor] = []
    ctl_steer_tokens: list[torch.Tensor] = []
    ctl_patch_tgts: list[torch.Tensor] = []
    ctl_steer_dlts: list[torch.Tensor] = []
    for sd in seq_data_list:
        n_p = sd.patch_targets[(layer, k)].shape[0]
        n_s = sd.steer_deltas[(layer, k)].shape[0]
        patch_tokens.append(sd.llm_tokens.expand(n_p, -1))
        steer_tokens.append(sd.llm_tokens.expand(n_s, -1))
        patch_tgts.append(sd.patch_targets[(layer, k)])
        steer_dlts.append(sd.steer_deltas[(layer, k)])
        if run_with_controls:
            ctl_p = sd.patch_controls[(layer, k)]
            ctl_s = sd.steer_controls[(layer, k)]
            ctl_patch_tokens.append(sd.llm_tokens.expand(ctl_p.shape[0], -1))
            ctl_steer_tokens.append(sd.llm_tokens.expand(ctl_s.shape[0], -1))
            ctl_patch_tgts.append(ctl_p)
            ctl_steer_dlts.append(ctl_s)
    n_patch = sum(t.shape[0] for t in patch_tgts)
    n_steer = sum(t.shape[0] for t in steer_dlts)
    tokens = torch.cat(
        patch_tokens + steer_tokens + ctl_patch_tokens + ctl_steer_tokens, dim=0
    )
    patch_all = torch.cat(patch_tgts + ctl_patch_tgts, dim=0) if ctl_patch_tgts else torch.cat(patch_tgts, dim=0)
    steer_all = torch.cat(steer_dlts + ctl_steer_dlts, dim=0) if ctl_steer_dlts else torch.cat(steer_dlts, dim=0)
    return tokens, n_patch, n_steer, patch_all, steer_all


# ── Result scattering helpers ─────────────────────────────────────────────────

def _scatter_ablation_results(
    P_abl_projected: np.ndarray,
    sub_batch: list[SequenceData],
    layer: int,
    n_hmm_tokens: int,
    ablation_kl: dict[str, dict[int, list[tuple[int, int, float]]]],
    ablation_kl_to_clean: dict[str, dict[int, list[tuple[int, int, float]]]],
) -> None:
    b_offset = 0
    for sd in sub_batch:
        n_projs = len(sd.ablation_projs[layer])
        P_slice_projected = P_abl_projected[b_offset:b_offset + n_projs]
        P_slice_hmm = _to_hmm_simplex(P_slice_projected, n_hmm_tokens)

        kl_abl = _kl(sd.P_opt[None], P_slice_hmm)
        ablation_kl["belief"][layer].append((sd.seq_idx, 0, float(kl_abl[0])))
        for draw_i, kl_v in enumerate(kl_abl[1:]):
            ablation_kl["random"][layer].append((sd.seq_idx, draw_i, float(kl_v)))

        kl_abl_to_clean = _kl(sd.P_clean_projected[None], P_slice_projected)
        ablation_kl_to_clean["belief"][layer].append((sd.seq_idx, 0, float(kl_abl_to_clean[0])))
        for draw_i, kl_v in enumerate(kl_abl_to_clean[1:]):
            ablation_kl_to_clean["random"][layer].append((sd.seq_idx, draw_i, float(kl_v)))

        b_offset += n_projs


def _scatter_combined_results(
    P_out_projected: np.ndarray,
    n_patch_total: int,
    n_steer_total: int,
    sub_batch: list[SequenceData],
    layer: int,
    k: int,
    n_hmm_tokens: int,
    patch_kl_to_target: dict,
    patch_kl_to_factual: dict,
    steer_kl_to_target: dict,
    steer_kl_to_factual: dict,
    patch_kl_to_clean: dict,
    steer_kl_to_clean: dict,
    baseline_kl_to_target_patch: dict,
    baseline_kl_to_target_steer: dict,
    n_past_consistent_draws: int,
    n_splice_draws: int,
    n_random_patch_draws: int,
    run_with_controls: bool,
    n_control_draws: int,
    patch_control_kl_to_target: dict | None,
    steer_control_kl_to_target: dict | None,
    patch_control_kl_to_factual: dict | None,
    steer_control_kl_to_factual: dict | None,
) -> None:
    P_patch_all_projected = P_out_projected[:n_patch_total]
    P_steer_all_projected = P_out_projected[n_patch_total:n_patch_total + n_steer_total]
    P_ctl_patch_all_projected = P_out_projected[n_patch_total + n_steer_total:
                                                 n_patch_total + n_steer_total + (
                                                     len(sub_batch) * n_control_draws if run_with_controls else 0)]
    P_ctl_steer_all_projected = P_out_projected[n_patch_total + n_steer_total + (
                                                     len(sub_batch) * n_control_draws if run_with_controls else 0):]
    P_patch_all_hmm = _to_hmm_simplex(P_patch_all_projected, n_hmm_tokens)
    P_steer_all_hmm = _to_hmm_simplex(P_steer_all_projected, n_hmm_tokens)
    P_ctl_patch_all_hmm = _to_hmm_simplex(P_ctl_patch_all_projected, n_hmm_tokens) if run_with_controls else None
    P_ctl_steer_all_hmm = _to_hmm_simplex(P_ctl_steer_all_projected, n_hmm_tokens) if run_with_controls else None

    cond_draw_counts = [
        ("optimal", 1),
        ("round_trip", 1),
        ("past_consistent", n_past_consistent_draws),
        ("splice", n_splice_draws),
        ("random", n_random_patch_draws),
    ]
    steer_cond_draw_counts = [
        ("optimal", 1),
        ("past_consistent", n_past_consistent_draws),
        ("splice", n_splice_draws),
        ("random", n_random_patch_draws),
    ]

    p_offset, s_offset = 0, 0
    for sd_idx, sd in enumerate(sub_batch):
        n_p = sd.patch_targets[(layer, k)].shape[0]
        n_s = sd.steer_deltas[(layer, k)].shape[0]
        P_patch_hmm = P_patch_all_hmm[p_offset:p_offset + n_p]
        P_steer_hmm = P_steer_all_hmm[s_offset:s_offset + n_s]
        P_patch_projected = P_patch_all_projected[p_offset:p_offset + n_p]
        P_steer_projected = P_steer_all_projected[s_offset:s_offset + n_s]

        patch_P_opts = sd.patch_target_P_opts[(layer, k)]
        steer_P_opts = sd.steer_target_P_opts[(layer, k)]

        kl_to_tgt = _kl(patch_P_opts, P_patch_hmm)
        kl_to_fac = _kl(sd.P_opt[None], P_patch_hmm)
        kl_to_cln = _kl(sd.P_clean_projected[None], P_patch_projected)
        bl_to_tgt = _kl(patch_P_opts, sd.P_clean_hmm[None].repeat(n_p, axis=0))

        b_idx = 0
        for cond, n_draws in cond_draw_counts:
            for draw_i in range(n_draws):
                patch_kl_to_target[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_tgt[b_idx])))
                patch_kl_to_factual[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_fac[b_idx])))
                patch_kl_to_clean[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_to_cln[b_idx])))
                baseline_kl_to_target_patch[cond][k][layer].append((sd.seq_idx, draw_i, float(bl_to_tgt[b_idx])))
                b_idx += 1

        kl_s_tgt = _kl(steer_P_opts, P_steer_hmm)
        kl_s_fac = _kl(sd.P_opt[None], P_steer_hmm)
        kl_s_cln = _kl(sd.P_clean_projected[None], P_steer_projected)
        bl_s_tgt = _kl(steer_P_opts, sd.P_clean_hmm[None].repeat(n_s, axis=0))

        s_idx = 0
        for cond, n_draws in steer_cond_draw_counts:
            for draw_i in range(n_draws):
                steer_kl_to_target[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_tgt[s_idx])))
                steer_kl_to_factual[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_fac[s_idx])))
                steer_kl_to_clean[cond][k][layer].append((sd.seq_idx, draw_i, float(kl_s_cln[s_idx])))
                baseline_kl_to_target_steer[cond][k][layer].append((sd.seq_idx, draw_i, float(bl_s_tgt[s_idx])))
                s_idx += 1

        if run_with_controls:
            ctl_p_start = sd_idx * n_control_draws
            P_ctl_p = P_ctl_patch_all_hmm[ctl_p_start:ctl_p_start + n_control_draws]
            P_ctl_s = P_ctl_steer_all_hmm[ctl_p_start:ctl_p_start + n_control_draws]

            patch_target_offsets = sd.patch_target_principled_offsets[(layer, k)]
            steer_target_offsets = sd.steer_target_principled_offsets[(layer, k)]

            for cond in PRINCIPLED_CONDITIONS:
                if cond not in patch_target_offsets:
                    continue
                offsets = patch_target_offsets[cond]
                # Use only first principled draw of each cond as the reference target P_opt.
                P_opt_ref = patch_P_opts[offsets[0]:offsets[0] + 1]
                kl_ctl_p_tgt = _kl(P_opt_ref, P_ctl_p)
                kl_ctl_p_fac = _kl(sd.P_opt[None], P_ctl_p)
                for draw_i in range(n_control_draws):
                    patch_control_kl_to_target[cond][k][layer].append(
                        (sd.seq_idx, draw_i, float(kl_ctl_p_tgt[draw_i]))
                    )
                    patch_control_kl_to_factual[cond][k][layer].append(
                        (sd.seq_idx, draw_i, float(kl_ctl_p_fac[draw_i]))
                    )
            for cond in PRINCIPLED_CONDITIONS:
                if cond not in steer_target_offsets:
                    continue
                offsets = steer_target_offsets[cond]
                P_opt_ref = steer_P_opts[offsets[0]:offsets[0] + 1]
                kl_ctl_s_tgt = _kl(P_opt_ref, P_ctl_s)
                kl_ctl_s_fac = _kl(sd.P_opt[None], P_ctl_s)
                for draw_i in range(n_control_draws):
                    steer_control_kl_to_target[cond][k][layer].append(
                        (sd.seq_idx, draw_i, float(kl_ctl_s_tgt[draw_i]))
                    )
                    steer_control_kl_to_factual[cond][k][layer].append(
                        (sd.seq_idx, draw_i, float(kl_ctl_s_fac[draw_i]))
                    )

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
    hmm: BeliefHMM,
    rng: np.random.Generator,
) -> np.ndarray:
    k = len(positions)
    initial_belief = seq_beliefs[positions[0]].astype(np.float32)
    _, new_beliefs = hmm.sample_continuation(initial_belief, k, rng)
    return new_beliefs.astype(np.float32)


def _random_targets(positions: list[int], rng: np.random.Generator, n_states: int = 3) -> np.ndarray:
    return rng.dirichlet(np.ones(n_states), size=len(positions)).astype(np.float32)


def _splice_targets(
    seq_idx: int,
    all_beliefs: np.ndarray,
    valid_donor_seqs: list[int],
    post_convergence_start: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a splice-target: take k consecutive HMM-evolved beliefs from a different donor seq.

    The donor's beliefs already evolve under the HMM, so the resulting target is internally
    consistent (unlike `random` which is i.i.d. Dirichlet) but conflicts with the prefix
    tokens the model has actually seen at the patched positions.
    """
    donors = [j for j in valid_donor_seqs if j != seq_idx]
    if not donors:
        donors = [j for j in valid_donor_seqs]  # degenerate: only one valid seq, allow self
    donor = int(rng.choice(donors))
    L_donor = all_beliefs.shape[1]
    t_max = L_donor - k
    t_min = post_convergence_start
    if t_max < t_min:
        t_min = max(0, t_max)
    t0 = int(rng.integers(t_min, t_max + 1))
    return all_beliefs[donor, t0:t0 + k].astype(np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_kl_vs_layer(
    agg: dict[str, dict[int, dict]],
    layer_indices: list[int],
    k: int,
    baseline_mean: float,
    title: str,
    hmm_subtitle: str,
    colors: dict[str, str],
    path: Path,
    save_html: bool = False,
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
        title=(
            f"{title} (k={k})<br>"
            f"<sup>{hmm_subtitle} | KL(P_opt(η_target) ∥ P_intervened) — log scale — mean ± stderr</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=500, width=820,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_kl_minus_control_vs_layer(
    agg_diff: dict[str, dict[int, dict]],
    layer_indices: list[int],
    k: int,
    title: str,
    hmm_subtitle: str,
    colors: dict[str, str],
    path: Path,
    save_html: bool = False,
) -> None:
    """Per-condition KL_to_target minus control KL_to_target (paired by seq).

    Negative values ⇒ principled intervention beats a random-direction control on
    hitting that condition's own target distribution.
    """
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for cond, color in colors.items():
        if cond not in agg_diff:
            continue
        means = [agg_diff[cond][l]["mean"] for l in layer_indices]
        stderrs = [agg_diff[cond][l]["stderr"] for l in layer_indices]
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

    fig.update_layout(
        title=(
            f"{title} (k={k})<br>"
            f"<sup>{hmm_subtitle} | KL(P_opt(η_C) ‖ LLM_int_C) − KL(P_opt(η_C) ‖ LLM_int_control) "
            "— negative = principled beats random control</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats]",
        height=500, width=820,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_heatmap(
    agg_by_layer: dict[int, dict],
    layer_indices: list[int],
    title: str,
    hmm_subtitle: str,
    path: Path,
    save_html: bool = False,
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
        title=f"{title}<br><sup>{hmm_subtitle}</sup>",
        xaxis_title="Layer",
        yaxis_title="Sequence",
        height=500, width=700,
        margin=dict(t=70, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_ablation_curve(
    agg_belief: dict[int, dict],
    agg_random: dict[int, dict],
    layer_indices: list[int],
    title: str,
    subtitle: str,
    hmm_subtitle: str,
    path: Path,
    save_html: bool = False,
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
        title=f"{title}<br><sup>{hmm_subtitle} | {subtitle}</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=460, width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_roundtrip_comparison(
    agg_patch: dict[str, dict[int, dict]],
    layer_indices: list[int],
    baseline_mean: float,
    k: int,
    hmm_subtitle: str,
    path: Path,
    save_html: bool = False,
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
              f"<sup>{hmm_subtitle} | "
              "round-trip ≈ optimal → H2A (subspace matters); ≈ baseline → H2B (clean beliefs matter)</sup>",
        xaxis_title="Layer",
        yaxis_title="KL [nats]",
        height=480, width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_crossing(
    agg_to_factual: dict[int, dict],
    agg_to_target: dict[int, dict],
    baseline_to_factual: float,
    agg_baseline_to_target: dict[int, dict],
    layer_indices: list[int],
    k: int,
    condition: str,
    title: str,
    hmm_subtitle: str,
    path: Path,
    save_html: bool = False,
) -> None:
    """Old-style crossing plot.

    Blue: KL(P_opt_factual ∥ P) — should go UP after patching with wrong belief.
    Orange: KL(P_opt_target ∥ P) — should go DOWN (model adopts injected belief).
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
            f"<sup>{hmm_subtitle} | Blue = KL to factual (↑ = moved away) | Orange = KL to target (↓ = adopted)</sup>"
        ),
        height=max(320, 220 * n_rows + 100),
        width=min(220 * n_cols + 220, 1400),
        margin=dict(t=90, b=60, l=60, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_causal_shift(
    agg_to_target: dict[str, dict[int, dict]],
    agg_baseline_to_target: dict[str, dict[int, dict]],
    layer_indices: list[int],
    k: int,
    title: str,
    hmm_subtitle: str,
    colors: dict[str, str],
    path: Path,
    save_html: bool = False,
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
            f"<sup>{hmm_subtitle} | ΔKL = KL_baseline − KL_intervened toward target: positive = causal</sup>"
        ),
        xaxis_title="Layer",
        yaxis_title="ΔKL [nats] (↑ = toward target)",
        height=460,
        width=780,
        margin=dict(t=80, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


# ── (layer × k) effect surfaces ───────────────────────────────────────────────

def _layer_k_matrix(
    by_k_by_layer: dict[int, dict[int, dict]],
    layer_indices: list[int],
    k_values: list[int],
    field: str = "mean",
) -> np.ndarray:
    """Build a [n_k, n_layer] matrix of `field` from a {k: {layer: agg}} dict."""
    n_k = len(k_values)
    n_l = len(layer_indices)
    out = np.full((n_k, n_l), np.nan, dtype=float)
    for i, k in enumerate(k_values):
        if k not in by_k_by_layer:
            continue
        per_layer = by_k_by_layer[k]
        for j, l in enumerate(layer_indices):
            if l in per_layer and field in per_layer[l]:
                out[i, j] = per_layer[l][field]
    return out


def _causal_shift_matrix(
    agg_to_target: dict[int, dict[int, dict]],
    agg_baseline_to_target: dict[int, dict[int, dict]],
    layer_indices: list[int],
    k_values: list[int],
) -> np.ndarray:
    """ΔKL[k, layer] = KL_baseline_to_target − KL_intervened_to_target."""
    Z_int = _layer_k_matrix(agg_to_target, layer_indices, k_values)
    Z_bl = _layer_k_matrix(agg_baseline_to_target, layer_indices, k_values)
    return Z_bl - Z_int


def _plot_layer_k_heatmap(
    Z: np.ndarray,
    layer_indices: list[int],
    k_values: list[int],
    title: str,
    subtitle: str,
    colorbar_title: str,
    hmm_subtitle: str,
    path: Path,
    diverging: bool = True,
    log_scale: bool = False,
    save_html: bool = False,
) -> None:
    """Heatmap with k on y-axis (low at bottom) and layer on x-axis."""
    Zp = Z.copy()
    if log_scale:
        Zp = np.log10(np.clip(Zp, 1e-10, None))
    finite = Zp[np.isfinite(Zp)]
    if finite.size == 0:
        zmin, zmax = -1.0, 1.0
    elif diverging:
        amax = float(np.nanmax(np.abs(finite)))
        zmin, zmax = -amax, amax
    else:
        zmin, zmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    colorscale = "RdBu_r" if diverging else "Viridis"
    range_str = f"range: [{zmin:.4g}, {zmax:.4g}]"
    fig = go.Figure(go.Heatmap(
        z=Zp,
        x=[str(l) for l in layer_indices],
        y=[str(k) for k in k_values],
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=colorbar_title),
    ))
    full_subtitle = f"{subtitle} — {range_str}" if subtitle else range_str
    if hmm_subtitle:
        full_subtitle = f"{hmm_subtitle} | {full_subtitle}"
    fig.update_layout(
        title=f"{title}<br><sup>{full_subtitle}</sup>",
        xaxis_title="Layer",
        yaxis_title="k (positions intervened)",
        height=520,
        width=900,
        margin=dict(t=90, b=60, l=70, r=40),
    )
    fig.update_yaxes(autorange=True)
    _save_fig(fig, path, save_html=save_html)


def _plot_layer_k_panel(
    matrices: dict[str, np.ndarray],
    layer_indices: list[int],
    k_values: list[int],
    title: str,
    subtitle: str,
    colorbar_title: str,
    hmm_subtitle: str,
    path: Path,
    diverging: bool = True,
    save_html: bool = False,
) -> None:
    """Side-by-side heatmap panel (one per condition) with shared color scale."""
    conds = list(matrices.keys())
    n = len(conds)
    if n == 0:
        return
    all_vals = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices.values()])
    if all_vals.size == 0:
        zmin, zmax = -1.0, 1.0
    elif diverging:
        amax = float(np.nanmax(np.abs(all_vals)))
        zmin, zmax = -amax, amax
    else:
        zmin, zmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    colorscale = "RdBu_r" if diverging else "Viridis"
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[c.replace("_", "-") for c in conds],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    for i, cond in enumerate(conds):
        fig.add_trace(go.Heatmap(
            z=matrices[cond],
            x=[str(l) for l in layer_indices],
            y=[str(k) for k in k_values],
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            showscale=(i == n - 1),
            colorbar=dict(title=colorbar_title) if i == n - 1 else None,
        ), row=1, col=i + 1)
        fig.update_xaxes(title_text="Layer", row=1, col=i + 1)
    fig.update_yaxes(title_text="k", row=1, col=1)
    range_str = f"shared range: [{zmin:.4g}, {zmax:.4g}]"
    full_subtitle = f"{subtitle} — {range_str}" if subtitle else range_str
    if hmm_subtitle:
        full_subtitle = f"{hmm_subtitle} | {full_subtitle}"
    fig.update_layout(
        title=f"{title}<br><sup>{full_subtitle}</sup>",
        height=480,
        width=max(420, 340 * n + 120),
        margin=dict(t=90, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _plot_curves_over_layer_by_k(
    Z: np.ndarray,
    layer_indices: list[int],
    k_values: list[int],
    title: str,
    subtitle: str,
    yaxis_title: str,
    hmm_subtitle: str,
    path: Path,
    log_y: bool = False,
    add_zero_line: bool = False,
    save_html: bool = False,
) -> None:
    """One curve per k, layer on x-axis, k mapped to Viridis."""
    layers_str = [str(l) for l in layer_indices]
    fig = go.Figure()
    if add_zero_line:
        fig.add_hline(y=0, line_color="black", line_width=1)
    n_k = len(k_values)
    for i, k in enumerate(k_values):
        t = i / max(n_k - 1, 1)
        r = int(68 + (253 - 68) * t)
        g = int(1 + (231 - 1) * t)
        b = int(84 + (37 - 84) * t)
        color = f"rgb({r},{g},{b})"
        fig.add_trace(go.Scatter(
            x=layers_str,
            y=list(Z[i, :]),
            name=f"k={k}",
            mode="lines",
            line=dict(color=color, width=1.5),
        ))
    if log_y:
        fig.update_yaxes(type="log")
    full_subtitle = f"{hmm_subtitle} | {subtitle}" if hmm_subtitle else subtitle
    fig.update_layout(
        title=f"{title}<br><sup>{full_subtitle}</sup>",
        xaxis_title="Layer",
        yaxis_title=yaxis_title,
        height=520, width=900,
        margin=dict(t=90, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _fit_linear_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²) for finite (x, y). Returns NaNs if insufficient points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    xs = x[mask].astype(float)
    ys = y[mask].astype(float)
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def _plot_curves_over_k_by_layer(
    Z: np.ndarray,
    layer_indices: list[int],
    k_values: list[int],
    title: str,
    subtitle: str,
    yaxis_title: str,
    hmm_subtitle: str,
    path: Path,
    log_y: bool = False,
    add_zero_line: bool = False,
    save_html: bool = False,
) -> None:
    """One curve per layer, k on x-axis, layer mapped to Viridis. Adds linear trend per layer with R²."""
    k_arr = np.asarray(k_values, dtype=float)
    fig = go.Figure()
    if add_zero_line:
        fig.add_hline(y=0, line_color="black", line_width=1)
    n_l = len(layer_indices)
    for j, l in enumerate(layer_indices):
        t = j / max(n_l - 1, 1)
        r = int(68 + (253 - 68) * t)
        g = int(1 + (231 - 1) * t)
        b = int(84 + (37 - 84) * t)
        color = f"rgb({r},{g},{b})"
        y = Z[:, j].astype(float)
        slope, intercept, r2 = _fit_linear_trend(k_arr, y)
        label = (
            f"L{l} (slope={slope:.3g}, R²={r2:.2f})"
            if np.isfinite(r2) else f"L{l}"
        )
        fig.add_trace(go.Scatter(
            x=[str(k) for k in k_values],
            y=list(y),
            name=label,
            legendgroup=f"L{l}",
            mode="lines",
            line=dict(color=color, width=1.5),
        ))
        if np.isfinite(slope):
            yhat = slope * k_arr + intercept
            fig.add_trace(go.Scatter(
                x=[str(k) for k in k_values],
                y=list(yhat),
                name=f"L{l} trend",
                legendgroup=f"L{l}",
                showlegend=False,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.6,
                hoverinfo="skip",
            ))
    if log_y:
        fig.update_yaxes(type="log")
    full_subtitle = f"{subtitle} — dashed = linear fit in k"
    if hmm_subtitle:
        full_subtitle = f"{hmm_subtitle} | {full_subtitle}"
    fig.update_layout(
        title=f"{title}<br><sup>{full_subtitle}</sup>",
        xaxis_title="k (positions intervened)",
        yaxis_title=yaxis_title,
        height=520, width=1100,
        margin=dict(t=90, b=60, l=70, r=40),
    )
    _save_fig(fig, path, save_html=save_html)


def _emit_kl_minus_control_layer_k(
    *,
    label: str,
    conditions: list[str],
    diff_records_by_cond: dict[str, dict[int, dict[int, list[tuple[int, int, float]]]]],
    layer_indices: list[int],
    k_values: list[int],
    fig_dir: Path,
    hmm_subtitle: str = "",
    save_html: bool = False,
) -> None:
    """(layer × k) heatmap & lines for KL_C_to_C − KL_ctl_to_C, per condition.

    `diff_records_by_cond[cond][k][layer]` should be the per-seq aggregated diff dict
    (output of _agg_paired_diff for each k, layer).
    """
    diff_mats: dict[str, np.ndarray] = {}
    for cond in conditions:
        if cond not in diff_records_by_cond:
            continue
        diff_mats[cond] = _layer_k_matrix(
            diff_records_by_cond[cond], layer_indices, k_values, field="mean",
        )

    for cond, Z in diff_mats.items():
        base = fig_dir / f"{label}_kl_minus_control_{cond}"
        _plot_layer_k_heatmap(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} KL minus control — {cond.replace('_','-')}",
            subtitle="ΔKL = KL_C_to_C − KL_ctl_to_C (negative = principled beats control)",
            colorbar_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_heatmap"),
            diverging=True,
            save_html=save_html,
        )
        _plot_curves_over_layer_by_k(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} KL minus control — {cond.replace('_','-')}",
            subtitle="ΔKL by layer, one curve per k (Viridis: low → high)",
            yaxis_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_k"),
            add_zero_line=True,
            save_html=save_html,
        )
        _plot_curves_over_k_by_layer(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} KL minus control — {cond.replace('_','-')}",
            subtitle="ΔKL by k, one curve per layer (Viridis: shallow → deep)",
            yaxis_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_layer"),
            add_zero_line=True,
            save_html=save_html,
        )

    if diff_mats:
        _plot_layer_k_panel(
            diff_mats, layer_indices, k_values,
            title=f"{label.capitalize()} KL minus control by condition",
            subtitle="ΔKL = KL_C_to_C − KL_ctl_to_C — shared diverging scale",
            colorbar_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=fig_dir / f"{label}_kl_minus_control_panel",
            diverging=True,
            save_html=save_html,
        )


def _emit_layer_k_effects(
    *,
    label: str,
    conditions: list[str],
    agg_to_target: dict[str, dict[int, dict[int, dict]]],
    agg_baseline_to_target: dict[str, dict[int, dict[int, dict]]],
    layer_indices: list[int],
    k_values: list[int],
    fig_dir: Path,
    hmm_subtitle: str = "",
    save_html: bool = False,
) -> None:
    """Produce all (layer × k) plots for a given intervention type (patching|steering)."""
    causal_shift: dict[str, np.ndarray] = {}
    abs_kl: dict[str, np.ndarray] = {}
    for cond in conditions:
        if cond not in agg_to_target:
            continue
        causal_shift[cond] = _causal_shift_matrix(
            agg_to_target[cond], agg_baseline_to_target[cond],
            layer_indices, k_values,
        )
        abs_kl[cond] = _layer_k_matrix(
            agg_to_target[cond], layer_indices, k_values,
        )

    # 1) Per-condition causal-shift heatmaps + line views
    for cond, Z in causal_shift.items():
        base = fig_dir / f"{label}_causal_shift_{cond}"
        _plot_layer_k_heatmap(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} causal shift — {cond.replace('_','-')}",
            subtitle="ΔKL = KL_baseline − KL_intervened to target (↑ = toward target)",
            colorbar_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_heatmap"),
            diverging=True,
            save_html=save_html,
        )
        _plot_curves_over_layer_by_k(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} causal shift — {cond.replace('_','-')}",
            subtitle="ΔKL toward target, one curve per k (Viridis: low k → high k)",
            yaxis_title="ΔKL [nats] (↑ = toward target)",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_k"),
            add_zero_line=True,
            save_html=save_html,
        )
        _plot_curves_over_k_by_layer(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} causal shift — {cond.replace('_','-')}",
            subtitle="ΔKL toward target, one curve per layer (Viridis: shallow → deep)",
            yaxis_title="ΔKL [nats] (↑ = toward target)",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_layer"),
            add_zero_line=True,
            save_html=save_html,
        )

    # 1b) Combined panel across conditions (shared diverging scale)
    if causal_shift:
        _plot_layer_k_panel(
            causal_shift, layer_indices, k_values,
            title=f"{label.capitalize()} causal shift by condition",
            subtitle="ΔKL toward target — shared diverging scale across panels",
            colorbar_title="ΔKL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=fig_dir / f"{label}_causal_shift_panel",
            diverging=True,
            save_html=save_html,
        )

    # 2) Gap heatmaps (and line views) — optimal vs other conditions
    if "optimal" in causal_shift:
        Z_opt = causal_shift["optimal"]
        for other in ["past_consistent", "random", "round_trip"]:
            if other not in causal_shift:
                continue
            Z_gap = Z_opt - causal_shift[other]
            base = fig_dir / f"{label}_gap_optimal_minus_{other}"
            _plot_layer_k_heatmap(
                Z_gap, layer_indices, k_values,
                title=f"{label.capitalize()} gap — optimal − {other.replace('_','-')}",
                subtitle="ΔKL_optimal − ΔKL_other (↑ = optimal pushes more)",
                colorbar_title="ΔΔKL [nats]",
                hmm_subtitle=hmm_subtitle,
                path=base.with_name(base.name + "_heatmap"),
                diverging=True,
                save_html=save_html,
            )
            _plot_curves_over_layer_by_k(
                Z_gap, layer_indices, k_values,
                title=f"{label.capitalize()} gap — optimal − {other.replace('_','-')}",
                subtitle="ΔΔKL by layer, one curve per k (Viridis: low → high)",
                yaxis_title="ΔΔKL [nats]",
                hmm_subtitle=hmm_subtitle,
                path=base.with_name(base.name + "_lines_by_k"),
                add_zero_line=True,
                save_html=save_html,
            )
            _plot_curves_over_k_by_layer(
                Z_gap, layer_indices, k_values,
                title=f"{label.capitalize()} gap — optimal − {other.replace('_','-')}",
                subtitle="ΔΔKL by k, one curve per layer (Viridis: shallow → deep)",
                yaxis_title="ΔΔKL [nats]",
                hmm_subtitle=hmm_subtitle,
                path=base.with_name(base.name + "_lines_by_layer"),
                add_zero_line=True,
                save_html=save_html,
            )

    # 3) Absolute KL_to_target heatmaps + lines (sequential, log scale)
    for cond, Z in abs_kl.items():
        base = fig_dir / f"{label}_abs_kl_to_target_{cond}"
        _plot_layer_k_heatmap(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} absolute KL to target — {cond.replace('_','-')}",
            subtitle="KL(P_opt(target) ∥ P_intervened) — log₁₀ scale",
            colorbar_title="log₁₀(KL)",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_heatmap"),
            diverging=False,
            log_scale=True,
            save_html=save_html,
        )
        _plot_curves_over_layer_by_k(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} absolute KL to target — {cond.replace('_','-')}",
            subtitle="KL by layer, one curve per k (log y)",
            yaxis_title="KL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_k"),
            log_y=True,
            save_html=save_html,
        )
        _plot_curves_over_k_by_layer(
            Z, layer_indices, k_values,
            title=f"{label.capitalize()} absolute KL to target — {cond.replace('_','-')}",
            subtitle="KL by k, one curve per layer (log y)",
            yaxis_title="KL [nats]",
            hmm_subtitle=hmm_subtitle,
            path=base.with_name(base.name + "_lines_by_layer"),
            log_y=True,
            save_html=save_html,
        )


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
    control_rng = np.random.default_rng(config.random_seed + 1)
    device = get_device(config.device)

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
    plot_data_dir = out_dir / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

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

    model = load_model(
        config.model_name,
        device,
        logger,
        n_ctx=config.n_ctx_override,
        model_n_devices=config.model_n_devices,
    )
    model_dtype: torch.dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        _patch_attention_dtype(model)
        logger.info("Patched attention softmax to stay in fp16")

    hmm = build_process_hmm(
        config.hmm.process_name,
        config.hmm.process_params,
        hmm_device=device,
    )
    logger.info(f"HMM             : {config.hmm.process_name} {config.hmm.process_params}")
    hmm_subtitle = format_hmm_process(config.hmm.process_name, config.hmm.process_params)
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

    # Control buckets: KL of (random-direction control intervention output)
    # against P_opt(η_target_for_cond_C) — same control used per (seq, layer, k),
    # scored against each principled condition's target.
    patch_control_kl_to_target = _make_cond_k_layer(PRINCIPLED_CONDITIONS) if config.run_with_controls else None
    patch_control_kl_to_factual = _make_cond_k_layer(PRINCIPLED_CONDITIONS) if config.run_with_controls else None
    steer_control_kl_to_target = _make_cond_k_layer(PRINCIPLED_CONDITIONS) if config.run_with_controls else None
    steer_control_kl_to_factual = _make_cond_k_layer(PRINCIPLED_CONDITIONS) if config.run_with_controls else None

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
    seq_exclude_set = set(config.seq_exclude)
    if seq_exclude_set:
        logger.info(f"Excluding sequences: {sorted(seq_exclude_set)}")

    for seq_i in range(N):
        if seq_i in seq_exclude_set:
            logger.info(f"  Seq {seq_i}/{N-1} — skipped (excluded)")
            continue
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
        P_clean_hmm = (probs_hmm / (probs_hmm.sum() + 1e-10)).astype(np.float32)
        probs_junk = max(float(1.0 - probs_hmm.sum()), 1e-10)
        P_clean_projected = np.append(probs_hmm, probs_junk).astype(np.float32)
        del logits

        eta_baseline = seq_beliefs[measure_pos + 1].astype(np.float32)
        P_opt = (eta_baseline @ emit.T).astype(np.float32)
        bkl = float(_kl(P_opt[None], P_clean_hmm[None])[0])
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
            P_clean_hmm=P_clean_hmm,
            P_clean_projected=P_clean_projected,
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
    n_patch_per_seq = (
        1  # optimal
        + 1  # round_trip
        + config.n_past_consistent_draws
        + config.n_splice_draws
        + config.n_random_patch_draws
    )
    n_steer_per_seq = (
        1  # optimal
        + config.n_past_consistent_draws
        + config.n_splice_draws
        + config.n_random_patch_draws
    )
    n_ctl = config.n_control_draws if config.run_with_controls else 0
    n_combined_per_seq = n_patch_per_seq + n_steer_per_seq + 2 * n_ctl

    valid_donor_seqs = [sd_d.seq_idx for sd_d in all_seq_data]

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
                eta_splice_list = [
                    _splice_targets(
                        seq_i, all_beliefs, valid_donor_seqs,
                        config.post_convergence_start, k, rng,
                    )
                    for _ in range(config.n_splice_draws)
                ]
                eta_random_list = [
                    _random_targets(positions, rng, n_states)
                    for _ in range(config.n_random_patch_draws)
                ]
                eta_control_list = (
                    [
                        _random_targets(positions, control_rng, n_states)
                        for _ in range(config.n_control_draws)
                    ]
                    if config.run_with_controls else []
                )

                round_trip_acts, round_trip_beliefs = _round_trip_targets(
                    encoder, clean_acts_k, decoder,
                )

                with torch.no_grad():
                    dec_optimal = decoder(torch.from_numpy(eta_optimal).float())
                    dec_past = [
                        decoder(torch.from_numpy(ep).float())
                        for ep in eta_past_list
                    ]
                    dec_splice = [
                        decoder(torch.from_numpy(es).float())
                        for es in eta_splice_list
                    ]
                    dec_random = [
                        decoder(torch.from_numpy(er).float())
                        for er in eta_random_list
                    ]
                    dec_control = [
                        decoder(torch.from_numpy(ec).float())
                        for ec in eta_control_list
                    ]

                sd.patch_targets[(layer, k)] = torch.stack(
                    [dec_optimal, torch.from_numpy(round_trip_acts)]
                    + dec_past + dec_splice + dec_random,
                    dim=0,
                )

                measure_belief_idx = -1
                patch_P_opt_list = [
                    (eta_optimal[measure_belief_idx] @ emit.T).astype(np.float32),
                    (round_trip_beliefs[measure_belief_idx] @ emit.T).astype(np.float32),
                ]
                # Track row offsets in patch_P_opts per principled condition.
                patch_offsets: dict[str, list[int]] = {"optimal": [0]}
                next_off = 2  # 0=optimal, 1=round_trip
                patch_offsets["past_consistent"] = list(range(next_off, next_off + len(eta_past_list)))
                for ep in eta_past_list:
                    patch_P_opt_list.append((ep[measure_belief_idx] @ emit.T).astype(np.float32))
                next_off += len(eta_past_list)
                patch_offsets["splice"] = list(range(next_off, next_off + len(eta_splice_list)))
                for es in eta_splice_list:
                    patch_P_opt_list.append((es[measure_belief_idx] @ emit.T).astype(np.float32))
                next_off += len(eta_splice_list)
                # random offsets — not used by control scatter but kept for completeness
                for er in eta_random_list:
                    patch_P_opt_list.append((er[measure_belief_idx] @ emit.T).astype(np.float32))
                sd.patch_target_P_opts[(layer, k)] = np.stack(patch_P_opt_list, axis=0)
                sd.patch_target_principled_offsets[(layer, k)] = patch_offsets

                with torch.no_grad():
                    source_beliefs = encoder(torch.from_numpy(clean_acts_k).float())
                    dec_source = decoder(source_beliefs)
                    steer_optimal = (dec_optimal - dec_source).unsqueeze(0)
                    steer_past = [
                        (dp - dec_source).unsqueeze(0)
                        for dp in dec_past
                    ]
                    steer_splice = [
                        (ds - dec_source).unsqueeze(0)
                        for ds in dec_splice
                    ]
                    steer_random = [
                        (dr - dec_source).unsqueeze(0)
                        for dr in dec_random
                    ]
                    steer_control = [
                        (dc - dec_source).unsqueeze(0)
                        for dc in dec_control
                    ]

                sd.steer_deltas[(layer, k)] = torch.cat(
                    [steer_optimal] + steer_past + steer_splice + steer_random, dim=0
                )

                steer_P_opt_list = [
                    (eta_optimal[measure_belief_idx] @ emit.T).astype(np.float32),
                ]
                steer_offsets: dict[str, list[int]] = {"optimal": [0]}
                next_off_s = 1
                steer_offsets["past_consistent"] = list(range(next_off_s, next_off_s + len(eta_past_list)))
                for ep in eta_past_list:
                    steer_P_opt_list.append((ep[measure_belief_idx] @ emit.T).astype(np.float32))
                next_off_s += len(eta_past_list)
                steer_offsets["splice"] = list(range(next_off_s, next_off_s + len(eta_splice_list)))
                for es in eta_splice_list:
                    steer_P_opt_list.append((es[measure_belief_idx] @ emit.T).astype(np.float32))
                next_off_s += len(eta_splice_list)
                for er in eta_random_list:
                    steer_P_opt_list.append((er[measure_belief_idx] @ emit.T).astype(np.float32))
                sd.steer_target_P_opts[(layer, k)] = np.stack(steer_P_opt_list, axis=0)
                sd.steer_target_principled_offsets[(layer, k)] = steer_offsets

                if config.run_with_controls and dec_control:
                    sd.patch_controls[(layer, k)] = torch.stack(dec_control, dim=0)
                    sd.steer_controls[(layer, k)] = torch.cat(steer_control, dim=0)

            del encoder, decoder

    control_etas_sha = ""
    if config.run_with_controls:
        import hashlib
        h = hashlib.sha256()
        for sd in all_seq_data:
            for layer in config.layer_indices:
                for k in config.k_values:
                    if (layer, k) in sd.patch_controls:
                        h.update(sd.patch_controls[(layer, k)].cpu().numpy().tobytes())
        control_etas_sha = h.hexdigest()
        logger.info(f"Control draws SHA256: {control_etas_sha[:16]} (n_control_draws={config.n_control_draws})")

    # ── Load checkpoint (resume) ──────────────────────────────────────────────
    completed_layers: list[int] = []
    if resume_dir is not None:
        chk_path = out_dir / CHECKPOINT_FILENAME
        if chk_path.exists():
            logger.info(f"Loading checkpoint: {chk_path}")
            chk = _load_checkpoint(
                chk_path, PATCH_CONDITIONS, STEER_CONDITIONS,
                config.k_values, config.layer_indices,
                PRINCIPLED_CONDITIONS if config.run_with_controls else None,
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
            if config.run_with_controls and "patch_control_kl_to_target" in chk:
                patch_control_kl_to_target = chk["patch_control_kl_to_target"]
                patch_control_kl_to_factual = chk["patch_control_kl_to_factual"]
                steer_control_kl_to_target = chk["steer_control_kl_to_target"]
                steer_control_kl_to_factual = chk["steer_control_kl_to_factual"]
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
                P_abl, batch, layer, n_vocab, ablation_kl, ablation_kl_to_clean,
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
                tb, np_b, ns_b, pt_b, sd_b = _build_combined_batch(
                    batch, layer, _k, config.run_with_controls,
                )
                n_ctl_p_tot = pt_b.shape[0] - np_b
                n_ctl_s_tot = sd_b.shape[0] - ns_b

                max_v = config.max_variants_per_pass
                if max_v is not None and tb.shape[0] > max_v:
                    # Variant-level chunking: split each of the 4 segments (patch_main,
                    # steer_main, patch_ctl, steer_ctl) into sub-passes of ≤ max_v rows.
                    pieces: list[np.ndarray] = []
                    # Segment 0: patch_main (replace)
                    pmt = tb[:np_b]
                    pmtgt = pt_b[:np_b]
                    for start in range(0, np_b, max_v):
                        end = min(start + max_v, np_b)
                        pieces.append(_run_combined(
                            model, pmt[start:end], layer, _pos,
                            end - start, 0, pmtgt[start:end], sd_b[:0],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    # Segment 1: steer_main (add)
                    smt = tb[np_b:np_b + ns_b]
                    smdlt = sd_b[:ns_b]
                    for start in range(0, ns_b, max_v):
                        end = min(start + max_v, ns_b)
                        pieces.append(_run_combined(
                            model, smt[start:end], layer, _pos,
                            0, end - start, pt_b[:0], smdlt[start:end],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    # Segment 2: patch_ctl (replace)
                    pct = tb[np_b + ns_b:np_b + ns_b + n_ctl_p_tot]
                    pctgt = pt_b[np_b:]
                    for start in range(0, n_ctl_p_tot, max_v):
                        end = min(start + max_v, n_ctl_p_tot)
                        pieces.append(_run_combined(
                            model, pct[start:end], layer, _pos,
                            end - start, 0, pctgt[start:end], sd_b[:0],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    # Segment 3: steer_ctl (add)
                    sct = tb[np_b + ns_b + n_ctl_p_tot:]
                    sctlt = sd_b[ns_b:]
                    for start in range(0, n_ctl_s_tot, max_v):
                        end = min(start + max_v, n_ctl_s_tot)
                        pieces.append(_run_combined(
                            model, sct[start:end], layer, _pos,
                            0, end - start, pt_b[:0], sctlt[start:end],
                            measure_pos, mid_tok_ids, device, model_dtype,
                        ))
                    P_out = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 0))
                else:
                    try:
                        P_out = _run_combined(
                            model, tb, layer, _pos,
                            np_b, ns_b, pt_b, sd_b,
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
                    P_out, np_b, ns_b, batch, layer, _k, n_vocab,
                    patch_kl_to_target, patch_kl_to_factual,
                    steer_kl_to_target, steer_kl_to_factual,
                    patch_kl_to_clean, steer_kl_to_clean,
                    baseline_kl_to_target_patch, baseline_kl_to_target_steer,
                    config.n_past_consistent_draws, config.n_splice_draws,
                    config.n_random_patch_draws,
                    config.run_with_controls, config.n_control_draws,
                    patch_control_kl_to_target, steer_control_kl_to_target,
                    patch_control_kl_to_factual, steer_control_kl_to_factual,
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
            patch_control_kl_to_target=patch_control_kl_to_target,
            patch_control_kl_to_factual=patch_control_kl_to_factual,
            steer_control_kl_to_target=steer_control_kl_to_target,
            steer_control_kl_to_factual=steer_control_kl_to_factual,
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

    if config.run_with_controls:
        agg_patch_ctl_to_target = _agg_cond_k_layer(patch_control_kl_to_target, PRINCIPLED_CONDITIONS)
        agg_patch_ctl_to_factual = _agg_cond_k_layer(patch_control_kl_to_factual, PRINCIPLED_CONDITIONS)
        agg_steer_ctl_to_target = _agg_cond_k_layer(steer_control_kl_to_target, PRINCIPLED_CONDITIONS)
        agg_steer_ctl_to_factual = _agg_cond_k_layer(steer_control_kl_to_factual, PRINCIPLED_CONDITIONS)
    else:
        agg_patch_ctl_to_target = None
        agg_patch_ctl_to_factual = None
        agg_steer_ctl_to_target = None
        agg_steer_ctl_to_factual = None

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
        "ablation_to_clean_projected": {
            cond: {str(l): v for l, v in by_layer.items()}
            for cond, by_layer in agg_ablation_to_clean.items()
        },
    }
    if config.run_with_controls:
        metrics["patching_control_to_target"] = _to_json(agg_patch_ctl_to_target, PRINCIPLED_CONDITIONS)
        metrics["patching_control_to_factual"] = _to_json(agg_patch_ctl_to_factual, PRINCIPLED_CONDITIONS)
        metrics["steering_control_to_target"] = _to_json(agg_steer_ctl_to_target, PRINCIPLED_CONDITIONS)
        metrics["steering_control_to_factual"] = _to_json(agg_steer_ctl_to_factual, PRINCIPLED_CONDITIONS)

        # Paired-diff aggregates: KL_C_to_X − KL_ctl_to_X per seq, then aggregated.
        # These mirror the new control-delta plots and let downstream code skip
        # re-implementing the per-seq pairing logic.
        agg_patch_diff_to_target_metrics: dict[str, dict[int, dict[int, dict]]] = {}
        agg_patch_diff_to_factual_metrics: dict[str, dict[int, dict[int, dict]]] = {}
        agg_steer_diff_to_target_metrics: dict[str, dict[int, dict[int, dict]]] = {}
        agg_steer_diff_to_factual_metrics: dict[str, dict[int, dict[int, dict]]] = {}
        for cond in PRINCIPLED_CONDITIONS:
            agg_patch_diff_to_target_metrics[cond] = {}
            agg_patch_diff_to_factual_metrics[cond] = {}
            agg_steer_diff_to_target_metrics[cond] = {}
            agg_steer_diff_to_factual_metrics[cond] = {}
            for k in config.k_values:
                agg_patch_diff_to_target_metrics[cond][k] = {
                    l: _agg_paired_diff(
                        patch_kl_to_target[cond][k][l],
                        patch_control_kl_to_target[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_patch_diff_to_factual_metrics[cond][k] = {
                    l: _agg_paired_diff(
                        patch_kl_to_factual[cond][k][l],
                        patch_control_kl_to_factual[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_steer_diff_to_target_metrics[cond][k] = {
                    l: _agg_paired_diff(
                        steer_kl_to_target[cond][k][l],
                        steer_control_kl_to_target[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_steer_diff_to_factual_metrics[cond][k] = {
                    l: _agg_paired_diff(
                        steer_kl_to_factual[cond][k][l],
                        steer_control_kl_to_factual[cond][k][l],
                    )
                    for l in config.layer_indices
                }
        metrics["patching_minus_control_to_target"] = _to_json(
            agg_patch_diff_to_target_metrics, PRINCIPLED_CONDITIONS
        )
        metrics["patching_minus_control_to_factual"] = _to_json(
            agg_patch_diff_to_factual_metrics, PRINCIPLED_CONDITIONS
        )
        metrics["steering_minus_control_to_target"] = _to_json(
            agg_steer_diff_to_target_metrics, PRINCIPLED_CONDITIONS
        )
        metrics["steering_minus_control_to_factual"] = _to_json(
            agg_steer_diff_to_factual_metrics, PRINCIPLED_CONDITIONS
        )
    raw_records = {
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
    if config.run_with_controls:
        raw_records["patch_control_kl_to_target"] = _serialize_ckl(patch_control_kl_to_target)
        raw_records["patch_control_kl_to_factual"] = _serialize_ckl(patch_control_kl_to_factual)
        raw_records["steer_control_kl_to_target"] = _serialize_ckl(steer_control_kl_to_target)
        raw_records["steer_control_kl_to_factual"] = _serialize_ckl(steer_control_kl_to_factual)
    _write_json(
        plot_data_dir / "raw_intervention_records.json",
        raw_records,
    )
    _write_json(
        plot_data_dir / "aggregated_plot_data.json",
        {
            "layer_indices": config.layer_indices,
            "k_values": config.k_values,
            "baseline_mean": agg_baseline["mean"],
            "metrics": metrics,
        },
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    provenance = {
        "random_seed": config.random_seed,
        "control_rng_seed": config.random_seed + 1 if config.run_with_controls else None,
        "n_random_ablation_draws": config.n_random_ablation_draws,
        "n_past_consistent_draws": config.n_past_consistent_draws,
        "n_splice_draws": config.n_splice_draws,
        "n_random_patch_draws": config.n_random_patch_draws,
        "run_with_controls": config.run_with_controls,
        "n_control_draws": config.n_control_draws,
        "control_etas_sha256": control_etas_sha,
        "patch_conditions": PATCH_CONDITIONS,
        "steer_conditions": STEER_CONDITIONS,
        "principled_conditions": PRINCIPLED_CONDITIONS,
        "seqs_used": [sd.seq_idx for sd in all_seq_data],
        "seqs_excluded": list(config.seq_exclude),
        "layer_indices": list(config.layer_indices),
        "k_values": list(config.k_values),
        "post_convergence_start": config.post_convergence_start,
        "train_eval_split": config.train_eval_split,
        "L_trunc": L_trunc,
        "measure_pos": measure_pos,
        "eval_act_start": eval_act_start,
        "model_name": config.model_name,
        "training_dir": str(training_dir),
        "hmm": {"process_name": config.hmm.process_name, "process_params": dict(config.hmm.process_params)},
    }
    _write_json(out_dir / "run_provenance.json", provenance)
    logger.info("Saved run_provenance.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    baseline_mean = agg_baseline["mean"]

    # Per-(cond, k, layer) paired diff: principled cond minus its random-direction control.
    if config.run_with_controls:
        agg_patch_diff_to_target: dict[str, dict[int, dict[int, dict]]] = {}
        agg_patch_diff_to_factual: dict[str, dict[int, dict[int, dict]]] = {}
        agg_steer_diff_to_target: dict[str, dict[int, dict[int, dict]]] = {}
        agg_steer_diff_to_factual: dict[str, dict[int, dict[int, dict]]] = {}
        for cond in PRINCIPLED_CONDITIONS:
            agg_patch_diff_to_target[cond] = {}
            agg_patch_diff_to_factual[cond] = {}
            agg_steer_diff_to_target[cond] = {}
            agg_steer_diff_to_factual[cond] = {}
            for k in config.k_values:
                agg_patch_diff_to_target[cond][k] = {
                    l: _agg_paired_diff(
                        patch_kl_to_target[cond][k][l],
                        patch_control_kl_to_target[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_patch_diff_to_factual[cond][k] = {
                    l: _agg_paired_diff(
                        patch_kl_to_factual[cond][k][l],
                        patch_control_kl_to_factual[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_steer_diff_to_target[cond][k] = {
                    l: _agg_paired_diff(
                        steer_kl_to_target[cond][k][l],
                        steer_control_kl_to_target[cond][k][l],
                    )
                    for l in config.layer_indices
                }
                agg_steer_diff_to_factual[cond][k] = {
                    l: _agg_paired_diff(
                        steer_kl_to_factual[cond][k][l],
                        steer_control_kl_to_factual[cond][k][l],
                    )
                    for l in config.layer_indices
                }
    else:
        agg_patch_diff_to_target = None
        agg_patch_diff_to_factual = None
        agg_steer_diff_to_target = None
        agg_steer_diff_to_factual = None

    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation patching: KL vs layer by condition",
            hmm_subtitle=hmm_subtitle,
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_kl_vs_layer_k{k}",
            save_html=config.save_html,
        )
        if config.run_with_controls:
            _plot_kl_minus_control_vs_layer(
                {cond: agg_patch_diff_to_target[cond][k] for cond in PRINCIPLED_CONDITIONS},
                config.layer_indices,
                k=k,
                title="Activation patching: KL_to_target − KL_control_to_target",
                hmm_subtitle=hmm_subtitle,
                colors={c: _PATCH_COLORS[c] for c in PRINCIPLED_CONDITIONS},
                path=fig_dir / f"patching_kl_minus_control_vs_layer_k{k}",
                save_html=config.save_html,
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
                hmm_subtitle=hmm_subtitle,
                path=fig_dir / f"patching_crossing_{cross_cond}_k{k}",
                save_html=config.save_html,
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_patch[cond][k] for cond in PATCH_CONDITIONS},
            layer_indices=config.layer_indices,
            k=k,
            title="Activation patching",
            hmm_subtitle=hmm_subtitle,
            colors=_PATCH_COLORS,
            path=fig_dir / f"patching_causal_shift_k{k}",
            save_html=config.save_html,
        )

    for k in config.k_values:
        _plot_kl_vs_layer(
            {cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            config.layer_indices,
            k=k,
            baseline_mean=baseline_mean,
            title="Activation steering: KL vs layer by condition",
            hmm_subtitle=hmm_subtitle,
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_kl_vs_layer_k{k}",
            save_html=config.save_html,
        )
        if config.run_with_controls:
            _plot_kl_minus_control_vs_layer(
                {cond: agg_steer_diff_to_target[cond][k] for cond in PRINCIPLED_CONDITIONS},
                config.layer_indices,
                k=k,
                title="Activation steering: KL_to_target − KL_control_to_target",
                hmm_subtitle=hmm_subtitle,
                colors={c: _STEER_COLORS[c] for c in PRINCIPLED_CONDITIONS},
                path=fig_dir / f"steering_kl_minus_control_vs_layer_k{k}",
                save_html=config.save_html,
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
                hmm_subtitle=hmm_subtitle,
                path=fig_dir / f"steering_crossing_{cross_cond}_k{k}",
                save_html=config.save_html,
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_steer[cond][k] for cond in STEER_CONDITIONS},
            layer_indices=config.layer_indices,
            k=k,
            title="Activation steering",
            hmm_subtitle=hmm_subtitle,
            colors=_STEER_COLORS,
            path=fig_dir / f"steering_causal_shift_k{k}",
            save_html=config.save_html,
        )

    k_max = max(config.k_values)
    _plot_heatmap(
        agg_patch_to_target["optimal"][k_max],
        config.layer_indices,
        title=f"Patching KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"heatmap_patching_optimal_k{k_max}",
        save_html=config.save_html,
    )
    _plot_heatmap(
        agg_steer_to_target["optimal"][k_max],
        config.layer_indices,
        title=f"Steering KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"heatmap_steering_optimal_k{k_max}",
        save_html=config.save_html,
    )

    _plot_ablation_curve(
        agg_ablation["belief"], agg_ablation["random"],
        config.layer_indices,
        title="Belief-subspace ablation: KL to optimal",
        subtitle="KL(P_opt ∥ P_ablated) — HMM-token simplex — log scale",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / "ablation_causal_importance",
        save_html=config.save_html,
    )
    _plot_ablation_curve(
        agg_ablation_to_clean["belief"], agg_ablation_to_clean["random"],
        config.layer_indices,
        title="Belief-subspace ablation: output perturbation (HMM-token + junk)",
        subtitle="KL(P_clean ∥ P_ablated) — HMM-token + junk projection — log scale",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / "ablation_kl_to_output_projected",
        save_html=config.save_html,
    )

    _plot_roundtrip_comparison(
        {cond: agg_patch_to_target[cond][k_max] for cond in ["optimal", "round_trip"]},
        config.layer_indices,
        baseline_mean=baseline_mean,
        k=k_max,
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"roundtrip_h2a_vs_h2b_k{k_max}",
        save_html=config.save_html,
    )

    _emit_layer_k_effects(
        label="patching",
        conditions=PATCH_CONDITIONS,
        agg_to_target=agg_patch_to_target,
        agg_baseline_to_target=agg_bl_to_target_patch,
        layer_indices=config.layer_indices,
        k_values=config.k_values,
        fig_dir=fig_dir,
        hmm_subtitle=hmm_subtitle,
        save_html=config.save_html,
    )
    if config.run_with_controls:
        _emit_kl_minus_control_layer_k(
            label="patching",
            conditions=PRINCIPLED_CONDITIONS,
            diff_records_by_cond=agg_patch_diff_to_target,
            layer_indices=config.layer_indices,
            k_values=config.k_values,
            fig_dir=fig_dir,
            hmm_subtitle=hmm_subtitle,
            save_html=config.save_html,
        )
    _emit_layer_k_effects(
        label="steering",
        conditions=STEER_CONDITIONS,
        agg_to_target=agg_steer_to_target,
        agg_baseline_to_target=agg_bl_to_target_steer,
        layer_indices=config.layer_indices,
        k_values=config.k_values,
        fig_dir=fig_dir,
        hmm_subtitle=hmm_subtitle,
        save_html=config.save_html,
    )
    if config.run_with_controls:
        _emit_kl_minus_control_layer_k(
            label="steering",
            conditions=PRINCIPLED_CONDITIONS,
            diff_records_by_cond=agg_steer_diff_to_target,
            layer_indices=config.layer_indices,
            k_values=config.k_values,
            fig_dir=fig_dir,
            hmm_subtitle=hmm_subtitle,
            save_html=config.save_html,
        )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
