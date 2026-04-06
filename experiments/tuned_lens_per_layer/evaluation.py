"""Evaluation metrics for the per-layer tuned lens experiment."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LayerMetrics:
    """Metrics for one layer, computed on the held-out test set."""

    layer: int

    # Global (averaged over all test positions)
    kl_final_vs_tuned: float = 0.0      # KL(final LLM dist || tuned lens dist)
    kl_hmm_vs_tuned: float = 0.0        # KL(HMM next-state || tuned lens dist)
    kl_hmm_vs_logit: float = 0.0        # KL(HMM next-state || raw logit lens dist)
    nll_tuned: float = 0.0              # NLL of actual next token under tuned lens
    nll_logit: float = 0.0              # NLL of actual next token under logit lens
    top1_agreement_tuned: float = 0.0   # fraction where tuned lens top-1 == final model top-1
    top1_agreement_logit: float = 0.0   # fraction where logit lens top-1 == final model top-1

    # HMM-target tuned lens metrics
    kl_hmm_vs_tuned_hmm: float = 0.0   # KL(HMM || HMM-target tuned lens)
    kl_final_vs_tuned_hmm: float = 0.0  # KL(final model || HMM-target tuned lens)

    # Per-position arrays (test set, averaged over test sequences)
    kl_final_vs_tuned_by_pos: np.ndarray = field(default_factory=lambda: np.array([]))
    kl_hmm_vs_tuned_by_pos: np.ndarray = field(default_factory=lambda: np.array([]))
    kl_hmm_vs_logit_by_pos: np.ndarray = field(default_factory=lambda: np.array([]))
    kl_hmm_vs_tuned_hmm_by_pos: np.ndarray = field(default_factory=lambda: np.array([]))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """KL(P || Q) per sample. p, q: (..., vocab_size)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def _nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Negative log-likelihood per sample. probs: (N, V), targets: (N,) integer indices."""
    p = np.clip(probs[np.arange(len(targets)), targets], eps, 1.0)
    return -np.log(p)


def compute_layer_metrics(
    layer: int,
    tuned_lens_probs: np.ndarray,
    logit_lens_probs: np.ndarray,
    final_model_probs: np.ndarray,
    hmm_probs: np.ndarray,
    next_tokens: np.ndarray,
    n_sequences: int,
    seq_length: int,
    tuned_lens_hmm_probs: np.ndarray | None = None,
) -> LayerMetrics:
    """Compute all evaluation metrics for a single layer.

    All probability arrays have shape (N, n_concepts) where N = n_sequences * seq_length.
    next_tokens: (N,) integer indices into the concept vocabulary.

    Parameters
    ----------
    tuned_lens_probs : (N, n_concepts) from model-target tuned lens
    logit_lens_probs : (N, n_concepts) from raw logit lens
    final_model_probs : (N, n_concepts) from final model output
    hmm_probs : (N, n_concepts) from HMM
    next_tokens : (N,) next token indices (in concept vocab space, 0..n_concepts-1)
    n_sequences, seq_length : for reshaping to per-position
    tuned_lens_hmm_probs : (N, n_concepts) from HMM-target tuned lens (optional)
    """
    N = tuned_lens_probs.shape[0]

    # Global metrics
    kl_final_tuned = float(_kl(final_model_probs, tuned_lens_probs).mean())
    kl_hmm_tuned = float(_kl(hmm_probs, tuned_lens_probs).mean())
    kl_hmm_logit = float(_kl(hmm_probs, logit_lens_probs).mean())

    nll_t = float(_nll(tuned_lens_probs, next_tokens).mean())
    nll_l = float(_nll(logit_lens_probs, next_tokens).mean())

    final_top1 = final_model_probs.argmax(axis=-1)
    tuned_top1 = tuned_lens_probs.argmax(axis=-1)
    logit_top1 = logit_lens_probs.argmax(axis=-1)
    top1_tuned = float((tuned_top1 == final_top1).mean())
    top1_logit = float((logit_top1 == final_top1).mean())

    # HMM-target tuned lens metrics
    kl_hmm_tuned_hmm = 0.0
    kl_final_tuned_hmm = 0.0
    if tuned_lens_hmm_probs is not None:
        kl_hmm_tuned_hmm = float(_kl(hmm_probs, tuned_lens_hmm_probs).mean())
        kl_final_tuned_hmm = float(_kl(final_model_probs, tuned_lens_hmm_probs).mean())

    # Per-position metrics (reshape to (n_seq, seq_len, ...) then average over sequences)
    def per_pos(values_flat: np.ndarray) -> np.ndarray:
        return values_flat.reshape(n_sequences, seq_length).mean(axis=0)

    kl_ft_pos = per_pos(_kl(final_model_probs, tuned_lens_probs))
    kl_ht_pos = per_pos(_kl(hmm_probs, tuned_lens_probs))
    kl_hl_pos = per_pos(_kl(hmm_probs, logit_lens_probs))

    # HMM-target tuned lens per-position
    kl_ht_hmm_pos = np.array([])
    if tuned_lens_hmm_probs is not None:
        kl_ht_hmm_pos = per_pos(_kl(hmm_probs, tuned_lens_hmm_probs))

    return LayerMetrics(
        layer=layer,
        kl_final_vs_tuned=kl_final_tuned,
        kl_hmm_vs_tuned=kl_hmm_tuned,
        kl_hmm_vs_logit=kl_hmm_logit,
        nll_tuned=nll_t,
        nll_logit=nll_l,
        top1_agreement_tuned=top1_tuned,
        top1_agreement_logit=top1_logit,
        kl_hmm_vs_tuned_hmm=kl_hmm_tuned_hmm,
        kl_final_vs_tuned_hmm=kl_final_tuned_hmm,
        kl_final_vs_tuned_by_pos=kl_ft_pos,
        kl_hmm_vs_tuned_by_pos=kl_ht_pos,
        kl_hmm_vs_logit_by_pos=kl_hl_pos,
        kl_hmm_vs_tuned_hmm_by_pos=kl_ht_hmm_pos,
    )
