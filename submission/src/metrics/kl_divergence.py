"""KL divergence utilities for comparing HMM and LLM next-token distributions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def extract_concept_probs(
    logits_flat: np.ndarray,
    concept_ids: list[int],
) -> np.ndarray:
    """Softmax-renormalise logits over a subset of token IDs.

    Parameters
    ----------
    logits_flat : np.ndarray
        Shape ``(N, full_vocab_size)`` — raw LLM logits.
    concept_ids : list[int]
        LLM token IDs corresponding to HMM emission symbols.

    Returns
    -------
    np.ndarray
        Shape ``(N, len(concept_ids))`` — probability distribution over the
        HMM vocabulary, renormalised so entries sum to 1.
    """
    concept_logits = logits_flat[:, concept_ids]
    return F.softmax(torch.tensor(concept_logits, dtype=torch.float32), dim=-1).numpy()


def compute_kl_divergence_batch(
    hmm_probs: np.ndarray,
    llm_probs: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position KL(HMM || LLM) averaged over sequences.

    Parameters
    ----------
    hmm_probs : np.ndarray
        Shape ``(n_sequences, seq_length, vocab_size)`` — HMM next-token
        probabilities (the *reference* distribution).
    llm_probs : np.ndarray
        Same shape — LLM next-token probabilities (the *approximation*).
    eps : float
        Floor for clipping to avoid log(0).

    Returns
    -------
    kl_mean : np.ndarray
        Shape ``(seq_length,)`` — mean KL across sequences at each position.
    kl_std : np.ndarray
        Shape ``(seq_length,)`` — std of KL across sequences at each position.

    Notes
    -----
    KL(P || Q) = Σ_v P(v) log(P(v) / Q(v)).  A low value means the LLM's
    predictions closely match the HMM's Bayes-optimal predictions.
    """
    p = np.clip(hmm_probs, eps, 1.0)
    q = np.clip(llm_probs, eps, 1.0)
    kl_per_pos_seq = np.sum(p * np.log(p / q), axis=-1)  # (n_seq, seq_len)
    return kl_per_pos_seq.mean(axis=0), kl_per_pos_seq.std(axis=0)
