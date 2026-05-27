"""Evaluation metrics for the per-layer tuned lens experiment.

Schema is per-lens. ``LayerMetrics.lenses`` is a dict keyed by lens name
(``"logit"``, ``"tuned_full"``, ``"tuned_concept"``, ``"tuned_hmm"``); each
entry has scalar KL/NLL/top-1 plus per-position curves. ``_kl`` and ``_nll``
defensively replace non-finite inputs (NaN/±inf) before clipping, so a
numerical pathology in the lens path can never silently materialise as
``Infinity`` in the saved metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


_LENSES_DEFAULT = ("logit", "tuned_full", "tuned_concept", "tuned_hmm")


@dataclass
class LayerMetrics:
    """Metrics for one layer, computed on the held-out test set.

    ``lenses[name]`` is a dict with keys::

        kl_final            float   KL(final model || lens)
        kl_hmm              float   KL(HMM        || lens)
        nll                 float   NLL of realised next token under lens
        top1_agreement      float   fraction where lens top-1 == final model top-1
        kl_final_by_pos     ndarray (seq_length,) per-position KL(final || lens)
        kl_hmm_by_pos       ndarray (seq_length,) per-position KL(HMM   || lens)
    """

    layer: int
    lenses: dict[str, dict] = field(default_factory=dict)


def _sanitize(arr: np.ndarray, eps: float) -> np.ndarray:
    """Replace NaN/+inf/-inf with safe values, then clip to ``[eps, 1.0]``.

    Guarantees: output is finite and in ``[eps, 1.0]``.
    """
    arr = np.nan_to_num(arr, nan=eps, posinf=1.0, neginf=eps)
    return np.clip(arr, eps, 1.0)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """KL(P || Q) per sample. p, q: (..., n_concepts).

    Output is finite by construction (sanitize → clip → bounded log).
    """
    p = _sanitize(p, eps)
    q = _sanitize(q, eps)
    return np.sum(p * np.log(p / q), axis=-1)


def _nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Negative log-likelihood per sample. probs: (N, V), targets: (N,) int."""
    p = probs[np.arange(len(targets)), targets]
    p = _sanitize(p, eps)
    return -np.log(p)


def compute_lens_metrics(
    lens_probs: np.ndarray,
    final_model_probs: np.ndarray,
    hmm_probs: np.ndarray,
    next_tokens: np.ndarray,
    n_sequences: int,
    seq_length: int,
) -> dict:
    """Compute all metrics for one lens against final + HMM references.

    All probability arrays are ``(N, n_concepts)`` with N = n_sequences * seq_length.
    """
    kl_f_per_sample = _kl(final_model_probs, lens_probs)
    kl_h_per_sample = _kl(hmm_probs, lens_probs)
    nll_per_sample = _nll(lens_probs, next_tokens)

    lens_top1 = lens_probs.argmax(axis=-1)
    final_top1 = final_model_probs.argmax(axis=-1)

    return {
        "kl_final": float(kl_f_per_sample.mean()),
        "kl_hmm": float(kl_h_per_sample.mean()),
        "nll": float(nll_per_sample.mean()),
        "top1_agreement": float((lens_top1 == final_top1).mean()),
        "kl_final_by_pos": kl_f_per_sample.reshape(n_sequences, seq_length).mean(axis=0),
        "kl_hmm_by_pos": kl_h_per_sample.reshape(n_sequences, seq_length).mean(axis=0),
    }


def compute_layer_metrics(
    layer: int,
    lens_probs: Mapping[str, np.ndarray],
    final_model_probs: np.ndarray,
    hmm_probs: np.ndarray,
    next_tokens: np.ndarray,
    n_sequences: int,
    seq_length: int,
) -> LayerMetrics:
    """Compute per-lens evaluation metrics for a single layer.

    Parameters
    ----------
    layer
        Layer index.
    lens_probs
        Dict mapping lens name -> (N, n_concepts) probability array.
    final_model_probs
        Reference: model's own concept-token softmax, shape (N, n_concepts).
    hmm_probs
        Reference: HMM Bayes-optimal next-token probs, shape (N, n_concepts).
    next_tokens
        Realised next-token integer indices (in concept-vocab space),
        shape (N,). Used for NLL.
    n_sequences, seq_length
        Used to reshape per-sample arrays into (n_sequences, seq_length)
        for per-position averaging.
    """
    out = LayerMetrics(layer=layer)
    for name, probs in lens_probs.items():
        out.lenses[name] = compute_lens_metrics(
            lens_probs=probs,
            final_model_probs=final_model_probs,
            hmm_probs=hmm_probs,
            next_tokens=next_tokens,
            n_sequences=n_sequences,
            seq_length=seq_length,
        )
    return out
