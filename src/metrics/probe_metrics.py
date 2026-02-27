from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from src.probes import ProbeResult
except ImportError:
    from probes import ProbeResult


def compute_kl(
    model_probs: np.ndarray,
    optimal_probs: np.ndarray,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position KL(model ‖ optimal) and its smoothed version.

    Returns (kl_raw, kl_smooth), both of shape (seq_len,).
    """
    kl_raw = np.sum(
        np.where(
            model_probs > 0,
            model_probs * np.log(model_probs / np.clip(optimal_probs, 1e-10, None)),
            0.0,
        ),
        axis=-1,
    )

    if smooth_window > 1:
        half = smooth_window // 2
        kl_padded = np.pad(kl_raw, (half, half), mode="edge")
        kernel = np.ones(smooth_window) / smooth_window
        kl_smooth = np.convolve(kl_padded, kernel, mode="valid")[: len(kl_raw)]
    else:
        kl_smooth = kl_raw

    return kl_raw, kl_smooth


def find_kl_threshold(
    model_probs: np.ndarray,
    optimal_probs: np.ndarray,
    fraction: float = 0.2,
    smooth_window: int = 5,
    min_position: int = 0,
) -> tuple[int, bool]:
    """
    Returns (t*, crossed) where t* is the first position (>= min_position) where
    the smoothed KL drops into the bottom `fraction` of its (max - min) range
    computed over [min_position:], and crossed=True.
    Falls back to the argmin of the smoothed KL (within [min_position:]) with
    crossed=False.  The returned index is always absolute (into the full sequence).

    model_probs:   (seq_len, vocab_size) — LLM next-token probabilities
    optimal_probs: (seq_len, vocab_size) — Bayesian-optimal next-token probabilities
    fraction:      threshold = min + fraction * (max - min); e.g. 0.2 means
                   "first time KL is within the bottom 20% of its total swing"
    smooth_window: rolling-mean window size applied before thresholding
    min_position:  ignore positions before this index when detecting the threshold
    """
    _, kl_smooth = compute_kl(model_probs, optimal_probs, smooth_window)

    kl_search = kl_smooth[min_position:]
    kl_min, kl_max = kl_search.min(), kl_search.max()
    threshold = kl_min + fraction * (kl_max - kl_min)

    below = np.where(kl_search <= threshold)[0]
    if len(below) > 0:
        return int(below[0]) + min_position, True
    return int(np.argmin(kl_search)) + min_position, False


def _column_cosine_similarity(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between columns of W_a and W_b.

    W_a: (d_model, n_states)
    W_b: (d_model, n_states)
    Returns: (n_states, n_states) — entry [i, j] = cos(angle(W_a[:, i], W_b[:, j]))
    """
    norms_a = np.linalg.norm(W_a, axis=0, keepdims=True)   # (1, n_states)
    norms_b = np.linalg.norm(W_b, axis=0, keepdims=True)   # (1, n_states)
    W_a_norm = W_a / np.clip(norms_a, 1e-10, None)
    W_b_norm = W_b / np.clip(norms_b, 1e-10, None)
    return W_a_norm.T @ W_b_norm                            # (n_states, n_states)


def compare_probes(probe_a: ProbeResult, probe_b: ProbeResult) -> dict:
    """
    Compare two probes trained on different sequences.

    Returns:
        test_mse_a, test_mse_b       — each probe's held-out MSE
        cross_mse_ab                 — probe A's weights applied to B's activations
        cross_mse_ba                 — probe B's weights applied to A's activations
        column_cosine_sim            — (n_states, n_states) pairwise cosine similarities
                                       between per-component weight vectors; entry [i, j]
                                       is the cosine similarity between the direction for
                                       belief component i in probe A and component j in probe B
    """
    device = next(probe_a.probe.parameters()).device

    act_a = torch.tensor(probe_a.activations, dtype=torch.float32, device=device)
    bs_a = torch.tensor(probe_a.gt_belief_states, dtype=torch.float32, device=device)
    act_b = torch.tensor(probe_b.activations, dtype=torch.float32, device=device)
    bs_b = torch.tensor(probe_b.gt_belief_states, dtype=torch.float32, device=device)

    with torch.no_grad():
        cross_mse_ab = F.mse_loss(probe_a.probe(act_b), bs_b).item()
        cross_mse_ba = F.mse_loss(probe_b.probe(act_a), bs_a).item()

    W_a = probe_a.probe.W.detach().cpu().numpy()
    W_b = probe_b.probe.W.detach().cpu().numpy()

    return {
        "test_mse_a": probe_a.test_mse,
        "test_mse_b": probe_b.test_mse,
        "cross_mse_ab": cross_mse_ab,
        "cross_mse_ba": cross_mse_ba,
        "column_cosine_sim": _column_cosine_similarity(W_a, W_b),
    }


def cross_mse_matrix(
    probes: list[ProbeResult],
    activations: list[np.ndarray],
    belief_states: list[np.ndarray],
) -> np.ndarray:
    """
    Compute an N×N matrix where entry [i, j] is the MSE of probe i
    evaluated on sequence j's activations and belief states.
    """
    n = len(probes)
    matrix = np.zeros((n, n))
    for i, pr in enumerate(probes):
        device = next(pr.probe.parameters()).device
        for j in range(n):
            act = torch.tensor(activations[j], dtype=torch.float32, device=device)
            bs = torch.tensor(belief_states[j], dtype=torch.float32, device=device)
            with torch.no_grad():
                matrix[i, j] = F.mse_loss(pr.probe(act), bs).item()
    return matrix


def pairwise_cosine_sim_matrix(probes: list[ProbeResult]) -> np.ndarray:
    """
    Compute an N×N matrix where entry [i, j] is the mean absolute diagonal
    cosine similarity between probe i and probe j's weight matrices.

    Entry [i, j] = mean(|diag(column_cosine_sim(W_i, W_j))|)

    A value near 1 indicates the two probes learned the same directions per
    belief component; near 0 indicates uncorrelated subspaces.
    """
    n = len(probes)
    matrix = np.zeros((n, n))
    for i in range(n):
        W_i = probes[i].probe.W.detach().cpu().numpy()
        for j in range(n):
            W_j = probes[j].probe.W.detach().cpu().numpy()
            cos_sim = _column_cosine_similarity(W_i, W_j)
            matrix[i, j] = float(np.abs(np.diag(cos_sim)).mean())
    return matrix
