from __future__ import annotations

import logging
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
    include_junk: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position KL divergence and its smoothed version.

    Both modes compute KL(optimal ‖ model).

    Standard mode (include_junk=False):
        Both arrays are (seq_len, n_emissions). Computes KL(optimal || model).

    Projected mode (include_junk=True):
        model_probs is (seq_len, n_emissions + 1) — last column is junk mass.
        optimal_probs is (seq_len, n_emissions) — ground-truth over emissions only.
        Computes KL(optimal || model_projected) where model_projected is the
        (n_emissions + 1)-vector from model_probs and the augmented ground truth
        has a zero junk entry.  The junk term drops out of the sum since
        optimal_gt[junk] = 0, so the result equals
            sum_e optimal[e] * log(optimal[e] / model[e])
        which naturally penalises junk mass through reduced model[e] probability.

    Returns (kl_raw, kl_smooth), both of shape (seq_len,).
    """
    if include_junk:
        p = optimal_probs                                        # (seq_len, n_emissions)
        q = model_probs[:, : optimal_probs.shape[-1]]           # emission columns only
        kl_raw = np.sum(
            np.where(
                p > 0,
                p * np.log(p / np.clip(q, 1e-10, None)),
                0.0,
            ),
            axis=-1,
        )
    else:
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
    include_junk: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[int, bool]:
    """
    Returns (t*, crossed) where t* is the first position (>= min_position) where
    the smoothed KL drops into the bottom `fraction` of its (max - min) range
    computed over [min_position:], and crossed=True.
    Falls back to the argmin of the smoothed KL (within [min_position:]) with
    crossed=False.  The returned index is always absolute (into the full sequence).

    model_probs:   (seq_len, vocab_size) or (seq_len, n_emissions+1) when include_junk=True
    optimal_probs: (seq_len, vocab_size) — Bayesian-optimal next-token probabilities
    fraction:      threshold = min + fraction * (max - min); e.g. 0.2 means
                   "first time KL is within the bottom 20% of its total swing"
    smooth_window: rolling-mean window size applied before thresholding
    min_position:  ignore positions before this index when detecting the threshold
    include_junk:  if True, use projected KL(optimal || model_projected); see compute_kl
    logger:        optional logger; if provided, logs the KL variant and junk mass stats
    """
    variant = "projected (include_junk=True)" if include_junk else "standard (include_junk=False)"
    if logger is not None:
        logger.info(f"  compute_kl variant : {variant}")
        if include_junk:
            junk_col = model_probs[:, -1]
            logger.info(
                f"  junk mass          : mean={junk_col.mean():.4f}"
                f"  std={junk_col.std():.4f}"
                f"  max={junk_col.max():.4f}"
            )

    _, kl_smooth = compute_kl(model_probs, optimal_probs, smooth_window, include_junk=include_junk)

    kl_min, kl_max = kl_smooth.min(), kl_smooth.max()
    threshold = kl_min + fraction * (kl_max - kl_min)

    kl_search = kl_smooth[min_position:]
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


def compute_activation_covariance(
    activations_list: list[np.ndarray],
    shrinkage: float = 0.1,
) -> np.ndarray:
    """
    Pool activations and compute a Ledoit-Wolf-style shrinkage-regularized
    covariance matrix.

    activations_list: list of (seq_len, d_model) float32 arrays
    shrinkage: alpha in (1-alpha)*Sigma + alpha*(tr(Sigma)/d)*I

    Returns: (d_model, d_model) float64 regularized covariance
    """
    pooled = np.concatenate(activations_list, axis=0).astype(np.float64)
    sigma = np.cov(pooled.T, ddof=1)
    d = sigma.shape[0]
    target = (np.trace(sigma) / d) * np.eye(d)
    return (1.0 - shrinkage) * sigma + shrinkage * target


def pairwise_principal_angles(
    probes: list[ProbeResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal angles between probe weight subspaces.

    For each probe, QR-decomposes W ∈ R^(d_model, n_states) to obtain an
    orthonormal basis Q ∈ R^(d_model, n_states). For each pair (i, j) the
    singular values of Q_i^T Q_j equal cos(theta_k) for the n_states
    principal angles, returned in ascending order (theta_1 ≤ theta_2 ≤ ...).

    Returns:
        angles_deg:          (N, N, n_states) principal angles in degrees
        mean_angle_matrix:   (N, N) mean of all n_states principal angles per pair
        mean_angle_2_matrix: (N, N) mean of the first two (smallest) principal angles
    """
    n = len(probes)
    n_states = probes[0].probe.W.shape[1]

    bases: list[np.ndarray] = []
    for pr in probes:
        W = pr.probe.W.detach().cpu().numpy()
        Q, _ = np.linalg.qr(W)
        bases.append(Q[:, :n_states])

    angles_deg = np.zeros((n, n, n_states))
    mean_angle_matrix = np.zeros((n, n))
    mean_angle_2_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M = bases[i].T @ bases[j]
            sv = np.linalg.svd(M, compute_uv=False)
            sv = np.clip(sv, -1.0, 1.0)
            angles = np.sort(np.degrees(np.arccos(sv)))
            angles_deg[i, j] = angles
            mean_angle_matrix[i, j] = float(angles.mean())
            mean_angle_2_matrix[i, j] = float(angles[:2].mean())

    return angles_deg, mean_angle_matrix, mean_angle_2_matrix


def pairwise_mahalanobis_cosine_sim(
    probes: list[ProbeResult],
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between probe weight directions in the
    Mahalanobis (whitened) space defined by sigma.

    Whitens each probe's weight matrix as W' = Sigma^{-1/2} W, then computes
    the same mean |diag| cosine similarity as pairwise_cosine_sim_matrix.

    sigma: (d_model, d_model) regularized activation covariance (float64)
    Returns: (N, N) matrix
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 1e-10, None)
    sigma_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

    n = len(probes)
    whitened: list[np.ndarray] = []
    for pr in probes:
        W = pr.probe.W.detach().cpu().numpy().astype(np.float64)
        whitened.append(sigma_inv_sqrt @ W)

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_sim = _column_cosine_similarity(whitened[i], whitened[j])
            matrix[i, j] = float(np.abs(np.diag(cos_sim)).mean())
    return matrix
