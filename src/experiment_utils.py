from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(out_dir: Path, name: str = "exp") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


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


def build_emission_matrix(hmm) -> np.ndarray:
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

    p0 = probs_full[0, mid_tok_ids].cpu().numpy().copy()
    if first_tok_id not in mid_tok_ids:
        p0[0] = probs_full[0, first_tok_id].item()
    out[0] = p0 / (p0.sum() + 1e-10)

    if seq_len > 1:
        mid = probs_full[1:seq_len, :][:, mid_tok_ids].cpu().numpy()
        out[1:] = mid / (mid.sum(axis=-1, keepdims=True) + 1e-10)

    return out
