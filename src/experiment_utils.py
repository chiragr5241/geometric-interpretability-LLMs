from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────── HuggingFace model wrapper ───────────────────────────


class WrappedHFModel:
    """HuggingFace causal LM wrapped to expose the TransformerLens API subset
    needed by the tuned-lens pipeline.

    Used for models not in the TransformerLens registry (e.g. Qwen3.5 VLMs).
    Only implements the slice of TL's interface that run_pipeline,
    train_tuned_lens, and extract_eval_weights actually call.
    """

    def __init__(self, hf_model, tokenizer, device: torch.device) -> None:
        self._hf_model = hf_model
        self._tokenizer = tokenizer
        self._device = device

        cfg = hf_model.config
        self.cfg = SimpleNamespace(
            n_layers=cfg.num_hidden_layers,
            d_model=cfg.hidden_size,
            device=str(device),
            n_ctx=getattr(cfg, "max_position_embeddings", 131072),
        )

        # TL convention: W_U [d_model, vocab_size].
        # HF convention: lm_head.weight [vocab_size, d_model].
        # We store a .T view so it stays synced with the model's device.
        lm_head = hf_model.lm_head
        self.unembed = SimpleNamespace(
            W_U=lm_head.weight.T.detach(),
            b_U=(lm_head.bias.detach()
                 if lm_head.bias is not None
                 else torch.zeros(lm_head.weight.shape[0],
                                  device=lm_head.weight.device,
                                  dtype=lm_head.weight.dtype)),
        )

        # Final layer norm (RMSNorm for Qwen/Llama family).
        self.ln_final = hf_model.model.norm

    def eval(self) -> "WrappedHFModel":
        self._hf_model.eval()
        return self

    def cpu(self) -> "WrappedHFModel":
        """Move backbone to CPU; update cfg.device so downstream code is consistent."""
        self._hf_model.cpu()
        self._device = torch.device("cpu")
        self.cfg.device = "cpu"
        return self

    def to_tokens(self, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
        """Tokenize -> [1, seq_len] int64 tensor on model device."""
        ids = self._tokenizer.encode(prompt, add_special_tokens=prepend_bos)
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def to_str_tokens(self, prompt: str, prepend_bos: bool = True) -> list[str]:
        ids = self._tokenizer.encode(prompt, add_special_tokens=prepend_bos)
        return self._tokenizer.convert_ids_to_tokens(ids)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(out_dir: Path, name: str = "exp") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def _extend_model_context(model, n_ctx: int) -> None:
    """Extend a loaded HookedTransformer to support longer sequences.

    Patches model.cfg.n_ctx and recomputes the registered buffers that are
    pre-allocated at load time based on the original n_ctx:
      - rotary_sin / rotary_cos  (one pair per attention layer, RoPE models)
      - mask                     (causal mask, one per attention layer)
    """
    model.cfg.n_ctx = n_ctx
    for block in model.blocks:
        attn = block.attn
        device = attn.IGNORE.device

        if hasattr(attn, "rotary_sin"):
            sin, cos = attn.calculate_sin_cos_rotary(
                model.cfg.rotary_dim,
                n_ctx,
                base=model.cfg.rotary_base,
                dtype=model.cfg.dtype,
            )
            attn.rotary_sin = sin.to(device)
            attn.rotary_cos = cos.to(device)

        if hasattr(attn, "mask"):
            causal_mask = torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool, device=device))
            window_size = getattr(model.cfg, "window_size", None)
            if window_size is not None:
                causal_mask = torch.triu(causal_mask, 1 - window_size)
            attn.mask = causal_mask


def load_model(
    model_name: str,
    device: torch.device,
    logger: logging.Logger,
    n_ctx: int | None = None,
):
    """Load a model for the tuned-lens pipeline.

    Tries TransformerLens first. Falls back to a WrappedHFModel for models
    not in the TL registry (e.g. Qwen3.5 VLMs).
    """
    import transformers
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        from huggingface_hub.constants import HF_HUB_CACHE
        transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE
    from transformer_lens import HookedTransformer

    dtype = torch.float32 if device.type == "cpu" else torch.float16
    n_devices = min(torch.cuda.device_count(), 4) if device.type == "cuda" else 1
    logger.info(f"Loading '{model_name}' on {device} with dtype={dtype}, n_devices={n_devices} ...")
    t0 = time.time()
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            dtype=dtype,
            device=str(device),
            n_devices=n_devices,
        )
        if n_ctx is not None:
            _extend_model_context(model, n_ctx)
        model.eval()
        elapsed = time.time() - t0
        logger.info(
            f"Model loaded via TransformerLens in {elapsed:.1f}s  |  "
            f"layers={model.cfg.n_layers}  d_model={model.cfg.d_model}  n_ctx={model.cfg.n_ctx}"
        )
        return model
    except ValueError as exc:
        if "not found" not in str(exc):
            raise
        logger.warning(
            f"TransformerLens does not support {model_name!r} — falling back to "
            f"HuggingFace AutoModelForCausalLM. (n_ctx override ignored.)"
        )
        return _load_model_hf(model_name, device, logger)


def _load_model_hf(
    model_name: str,
    device: torch.device,
    logger: logging.Logger,
) -> WrappedHFModel:
    """Load a model not supported by TransformerLens via HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer for {model_name!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading {model_name!r} via AutoModelForCausalLM ...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    )
    hf_model.eval()
    elapsed = time.time() - t0

    wrapper = WrappedHFModel(hf_model, tokenizer, device)
    logger.info(
        f"Model loaded via HuggingFace in {elapsed:.1f}s  |  "
        f"layers={wrapper.cfg.n_layers}  d_model={wrapper.cfg.d_model}"
    )
    return wrapper


def get_concept_token_ids(model, concepts: list[str]) -> dict[str, int]:
    """Get the LLM token ID for each concept (space-prefixed).

    Works with both HookedTransformer and WrappedHFModel.

    Parameters
    ----------
    model
        HookedTransformer or WrappedHFModel.
    concepts : list[str]
        HMM symbol names, e.g. ``["A", "B", "C"]``.

    Returns
    -------
    dict mapping concept name → LLM token ID.
    """
    concept_to_id: dict[str, int] = {}
    for concept in concepts:
        spaced = f" {concept}"
        if isinstance(model, WrappedHFModel):
            ids = model._tokenizer.encode(spaced, add_special_tokens=False)
        else:
            ids = model.to_tokens(spaced, prepend_bos=False)[0].tolist()
        concept_to_id[concept] = int(ids[-1])
    return concept_to_id


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination) score."""
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-10))


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

    first_text = ordered[0]
    first_ids = model.to_tokens(first_text, prepend_bos=False)[0].tolist()
    assert len(first_ids) == 1, (
        f"First HMM token '{first_text}' tokenises to >1 LLM tokens: {first_ids}"
    )

    mid_ids: list[int] = []
    for token_str in ordered:
        spaced = " " + token_str
        tids = model.to_tokens(spaced, prepend_bos=False)[0].tolist()
        assert len(tids) == 1, (
            f"Space-prefixed HMM token {spaced!r} tokenises to >1 LLM tokens: {tids}"
        )
        mid_ids.append(tids[0])

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


def get_model_probs_projected(
    logits: torch.Tensor,
    first_tok_id: int,
    mid_tok_ids: list[int],
    seq_len: int,
) -> np.ndarray:
    """
    Project next-token probabilities onto a (n_hmm_tokens + 1)-simplex.

    Returns (seq_len, n_hmm_tokens + 1) where the last column is junk mass
    (all probability not assigned to any emission token).  Probabilities are
    NOT renormalised — they sum to 1 with junk absorbing the remainder.

    logits shape: (1, llm_seq_len, vocab_size)

    Position 0: uses first_tok_id for emission 0 (no leading space).
    Positions 1+: uses mid_tok_ids (space-prefixed) for all emissions.
    """
    n = len(mid_tok_ids)
    with torch.no_grad():
        probs_full = F.softmax(logits[0, :seq_len, :].float(), dim=-1)   # (seq_len, vocab_size)

    out = np.empty((seq_len, n + 1), dtype=np.float32)

    emission_ids_pos0 = list(mid_tok_ids)
    if first_tok_id not in mid_tok_ids:
        emission_ids_pos0[0] = first_tok_id
    p0 = probs_full[0, emission_ids_pos0].cpu().numpy().copy()
    out[0, :n] = p0
    out[0, n] = max(0.0, 1.0 - float(p0.sum()))

    if seq_len > 1:
        mid = probs_full[1:seq_len, :][:, mid_tok_ids].cpu().numpy()
        out[1:, :n] = mid
        out[1:, n] = np.clip(1.0 - mid.sum(axis=-1), 0.0, None)

    return out
