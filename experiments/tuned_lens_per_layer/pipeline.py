"""Core pipeline: forward pass, tuned lens training, and evaluation.

This pipeline trains up to three tuned-lens variants per run (full-vocab,
concept-only, HMM-target) plus the raw logit lens, and evaluates all four
on a single shared concept-only KL/NLL/top-1 schema. Memory-optimized for
long sequences and large models on A40 (48 GB):

  * KV-cached chunked forward pass (avoids materializing the full sequence
    at once and keeps activations only at concept-token positions).
  * No full-vocab logits tensor: target log-probs for the canonical
    full-vocabulary tuned lens are computed per batch from cached final-layer
    residuals (``unembed @ ln_final(h)``), eliminating the ~24 GB target tensor.
  * GPU pseudoinverse OLS for the belief-state R² probe (replaces sklearn's
    CPU LinearRegression; ~10-100x faster).
  * fp32 throughout the lens-application path; ``_kl`` / ``_nll`` defensively
    sanitise NaN/±inf so a numerical pathology can never silently emit
    ``Infinity`` into the metrics.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data_generation import generate_hmm_sequences
from experiment_utils import WrappedHFModel, get_concept_token_ids

from .config import TunedLensConfig
from .evaluation import LayerMetrics, compute_layer_metrics
from .tuned_lens import (
    EvalWeights,
    apply_logit_lens,
    apply_tuned_lens,
    extract_eval_weights,
    load_translator,
    save_translators,
    train_tuned_lens,
    train_tuned_lens_concept,
)
from . import plotting

logger = logging.getLogger(__name__)


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, s = divmod(s, 60)
        return f"{int(m)}m{s:04.1f}s"
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


@contextmanager
def _timed(stage: str):
    """Context manager that logs wall-clock time for a stage."""
    logger.info(f"[start] {stage}")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        logger.info(f"[done ] {stage}  ({_fmt_secs(dt)})")


# ─────────────────────────── forward pass ────────────────────────────────────


def _forward_pass(
    model,
    walk_concepts: list[list[str]],
    vocab_tokens: list[str],
    layer_indices: list[int],
    n_sequences: int,
    chunk_size: int = 2048,
    concept_ids: list[int] | None = None,
) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    """KV-cached chunked forward storing activations only at concept positions.

    Dispatches to the HF path for WrappedHFModel instances; uses the
    TransformerLens hook API otherwise.
    """
    if isinstance(model, WrappedHFModel):
        if concept_ids is None:
            raise ValueError("concept_ids is required for the HuggingFace forward path")
        return _forward_pass_hf(
            model, walk_concepts, vocab_tokens, layer_indices,
            n_sequences, chunk_size, concept_ids,
        )
    return _forward_pass_tl(
        model, walk_concepts, vocab_tokens, layer_indices, n_sequences, chunk_size,
    )


def _forward_pass_tl(
    model,
    walk_concepts: list[list[str]],
    vocab_tokens: list[str],
    layer_indices: list[int],
    n_sequences: int,
    chunk_size: int,
) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    """TransformerLens KV-cached forward pass."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    letter_set = set(vocab_tokens)
    device = model.cfg.device
    seq_activations: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}
    n_concepts_per_seq: list[int] = []
    hook_names = {l: f"blocks.{l}.hook_resid_post" for l in layer_indices}
    seq_times: list[float] = []

    for seq_idx in tqdm(range(n_sequences), desc="Forward pass (TL KV-cached)"):
        seq_t0 = time.time()
        seq_concepts = walk_concepts[seq_idx]
        prompt = seq_concepts[0] + " " + " ".join(seq_concepts[1:])
        input_ids = model.to_tokens(prompt, prepend_bos=True).to(device)
        str_tokens = model.to_str_tokens(prompt, prepend_bos=True)

        positions = np.array(
            [i for i, tok in enumerate(str_tokens) if tok.strip() in letter_set],
            dtype=np.int64,
        )
        n_use = min(len(positions), len(seq_concepts))
        positions = positions[:n_use]
        n_concepts_per_seq.append(int(n_use))

        seq_len = input_ids.shape[1]
        kv_cache = HookedTransformerKeyValueCache.init_cache(
            model.cfg, device, batch_size=1
        )
        chunk_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_tokens = input_ids[:, chunk_start:chunk_end]
            chunk_pos_global = np.arange(chunk_start, chunk_end)
            local_idx = np.where(np.isin(chunk_pos_global, positions))[0]
            need_capture = local_idx.size > 0

            captured: dict[int, torch.Tensor] = {}
            fwd_hooks: list = []
            if need_capture:
                local_idx_t = torch.from_numpy(local_idx).to(device=device, dtype=torch.long)

                def make_hook(li: int, idx_t: torch.Tensor):
                    def hook_fn(activation, hook):
                        captured[li] = activation[0].index_select(0, idx_t).detach()
                    return hook_fn

                for l in layer_indices:
                    fwd_hooks.append((hook_names[l], make_hook(l, local_idx_t)))

            with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
                _ = model(
                    chunk_tokens,
                    past_kv_cache=kv_cache,
                    return_type=None,
                    stop_at_layer=model.cfg.n_layers,
                )

            if need_capture:
                for l in layer_indices:
                    chunk_layer_acts[l].append(captured[l])
            del captured

        del kv_cache
        torch.cuda.empty_cache()

        d_model = model.cfg.d_model
        for l in layer_indices:
            if chunk_layer_acts[l]:
                merged = torch.cat(chunk_layer_acts[l], dim=0)
                seq_activations[l].append(merged.to(torch.float32).cpu().numpy())
            else:
                seq_activations[l].append(np.zeros((0, d_model), dtype=np.float32))
            chunk_layer_acts[l].clear()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seq_dt = time.time() - seq_t0
        seq_times.append(seq_dt)
        if seq_idx == 0 or (seq_idx + 1) % max(1, n_sequences // 10) == 0:
            avg = sum(seq_times) / len(seq_times)
            remaining = avg * (n_sequences - seq_idx - 1)
            logger.info(
                f"  seq {seq_idx + 1}/{n_sequences}: {_fmt_secs(seq_dt)} "
                f"(avg {_fmt_secs(avg)}, ETA {_fmt_secs(remaining)})"
            )

    return seq_activations, n_concepts_per_seq


def _forward_pass_hf(
    model: WrappedHFModel,
    walk_concepts: list[list[str]],
    vocab_tokens: list[str],
    layer_indices: list[int],
    n_sequences: int,
    chunk_size: int,
    concept_ids: list[int],
) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    """HuggingFace-native forward pass for models not supported by TransformerLens.

    Uses the same pattern as experiments/notebooks/r2_qwen35_9b.ipynb:
      - register_forward_hook on hf_model.model.layers[l]
      - out[0] gives the post-layer hidden states
      - Native past_key_values KV caching across chunks

    Prompt format: " A B C ..." (leading space so all tokens are space-prefixed
    and their IDs match the space-prefixed concept_ids from get_concept_token_ids).
    """
    hf_model = model._hf_model
    tokenizer = model._tokenizer
    device = model._device

    # Build a set of token IDs to match. Include both space-prefixed (mid-sequence)
    # and non-space-prefixed (possible first token) variants to be safe.
    _nospace_ids = {
        tokenizer.encode(c, add_special_tokens=False)[-1] for c in vocab_tokens
    }
    concept_id_set = set(concept_ids) | _nospace_ids

    seq_activations: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}
    n_concepts_per_seq: list[int] = []
    seq_times: list[float] = []

    for seq_idx in tqdm(range(n_sequences), desc="Forward pass (HF KV-cached)"):
        seq_t0 = time.time()
        seq_concepts = walk_concepts[seq_idx]

        # Leading space ensures all concept tokens are space-prefixed in the tokenizer.
        prompt = " " + " ".join(seq_concepts)
        input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

        ids_np = np.array(input_ids_list, dtype=np.int64)
        positions = np.where(np.isin(ids_np, list(concept_id_set)))[0].astype(np.int64)
        n_use = min(len(positions), len(seq_concepts))
        positions = positions[:n_use]
        n_concepts_per_seq.append(int(n_use))

        seq_len = input_ids.shape[1]
        past_kv = None
        chunk_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_tokens = input_ids[:, chunk_start:chunk_end]
            chunk_pos_global = np.arange(chunk_start, chunk_end)
            local_idx = np.where(np.isin(chunk_pos_global, positions))[0]
            need_capture = local_idx.size > 0

            captured: dict[int, torch.Tensor] = {}
            hooks: list = []

            if need_capture:
                local_idx_t = torch.from_numpy(local_idx).to(device=device, dtype=torch.long)

                def make_hook(li: int, idx_t: torch.Tensor):
                    def hook_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        captured[li] = h[0].index_select(0, idx_t).detach()
                    return hook_fn

                for l in layer_indices:
                    hooks.append(
                        hf_model.model.layers[l].register_forward_hook(
                            make_hook(l, local_idx_t)
                        )
                    )

            with torch.no_grad():
                out = hf_model.model(
                    chunk_tokens,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            for h in hooks:
                h.remove()

            if need_capture:
                for l in layer_indices:
                    if l in captured:
                        chunk_layer_acts[l].append(captured[l])
            del captured

            past_kv = out.past_key_values
            del out
            torch.cuda.empty_cache()

        del past_kv
        torch.cuda.empty_cache()

        d_model = model.cfg.d_model
        for l in layer_indices:
            if chunk_layer_acts[l]:
                merged = torch.cat(chunk_layer_acts[l], dim=0)
                seq_activations[l].append(merged.to(torch.float32).cpu().numpy())
            else:
                seq_activations[l].append(np.zeros((0, d_model), dtype=np.float32))
            chunk_layer_acts[l].clear()
        torch.cuda.empty_cache()

        seq_dt = time.time() - seq_t0
        seq_times.append(seq_dt)
        if seq_idx == 0 or (seq_idx + 1) % max(1, n_sequences // 10) == 0:
            avg = sum(seq_times) / len(seq_times)
            remaining = avg * (n_sequences - seq_idx - 1)
            logger.info(
                f"  seq {seq_idx + 1}/{n_sequences}: {_fmt_secs(seq_dt)} "
                f"(avg {_fmt_secs(avg)}, ETA {_fmt_secs(remaining)})"
            )

    return seq_activations, n_concepts_per_seq


# ────────────────────────────── helpers ──────────────────────────────────────


def _final_concept_probs(
    final_resid: np.ndarray,
    eval_weights: EvalWeights,
    batch_size: int = 1024,
) -> np.ndarray:
    """Apply ln_final + concept-only unembed + softmax (fp32) to cached residuals."""
    device = eval_weights.device
    W_c = eval_weights.W_c.to(device)
    b_c = eval_weights.b_c.to(device)
    ln_final = eval_weights.ln_final.to(device)
    h_t = torch.from_numpy(final_resid).to(torch.float32)
    out = []
    with torch.no_grad():
        for s in range(0, h_t.shape[0], batch_size):
            h = h_t[s:s + batch_size].to(device)
            normed = ln_final(h).to(torch.float32)
            logits_c = normed @ W_c + b_c
            out.append(F.softmax(logits_c, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _final_concept_logits(
    final_resid: np.ndarray,
    eval_weights: EvalWeights,
    batch_size: int = 1024,
) -> np.ndarray:
    """Concept-only logits (no softmax) from cached final residuals."""
    device = eval_weights.device
    W_c = eval_weights.W_c.to(device)
    b_c = eval_weights.b_c.to(device)
    ln_final = eval_weights.ln_final.to(device)
    h_t = torch.from_numpy(final_resid).to(torch.float32)
    out = []
    with torch.no_grad():
        for s in range(0, h_t.shape[0], batch_size):
            h = h_t[s:s + batch_size].to(device)
            normed = ln_final(h).to(torch.float32)
            out.append((normed @ W_c + b_c).cpu().numpy())
    return np.concatenate(out, axis=0)


def _r2_belief_probe_gpu(
    train_activations: dict[int, np.ndarray],
    test_activations: dict[int, np.ndarray],
    train_beliefs: np.ndarray,
    test_beliefs: np.ndarray,
    layer_indices: list[int],
    n_test_sequences: int,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """Belief-state OLS probe via torch.linalg.pinv on GPU.

    Returns (r2_per_layer, r2_per_seq_per_layer) where r2_per_seq_per_layer[L]
    is a (n_test_sequences,) array of per-sequence R² values used for CI bands.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y_tr = torch.from_numpy(train_beliefs).to(device=device, dtype=torch.float32)
    Y_te = torch.from_numpy(test_beliefs).to(device=device, dtype=torch.float32)
    Y_te_centered = Y_te - Y_te.mean(dim=0, keepdim=True)
    ss_tot = (Y_te_centered ** 2).sum().item()

    n_flat, belief_dim = Y_te.shape
    seq_len = n_flat // n_test_sequences
    Y_te_seq = Y_te.reshape(n_test_sequences, seq_len, belief_dim)

    r2: dict[int, float] = {}
    r2_per_seq: dict[int, np.ndarray] = {}
    for layer in layer_indices:
        X_tr = torch.from_numpy(train_activations[layer]).to(device=device, dtype=torch.float32)
        X_te = torch.from_numpy(test_activations[layer]).to(device=device, dtype=torch.float32)
        ones_tr = torch.ones(X_tr.shape[0], 1, device=device, dtype=X_tr.dtype)
        ones_te = torch.ones(X_te.shape[0], 1, device=device, dtype=X_te.dtype)
        X_tr_b = torch.cat([X_tr, ones_tr], dim=1)
        X_te_b = torch.cat([X_te, ones_te], dim=1)
        W = torch.linalg.pinv(X_tr_b) @ Y_tr
        pred = X_te_b @ W
        ss_res = ((pred - Y_te) ** 2).sum().item()
        r2[layer] = float(1.0 - ss_res / (ss_tot + 1e-10))

        # Per-sequence R²: reshape residuals to (n_seq, seq_len, belief_dim)
        pred_seq = pred.reshape(n_test_sequences, seq_len, belief_dim)
        Y_te_seq_mean = Y_te_seq.mean(dim=1, keepdim=True)
        ss_res_seq = ((pred_seq - Y_te_seq) ** 2).sum(dim=(1, 2))
        ss_tot_seq = ((Y_te_seq - Y_te_seq_mean) ** 2).sum(dim=(1, 2))
        r2_seq = (1.0 - ss_res_seq / (ss_tot_seq + 1e-10)).cpu().numpy()
        r2_per_seq[layer] = r2_seq

        del X_tr, X_te, X_tr_b, X_te_b, W, pred, pred_seq, ss_res_seq, ss_tot_seq, r2_seq
        torch.cuda.empty_cache()

    del Y_tr, Y_te, Y_te_centered, Y_te_seq
    torch.cuda.empty_cache()
    return r2, r2_per_seq


# ─────────────────────────────── pipeline ───────────────────────────────────


def run_pipeline(
    model,
    config: TunedLensConfig,
    output_dir: Path,
    release_backbone: bool = True,
) -> dict:
    """Run the full tuned lens per-layer experiment.

    Parameters
    ----------
    model
        HookedTransformer on GPU (must still be on GPU at call time).
    config
        Experiment configuration.
    output_dir
        Where to write all artifacts.
    release_backbone
        If True (default), move the model to CPU after training completes
        and before evaluation begins, freeing ~18 GB of GPU memory.
        Set to False when running multiple configs in a sweep so the model
        stays on GPU for the next config's forward pass.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    pipeline_t0 = time.time()
    logger.info(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Resolve which lenses we'll train this run
    train_tuned_full = config.train_tuned_full
    train_tuned_concept = config.train_tuned_concept
    train_tuned_hmm = config.train_tuned_hmm
    if config.train_hmm_target is not None:  # backwards-compat
        train_tuned_hmm = bool(config.train_hmm_target)
    logger.info(
        f"Lens variants: logit (always), "
        f"tuned_full={train_tuned_full}, tuned_concept={train_tuned_concept}, "
        f"tuned_hmm={train_tuned_hmm}"
    )

    # ── 1. HMM data ────────────────────────────────────────────────────────
    with _timed("HMM sequence generation"):
        hmm_data = generate_hmm_sequences(
            process_name=config.process_name,
            process_params=config.process_params,
            n_sequences=config.n_sequences,
            seq_length=config.seq_length,
            random_seed=config.random_seed,
        )
    tokens = torch.from_numpy(hmm_data.tokens)
    belief_states_all = hmm_data.belief_states
    obs_probs_all = hmm_data.obs_probs

    walk_concepts = [[config.vocab_tokens[int(t)] for t in seq] for seq in tokens]
    logger.info(f"Generated {config.n_sequences} sequences of length {config.seq_length}")

    # ── 2. Token IDs + eval weights ────────────────────────────────────────
    concept_to_id = get_concept_token_ids(model, config.vocab_tokens)
    concept_ids = [concept_to_id[c] for c in config.vocab_tokens]
    logger.info(f"Concept token IDs: {dict(zip(config.vocab_tokens, concept_ids))}")

    # Extract ln_final + concept-column weights while model is still on GPU.
    # These tiny tensors (~KB) replace the backbone (18 GB) for all evaluation
    # steps, so the backbone can be freed after training.
    eval_weights = extract_eval_weights(model, concept_ids)
    logger.info(
        f"Eval weights extracted: W_c{list(eval_weights.W_c.shape)}, "
        f"device={eval_weights.device}"
    )

    # ── 3. Forward pass ────────────────────────────────────────────────────
    final_layer_idx = config.layer_indices[-1]
    if final_layer_idx != model.cfg.n_layers - 1:
        logger.warning(
            f"layer_indices[-1]={final_layer_idx} != n_layers-1={model.cfg.n_layers - 1}; "
            f"the canonical full-vocab tuned lens target will use this earlier "
            f"layer's residual rather than the model's true final residual."
        )
    logger.info(
        f"Running KV-cached chunked forward pass "
        f"(chunk_size={config.forward_chunk_size})..."
    )
    with _timed(f"forward pass ({config.n_sequences} seqs × {config.seq_length} concepts)"):
        seq_activations, n_concepts_list = _forward_pass(
            model, walk_concepts, config.vocab_tokens,
            config.layer_indices, config.n_sequences,
            chunk_size=config.forward_chunk_size,
            concept_ids=concept_ids,
        )

    seq_len_actual = n_concepts_list[0]
    if any(n != seq_len_actual for n in n_concepts_list):
        raise RuntimeError(
            f"Inconsistent concept-position counts across sequences: {set(n_concepts_list)}"
        )
    if seq_len_actual != config.seq_length:
        logger.warning(f"Actual seq_length={seq_len_actual}, expected {config.seq_length}")

    # ── 4. Train/test split ────────────────────────────────────────────────
    n_train = config.n_train_sequences
    n_test = config.n_sequences - n_train
    logger.info(f"Train/test split: {n_train} train, {n_test} test sequences")

    train_window = config.train_pos_window
    if train_window is not None:
        w_start, w_end = int(train_window[0]), int(train_window[1])
        if w_end > seq_len_actual or w_start < 0 or w_start >= w_end:
            raise ValueError(
                f"train_pos_window {train_window} invalid for seq_length {seq_len_actual}"
            )
        logger.info(f"Restricting TRAINING positions to [{w_start}, {w_end})")

    def concat_seqs(seq_list, start, end, window=None):
        sliced = seq_list[start:end]
        if window is not None:
            ws, we = window
            sliced = [s[ws:we] for s in sliced]
        return np.concatenate(sliced, axis=0)

    train_activations: dict[int, np.ndarray] = {}
    test_activations: dict[int, np.ndarray] = {}
    for layer in config.layer_indices:
        arrs = seq_activations[layer]
        train_activations[layer] = concat_seqs(arrs, 0, n_train, window=train_window)
        test_activations[layer] = concat_seqs(arrs, n_train, config.n_sequences)
    del seq_activations

    train_final_resid = train_activations[final_layer_idx]
    test_final_resid = test_activations[final_layer_idx]

    # HMM probs / next tokens for evaluation (test set)
    test_obs_probs_flat = obs_probs_all[n_train:, :seq_len_actual, :].reshape(
        -1, obs_probs_all.shape[-1]
    )
    test_tokens = hmm_data.tokens[n_train:, :seq_len_actual]
    test_next_tokens_full = np.zeros((n_test, seq_len_actual), dtype=np.int64)
    test_next_tokens_full[:, :-1] = test_tokens[:, 1:]
    test_next_tokens_full[:, -1] = 0
    test_next_tokens_flat = test_next_tokens_full.reshape(-1)

    n_train_pos = train_activations[final_layer_idx].shape[0]
    n_test_pos = test_activations[final_layer_idx].shape[0]
    logger.info(f"Train positions: {n_train_pos}, Test positions: {n_test_pos}")

    # ── 5. Train tuned lenses ──────────────────────────────────────────────
    # Each variant is saved to disk immediately after training and removed
    # from CPU RAM (~64 MB/layer × 28 layers per variant = ~1.8 GB freed per
    # variant). Translators are reloaded from disk one layer at a time during
    # evaluation, keeping peak CPU RAM near ~64 MB regardless of layer count.
    optimizer_name = config.tuned_lens_optimizer
    loss_curves_by_lens: dict[str, dict] = {}
    trained_lens_names: list[str] = []

    if train_tuned_full:
        with _timed(f"train tuned_full ({len(config.layer_indices)} layers × {config.tuned_lens_epochs} epochs)"):
            tr, lc = train_tuned_lens(
                activations_by_layer=train_activations,
                model=model,
                layers=config.layer_indices,
                target_final_resid=train_final_resid,
                n_epochs=config.tuned_lens_epochs,
                lr=config.tuned_lens_lr,
                batch_size=config.tuned_lens_batch_size,
                optimizer_name=optimizer_name,
                use_bf16=config.use_bf16,
            )
        save_translators(tr, output_dir / "translators_tuned_full")
        logger.info(f"  Saved tuned_full translators ({len(tr)} layers)")
        loss_curves_by_lens["tuned_full"] = lc
        trained_lens_names.append("tuned_full")
        del tr
        torch.cuda.empty_cache()

    if train_tuned_concept:
        train_concept_logits = _final_concept_logits(train_final_resid, eval_weights)
        with _timed(f"train tuned_concept ({len(config.layer_indices)} layers × {config.tuned_lens_epochs} epochs)"):
            tr, lc = train_tuned_lens_concept(
                activations_by_layer=train_activations,
                model=model,
                concept_ids=concept_ids,
                layers=config.layer_indices,
                target_concept_values=train_concept_logits,
                target_is_probs=False,
                n_epochs=config.tuned_lens_epochs,
                lr=config.tuned_lens_lr,
                batch_size=config.tuned_lens_batch_size,
                optimizer_name=optimizer_name,
                use_bf16=config.use_bf16,
            )
        save_translators(tr, output_dir / "translators_tuned_concept")
        logger.info(f"  Saved tuned_concept translators ({len(tr)} layers)")
        loss_curves_by_lens["tuned_concept"] = lc
        trained_lens_names.append("tuned_concept")
        del tr
        torch.cuda.empty_cache()

    if train_tuned_hmm:
        if train_window is not None:
            train_obs_probs = obs_probs_all[:n_train, w_start:w_end, :]
        else:
            train_obs_probs = obs_probs_all[:n_train, :seq_len_actual, :]
        train_obs_probs_flat = train_obs_probs.reshape(-1, train_obs_probs.shape[-1])
        with _timed(f"train tuned_hmm ({len(config.layer_indices)} layers × {config.tuned_lens_epochs} epochs)"):
            tr, lc = train_tuned_lens_concept(
                activations_by_layer=train_activations,
                model=model,
                concept_ids=concept_ids,
                layers=config.layer_indices,
                target_concept_values=train_obs_probs_flat,
                target_is_probs=True,
                n_epochs=config.tuned_lens_epochs,
                lr=config.tuned_lens_lr,
                batch_size=config.tuned_lens_batch_size,
                optimizer_name=optimizer_name,
                use_bf16=config.use_bf16,
            )
        save_translators(tr, output_dir / "translators_tuned_hmm")
        logger.info(f"  Saved tuned_hmm translators ({len(tr)} layers)")
        loss_curves_by_lens["tuned_hmm"] = lc
        trained_lens_names.append("tuned_hmm")
        del tr
        torch.cuda.empty_cache()

    for lens_name, lc in loss_curves_by_lens.items():
        plotting.plot_training_loss(
            lc, figures_dir / f"training_loss_{lens_name}.png", label=lens_name,
        )

    # ── 5b. Release backbone ───────────────────────────────────────────────
    # All translators are saved. Evaluation only needs eval_weights (ln_final +
    # concept-column W_U) which were extracted earlier. Move the backbone to
    # CPU to free GPU memory before the evaluation loop.
    if release_backbone:
        with _timed("release backbone to CPU"):
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Backbone moved to CPU — GPU free for evaluation.")

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    logger.info("Evaluating on held-out test set...")

    # final_concept_probs uses eval_weights (backbone may now be on CPU)
    final_concept_probs = _final_concept_probs(test_final_resid, eval_weights)

    test_beliefs_flat = belief_states_all[n_train:, :seq_len_actual].reshape(
        -1, belief_states_all.shape[-1]
    )
    if train_window is not None:
        train_beliefs_flat = belief_states_all[:n_train, w_start:w_end].reshape(
            -1, belief_states_all.shape[-1]
        )
    else:
        train_beliefs_flat = belief_states_all[:n_train, :seq_len_actual].reshape(
            -1, belief_states_all.shape[-1]
        )

    with _timed(f"belief-state R² probe (GPU pinv, {len(config.layer_indices)} layers)"):
        r2_per_layer, r2_per_seq_per_layer = _r2_belief_probe_gpu(
            train_activations, test_activations,
            train_beliefs_flat, test_beliefs_flat,
            config.layer_indices,
            n_test_sequences=n_test,
        )

    all_metrics: list[LayerMetrics] = []
    eval_t0 = time.time()
    for layer in tqdm(config.layer_indices, desc="Evaluating layers"):
        lens_probs: dict[str, np.ndarray] = {}
        lens_probs["logit"] = apply_logit_lens(test_activations[layer], eval_weights, layer=layer)

        # Load each trained translator from disk, apply, then release immediately.
        for lens_name in trained_lens_names:
            t = load_translator(output_dir / f"translators_{lens_name}", layer)
            lens_probs[lens_name] = apply_tuned_lens(
                test_activations[layer], t, eval_weights, layer=layer,
            )
            del t
        torch.cuda.empty_cache()

        m = compute_layer_metrics(
            layer=layer,
            lens_probs=lens_probs,
            final_model_probs=final_concept_probs,
            hmm_probs=test_obs_probs_flat,
            next_tokens=test_next_tokens_flat,
            n_sequences=n_test,
            seq_length=seq_len_actual,
        )
        all_metrics.append(m)

        line = f"  Layer {layer:2d}: "
        for lens_name in ["logit", "tuned_full", "tuned_concept", "tuned_hmm"]:
            if lens_name in m.lenses:
                ln = m.lenses[lens_name]
                line += (
                    f"{lens_name}[KL_h={ln['kl_hmm']:.4f},KL_f={ln['kl_final']:.4f}] "
                )
        logger.info(line.rstrip())
    logger.info(f"[done ] layer evaluation  ({_fmt_secs(time.time() - eval_t0)})")

    # ── 7. Plots ───────────────────────────────────────────────────────────
    logger.info("Generating plots...")
    plots_t0 = time.time()

    plotting.plot_kl_hmm_by_layer(all_metrics, figures_dir / "kl_hmm_by_layer.png")
    plotting.plot_kl_final_by_layer(all_metrics, figures_dir / "kl_final_by_layer.png")
    plotting.plot_nll_by_layer(all_metrics, figures_dir / "nll_by_layer.png")
    plotting.plot_top1_agreement_by_layer(all_metrics, figures_dir / "top1_agreement_by_layer.png")
    plotting.plot_r2_belief_by_layer(r2_per_layer, figures_dir / "r2_belief_by_layer.png")

    available_lenses = list(all_metrics[0].lenses.keys()) if all_metrics else []
    for lens_name in available_lenses:
        plotting.plot_kl_hmm_by_position(
            all_metrics, config.layer_indices, lens_name,
            figures_dir / f"kl_hmm_by_position_{lens_name}.png",
        )
        plotting.plot_kl_final_by_position(
            all_metrics, config.layer_indices, lens_name,
            figures_dir / f"kl_final_by_position_{lens_name}.png",
        )

    plotting.plot_summary(
        all_metrics, r2_per_layer, figures_dir / "summary.png",
        title=f"Tuned Lens — {config.process_name} {config.process_params}",
    )
    logger.info(f"[done ] plots  ({_fmt_secs(time.time() - plots_t0)})")

    # ── 8. Artifacts ───────────────────────────────────────────────────────
    # Translators were already saved to disk in section 5 (save_translators).
    logger.info("Saving artifacts...")
    save_t0 = time.time()

    metrics_summary = []
    for m in all_metrics:
        entry = {"layer": m.layer, "r2_belief_probe": r2_per_layer.get(m.layer)}
        # Flatten per-lens scalars for easy CSV/dataframe loading
        for lens_name, vals in m.lenses.items():
            entry[f"{lens_name}_kl_final"] = vals["kl_final"]
            entry[f"{lens_name}_kl_hmm"] = vals["kl_hmm"]
            entry[f"{lens_name}_nll"] = vals["nll"]
            entry[f"{lens_name}_top1_agreement"] = vals["top1_agreement"]
        metrics_summary.append(entry)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Per-position arrays for all (layer, lens, metric)
    npz_dict: dict[str, np.ndarray] = {}
    for m in all_metrics:
        for lens_name, vals in m.lenses.items():
            npz_dict[f"kl_final_vs_{lens_name}_layer{m.layer}"] = vals["kl_final_by_pos"]
            npz_dict[f"kl_hmm_vs_{lens_name}_layer{m.layer}"] = vals["kl_hmm_by_pos"]
    for layer, r2_seq in r2_per_seq_per_layer.items():
        npz_dict[f"r2_probe_per_seq_layer{layer}"] = r2_seq
    np.savez(output_dir / "per_position_metrics.npz", **npz_dict)

    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "experiment_name": config.experiment_name,
            "model_name": config.model_name,
            "process_name": config.process_name,
            "process_params": config.process_params,
            "vocab_tokens": config.vocab_tokens,
            "seq_length": config.seq_length,
            "n_sequences": config.n_sequences,
            "n_train_sequences": config.n_train_sequences,
            "train_pos_window": config.train_pos_window,
            "tuned_lens_epochs": config.tuned_lens_epochs,
            "tuned_lens_lr": config.tuned_lens_lr,
            "tuned_lens_batch_size": config.tuned_lens_batch_size,
            "tuned_lens_optimizer": config.tuned_lens_optimizer,
            "train_tuned_full": train_tuned_full,
            "train_tuned_concept": train_tuned_concept,
            "train_tuned_hmm": train_tuned_hmm,
            "layer_indices": config.layer_indices,
            "random_seed": config.random_seed,
            "forward_chunk_size": config.forward_chunk_size,
        }, f, indent=2)

    with open(output_dir / "training_losses.json", "w") as f:
        json.dump(
            {ln: {str(k): v for k, v in lc.items()}
             for ln, lc in loss_curves_by_lens.items()},
            f,
        )
    logger.info(f"[done ] artifact serialization  ({_fmt_secs(time.time() - save_t0)})")

    # ── 9. Report ──────────────────────────────────────────────────────────
    logger.info("Writing report...")
    _write_report(output_dir, config, all_metrics, r2_per_layer)

    total_elapsed = time.time() - pipeline_t0
    logger.info(f"All artifacts saved to {output_dir}")
    logger.info(f"=== Pipeline complete in {_fmt_secs(total_elapsed)} ===")
    return {
        "metrics": metrics_summary,
        "r2_per_layer": r2_per_layer,
        "output_dir": str(output_dir),
        "elapsed_seconds": total_elapsed,
    }


def _write_report(
    output_dir: Path,
    config: TunedLensConfig,
    metrics: list[LayerMetrics],
    r2_per_layer: dict[int, float],
) -> None:
    """Concise markdown report covering all available lens variants."""
    lenses = list(metrics[0].lenses.keys()) if metrics else []
    best_r2_layer = max(r2_per_layer, key=r2_per_layer.get) if r2_per_layer else None

    lines = [
        f"# Tuned Lens Per-Layer Experiment Report\n",
        "## Configuration\n",
        f"- Model: `{config.model_name}`",
        f"- HMM: `{config.process_name}` {config.process_params}",
        f"- Vocab: `{config.vocab_tokens}`",
        f"- seq_length={config.seq_length}, n_sequences={config.n_sequences}, "
        f"n_train_sequences={config.n_train_sequences}",
        f"- train_pos_window={config.train_pos_window}",
        f"- Layers analyzed: {config.layer_indices[0]}–{config.layer_indices[-1]} "
        f"({len(config.layer_indices)} layers)",
        f"- Tuned-lens variants trained: {[l for l in lenses if l != 'logit']}",
        f"- Optimizer: {config.tuned_lens_optimizer}, "
        f"epochs={config.tuned_lens_epochs}, lr={config.tuned_lens_lr}",
        "",
        "## Per-layer best lens by KL(HMM || lens)\n",
    ]
    for lens_name in lenses:
        best = min(metrics, key=lambda m: m.lenses[lens_name]["kl_hmm"])
        lines.append(
            f"- **{lens_name}**: layer {best.layer} (KL_HMM={best.lenses[lens_name]['kl_hmm']:.4f}, "
            f"KL_final={best.lenses[lens_name]['kl_final']:.4f})"
        )
    if best_r2_layer is not None:
        lines.append(
            f"- **belief R²**: layer {best_r2_layer} (R²={r2_per_layer[best_r2_layer]:.4f})"
        )

    lines += ["", "## Per-layer table (KL(HMM || lens))\n"]
    header = "| Layer |"
    for lens_name in lenses:
        header += f" {lens_name} |"
    header += " R² |"
    lines.append(header)
    lines.append("|---" * (len(lenses) + 2) + "|")
    for m in metrics:
        row = f"| {m.layer} |"
        for lens_name in lenses:
            row += f" {m.lenses[lens_name]['kl_hmm']:.4f} |"
        r2 = r2_per_layer.get(m.layer, float("nan"))
        row += f" {r2:.4f} |"
        lines.append(row)

    lines += ["", "## Saved artifacts\n",
              "- `metrics.json` — flat per-lens scalars + R²",
              "- `per_position_metrics.npz` — per-position curves keyed `kl_<final|hmm>_vs_<lens>_layer<L>`",
              "- `training_losses.json` — per-epoch KL training loss per lens × layer",
              "- `translators_<lens>/` — saved translator state dicts",
              "- `figures/` — see naming convention in `plotting.py`",
              "- `config.json` — full run config",
              ""]

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
