"""Activation collection and disk caching.

Runs the model forward pass over HMM sequences, collects per-layer
residual-stream activations at letter-aligned positions, and saves
everything to a single .npz file for offline analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data_generation import generate_hmm_sequences, HMMSequenceData
from experiment_utils import get_concept_token_ids
from metrics.kl_divergence import compute_kl_divergence_batch, extract_concept_probs


@dataclass
class CachedActivations:
    """In-memory container mirroring the on-disk .npz layout."""
    activations: dict[int, np.ndarray]  # layer -> (n_points, d_model)
    belief_states: np.ndarray           # (n_points, n_states)
    obs_probs: np.ndarray               # (n_points, n_vocab)
    concept_logits: np.ndarray          # (n_points, n_concepts)
    concept_ids: list[int]
    token_positions: np.ndarray         # (n_points,)  position index within seq
    seq_indices: np.ndarray             # (n_points,)  which sequence
    layer_indices: list[int]
    n_sequences: int
    seq_length: int
    vocab_tokens: list[str]
    process_name: str
    process_params: dict[str, float]


def collect_activations(
    process_name: str,
    process_params: dict[str, float],
    vocab_tokens: list[str],
    model,
    layer_indices: list[int],
    n_sequences: int,
    seq_length: int,
    random_seed: int,
    logger: logging.Logger,
) -> CachedActivations:
    """Run forward passes and collect all per-layer activations."""

    hmm_data = generate_hmm_sequences(
        process_name=process_name,
        process_params=process_params,
        n_sequences=n_sequences,
        seq_length=seq_length,
        random_seed=random_seed,
    )
    hmm = hmm_data.hmm
    logger.info(
        f"HMM: {process_name}, vocab_size={hmm.vocab_size}, "
        f"num_states={hmm.num_states}"
    )

    tokens = torch.from_numpy(hmm_data.tokens)
    walk_concepts = [
        [vocab_tokens[int(t)] for t in seq] for seq in tokens
    ]

    concept_to_id = get_concept_token_ids(model, vocab_tokens)
    concept_ids = [concept_to_id[c] for c in vocab_tokens]
    logger.info(f"Concept token IDs: {dict(zip(vocab_tokens, concept_ids))}")

    letter_set = set(vocab_tokens)
    all_activations = {layer: [] for layer in layer_indices}
    all_concept_logits = []
    all_beliefs = []
    all_obs_probs = []
    all_positions = []
    all_seq_indices = []

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]

    for seq_idx in tqdm(range(n_sequences), desc="Collecting activations"):
        seq_concepts = walk_concepts[seq_idx]
        beliefs = hmm_data.belief_states[seq_idx]
        obs_p = hmm_data.obs_probs[seq_idx]

        prompt = seq_concepts[0] + " " + " ".join(seq_concepts[1:])
        input_ids = model.to_tokens(prompt, prepend_bos=True).to(
            model.embed.W_E.device
        )
        str_tokens = model.to_str_tokens(prompt, prepend_bos=True)

        positions = [
            i for i, tok in enumerate(str_tokens) if tok.strip() in letter_set
        ]
        n_use = min(len(positions), len(seq_concepts))
        positions = positions[:n_use]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids, names_filter=hook_names, return_type=None,
            )

            for layer in layer_indices:
                hook = f"blocks.{layer}.hook_resid_post"
                acts = cache[hook][0][positions].cpu().float().numpy()
                all_activations[layer].append(acts)

            last_resid = cache[f"blocks.{layer_indices[-1]}.hook_resid_post"]
            last_resid = last_resid.to(model.unembed.W_U.device)
            full_logits = model.unembed(model.ln_final(last_resid))
            seq_full_logits = full_logits[0, positions].cpu().float()
            concept_ids_t = torch.tensor(concept_ids)
            seq_concept_logits = seq_full_logits[:, concept_ids_t].numpy()
            all_concept_logits.append(seq_concept_logits)

            del full_logits, last_resid, cache

        all_beliefs.append(beliefs[:n_use])
        all_obs_probs.append(obs_p[:n_use])
        all_positions.append(np.arange(n_use))
        all_seq_indices.append(np.full(n_use, seq_idx))
        torch.cuda.empty_cache()

    for layer in layer_indices:
        all_activations[layer] = np.concatenate(all_activations[layer], axis=0)

    result = CachedActivations(
        activations=all_activations,
        belief_states=np.concatenate(all_beliefs, axis=0),
        obs_probs=np.concatenate(all_obs_probs, axis=0),
        concept_logits=np.concatenate(all_concept_logits, axis=0),
        concept_ids=concept_ids,
        token_positions=np.concatenate(all_positions, axis=0),
        seq_indices=np.concatenate(all_seq_indices, axis=0),
        layer_indices=layer_indices,
        n_sequences=n_sequences,
        seq_length=seq_length,
        vocab_tokens=vocab_tokens,
        process_name=process_name,
        process_params=process_params,
    )

    n_total = result.belief_states.shape[0]
    logger.info(f"Collected {n_total} datapoints across {n_sequences} sequences")
    logger.info(
        f"Activation shape per layer: {all_activations[layer_indices[0]].shape}"
    )
    return result


def save_cache(cache: CachedActivations, path: Path) -> None:
    """Persist activations and metadata to a single .npz file.

    Uses uncompressed savez for speed — activation arrays are large and
    compress poorly (random-looking float16/32 values).
    """
    arrays = {}
    for layer in cache.layer_indices:
        arrays[f"acts_layer_{layer}"] = cache.activations[layer]
    arrays["belief_states"] = cache.belief_states
    arrays["obs_probs"] = cache.obs_probs
    arrays["concept_logits"] = cache.concept_logits
    arrays["concept_ids"] = np.array(cache.concept_ids)
    arrays["token_positions"] = cache.token_positions
    arrays["seq_indices"] = cache.seq_indices
    arrays["layer_indices"] = np.array(cache.layer_indices)
    np.savez(path, **arrays)


def load_cache(
    path: Path,
    vocab_tokens: list[str],
    process_name: str,
    process_params: dict[str, float],
    n_sequences: int,
    seq_length: int,
) -> CachedActivations:
    """Load a previously saved .npz cache."""
    data = np.load(path)
    layer_indices = data["layer_indices"].tolist()
    activations = {}
    for layer in layer_indices:
        activations[layer] = data[f"acts_layer_{layer}"]

    return CachedActivations(
        activations=activations,
        belief_states=data["belief_states"],
        obs_probs=data["obs_probs"],
        concept_logits=data["concept_logits"],
        concept_ids=data["concept_ids"].tolist(),
        token_positions=data["token_positions"],
        seq_indices=data["seq_indices"],
        layer_indices=layer_indices,
        n_sequences=n_sequences,
        seq_length=seq_length,
        vocab_tokens=vocab_tokens,
        process_name=process_name,
        process_params=process_params,
    )
