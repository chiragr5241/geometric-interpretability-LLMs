"""HMM data generation utilities.

Wraps simplexity's HMM builder and sequence generator with convenience
functions used across experiments (belief state sweep, probes, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.generator import (
    generate_data_batch_with_full_history,
)
from simplexity.utils.pytorch_utils import jax_to_torch


@dataclass
class HMMSequenceData:
    """Container for generated HMM sequence data, converted to numpy."""

    tokens: np.ndarray
    """Integer token IDs.  Shape: (n_sequences, seq_length)."""

    belief_states: np.ndarray
    """Posterior belief states aligned with *tokens*.
    Shape: (n_sequences, seq_length, n_states).
    ``belief_states[b, t]`` is the belief **after** observing ``tokens[b, t]``."""

    obs_probs: np.ndarray
    """HMM next-token probabilities derived from *belief_states*.
    Shape: (n_sequences, seq_length, vocab_size).
    ``obs_probs[b, t]`` = P(next_token | belief at position t), i.e. predicts
    the token at position t+1."""

    hmm: object
    """The simplexity HiddenMarkovModel instance."""


def generate_hmm_sequences(
    process_name: str,
    process_params: dict[str, float],
    n_sequences: int,
    seq_length: int,
    random_seed: int = 42,
) -> HMMSequenceData:
    """Build an HMM and generate sequences with full belief-state history.

    Parameters
    ----------
    process_name : str
        Name recognised by ``simplexity.generative_processes.builder``
        (e.g. ``"mess3"``, ``"leopard"``).
    process_params : dict
        HMM-specific parameters (e.g. ``{"x": 0.05, "a": 0.85}``).
    n_sequences : int
        Number of independent sequences to generate.
    seq_length : int
        Desired number of input tokens per sequence.  Internally we request
        ``seq_length + 1`` from simplexity so that, after the input/label
        split, ``inputs`` has exactly *seq_length* tokens.
    random_seed : int
        JAX PRNG seed.

    Returns
    -------
    HMMSequenceData
        Structured container with tokens, belief states, observation
        probabilities, and the HMM object.
    """
    hmm = build_hidden_markov_model(
        process_name, process_params=process_params, device=None,
    )

    key = jax.random.PRNGKey(random_seed)
    gen_states = jnp.tile(hmm.initial_state, (n_sequences, 1))

    # +1 so that after simplexity's input/label split, inputs has seq_length tokens.
    gen_result = generate_data_batch_with_full_history(
        gen_states, hmm, n_sequences, seq_length + 1, key,
    )

    tokens_jax = gen_result["inputs"]
    belief_states_jax = gen_result["belief_states"]

    obs_probs_jax = compute_observation_probs(belief_states_jax, hmm)

    return HMMSequenceData(
        tokens=jax_to_torch(tokens_jax).cpu().numpy(),
        belief_states=jax_to_torch(belief_states_jax).cpu().numpy(),
        obs_probs=jax_to_torch(obs_probs_jax).cpu().numpy(),
        hmm=hmm,
    )


def compute_observation_probs(
    belief_states: jnp.ndarray,
    hmm,
) -> jnp.ndarray:
    """Compute P(next_token | belief) for every position via einsum.

    Parameters
    ----------
    belief_states : jnp.ndarray
        Shape ``(batch, seq_len, n_states)``.
    hmm
        Simplexity ``HiddenMarkovModel`` with ``.transition_matrices`` of
        shape ``(vocab_size, n_states, n_states)``.

    Returns
    -------
    jnp.ndarray
        Shape ``(batch, seq_len, vocab_size)``.
        Entry ``[b, t, v]`` = Σ_s belief[b,t,s] · Σ_{s'} T[v,s,s']
        = P(emit v | belief at time t).
    """
    return jnp.einsum("bns,vst->bnv", belief_states, hmm.transition_matrices)
