"""Core pipeline: forward pass, KL divergence, and probe training for a single config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data_generation import generate_hmm_sequences
from experiment_utils import get_concept_token_ids
from metrics.kl_divergence import (
    compute_kl_divergence_batch,
    extract_concept_probs,
    extract_concept_probs_all_vocab,
)

from .config import BeliefStateSweepConfig, make_config_label
from .pca import plot_pca, plot_pca_3d
from .probes import train_probes
from .results import ConfigResult


def run_single_config(
    process_name: str,
    process_params: dict[str, float],
    vocab_tokens: list[str],
    model,
    config: BeliefStateSweepConfig,
    logger,
    pca_plot_path: Path | None = None,
    pca_3d_plot_path: Path | None = None,
    seq_length: int | None = None,
    n_sequences: int | None = None,
) -> ConfigResult:
    """Run the full pipeline for one HMM parameter combination."""
    seq_length = seq_length if seq_length is not None else config.seq_length
    n_sequences = n_sequences if n_sequences is not None else config.n_sequences
    label = make_config_label(process_name, process_params)
    n_vocab = len(vocab_tokens)

    # 1. Build HMM and generate sequences
    hmm_data = generate_hmm_sequences(
        process_name=process_name,
        process_params=process_params,
        n_sequences=n_sequences,
        seq_length=seq_length,
        random_seed=config.random_seed,
    )
    hmm = hmm_data.hmm
    logger.info(f"  HMM: vocab_size={hmm.vocab_size}, num_states={hmm.num_states}")

    tokens = torch.from_numpy(hmm_data.tokens)
    belief_states_all = hmm_data.belief_states
    obs_probs_all = hmm_data.obs_probs

    walk_concepts = [[vocab_tokens[int(t)] for t in seq] for seq in tokens]
    logger.info(f"  Generated {n_sequences} sequences of length {seq_length}")

    # 2. Resolve LLM token IDs
    concept_to_id = get_concept_token_ids(model, vocab_tokens)
    concept_ids = [concept_to_id[c] for c in vocab_tokens]
    logger.info(f"  Concept token IDs: {dict(zip(vocab_tokens, concept_ids))}")

    # 3. Forward pass with activation caching
    letter_set = set(vocab_tokens)
    all_activations = {layer: [] for layer in config.layer_indices}
    all_logits = []
    all_beliefs = []

    for seq_idx in tqdm(
        range(n_sequences), desc=f"  Forward pass ({label})"
    ):
        seq_concepts = walk_concepts[seq_idx]
        beliefs = belief_states_all[seq_idx]

        prompt = seq_concepts[0] + " " + " ".join(seq_concepts[1:])
        input_ids = model.to_tokens(prompt, prepend_bos=True).to(
            model.embed.W_E.device
        )
        str_tokens = model.to_str_tokens(prompt, prepend_bos=True)

        positions = [
            i for i, tok in enumerate(str_tokens) if tok.strip() in letter_set
        ]
        n_use = min(len(positions), len(seq_concepts))
        if len(positions) != len(seq_concepts):
            logger.warning(
                f"  Seq {seq_idx}: expected {len(seq_concepts)} letter positions, "
                f"got {len(positions)}"
            )
        positions = positions[:n_use]

        hook_names = [
            f"blocks.{l}.hook_resid_post" for l in config.layer_indices
        ]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_names,
                return_type=None,
            )

            for layer in config.layer_indices:
                hook_name = f"blocks.{layer}.hook_resid_post"
                layer_acts = cache[hook_name][0]
                letter_acts = layer_acts[positions].cpu().float().numpy()
                all_activations[layer].append(letter_acts)

            last_resid = cache[f"blocks.{config.layer_indices[-1]}.hook_resid_post"]
            last_resid = last_resid.to(model.unembed.W_U.device)
            logits_out = model.unembed(model.ln_final(last_resid))
            seq_logits = logits_out[0, positions].cpu().float().detach().numpy()
            all_logits.append(seq_logits)
            del logits_out, last_resid, cache

        all_beliefs.append(beliefs[:n_use])
        torch.cuda.empty_cache()

    # Concatenate across sequences
    for layer in config.layer_indices:
        all_activations[layer] = np.concatenate(all_activations[layer], axis=0)
    all_logits_flat = np.concatenate(all_logits, axis=0)
    all_beliefs_flat = np.concatenate(all_beliefs, axis=0)

    logger.info(f"  Total datapoints: {len(all_beliefs_flat)}")

    # 4. KL divergence
    llm_probs = extract_concept_probs(all_logits_flat, concept_ids)
    llm_probs_all_vocab = extract_concept_probs_all_vocab(all_logits_flat, concept_ids)

    n_total = llm_probs.shape[0]
    if n_total != n_sequences * seq_length:
        logger.warning(
            f"  LLM produced {n_total} data points but expected "
            f"{n_sequences}×{seq_length}={n_sequences * seq_length}. "
            f"KL will be computed on the first {n_total} points only."
        )
    llm_probs_3d = llm_probs.reshape(n_sequences, -1, n_vocab)
    llm_probs_all_vocab_3d = llm_probs_all_vocab.reshape(n_sequences, -1, n_vocab)
    hmm_probs = obs_probs_all[:, :llm_probs_3d.shape[1], :]

    kl_mean, kl_std = compute_kl_divergence_batch(hmm_probs, llm_probs_3d)
    kl_all_vocab_mean, kl_all_vocab_std = compute_kl_divergence_batch(
        hmm_probs, llm_probs_all_vocab_3d
    )
    logger.info(
        f"  Mean KL (renorm): {kl_mean.mean():.4f}, "
        f"Mean KL (all-vocab): {kl_all_vocab_mean.mean():.4f}"
    )

    # 5. Linear probes
    probe_results = train_probes(
        all_activations=all_activations,
        belief_states_flat=all_beliefs_flat,
        layer_indices=config.layer_indices,
        test_size=config.probe.test_size,
        random_state=config.probe.random_state,
    )

    logger.info(
        f"  Best R²={probe_results.r2_per_layer[probe_results.best_layer]:.4f} "
        f"at layer {probe_results.best_layer}"
    )

    # 6. PCA plots
    if pca_plot_path is not None:
        plot_pca(
            all_activations=all_activations,
            belief_states_flat=all_beliefs_flat,
            layers=config.layer_indices,
            title=f"PCA of Residual Stream by Layer (colored by belief) — {label}",
            path=pca_plot_path,
        )

    if pca_3d_plot_path is not None:
        plot_pca_3d(
            all_activations=all_activations,
            belief_states_flat=all_beliefs_flat,
            layers=config.layer_indices,
            title=f"3D PCA of Residual Stream by Layer (colored by belief) — {label}",
            path=pca_3d_plot_path,
        )

    return ConfigResult(
        process_name=process_name,
        process_params=process_params,
        label=label,
        vocab_tokens=vocab_tokens,
        belief_states_flat=all_beliefs_flat,
        kl_mean=kl_mean,
        kl_std=kl_std,
        kl_all_vocab_mean=kl_all_vocab_mean,
        kl_all_vocab_std=kl_all_vocab_std,
        r2_per_layer=probe_results.r2_per_layer,
        mse_per_layer=probe_results.mse_per_layer,
        predicted_beliefs=probe_results.best_layer_predicted,
        predicted_beliefs_gt=probe_results.best_layer_gt,
    )
