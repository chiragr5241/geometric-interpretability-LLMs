"""Part C: Causal interventions — projection ablation.

Hooks into the TransformerLens forward pass to ablate either the
belief-aligned or orthogonal component at a chosen layer, then measures
downstream effects on final predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data_generation import generate_hmm_sequences
from experiment_utils import get_concept_token_ids

from .decomposition import DecompositionResult


@dataclass
class AblationResult:
    """Results of ablating one component at one layer."""
    layer: int
    condition: str  # "ablate_orth", "ablate_belief", "mean_ablate_orth"

    kl_vs_original_mean: float   # KL(original || ablated) averaged over points
    kl_vs_original_std: float
    kl_vs_hmm_mean: float        # KL(HMM || ablated)
    kl_vs_hmm_std: float
    kl_vs_hmm_baseline: float    # KL(HMM || original) for reference
    top1_accuracy_ablated: float
    top1_accuracy_original: float
    norm_change_mean: float      # L2 norm of activation change
    norm_change_std: float


@dataclass
class InterventionResults:
    """All ablation results across layers and conditions."""
    results: list[AblationResult]

    def to_dict(self) -> dict:
        rows = []
        for r in self.results:
            rows.append({
                "layer": r.layer,
                "condition": r.condition,
                "kl_vs_original_mean": r.kl_vs_original_mean,
                "kl_vs_original_std": r.kl_vs_original_std,
                "kl_vs_hmm_mean": r.kl_vs_hmm_mean,
                "kl_vs_hmm_std": r.kl_vs_hmm_std,
                "kl_vs_hmm_baseline": r.kl_vs_hmm_baseline,
                "top1_accuracy_ablated": r.top1_accuracy_ablated,
                "top1_accuracy_original": r.top1_accuracy_original,
                "norm_change_mean": r.norm_change_mean,
                "norm_change_std": r.norm_change_std,
            })
        return {"ablation_results": rows}


def _get_concept_probs(logits: torch.Tensor, concept_ids: list[int]) -> np.ndarray:
    """Extract and normalise concept probabilities from full logits."""
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        concept_probs = probs[:, concept_ids].cpu().numpy()
        concept_probs = concept_probs / (concept_probs.sum(axis=-1, keepdims=True) + 1e-10)
    return concept_probs


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """KL(p || q) per row."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def run_ablation_sweep(
    model,
    decomposition: DecompositionResult,
    process_name: str,
    process_params: dict[str, float],
    vocab_tokens: list[str],
    layer_indices: list[int],
    n_sequences: int,
    seq_length: int,
    random_seed: int,
    logger: logging.Logger,
) -> InterventionResults:
    """Run projection ablation at each layer and measure downstream effects.

    For each layer, we run three conditions:
    1. ablate_orth: remove orthogonal component (keep only belief)
    2. ablate_belief: remove belief component (keep only orthogonal)
    3. mean_ablate_orth: replace orthogonal component with its mean
    """
    hmm_data = generate_hmm_sequences(
        process_name=process_name,
        process_params=process_params,
        n_sequences=n_sequences,
        seq_length=seq_length,
        random_seed=random_seed + 1000,  # different seed from training data
    )
    tokens = torch.from_numpy(hmm_data.tokens)
    walk_concepts = [
        [vocab_tokens[int(t)] for t in seq] for seq in tokens
    ]

    concept_to_id = get_concept_token_ids(model, vocab_tokens)
    concept_ids = [concept_to_id[c] for c in vocab_tokens]
    letter_set = set(vocab_tokens)

    # Precompute mean orthogonal component per layer (from training data)
    orth_means = {}
    for layer in layer_indices:
        ld = decomposition.layers[layer]
        orth_means[layer] = ld.h_orth.mean(axis=0)

    all_results: list[AblationResult] = []

    for intervention_layer in tqdm(layer_indices, desc="Ablation sweep"):
        ld = decomposition.layers[intervention_layer]
        Q_basis = torch.from_numpy(ld.Q).float()  # (d_model, n_states)
        orth_mean_vec = torch.from_numpy(orth_means[intervention_layer]).float()

        conditions = {
            "ablate_orth": ("project_belief", Q_basis, None),
            "ablate_belief": ("project_orth", Q_basis, None),
            "mean_ablate_orth": ("mean_replace_orth", Q_basis, orth_mean_vec),
        }

        for cond_name, (mode, Q_b, orth_mean) in conditions.items():
            orig_probs_all = []
            ablated_probs_all = []
            hmm_probs_all = []
            norm_changes = []

            for seq_idx in range(n_sequences):
                seq_concepts = walk_concepts[seq_idx]
                obs_p = hmm_data.obs_probs[seq_idx]

                prompt = seq_concepts[0] + " " + " ".join(seq_concepts[1:])
                input_ids = model.to_tokens(prompt, prepend_bos=True).to(
                    model.embed.W_E.device
                )
                str_tokens = model.to_str_tokens(prompt, prepend_bos=True)
                positions = [
                    i for i, tok in enumerate(str_tokens)
                    if tok.strip() in letter_set
                ]
                n_use = min(len(positions), len(seq_concepts))
                positions = positions[:n_use]

                # Original forward pass (use run_with_hooks for multi-GPU compat)
                with torch.no_grad():
                    orig_logits = model.run_with_hooks(input_ids, fwd_hooks=[])
                    orig_logits_at_pos = orig_logits[0, positions]

                # Ablated forward pass with hook — uses Q basis, device-aware
                def make_hook(mode, Q_cpu, orth_mean_cpu, positions_list):
                    def hook_fn(value, hook):
                        dev = value.device
                        v = value.clone()
                        pos_tensor = torch.tensor(positions_list, device=dev)
                        h = v[0, pos_tensor].float()
                        Q = Q_cpu.to(dev)

                        # Project via Q: h_belief = (h @ Q) @ Q.T
                        coords = h @ Q              # (n_pos, n_states)
                        h_belief = coords @ Q.T     # (n_pos, d_model)

                        if mode == "project_belief":
                            h_new = h_belief
                        elif mode == "project_orth":
                            h_new = h - h_belief
                        elif mode == "mean_replace_orth":
                            h_new = h_belief + orth_mean_cpu.to(dev).unsqueeze(0)

                        v[0, pos_tensor] = h_new.to(value.dtype)
                        return v
                    return hook_fn

                hook_name = f"blocks.{intervention_layer}.hook_resid_post"
                hook_fn = make_hook(mode, Q_b, orth_mean, positions)

                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(
                        input_ids,
                        fwd_hooks=[(hook_name, hook_fn)],
                    )
                    ablated_logits_at_pos = ablated_logits[0, positions]

                orig_p = _get_concept_probs(orig_logits_at_pos, concept_ids)
                ablated_p = _get_concept_probs(ablated_logits_at_pos, concept_ids)
                hmm_p = obs_p[:n_use]

                orig_probs_all.append(orig_p)
                ablated_probs_all.append(ablated_p)
                hmm_probs_all.append(hmm_p)

                with torch.no_grad():
                    norm_diff = (
                        (ablated_logits_at_pos - orig_logits_at_pos)
                        .float().norm(dim=-1).cpu().numpy()
                    )
                norm_changes.append(norm_diff)

                del orig_logits, ablated_logits
                torch.cuda.empty_cache()

            orig_all = np.concatenate(orig_probs_all, axis=0)
            ablated_all = np.concatenate(ablated_probs_all, axis=0)
            hmm_all = np.concatenate(hmm_probs_all, axis=0)
            norms = np.concatenate(norm_changes)

            kl_orig = _kl_divergence(orig_all, ablated_all)
            kl_hmm_ablated = _kl_divergence(hmm_all, ablated_all)
            kl_hmm_orig = _kl_divergence(hmm_all, orig_all)

            top1_orig = (orig_all.argmax(axis=-1) == hmm_all.argmax(axis=-1)).mean()
            top1_ablated = (ablated_all.argmax(axis=-1) == hmm_all.argmax(axis=-1)).mean()

            all_results.append(AblationResult(
                layer=intervention_layer,
                condition=cond_name,
                kl_vs_original_mean=float(np.mean(kl_orig)),
                kl_vs_original_std=float(np.std(kl_orig)),
                kl_vs_hmm_mean=float(np.mean(kl_hmm_ablated)),
                kl_vs_hmm_std=float(np.std(kl_hmm_ablated)),
                kl_vs_hmm_baseline=float(np.mean(kl_hmm_orig)),
                top1_accuracy_ablated=float(top1_ablated),
                top1_accuracy_original=float(top1_orig),
                norm_change_mean=float(np.mean(norms)),
                norm_change_std=float(np.std(norms)),
            ))

            logger.info(
                f"  Layer {intervention_layer:2d} [{cond_name}]: "
                f"KL(orig||abl)={np.mean(kl_orig):.4f}  "
                f"KL(HMM||abl)={np.mean(kl_hmm_ablated):.4f}  "
                f"top1_abl={top1_ablated:.3f}"
            )

    return InterventionResults(results=all_results)
