"""Core pipeline: forward pass, tuned lens training, and evaluation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data_generation import generate_hmm_sequences
from experiment_utils import get_concept_token_ids
from metrics.kl_divergence import extract_concept_probs

from .config import TunedLensConfig
from .evaluation import LayerMetrics, compute_layer_metrics
from .tuned_lens import (
    apply_logit_lens,
    apply_tuned_lens,
    train_tuned_lens,
)
from .plotting import (
    plot_comparison,
    plot_layer_vs_kl_final,
    plot_layer_vs_kl_hmm,
    plot_nll_by_layer,
    plot_token_position_vs_kl,
    plot_training_loss,
)

logger = logging.getLogger(__name__)


def _forward_pass(
    model,
    walk_concepts: list[list[str]],
    vocab_tokens: list[str],
    layer_indices: list[int],
    n_sequences: int,
) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    """Run forward pass, return per-sequence activations and logits."""
    letter_set = set(vocab_tokens)
    seq_activations = {layer: [] for layer in layer_indices}
    seq_logits = []

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_indices]

    for seq_idx in tqdm(range(n_sequences), desc="Forward pass"):
        seq_concepts = walk_concepts[seq_idx]
        prompt = seq_concepts[0] + " " + " ".join(seq_concepts[1:])
        input_ids = model.to_tokens(prompt, prepend_bos=True).to(model.embed.W_E.device)
        str_tokens = model.to_str_tokens(prompt, prepend_bos=True)

        positions = [i for i, tok in enumerate(str_tokens) if tok.strip() in letter_set]
        n_use = min(len(positions), len(seq_concepts))
        positions = positions[:n_use]

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=hook_names, return_type=None)

            for layer in layer_indices:
                hook_name = f"blocks.{layer}.hook_resid_post"
                layer_acts = cache[hook_name][0][positions].cpu().float().numpy()
                seq_activations[layer].append(layer_acts)

            last_resid = cache[f"blocks.{layer_indices[-1]}.hook_resid_post"]
            last_resid = last_resid.to(model.unembed.W_U.device)
            logits_out = model.unembed(model.ln_final(last_resid))
            seq_logits.append(logits_out[0, positions].cpu().float().detach().numpy())
            del logits_out, last_resid, cache

        torch.cuda.empty_cache()

    return seq_activations, seq_logits


def run_pipeline(
    model,
    config: TunedLensConfig,
    output_dir: Path,
) -> dict:
    """Run the full tuned lens per-layer experiment.

    Returns a summary dict with key metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ── 1. Generate HMM data ────────────────────────────────────────────────
    logger.info("Generating HMM sequences...")
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

    # ── 2. Resolve token IDs ────────────────────────────────────────────────
    concept_to_id = get_concept_token_ids(model, config.vocab_tokens)
    concept_ids = [concept_to_id[c] for c in config.vocab_tokens]
    logger.info(f"Concept token IDs: {dict(zip(config.vocab_tokens, concept_ids))}")

    # ── 3. Forward pass ─────────────────────────────────────────────────────
    logger.info("Running forward pass with activation caching...")
    seq_activations, seq_logits = _forward_pass(
        model, walk_concepts, config.vocab_tokens, config.layer_indices, config.n_sequences,
    )

    # Verify sequence lengths
    seq_len_actual = seq_logits[0].shape[0]
    if seq_len_actual != config.seq_length:
        logger.warning(f"Actual seq_length={seq_len_actual}, expected {config.seq_length}")

    # ── 4. Train/test split by sequence ─────────────────────────────────────
    n_train = config.n_train_sequences
    n_test = config.n_sequences - n_train
    logger.info(f"Train/test split: {n_train} train, {n_test} test sequences")

    def concat_seqs(seq_list: list[np.ndarray], start: int, end: int) -> np.ndarray:
        return np.concatenate(seq_list[start:end], axis=0)

    # Training data: flatten train sequences
    train_activations = {}
    test_activations = {}
    for layer in config.layer_indices:
        arrs = seq_activations[layer]
        train_activations[layer] = concat_seqs(arrs, 0, n_train)
        test_activations[layer] = concat_seqs(arrs, n_train, config.n_sequences)

    train_logits = concat_seqs(seq_logits, 0, n_train)
    test_logits = concat_seqs(seq_logits, n_train, config.n_sequences)

    # HMM probs for test set
    test_obs_probs = obs_probs_all[n_train:, :seq_len_actual, :]
    test_obs_probs_flat = test_obs_probs.reshape(-1, test_obs_probs.shape[-1])

    # Next tokens for test set (for NLL): token at position t+1
    test_tokens = hmm_data.tokens[n_train:, :seq_len_actual]
    # Shift: the next token for position t is the token at position t+1
    # For the last position we have no next token, so we use the seq_length-1 positions
    # Actually, obs_probs at position t predicts token at t+1, so we need tokens at t+1
    # But we only have seq_length tokens (0..seq_length-1).
    # The activation at position t predicts the next token, which is at t+1.
    # We'll use positions 0..seq_length-2 for NLL, or simply use all positions
    # where we have the next token available.
    # For simplicity, let's use the token at each position as the "current" token
    # and compute NLL for the model's prediction of the next token.
    # Actually, looking at the pipeline: obs_probs[b,t] = P(next_token | belief at t)
    # and the activation at concept position t is used to predict what comes after t.
    # The next token after position t is tokens[b, t+1] (if available).
    # For positions 0..seq_length-2 we have a valid next token.
    # For position seq_length-1, we don't. Let's handle this:
    test_next_tokens_full = np.zeros((n_test, seq_len_actual), dtype=np.int64)
    test_next_tokens_full[:, :-1] = test_tokens[:, 1:]
    test_next_tokens_full[:, -1] = 0  # placeholder for last position
    test_next_tokens_flat = test_next_tokens_full.reshape(-1)

    logger.info(f"Train points: {train_logits.shape[0]}, Test points: {test_logits.shape[0]}")

    # ── 5. Train tuned lens on TRAINING data ────────────────────────────────
    logger.info("Training tuned lens translators...")
    translators, loss_curves = train_tuned_lens(
        activations_by_layer=train_activations,
        model=model,
        layers=config.layer_indices,
        target_logits=train_logits,
        n_epochs=config.tuned_lens_epochs,
        lr=config.tuned_lens_lr,
        batch_size=config.tuned_lens_batch_size,
    )

    plot_training_loss(loss_curves, figures_dir / "training_loss.png")

    # ── 6. Evaluate on TEST data ────────────────────────────────────────────
    logger.info("Evaluating on held-out test set...")

    # Final model's concept probs on test set
    final_concept_probs = extract_concept_probs(test_logits, concept_ids)

    # Also train belief-state probes for the comparison plot
    from sklearn.linear_model import LinearRegression
    test_beliefs_flat = belief_states_all[n_train:, :seq_len_actual].reshape(-1, belief_states_all.shape[-1])
    train_beliefs_flat = belief_states_all[:n_train, :seq_len_actual].reshape(-1, belief_states_all.shape[-1])

    r2_per_layer: dict[int, float] = {}
    for layer in config.layer_indices:
        reg = LinearRegression()
        reg.fit(train_activations[layer], train_beliefs_flat)
        y_pred = reg.predict(test_activations[layer])
        ss_res = np.sum((y_pred - test_beliefs_flat) ** 2)
        ss_tot = np.sum((test_beliefs_flat - test_beliefs_flat.mean(axis=0)) ** 2)
        r2_per_layer[layer] = float(1.0 - ss_res / (ss_tot + 1e-10))

    all_metrics: list[LayerMetrics] = []

    for layer in tqdm(config.layer_indices, desc="Evaluating layers"):
        # Raw logit lens
        logit_probs = apply_logit_lens(test_activations[layer], model, concept_ids)

        # Tuned lens
        tuned_probs = apply_tuned_lens(
            test_activations[layer], translators[layer], model, concept_ids,
        )

        m = compute_layer_metrics(
            layer=layer,
            tuned_lens_probs=tuned_probs,
            logit_lens_probs=logit_probs,
            final_model_probs=final_concept_probs,
            hmm_probs=test_obs_probs_flat,
            next_tokens=test_next_tokens_flat,
            n_sequences=n_test,
            seq_length=seq_len_actual,
        )
        all_metrics.append(m)

        logger.info(
            f"  Layer {layer:2d}: "
            f"KL(final||tuned)={m.kl_final_vs_tuned:.4f}  "
            f"KL(HMM||tuned)={m.kl_hmm_vs_tuned:.4f}  "
            f"KL(HMM||logit)={m.kl_hmm_vs_logit:.4f}  "
            f"top1_tuned={m.top1_agreement_tuned:.3f}"
        )

    # ── 7. Generate plots ───────────────────────────────────────────────────
    logger.info("Generating plots...")

    plot_layer_vs_kl_final(all_metrics, figures_dir / "layer_vs_kl_final_tuned.png")
    plot_layer_vs_kl_hmm(all_metrics, figures_dir / "layer_vs_kl_hmm.png")
    plot_nll_by_layer(all_metrics, figures_dir / "nll_by_layer.png")

    # Token position plots for a selection of layers
    n_layers = len(config.layer_indices)
    selected = [config.layer_indices[i] for i in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, -1]]

    plot_token_position_vs_kl(
        all_metrics, selected, figures_dir / "token_pos_kl_hmm_tuned.png",
        metric_type="kl_hmm_vs_tuned",
    )
    plot_token_position_vs_kl(
        all_metrics, selected, figures_dir / "token_pos_kl_final_tuned.png",
        metric_type="kl_final_vs_tuned",
    )

    plot_comparison(all_metrics, r2_per_layer, figures_dir / "comparison.png")

    # ── 8. Save artifacts ───────────────────────────────────────────────────
    logger.info("Saving artifacts...")

    # Save translators
    translators_dir = output_dir / "translators"
    translators_dir.mkdir(exist_ok=True)
    for layer, translator in translators.items():
        torch.save(translator.state_dict(), translators_dir / f"layer_{layer}.pt")

    # Save metrics as JSON
    metrics_summary = []
    for m in all_metrics:
        metrics_summary.append({
            "layer": m.layer,
            "kl_final_vs_tuned": m.kl_final_vs_tuned,
            "kl_hmm_vs_tuned": m.kl_hmm_vs_tuned,
            "kl_hmm_vs_logit": m.kl_hmm_vs_logit,
            "nll_tuned": m.nll_tuned,
            "nll_logit": m.nll_logit,
            "top1_agreement_tuned": m.top1_agreement_tuned,
            "top1_agreement_logit": m.top1_agreement_logit,
            "r2_belief_probe": r2_per_layer.get(m.layer),
        })

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save per-position KL arrays
    np.savez(
        output_dir / "per_position_metrics.npz",
        **{f"kl_final_tuned_layer{m.layer}": m.kl_final_vs_tuned_by_pos for m in all_metrics},
        **{f"kl_hmm_tuned_layer{m.layer}": m.kl_hmm_vs_tuned_by_pos for m in all_metrics},
        **{f"kl_hmm_logit_layer{m.layer}": m.kl_hmm_vs_logit_by_pos for m in all_metrics},
    )

    # Save config
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
            "tuned_lens_epochs": config.tuned_lens_epochs,
            "tuned_lens_lr": config.tuned_lens_lr,
            "tuned_lens_batch_size": config.tuned_lens_batch_size,
            "layer_indices": config.layer_indices,
            "random_seed": config.random_seed,
        }, f, indent=2)

    # Save training loss curves
    with open(output_dir / "training_losses.json", "w") as f:
        json.dump({str(k): v for k, v in loss_curves.items()}, f)

    # ── 9. Generate markdown report ─────────────────────────────────────────
    logger.info("Writing report...")
    _write_report(output_dir, config, all_metrics, r2_per_layer, loss_curves)

    logger.info(f"All artifacts saved to {output_dir}")
    return {
        "metrics": metrics_summary,
        "r2_per_layer": r2_per_layer,
        "output_dir": str(output_dir),
    }


def _write_report(
    output_dir: Path,
    config: TunedLensConfig,
    metrics: list[LayerMetrics],
    r2_per_layer: dict[int, float],
    loss_curves: dict[int, list[float]],
) -> None:
    """Write a concise markdown report."""

    # Find key layers
    best_tuned_layer = min(metrics, key=lambda m: m.kl_final_vs_tuned)
    best_hmm_tuned_layer = min(metrics, key=lambda m: m.kl_hmm_vs_tuned)
    best_r2_layer = max(r2_per_layer, key=r2_per_layer.get)
    worst_tuned_layer = max(metrics, key=lambda m: m.kl_final_vs_tuned)

    # Check if early layers can reconstruct
    early_layers = [m for m in metrics if m.layer <= len(config.layer_indices) // 3]
    late_layers = [m for m in metrics if m.layer >= 2 * len(config.layer_indices) // 3]
    mid_layers = [m for m in metrics if len(config.layer_indices) // 3 < m.layer < 2 * len(config.layer_indices) // 3]

    avg_early_kl = np.mean([m.kl_final_vs_tuned for m in early_layers]) if early_layers else float("nan")
    avg_late_kl = np.mean([m.kl_final_vs_tuned for m in late_layers]) if late_layers else float("nan")

    report = f"""# Tuned Lens Per-Layer Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `{config.model_name}` |
| HMM Process | `{config.process_name}` |
| Process Parameters | `{config.process_params}` |
| Vocabulary | `{config.vocab_tokens}` |
| Sequence Length | {config.seq_length} |
| Sequences (total) | {config.n_sequences} |
| Train / Test Split | {config.n_train_sequences} / {config.n_sequences - config.n_train_sequences} sequences |
| Layers Analyzed | {config.layer_indices[0]}–{config.layer_indices[-1]} ({len(config.layer_indices)} layers) |
| Tuned Lens Epochs | {config.tuned_lens_epochs} |
| Learning Rate | {config.tuned_lens_lr} |
| Batch Size | {config.tuned_lens_batch_size} |
| Random Seed | {config.random_seed} |

## Implementation Choices

- **Tuned lens variant**: Faithful full-vocabulary version (arXiv:2303.08112). Each per-layer
  affine translator T_l: R^{{d_model}} → R^{{d_model}} is identity-initialized and trained to
  minimize KL(p_final || p_lens) over the entire vocabulary (~128K tokens).
- **Pipeline**: h_l → T_l(h_l) → ln_final → W_U·(·) + b_U → softmax
- **Train/test split**: By sequence (first {config.n_train_sequences} sequences train,
  remaining {config.n_sequences - config.n_train_sequences} held out). No data leakage across sequences.
- **Belief-state probes**: Ordinary least squares (LinearRegression) from activations → beliefs,
  trained on the same train sequences, evaluated on the same test sequences.
- **LR schedule**: Cosine annealing over {config.tuned_lens_epochs} epochs.

## Main Results

### Best layers

| Criterion | Layer | Value |
|-----------|-------|-------|
| Lowest KL(final \\|\\| tuned) | {best_tuned_layer.layer} | {best_tuned_layer.kl_final_vs_tuned:.4f} |
| Lowest KL(HMM \\|\\| tuned) | {best_hmm_tuned_layer.layer} | {best_hmm_tuned_layer.kl_hmm_vs_tuned:.4f} |
| Highest belief-state R² | {best_r2_layer} | {r2_per_layer[best_r2_layer]:.4f} |
| Highest top-1 agreement (tuned) | {max(metrics, key=lambda m: m.top1_agreement_tuned).layer} | {max(metrics, key=lambda m: m.top1_agreement_tuned).top1_agreement_tuned:.3f} |

### Layer-group averages: KL(final || tuned lens)

| Group | Layers | Mean KL |
|-------|--------|---------|
| Early (first third) | {early_layers[0].layer}–{early_layers[-1].layer} | {avg_early_kl:.4f} |
| Late (last third) | {late_layers[0].layer}–{late_layers[-1].layer} | {avg_late_kl:.4f} |

### Per-layer metrics table

| Layer | KL(final\\|\\|tuned) | KL(HMM\\|\\|tuned) | KL(HMM\\|\\|logit) | NLL(tuned) | Top-1(tuned) | R² |
|-------|-------|-------|-------|-------|-------|-------|
"""

    for m in metrics:
        r2 = r2_per_layer.get(m.layer, float("nan"))
        report += (
            f"| {m.layer} | {m.kl_final_vs_tuned:.4f} | {m.kl_hmm_vs_tuned:.4f} | "
            f"{m.kl_hmm_vs_logit:.4f} | {m.nll_tuned:.4f} | {m.top1_agreement_tuned:.3f} | "
            f"{r2:.4f} |\n"
        )

    report += f"""
## Plots

- `figures/layer_vs_kl_final_tuned.png` — Layer vs KL(final model || tuned lens)
- `figures/layer_vs_kl_hmm.png` — Layer vs KL(HMM || tuned lens) and KL(HMM || logit lens)
- `figures/nll_by_layer.png` — Next-token NLL by layer
- `figures/token_pos_kl_hmm_tuned.png` — Token position vs KL(HMM || tuned) for selected layers
- `figures/token_pos_kl_final_tuned.png` — Token position vs KL(final || tuned) for selected layers
- `figures/comparison.png` — 4-panel comparison (tuned lens, logit lens, top-1, R²)
- `figures/training_loss.png` — Per-layer training loss curves

## Interpretation

### Do early layers already contain the final prediction?

"""

    if avg_early_kl < 2 * avg_late_kl:
        report += (
            "Early layers can already reconstruct final predictions reasonably well via the "
            "tuned lens (early-layer KL is within 2x of late-layer KL). This suggests that "
            "a substantial portion of the predictive information is present early in the network, "
            "and that later layers may primarily perform readout, linearization, or cleanup of "
            "representations that are already informationally rich.\n\n"
        )
    else:
        report += (
            "Early layers show substantially higher KL(final || tuned) than late layers, "
            "indicating that later layers perform genuinely new predictive computation rather "
            "than merely re-expressing information already present. The tuned lens cannot fully "
            "recover the final predictions from early representations alone.\n\n"
        )

    report += """### Tuned lens vs raw logit lens

If the tuned lens significantly outperforms the raw logit lens at early layers, this
indicates that the information *is present* in the residual stream but is not yet in
the format that the final unembedding expects. The tuned lens "decodes" this latent
information by learning the correct affine transformation, whereas the raw logit lens
fails because it applies the final-layer readout to an intermediate representation
that has not yet been aligned with the unembedding matrix.

### Tension with belief-state probe R²

"""

    best_r2 = r2_per_layer[best_r2_layer]
    report += (
        f"The belief-state probe achieves its highest R² at layer {best_r2_layer} "
        f"(R²={best_r2:.4f}). "
    )

    report += """
If belief-state R² peaks early but declines in later layers, while the tuned lens
continues to improve, this apparent tension can be explained as follows:

- **Belief-state R² measures linear decodability of HMM beliefs**, which may peak
  when the residual stream geometry most closely mirrors the belief simplex.
- **The tuned lens measures reconstruction of the model's full output distribution**,
  which requires not just belief information but also the correct formatting for the
  unembedding matrix.
- Later layers may transform the representation into a form that is better aligned
  with the unembedding but less linearly aligned with the belief simplex geometry.
  The information is not lost — it is re-encoded.
- This does NOT necessarily mean later layers add new *information*; they may add
  new *computation* that reformats existing information for the final readout.

**Caution**: These observations are specific to the tested HMM process and model.
Generalization to other processes or models should not be assumed without additional
experiments.

## Saved Artifacts

- `metrics.json` — Per-layer evaluation metrics
- `per_position_metrics.npz` — Per-position KL arrays for all layers
- `training_losses.json` — Training loss curves
- `translators/` — Saved translator state dicts (one per layer)
- `config.json` — Experiment configuration
- `figures/` — All plots
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)
