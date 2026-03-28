"""Main orchestration pipeline for the later-layer computation experiment.

Ties together activation caching, decomposition, decoder analysis,
causal intervention, interpretation ranking, and report generation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .cache import CachedActivations, collect_activations, save_cache, load_cache
from .config import HMMEntry, LaterLayerConfig, resolve_vocab_tokens
from .decomposition import build_decompositions, DecompositionResult
from .decoders import compute_targets, run_decoders, DecoderResults
from .intervention import run_ablation_sweep, InterventionResults
from .analysis import rank_interpretations, generate_report
from .plotting import (
    plot_probe_r2_and_variance,
    plot_decoder_r2,
    plot_ablation_sweep,
    plot_token_position_effects,
    plot_summary_comparison,
)


def run_pipeline(
    entry: HMMEntry,
    model,
    config: LaterLayerConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Run the full later-layer computation investigation for one HMM config.

    Returns a summary dict suitable for JSON serialisation.
    """
    from simplexity.generative_processes.builder import build_hidden_markov_model

    seq_length = entry.seq_length or config.seq_length
    n_sequences = entry.n_sequences or config.n_sequences
    layer_indices = config.layer_indices

    hmm_temp = build_hidden_markov_model(
        entry.process_name, process_params=entry.process_params, device=None,
    )
    vocab_tokens = resolve_vocab_tokens(hmm_temp, entry, config.default_vocab_tokens)
    n_states = hmm_temp.num_states

    label = f"{entry.process_name}_" + "_".join(
        f"{k}{v}" for k, v in sorted(entry.process_params.items())
    )
    config_dir = out_dir / "configs" / label
    (config_dir / "figures").mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Config: {label}")
    logger.info(f"Process: {entry.process_name}, params: {entry.process_params}")
    logger.info(f"Vocab: {vocab_tokens} ({hmm_temp.vocab_size} symbols, {n_states} states)")
    logger.info(f"Sequences: {n_sequences} × {seq_length}")
    logger.info(f"{'=' * 60}")

    # ── Phase 0: Activation collection ────────────────────────────────────────
    cache_path = config_dir / "cached_activations.npz"
    if cache_path.exists():
        logger.info("Loading cached activations from disk...")
        cached = load_cache(
            cache_path, vocab_tokens, entry.process_name,
            entry.process_params, n_sequences, seq_length,
        )
    else:
        logger.info("Collecting activations (forward passes)...")
        cached = collect_activations(
            process_name=entry.process_name,
            process_params=entry.process_params,
            vocab_tokens=vocab_tokens,
            model=model,
            layer_indices=layer_indices,
            n_sequences=n_sequences,
            seq_length=seq_length,
            random_seed=config.random_seed,
            logger=logger,
        )
        logger.info(f"Saving activation cache to {cache_path}")
        save_cache(cached, cache_path)

    # ── Phase 1 (Part A): Belief subspace decomposition ───────────────────────
    logger.info("Building belief subspace decompositions...")
    decomposition = build_decompositions(
        activations=cached.activations,
        belief_states=cached.belief_states,
        layer_indices=layer_indices,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    for layer in layer_indices:
        ld = decomposition.layers[layer]
        logger.info(
            f"  Layer {layer:2d}: R²={ld.probe_r2:.4f}  "
            f"var_belief={ld.var_belief:.4f}  var_orth={ld.var_orth:.4f}"
        )

    # Free the raw activations now that decomposition holds h_belief/h_orth
    import gc
    del cached.activations
    gc.collect()

    plot_probe_r2_and_variance(
        decomposition, layer_indices,
        config_dir / "figures" / "probe_r2_and_variance.png",
    )

    # ── Phase 2 (Part B): Predictive residual analysis ────────────────────────
    logger.info("Computing prediction targets...")
    hmm_for_targets = build_hidden_markov_model(
        entry.process_name, process_params=entry.process_params, device=None,
    )
    targets = compute_targets(
        concept_logits=cached.concept_logits,
        obs_probs=cached.obs_probs,
        belief_states=cached.belief_states,
        token_positions=cached.token_positions,
        hmm=hmm_for_targets,
    )

    logger.info("Running decoder analysis...")
    decoder_results = run_decoders(
        decomposition=decomposition,
        targets=targets,
        layer_indices=layer_indices,
        test_size=config.decoder_test_size,
        n_pca_components=n_states,
    )

    for layer in layer_indices:
        lr = decoder_results.layers[layer]
        orth_logit = lr.scores.get("orth", {}).get("concept_logits")
        belief_logit = lr.scores.get("belief", {}).get("concept_logits")
        if orth_logit and belief_logit:
            logger.info(
                f"  Layer {layer:2d}: R²(concept_logits) "
                f"belief={belief_logit.r2:.4f}  orth={orth_logit.r2:.4f}"
            )

    plot_decoder_r2(
        decoder_results, layer_indices,
        config_dir / "figures" / "decoder_r2.png",
    )

    # ── Phase 3 (Part C): Causal intervention ─────────────────────────────────
    causal_n = min(config.causal_n_sequences, n_sequences)
    causal_layers = _select_causal_layers(layer_indices)

    logger.info(
        f"Running causal interventions ({len(causal_layers)} layers, "
        f"{causal_n} sequences)..."
    )
    try:
        intervention_results = run_ablation_sweep(
            model=model,
            decomposition=decomposition,
            process_name=entry.process_name,
            process_params=entry.process_params,
            vocab_tokens=vocab_tokens,
            layer_indices=causal_layers,
            n_sequences=causal_n,
            seq_length=seq_length,
            random_seed=config.random_seed,
            logger=logger,
        )
    except Exception as e:
        import traceback
        logger.error(f"Causal intervention failed: {e}")
        logger.error(traceback.format_exc())
        intervention_results = InterventionResults(results=[])

    if intervention_results.results:
        plot_ablation_sweep(
            intervention_results, causal_layers,
            config_dir / "figures" / "ablation_sweep.png",
        )

    # ── Phase 4 (Part D): Interpretation ranking ──────────────────────────────
    logger.info("Ranking interpretations...")
    interpretations = rank_interpretations(
        decoder_results, intervention_results, layer_indices,
    )
    for i, interp in enumerate(interpretations, 1):
        logger.info(f"  #{i}: {interp.name} (score={interp.score:.3f})")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_token_position_effects(
        decoder_results, decomposition, cached.token_positions,
        layer_indices, config_dir / "figures" / "token_position_effects.png",
    )
    plot_summary_comparison(
        decoder_results, intervention_results, decomposition,
        layer_indices, config_dir / "figures" / "summary_comparison.png",
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = generate_report(
        interpretations=interpretations,
        decoder_results=decoder_results,
        intervention_results=intervention_results,
        decomposition_result=decomposition,
        layer_indices=layer_indices,
        config_label=label,
        process_name=entry.process_name,
        process_params=entry.process_params,
    )
    report_path = config_dir / "report.md"
    report_path.write_text(report)
    logger.info(f"Report written to {report_path}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = _build_metrics(
        decomposition, decoder_results, intervention_results,
        interpretations, layer_indices, label, entry,
    )
    with open(config_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _select_causal_layers(layer_indices: list[int]) -> list[int]:
    """Select a subset of layers for causal intervention to save compute.

    Picks every 3rd layer plus the first and last.
    """
    selected = set()
    selected.add(layer_indices[0])
    selected.add(layer_indices[-1])
    for i in range(0, len(layer_indices), 3):
        selected.add(layer_indices[i])
    return sorted(selected)


def _build_metrics(decomposition, decoder_results, intervention_results,
                   interpretations, layer_indices, label, entry) -> dict:
    metrics = {
        "label": label,
        "process_name": entry.process_name,
        "process_params": entry.process_params,
        "probe_r2_by_layer": {
            str(l): decomposition.layers[l].probe_r2 for l in layer_indices
        },
        "var_belief_by_layer": {
            str(l): decomposition.layers[l].var_belief for l in layer_indices
        },
        "var_orth_by_layer": {
            str(l): decomposition.layers[l].var_orth for l in layer_indices
        },
        "decoder_r2": {},
        "ablation": intervention_results.to_dict(),
        "interpretations": [
            {"name": i.name, "score": i.score,
             "evidence_for": i.evidence_for,
             "evidence_against": i.evidence_against}
            for i in interpretations
        ],
    }

    for layer in layer_indices:
        lr = decoder_results.layers.get(layer)
        if lr is None:
            continue
        layer_scores = {}
        for comp, tgt_scores in lr.scores.items():
            for tgt, score in tgt_scores.items():
                key = f"{comp}/{tgt}"
                if key not in layer_scores:
                    layer_scores[key] = {}
                layer_scores[key] = {"r2": score.r2, "mse": score.mse}
        metrics["decoder_r2"][str(layer)] = layer_scores

    return metrics
