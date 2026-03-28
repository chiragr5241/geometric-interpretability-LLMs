"""Part D: Interpretation ranking and report generation.

Ranks candidate interpretations of later-layer computation based on
evidence from decoder results (Part B) and causal interventions (Part C).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decoders import DecoderResults, COMPONENT_NAMES, TARGET_NAMES
from .intervention import InterventionResults


@dataclass
class Interpretation:
    name: str
    description: str
    evidence_for: list[str]
    evidence_against: list[str]
    score: float  # higher = more supported


def rank_interpretations(
    decoder_results: DecoderResults,
    intervention_results: InterventionResults,
    layer_indices: list[int],
) -> list[Interpretation]:
    """Rank candidate interpretations by evidence strength."""

    interpretations = []

    # Collect summary statistics
    orth_r2 = {}   # target -> mean R² of orth decoder across layers
    belief_r2 = {}
    full_r2 = {}
    orth_pca_r2 = {}

    for tgt in TARGET_NAMES:
        orth_vals, belief_vals, full_vals, pca_vals = [], [], [], []
        for layer in layer_indices:
            lr = decoder_results.layers.get(layer)
            if lr is None:
                continue
            for comp, vals in [("orth", orth_vals), ("belief", belief_vals),
                               ("full", full_vals), ("orth_pca", pca_vals)]:
                s = lr.scores.get(comp, {}).get(tgt)
                if s is not None:
                    vals.append(s.r2)
        orth_r2[tgt] = np.mean(orth_vals) if orth_vals else 0
        belief_r2[tgt] = np.mean(belief_vals) if belief_vals else 0
        full_r2[tgt] = np.mean(full_vals) if full_vals else 0
        orth_pca_r2[tgt] = np.mean(pca_vals) if pca_vals else 0

    # Ablation statistics
    ablate_orth_kl = []
    ablate_belief_kl = []
    mean_ablate_orth_kl = []
    for r in intervention_results.results:
        if r.condition == "ablate_orth":
            ablate_orth_kl.append(r.kl_vs_original_mean)
        elif r.condition == "ablate_belief":
            ablate_belief_kl.append(r.kl_vs_original_mean)
        elif r.condition == "mean_ablate_orth":
            mean_ablate_orth_kl.append(r.kl_vs_original_mean)

    avg_ablate_orth = np.mean(ablate_orth_kl) if ablate_orth_kl else 0
    avg_ablate_belief = np.mean(ablate_belief_kl) if ablate_belief_kl else 0

    # Late-layer vs early-layer ablation effect
    n_third = max(1, len(layer_indices) // 3)
    early_layers = layer_indices[:n_third]
    late_layers = layer_indices[-n_third:]

    early_orth_kl = [r.kl_vs_original_mean for r in intervention_results.results
                     if r.condition == "ablate_orth" and r.layer in early_layers]
    late_orth_kl = [r.kl_vs_original_mean for r in intervention_results.results
                    if r.condition == "ablate_orth" and r.layer in late_layers]

    # --- 1. Output readout / logit refinement ---
    ev_for, ev_against = [], []
    logit_orth = orth_r2.get("concept_logits", 0)
    hmm_orth = orth_r2.get("hmm_next_token", 0)
    if logit_orth > 0.1:
        ev_for.append(f"Orth component predicts concept logits (R²={logit_orth:.3f})")
    if hmm_orth < 0.1:
        ev_for.append(f"Orth component does NOT predict HMM probs (R²={hmm_orth:.3f})")
    else:
        ev_against.append(f"Orth also predicts HMM probs (R²={hmm_orth:.3f})")
    if avg_ablate_orth > 0.01:
        ev_for.append(f"Ablating orth changes logits (mean KL={avg_ablate_orth:.4f})")
    score = logit_orth * 0.5 + (1 - hmm_orth) * 0.3 + min(avg_ablate_orth * 10, 0.2)
    interpretations.append(Interpretation(
        name="Output readout / logit refinement",
        description="Later layers reformat belief state information into the "
                    "unembedding-compatible format needed for token prediction.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    # --- 2. Uncertainty tracking ---
    ev_for, ev_against = [], []
    ent_orth = orth_r2.get("belief_entropy", 0)
    ent_belief = belief_r2.get("belief_entropy", 0)
    if ent_orth > 0.1:
        ev_for.append(f"Orth component predicts belief entropy (R²={ent_orth:.3f})")
    else:
        ev_against.append(f"Orth does not predict entropy well (R²={ent_orth:.3f})")
    if ent_belief > ent_orth:
        ev_against.append("Belief component predicts entropy better")
    score = ent_orth * 0.7 + (ent_orth > ent_belief) * 0.3
    interpretations.append(Interpretation(
        name="Uncertainty tracking",
        description="Later layers encode a representation of predictive "
                    "uncertainty beyond what the belief state provides.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    # --- 3. Multi-step predictive information ---
    ev_for, ev_against = [], []
    ms2_orth = orth_r2.get("multi_step_2", 0)
    ms4_orth = orth_r2.get("multi_step_4", 0)
    ms2_belief = belief_r2.get("multi_step_2", 0)
    ms4_belief = belief_r2.get("multi_step_4", 0)
    if ms2_orth > ms2_belief + 0.01:
        ev_for.append(f"Orth predicts 2-step better than belief ({ms2_orth:.3f} vs {ms2_belief:.3f})")
    else:
        ev_against.append(f"Belief predicts 2-step as well or better")
    if ms4_orth > ms4_belief + 0.01:
        ev_for.append(f"Orth predicts 4-step better than belief ({ms4_orth:.3f} vs {ms4_belief:.3f})")
    score = max(ms2_orth - ms2_belief, 0) * 0.5 + max(ms4_orth - ms4_belief, 0) * 0.5
    interpretations.append(Interpretation(
        name="Multi-step predictive information",
        description="Later layers encode future predictions beyond one-step "
                    "that are not captured by the current belief state.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    # --- 4. Compression / redistribution ---
    ev_for, ev_against = [], []
    if late_orth_kl and early_orth_kl:
        late_avg = np.mean(late_orth_kl)
        early_avg = np.mean(early_orth_kl)
        if late_avg > early_avg * 1.5:
            ev_for.append(
                f"Ablating orth costs more in late layers "
                f"({late_avg:.4f} vs {early_avg:.4f})"
            )
        else:
            ev_against.append(
                f"Ablation cost similar across layers "
                f"(late={late_avg:.4f}, early={early_avg:.4f})"
            )
    score = 0.3 if ev_for else 0.1
    interpretations.append(Interpretation(
        name="Compression / redistribution",
        description="Later layers compress or redistribute information "
                    "across the residual stream, increasing importance "
                    "of non-belief directions.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    # --- 5. Synchronization-progress information ---
    ev_for, ev_against = [], []
    pos_orth = orth_r2.get("token_position", 0)
    pos_belief = belief_r2.get("token_position", 0)
    if pos_orth > 0.1:
        ev_for.append(f"Orth predicts token position (R²={pos_orth:.3f})")
    else:
        ev_against.append(f"Orth does not encode position well (R²={pos_orth:.3f})")
    score = pos_orth * 0.7
    interpretations.append(Interpretation(
        name="Synchronization-progress information",
        description="Later layers track how far along the sequence the model "
                    "has progressed, encoding synchronization depth.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    # --- 6. Residual correction ---
    ev_for, ev_against = [], []
    res_orth = orth_r2.get("logit_residual", 0)
    if res_orth > 0.1:
        ev_for.append(
            f"Orth predicts logit residual (what belief can't explain) "
            f"with R²={res_orth:.3f}"
        )
    else:
        ev_against.append(f"Orth doesn't predict logit residual well (R²={res_orth:.3f})")
    if avg_ablate_orth > 0.01:
        ev_for.append(f"Ablating orth has causal effect (KL={avg_ablate_orth:.4f})")
    score = res_orth * 0.6 + min(avg_ablate_orth * 10, 0.4)
    interpretations.append(Interpretation(
        name="Residual correction on top of belief state",
        description="Later layers apply a correction to predictions that "
                    "goes beyond what the belief-state probe captures, "
                    "possibly encoding model-specific learned biases.",
        evidence_for=ev_for, evidence_against=ev_against, score=score,
    ))

    interpretations.sort(key=lambda x: x.score, reverse=True)
    return interpretations


def generate_report(
    interpretations: list[Interpretation],
    decoder_results: DecoderResults,
    intervention_results: InterventionResults,
    decomposition_result,
    layer_indices: list[int],
    config_label: str,
    process_name: str,
    process_params: dict,
) -> str:
    """Generate a markdown report summarising all findings."""

    lines = [
        f"# Later-Layer Computation Investigation: {config_label}",
        "",
        "## Experiment Configuration",
        "",
        f"- **HMM process**: `{process_name}`",
        f"- **Parameters**: `{process_params}`",
        f"- **Layers analysed**: {min(layer_indices)}–{max(layer_indices)} "
        f"({len(layer_indices)} layers)",
        "",
        "## Part A: Belief Subspace Decomposition",
        "",
        "At each layer, a linear probe was trained to predict HMM belief states "
        "from residual-stream activations. The probe weight matrix defines a "
        "belief-relevant subspace (via QR decomposition). Each activation is "
        "decomposed into a belief-aligned component (h_belief) and an orthogonal "
        "complement (h_orth).",
        "",
        "| Layer | Probe R² | Variance (belief) | Variance (orth) |",
        "|-------|----------|-------------------|-----------------|",
    ]

    for layer in layer_indices:
        ld = decomposition_result.layers[layer]
        lines.append(
            f"| {layer:2d} | {ld.probe_r2:.4f} | "
            f"{ld.var_belief:.4f} | {ld.var_orth:.4f} |"
        )

    lines += [
        "",
        "## Part B: Predictive Residual Analysis",
        "",
        "Linear decoders were trained from each component to predict multiple targets. "
        "The key question: *what can h_orth predict that h_belief cannot?*",
        "",
    ]

    # Summary table: averaged across layers
    lines.append("### Mean R² across layers (by component × target)")
    lines.append("")
    header = "| Target |"
    sep = "|--------|"
    for comp in COMPONENT_NAMES:
        header += f" {comp} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for tgt in TARGET_NAMES:
        row = f"| {tgt} |"
        for comp in COMPONENT_NAMES:
            vals = []
            for layer in layer_indices:
                lr = decoder_results.layers.get(layer)
                if lr is None:
                    continue
                s = lr.scores.get(comp, {}).get(tgt)
                if s is not None:
                    vals.append(s.r2)
            mean_r2 = np.mean(vals) if vals else float("nan")
            row += f" {mean_r2:.4f} |"
        lines.append(row)

    lines += [
        "",
        "## Part C: Causal Interventions",
        "",
        "Projection ablation: at each layer, the orthogonal or belief component "
        "was ablated during the forward pass, and downstream logit changes were measured.",
        "",
        "| Layer | Condition | KL(orig‖abl) | KL(HMM‖abl) | KL(HMM‖orig) | Top-1 (abl) | Top-1 (orig) |",
        "|-------|-----------|-------------|-------------|--------------|-------------|-------------|",
    ]

    for r in intervention_results.results:
        lines.append(
            f"| {r.layer:2d} | {r.condition} | "
            f"{r.kl_vs_original_mean:.4f} | {r.kl_vs_hmm_mean:.4f} | "
            f"{r.kl_vs_hmm_baseline:.4f} | {r.top1_accuracy_ablated:.3f} | "
            f"{r.top1_accuracy_original:.3f} |"
        )

    lines += [
        "",
        "## Part D: Interpretation Ranking",
        "",
        "Candidate interpretations ranked by evidence strength:",
        "",
    ]

    for i, interp in enumerate(interpretations, 1):
        lines.append(f"### {i}. {interp.name} (score: {interp.score:.3f})")
        lines.append("")
        lines.append(f"*{interp.description}*")
        lines.append("")
        if interp.evidence_for:
            lines.append("**Evidence for:**")
            for e in interp.evidence_for:
                lines.append(f"- {e}")
        if interp.evidence_against:
            lines.append("**Evidence against:**")
            for e in interp.evidence_against:
                lines.append(f"- {e}")
        lines.append("")

    lines += [
        "## Conclusions",
        "",
        "### Is the later-layer signal merely a reformatted belief state?",
        "",
        _conclusion_paragraph(interpretations, intervention_results, layer_indices),
        "",
        "### Main limitations",
        "",
        "- The belief subspace is very low-dimensional (n_states) relative to "
        "d_model, giving the orthogonal complement far more capacity. "
        "The orth_pca control (matching dimensionality) should be checked "
        "for all claims about orthogonal predictive power.",
        "- Ablation creates out-of-distribution activations. The mean_ablate_orth "
        "condition partially addresses this but cannot fully eliminate it.",
        "- Results are specific to the tested HMM process and model.",
        "",
    ]

    return "\n".join(lines)


def _conclusion_paragraph(interpretations, intervention_results, layer_indices) -> str:
    """Generate a plain-English conclusion based on the evidence."""
    top = interpretations[0] if interpretations else None
    if top is None:
        return "Insufficient evidence to draw conclusions."

    ablate_orth_effects = [
        r.kl_vs_original_mean for r in intervention_results.results
        if r.condition == "ablate_orth"
    ]
    mean_effect = np.mean(ablate_orth_effects) if ablate_orth_effects else 0

    if mean_effect < 0.005:
        return (
            "The causal evidence suggests that ablating the orthogonal component "
            "has minimal effect on downstream predictions (mean KL < 0.005). "
            "This is consistent with the later-layer signal being primarily a "
            "reformatted version of the belief state, rather than genuinely "
            "additional computation. The top-ranked interpretation is: "
            f"**{top.name}**."
        )
    else:
        return (
            f"The causal evidence suggests that the orthogonal component carries "
            f"genuine predictive information (mean ablation KL = {mean_effect:.4f}). "
            f"This goes beyond mere reformatting of the belief state. "
            f"The strongest interpretation is: **{top.name}**."
        )
