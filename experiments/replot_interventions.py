"""Regenerate plots from a completed single_seq_interventions run.

Usage:
    python replot_interventions.py <out_dir>

Loads the phase_c_checkpoint.json and aggregated_plot_data.json from <out_dir>,
re-runs the aggregation and all plotting functions. No model loading required.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ── Bootstrap path so we can import from the experiments dir ──────────────────
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from single_seq_interventions import (
    PATCH_CONDITIONS,
    STEER_CONDITIONS,
    PRINCIPLED_CONDITIONS,
    CHECKPOINT_FILENAME,
    _PATCH_COLORS,
    _STEER_COLORS,
    _load_checkpoint,
    _agg_records,
    _agg_paired_diff,
    _plot_kl_vs_layer,
    _plot_kl_minus_control_vs_layer,
    _plot_heatmap,
    _plot_ablation_curve,
    _plot_roundtrip_comparison,
    _plot_crossing,
    _plot_causal_shift,
    _emit_layer_k_effects,
    _emit_kl_minus_control_layer_k,
)
from plot_titles import format_hmm_process


def _load_config(out_dir: Path) -> dict:
    import yaml
    with open(out_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def _agg_cond_k_layer(src: dict, conds: list, k_values: list, layer_indices: list) -> dict:
    return {
        cond: {k: {l: _agg_records(src[cond][k][l]) for l in layer_indices} for k in k_values}
        for cond in conds
    }


def main(out_dir: Path) -> None:
    cfg = _load_config(out_dir)
    layer_indices: list[int] = cfg["layer_indices"]
    k_values: list[int] = cfg["k_values"]
    run_with_controls: bool = cfg.get("run_with_controls", False)
    save_html: bool = cfg.get("save_html", False)

    hmm_subtitle = format_hmm_process(
        cfg["hmm"]["process_name"], cfg["hmm"]["process_params"]
    )

    # Load baseline mean from aggregated data
    with open(out_dir / "plot_data" / "aggregated_plot_data.json") as f:
        plot_data = json.load(f)
    baseline_mean: float = plot_data["baseline_mean"]

    # Load full checkpoint (raw per-seq records)
    chk_path = out_dir / CHECKPOINT_FILENAME
    print(f"Loading checkpoint: {chk_path}")
    chk = _load_checkpoint(
        chk_path,
        PATCH_CONDITIONS,
        STEER_CONDITIONS,
        k_values,
        layer_indices,
        PRINCIPLED_CONDITIONS if run_with_controls else None,
    )

    patch_kl_to_target = chk["patch_kl_to_target"]
    patch_kl_to_factual = chk["patch_kl_to_factual"]
    patch_kl_to_clean = chk["patch_kl_to_clean"]
    baseline_kl_to_target_patch = chk["baseline_kl_to_target_patch"]
    steer_kl_to_target = chk["steer_kl_to_target"]
    steer_kl_to_factual = chk["steer_kl_to_factual"]
    steer_kl_to_clean = chk["steer_kl_to_clean"]
    baseline_kl_to_target_steer = chk["baseline_kl_to_target_steer"]
    ablation_kl = chk["ablation_kl"]
    ablation_kl_to_clean = chk["ablation_kl_to_clean"]

    if run_with_controls:
        patch_control_kl_to_target = chk["patch_control_kl_to_target"]
        patch_control_kl_to_factual = chk["patch_control_kl_to_factual"]
        steer_control_kl_to_target = chk["steer_control_kl_to_target"]
        steer_control_kl_to_factual = chk["steer_control_kl_to_factual"]

    # Aggregate
    print("Aggregating ...")
    agg_patch_to_target = _agg_cond_k_layer(patch_kl_to_target, PATCH_CONDITIONS, k_values, layer_indices)
    agg_patch_to_factual = _agg_cond_k_layer(patch_kl_to_factual, PATCH_CONDITIONS, k_values, layer_indices)
    agg_bl_to_target_patch = _agg_cond_k_layer(baseline_kl_to_target_patch, PATCH_CONDITIONS, k_values, layer_indices)
    agg_steer_to_target = _agg_cond_k_layer(steer_kl_to_target, STEER_CONDITIONS, k_values, layer_indices)
    agg_steer_to_factual = _agg_cond_k_layer(steer_kl_to_factual, STEER_CONDITIONS, k_values, layer_indices)
    agg_bl_to_target_steer = _agg_cond_k_layer(baseline_kl_to_target_steer, STEER_CONDITIONS, k_values, layer_indices)

    agg_ablation = {
        "belief": {l: _agg_records(ablation_kl["belief"][l]) for l in layer_indices},
        "random": {l: _agg_records(ablation_kl["random"][l]) for l in layer_indices},
    }
    agg_ablation_to_clean = {
        "belief": {l: _agg_records(ablation_kl_to_clean["belief"][l]) for l in layer_indices},
        "random": {l: _agg_records(ablation_kl_to_clean["random"][l]) for l in layer_indices},
    }

    if run_with_controls:
        agg_patch_diff_to_target: dict = {}
        agg_patch_diff_to_factual: dict = {}
        agg_steer_diff_to_target: dict = {}
        agg_steer_diff_to_factual: dict = {}
        for cond in PRINCIPLED_CONDITIONS:
            agg_patch_diff_to_target[cond] = {
                k: {l: _agg_paired_diff(patch_kl_to_target[cond][k][l], patch_control_kl_to_target[cond][k][l])
                    for l in layer_indices}
                for k in k_values
            }
            agg_patch_diff_to_factual[cond] = {
                k: {l: _agg_paired_diff(patch_kl_to_factual[cond][k][l], patch_control_kl_to_factual[cond][k][l])
                    for l in layer_indices}
                for k in k_values
            }
            agg_steer_diff_to_target[cond] = {
                k: {l: _agg_paired_diff(steer_kl_to_target[cond][k][l], steer_control_kl_to_target[cond][k][l])
                    for l in layer_indices}
                for k in k_values
            }
            agg_steer_diff_to_factual[cond] = {
                k: {l: _agg_paired_diff(steer_kl_to_factual[cond][k][l], steer_control_kl_to_factual[cond][k][l])
                    for l in layer_indices}
                for k in k_values
            }

    # Plots
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots -> {fig_dir}")

    k_max = max(k_values)

    for k in k_values:
        _plot_kl_vs_layer(
            {cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            layer_indices, k=k, baseline_mean=baseline_mean,
            title="Activation patching: KL vs layer by condition",
            hmm_subtitle=hmm_subtitle, colors=_PATCH_COLORS,
            path=fig_dir / f"patching_kl_vs_layer_k{k}", save_html=save_html,
        )
        if run_with_controls:
            _plot_kl_minus_control_vs_layer(
                {cond: agg_patch_diff_to_target[cond][k] for cond in PRINCIPLED_CONDITIONS},
                layer_indices, k=k,
                title="Activation patching: KL_to_target − KL_control_to_target",
                hmm_subtitle=hmm_subtitle,
                colors={c: _PATCH_COLORS[c] for c in PRINCIPLED_CONDITIONS},
                path=fig_dir / f"patching_kl_minus_control_vs_layer_k{k}", save_html=save_html,
            )
        for cross_cond in ["past_consistent", "random"]:
            _plot_crossing(
                agg_to_factual=agg_patch_to_factual[cross_cond][k],
                agg_to_target=agg_patch_to_target[cross_cond][k],
                baseline_to_factual=baseline_mean,
                agg_baseline_to_target=agg_bl_to_target_patch[cross_cond][k],
                layer_indices=layer_indices, k=k, condition=cross_cond,
                title="Activation patching", hmm_subtitle=hmm_subtitle,
                path=fig_dir / f"patching_crossing_{cross_cond}_k{k}", save_html=save_html,
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_patch_to_target[cond][k] for cond in PATCH_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_patch[cond][k] for cond in PATCH_CONDITIONS},
            layer_indices=layer_indices, k=k, title="Activation patching",
            hmm_subtitle=hmm_subtitle, colors=_PATCH_COLORS,
            path=fig_dir / f"patching_causal_shift_k{k}", save_html=save_html,
        )

    for k in k_values:
        _plot_kl_vs_layer(
            {cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            layer_indices, k=k, baseline_mean=baseline_mean,
            title="Activation steering: KL vs layer by condition",
            hmm_subtitle=hmm_subtitle, colors=_STEER_COLORS,
            path=fig_dir / f"steering_kl_vs_layer_k{k}", save_html=save_html,
        )
        if run_with_controls:
            _plot_kl_minus_control_vs_layer(
                {cond: agg_steer_diff_to_target[cond][k] for cond in PRINCIPLED_CONDITIONS},
                layer_indices, k=k,
                title="Activation steering: KL_to_target − KL_control_to_target",
                hmm_subtitle=hmm_subtitle,
                colors={c: _STEER_COLORS[c] for c in PRINCIPLED_CONDITIONS},
                path=fig_dir / f"steering_kl_minus_control_vs_layer_k{k}", save_html=save_html,
            )
        for cross_cond in ["past_consistent", "random"]:
            _plot_crossing(
                agg_to_factual=agg_steer_to_factual[cross_cond][k],
                agg_to_target=agg_steer_to_target[cross_cond][k],
                baseline_to_factual=baseline_mean,
                agg_baseline_to_target=agg_bl_to_target_steer[cross_cond][k],
                layer_indices=layer_indices, k=k, condition=cross_cond,
                title="Activation steering", hmm_subtitle=hmm_subtitle,
                path=fig_dir / f"steering_crossing_{cross_cond}_k{k}", save_html=save_html,
            )
        _plot_causal_shift(
            agg_to_target={cond: agg_steer_to_target[cond][k] for cond in STEER_CONDITIONS},
            agg_baseline_to_target={cond: agg_bl_to_target_steer[cond][k] for cond in STEER_CONDITIONS},
            layer_indices=layer_indices, k=k, title="Activation steering",
            hmm_subtitle=hmm_subtitle, colors=_STEER_COLORS,
            path=fig_dir / f"steering_causal_shift_k{k}", save_html=save_html,
        )

    _plot_heatmap(
        agg_patch_to_target["optimal"][k_max], layer_indices,
        title=f"Patching KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"heatmap_patching_optimal_k{k_max}", save_html=save_html,
    )
    _plot_heatmap(
        agg_steer_to_target["optimal"][k_max], layer_indices,
        title=f"Steering KL heatmap — optimal (k={k_max}, log₁₀ scale)",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"heatmap_steering_optimal_k{k_max}", save_html=save_html,
    )

    _plot_ablation_curve(
        agg_ablation["belief"], agg_ablation["random"], layer_indices,
        title="Belief-subspace ablation: KL to optimal",
        subtitle="KL(P_opt ∥ P_ablated) — HMM-token simplex — log scale",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / "ablation_causal_importance", save_html=save_html,
    )
    _plot_ablation_curve(
        agg_ablation_to_clean["belief"], agg_ablation_to_clean["random"], layer_indices,
        title="Belief-subspace ablation: output perturbation (HMM-token + junk)",
        subtitle="KL(P_clean ∥ P_ablated) — HMM-token + junk projection — log scale",
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / "ablation_kl_to_output_projected", save_html=save_html,
    )

    _plot_roundtrip_comparison(
        {cond: agg_patch_to_target[cond][k_max] for cond in ["optimal", "round_trip"]},
        layer_indices, baseline_mean=baseline_mean, k=k_max,
        hmm_subtitle=hmm_subtitle,
        path=fig_dir / f"roundtrip_h2a_vs_h2b_k{k_max}", save_html=save_html,
    )

    _emit_layer_k_effects(
        label="patching", conditions=PATCH_CONDITIONS,
        agg_to_target=agg_patch_to_target,
        agg_baseline_to_target=agg_bl_to_target_patch,
        layer_indices=layer_indices, k_values=k_values,
        fig_dir=fig_dir, hmm_subtitle=hmm_subtitle, save_html=save_html,
    )
    if run_with_controls:
        _emit_kl_minus_control_layer_k(
            label="patching", conditions=PRINCIPLED_CONDITIONS,
            diff_records_by_cond=agg_patch_diff_to_target,
            layer_indices=layer_indices, k_values=k_values,
            fig_dir=fig_dir, hmm_subtitle=hmm_subtitle, save_html=save_html,
        )
    _emit_layer_k_effects(
        label="steering", conditions=STEER_CONDITIONS,
        agg_to_target=agg_steer_to_target,
        agg_baseline_to_target=agg_bl_to_target_steer,
        layer_indices=layer_indices, k_values=k_values,
        fig_dir=fig_dir, hmm_subtitle=hmm_subtitle, save_html=save_html,
    )
    if run_with_controls:
        _emit_kl_minus_control_layer_k(
            label="steering", conditions=PRINCIPLED_CONDITIONS,
            diff_records_by_cond=agg_steer_diff_to_target,
            layer_indices=layer_indices, k_values=k_values,
            fig_dir=fig_dir, hmm_subtitle=hmm_subtitle, save_html=save_html,
        )

    n_figs = len(list(fig_dir.glob("*.png")))
    print(f"Done. {n_figs} PNGs written to {fig_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <out_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())
