"""Combine R² + tuned-lens KL per layer across all configs of each process.

Reads every metrics.json under outputs/SPAR/, groups configs by process
(wing, strata, arch, spiral, mess3), and emits one dual-axis figure per
process with all configs overlaid.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SPAR = Path("/u/chirag5241/comp_mech_test/geometric-interpretability-LLMs/outputs/SPAR")
OUT = SPAR / "r2_kl_per_process"
OUT.mkdir(exist_ok=True)

PROCESSES = ["wing", "strata", "arch", "spiral", "mess3"]


def process_of(config_name: str) -> str | None:
    for p in PROCESSES:
        if config_name.startswith(p + "_") or config_name == p:
            return p
    return None


def param_key(process: str, config_name: str) -> tuple[float, ...]:
    """Extract numeric parameters (in order) from a config name for sorting."""
    suffix = config_name[len(process) + 1:] if config_name.startswith(process + "_") else config_name
    nums = re.findall(r"[-+]?\d*\.?\d+", suffix)
    return tuple(float(n) for n in nums)


def load_run_configs() -> dict[str, list[tuple[str, Path]]]:
    """Return {process: [(config_name, metrics_path), ...]} deduped across runs.

    When the same config name appears in multiple runs, prefer the most
    recent run directory (timestamped names sort after 'zoo_a40').
    Configs are sorted by their numeric parameter values (ascending).
    """
    candidates: dict[str, dict[str, Path]] = defaultdict(dict)
    for metrics_file in SPAR.rglob("metrics.json"):
        # Path: .../SPAR/<run>/(configs/)?<config>/metrics.json
        rel = metrics_file.relative_to(SPAR).parts
        run = rel[0]
        cfg = rel[2] if len(rel) >= 4 and rel[1] == "configs" else rel[1]
        proc = process_of(cfg)
        if proc is None:
            continue
        # Preserve the run with the lexicographically later name (newer timestamp).
        existing = candidates[proc].get(cfg)
        if existing is None or run > existing.parts[len(SPAR.parts)]:
            candidates[proc][cfg] = metrics_file

    return {
        p: sorted(d.items(), key=lambda kv: param_key(p, kv[0]))
        for p, d in candidates.items()
    }


def plot_process(
    process: str,
    configs: list[tuple[str, Path]],
    *,
    include_final_vs_tuned: bool,
    out_path: Path,
) -> None:
    if not configs:
        return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    # Per-metric colormap; shade encodes the parameter value
    # (light = smallest param → dark = largest param).
    # Avoid the very lightest end so lines stay visible on white.
    n = len(configs)
    def shades(name: str) -> list:
        cm = plt.get_cmap(name)
        return [cm(0.30 + 0.65 * (i / max(n - 1, 1))) for i in range(n)]
    r2_shades = shades("Greys")
    kl_tuned_shades = shades("Blues")
    kl_tuned_hmm_shades = shades("Greens")
    kl_hmm_final_shades = shades("Reds")
    kl_final_tuned_shades = shades("Oranges")

    for i, (cfg, mpath) in enumerate(configs):
        with open(mpath) as f:
            metrics = json.load(f)
        layers = [m["layer"] for m in metrics]
        r2 = [m["r2_belief_probe"] for m in metrics]
        kl_tuned = [m["kl_hmm_vs_tuned"] for m in metrics]
        kl_tuned_hmm = [m["kl_hmm_vs_tuned_hmm"] for m in metrics]
        # KL(HMM || model_final): tuned_lens at the top layer collapses to
        # the model's own final distribution, so kl_hmm_vs_tuned[-1] equals
        # KL(HMM || model_final). Equivalently, kl_hmm_vs_logit[-1].
        kl_hmm_final = metrics[-1]["kl_hmm_vs_tuned"]
        kl_final_tuned = [m["kl_final_vs_tuned"] for m in metrics]

        # Strip process prefix from label for compactness.
        label = re.sub(rf"^{process}_", "", cfg)

        ax1.plot(layers, r2, "o-", color=r2_shades[i], linewidth=1.8, label=label)
        ax2.plot(layers, kl_tuned, "s--", color=kl_tuned_shades[i], linewidth=1.4, alpha=0.9)
        ax2.plot(layers, kl_tuned_hmm, "^:", color=kl_tuned_hmm_shades[i], linewidth=1.4, alpha=0.9)
        ax2.axhline(kl_hmm_final, color=kl_hmm_final_shades[i], linewidth=1.2,
                    linestyle="-.", alpha=0.85)
        if include_final_vs_tuned:
            ax2.plot(layers, kl_final_tuned, "D-", color=kl_final_tuned_shades[i],
                     linewidth=1.4, alpha=0.9, markersize=4)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("R² (linear probe → belief state)  [left axis, black]")
    right_label = (
        "KL  [right axis;  blue=HMM‖tuned,  green=HMM‖tuned_hmm,  "
        "red=HMM‖model_final"
    )
    if include_final_vs_tuned:
        right_label += ",  orange=model_final‖tuned"
    right_label += "]"
    ax2.set_ylabel(right_label)
    ax1.grid(True, alpha=0.3)
    title = (
        f"Belief-State R² vs Tuned-Lens KL per Layer — {process} "
        f"({len(configs)} configs; darker shade = larger param)"
    )
    if include_final_vs_tuned:
        title += "  [+ KL(model_final‖tuned)]"
    ax1.set_title(title)

    # Legend by config (parameter ordering), plus a metric-color key.
    cfg_legend = ax1.legend(title="config", fontsize=8, loc="lower right", ncol=1)
    style_handles = [
        plt.Line2D([], [], color="black", marker="o", linestyle="-",
                   label="R² — left axis"),
        plt.Line2D([], [], color="steelblue", marker="s", linestyle="--",
                   label="KL(HMM‖tuned) — right axis"),
        plt.Line2D([], [], color="seagreen", marker="^", linestyle=":",
                   label="KL(HMM‖tuned_hmm) — right axis"),
        plt.Line2D([], [], color="firebrick", linestyle="-.",
                   label="KL(HMM‖model_final) — right axis"),
    ]
    if include_final_vs_tuned:
        style_handles.append(
            plt.Line2D([], [], color="darkorange", marker="D", linestyle="-",
                       label="KL(model_final‖tuned) — right axis")
        )
    ax2.legend(handles=style_handles, fontsize=8, loc="upper left")
    ax1.add_artist(cfg_legend)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(configs)} configs)")


def main() -> None:
    grouped = load_run_configs()
    for process in PROCESSES:
        cfgs = grouped.get(process, [])
        print(f"{process}: {len(cfgs)} configs")
        plot_process(
            process, cfgs,
            include_final_vs_tuned=False,
            out_path=OUT / f"r2_kl_{process}.png",
        )
        plot_process(
            process, cfgs,
            include_final_vs_tuned=True,
            out_path=OUT / f"r2_kl_{process}_with_final_tuned.png",
        )


if __name__ == "__main__":
    main()
