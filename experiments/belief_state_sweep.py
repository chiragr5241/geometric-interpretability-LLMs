#!/usr/bin/env python3
"""
Belief state geometry sweep — parameter grid search across HMM processes.

Sweeps over multiple HMM processes (mess3, leopard, etc.) with cartesian product
parameter grids. For each configuration, generates sequences, runs them through
an LLM, computes KL divergence, trains linear regression probes (evaluated by R²),
and produces per-config plots plus a cross-config comparison.

Usage:
    python experiments/belief_state_sweep.py experiments/configs/belief_state_sweep.yaml
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, HMMConfig, apply_runtime_overrides, setup_output_dir
from experiment_utils import get_device, load_model, setup_logging
from visualization import _to_barycentric, _belief_colors

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.generator import (
    generate_data_batch_with_full_history,
)
from simplexity.utils.pytorch_utils import jax_to_torch

from tqdm.auto import tqdm


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class SweepEntry:
    process_name: str
    param_grid: dict[str, list[float]]
    vocab_tokens: list[str] | None = None
    seq_length: int | None = None
    n_sequences: int | None = None


@dataclass
class BeliefStateSweepConfig(ExperimentConfig):
    seq_length: int = 2000
    n_sequences: int = 10
    random_seed: int = 42
    layer_indices: list[int] = field(default_factory=lambda: list(range(28)))
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sweeps: list[SweepEntry] = field(default_factory=list)
    default_vocab_tokens: dict[int, list[str]] = field(
        default_factory=lambda: {2: ["A", "B"], 3: ["A", "B", "C"]}
    )
    n_ctx_override: int | None = None


def load_sweep_config(path: str) -> BeliefStateSweepConfig:
    """Custom config loader that handles nested sweep entries."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    hmm_raw = raw.pop("hmm", {"process_name": "sweep", "process_params": {}})
    probe_raw = raw.pop("probe", {})
    sweeps_raw = raw.pop("sweeps", [])
    default_vocab_raw = raw.pop("default_vocab_tokens", {2: ["A", "B"], 3: ["A", "B", "C"]})

    default_vocab = {int(k): v for k, v in default_vocab_raw.items()}
    hmm = HMMConfig(**hmm_raw)
    probe = ProbeConfig(**probe_raw)
    sweeps = [SweepEntry(**s) for s in sweeps_raw]

    return BeliefStateSweepConfig(
        hmm=hmm,
        probe=probe,
        sweeps=sweeps,
        default_vocab_tokens=default_vocab,
        **raw,
    )


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ConfigResult:
    """Results from a single HMM parameter configuration."""
    process_name: str
    process_params: dict[str, float]
    label: str
    vocab_tokens: list[str]
    belief_states_flat: np.ndarray      # (total_points, n_states)
    kl_mean: np.ndarray                 # (seq_len,)
    kl_std: np.ndarray                  # (seq_len,)
    r2_per_layer: dict[int, float]
    mse_per_layer: dict[int, float]


# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_param_grid(entry: SweepEntry) -> list[dict[str, float]]:
    """Cartesian product of param_grid values."""
    keys = sorted(entry.param_grid.keys())
    values = [entry.param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def make_config_label(process_name: str, params: dict[str, float]) -> str:
    """e.g. 'mess3_a0.6_x0.15'."""
    parts = [process_name]
    for k in sorted(params.keys()):
        parts.append(f"{k}{params[k]}")
    return "_".join(parts)


def resolve_vocab_tokens(
    hmm, entry: SweepEntry, defaults: dict[int, list[str]]
) -> list[str]:
    """Determine token labels for this HMM's vocabulary."""
    if entry.vocab_tokens is not None:
        assert len(entry.vocab_tokens) == hmm.vocab_size
        return entry.vocab_tokens
    if hmm.vocab_size in defaults:
        return defaults[hmm.vocab_size]
    return [chr(65 + i) for i in range(hmm.vocab_size)]


def get_concept_token_ids(model, concepts: list[str]) -> dict[str, int]:
    """Get the LLM token ID for each concept (space-prefixed)."""
    concept_to_id = {}
    for concept in concepts:
        spaced = f" {concept}"
        ids = model.to_tokens(spaced, prepend_bos=False)[0]
        concept_to_id[concept] = ids[-1].item()
    return concept_to_id


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ── Per-config pipeline ──────────────────────────────────────────────────────

def run_single_config(
    process_name: str,
    process_params: dict[str, float],
    vocab_tokens: list[str],
    model,
    config: BeliefStateSweepConfig,
    logger,
    pca_plot_path: Path | None = None,
    seq_length: int | None = None,
    n_sequences: int | None = None,
) -> ConfigResult:
    """Run the full pipeline for one HMM parameter combination."""
    seq_length = seq_length if seq_length is not None else config.seq_length
    n_sequences = n_sequences if n_sequences is not None else config.n_sequences
    label = make_config_label(process_name, process_params)
    n_vocab = len(vocab_tokens)

    # 1. Build HMM
    hmm = build_hidden_markov_model(
        process_name, process_params=process_params, device=None,
    )
    logger.info(f"  HMM: vocab_size={hmm.vocab_size}, num_states={hmm.num_states}")

    # 2. Generate sequences
    # Pass seq_length + 1 so that inputs has exactly seq_length tokens
    sequence_len = seq_length + 1
    key = jax.random.PRNGKey(config.random_seed)
    gen_states = jnp.tile(hmm.initial_state, (n_sequences, 1))

    gen_result = generate_data_batch_with_full_history(
        gen_states, hmm, n_sequences, sequence_len, key,
    )
    inputs_jax = gen_result["inputs"]
    belief_states_jax = gen_result["belief_states"]
    obs_probs_all_jax = jnp.einsum(
        "bns,vst->bnv", belief_states_jax, hmm.transition_matrices
    )

    tokens = jax_to_torch(inputs_jax)
    # belief_states_jax: (n_seq, seq_length, n_states) — aligned with input tokens
    belief_states_all = jax_to_torch(belief_states_jax).cpu().numpy()

    # HMM observation probs: P(next_token | belief)
    obs_probs_all = jax_to_torch(obs_probs_all_jax)

    walk_concepts = [[vocab_tokens[int(t)] for t in seq] for seq in tokens]
    logger.info(
        f"  Generated {n_sequences} sequences of length {seq_length}"
    )

    # 3. Resolve LLM token IDs
    concept_to_id = get_concept_token_ids(model, vocab_tokens)
    concept_ids = [concept_to_id[c] for c in vocab_tokens]
    logger.info(f"  Concept token IDs: {dict(zip(vocab_tokens, concept_ids))}")

    # 4. Forward pass with activation caching
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

            # Manually compute logits (matches notebook approach for
            # multi-GPU compatibility: explicit device transfer).
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

    # 5. KL divergence
    concept_logits = all_logits_flat[:, concept_ids]
    llm_probs = F.softmax(torch.tensor(concept_logits), dim=-1).numpy()

    llm_probs_3d = llm_probs.reshape(
        n_sequences, seq_length, n_vocab
    )
    # obs_probs_all is already aligned with input positions (no offset needed)
    hmm_probs = obs_probs_all.cpu().numpy()

    eps = 1e-10
    p = np.clip(hmm_probs, eps, 1.0)
    q = np.clip(llm_probs_3d, eps, 1.0)

    kl_per_pos_seq = np.sum(p * np.log(p / q), axis=-1)
    kl_mean = kl_per_pos_seq.mean(axis=0)
    kl_std = kl_per_pos_seq.std(axis=0)

    logger.info(f"  Mean KL: {kl_mean.mean():.4f}")

    # 6. Linear probes with R² (primary) and MSE (legacy)
    r2_per_layer: dict[int, float] = {}
    mse_per_layer: dict[int, float] = {}

    for layer in config.layer_indices:
        acts = all_activations[layer]
        beliefs = all_beliefs_flat

        X_train, X_test, y_train, y_test = train_test_split(
            acts,
            beliefs,
            test_size=config.probe.test_size,
            random_state=config.probe.random_state,
        )

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        y_pred_test = reg.predict(X_test)

        mse_test = float(np.mean((y_pred_test - y_test) ** 2))
        r2_test = compute_r2(y_test, y_pred_test)

        r2_per_layer[layer] = r2_test
        mse_per_layer[layer] = mse_test

    best_layer = max(r2_per_layer, key=r2_per_layer.get)
    logger.info(
        f"  Best R²={r2_per_layer[best_layer]:.4f} at layer {best_layer}"
    )

    if pca_plot_path is not None:
        plot_pca(
            all_activations=all_activations,
            belief_states_flat=all_beliefs_flat,
            layers=config.layer_indices,
            title=f"PCA of Residual Stream by Layer (colored by belief) — {label}",
            path=pca_plot_path,
        )

    return ConfigResult(
        process_name=process_name,
        process_params=process_params,
        label=label,
        vocab_tokens=vocab_tokens,
        belief_states_flat=all_beliefs_flat,
        kl_mean=kl_mean,
        kl_std=kl_std,
        r2_per_layer=r2_per_layer,
        mse_per_layer=mse_per_layer,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_belief_simplex(
    belief_states: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot ground truth belief states on a 2D simplex (barycentric coords)."""
    x, y = _to_barycentric(belief_states)
    colors_rgb = _belief_colors(belief_states)
    # Convert "rgb(r,g,b)" strings to matplotlib-compatible tuples
    mpl_colors = []
    for c in colors_rgb:
        parts = c.replace("rgb(", "").replace(")", "").split(",")
        mpl_colors.append((int(parts[0]) / 255, int(parts[1]) / 255, int(parts[2]) / 255))

    sqrt3 = np.sqrt(3)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Simplex outline
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, sqrt3 / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1)

    # Scatter
    ax.scatter(x, y, c=mpl_colors, s=1, alpha=0.5)

    # Vertex labels
    ax.text(-0.06, -0.04, "S0", fontsize=9, ha="center")
    ax.text(1.06, -0.04, "S1", fontsize=9, ha="center")
    ax.text(0.5, sqrt3 / 2 + 0.04, "S2", fontsize=9, ha="center")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, sqrt3 / 2 + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kl_divergence(
    kl_mean: np.ndarray,
    kl_std: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot KL divergence over sequence position."""
    seq_len = len(kl_mean)
    positions = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(positions, kl_mean, linewidth=0.8)
    ax.fill_between(
        positions, kl_mean - kl_std, kl_mean + kl_std, alpha=0.2
    )
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("KL(HMM || LLM)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlim(1, seq_len)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r2_by_layer(
    r2_per_layer: dict[int, float],
    title: str,
    path: Path,
) -> None:
    """Bar chart of R² values per layer."""
    layers = sorted(r2_per_layer.keys())
    values = [r2_per_layer[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(layers)), values, tick_label=[str(l) for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (test set)")
    ax.set_title(title)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca(
    all_activations: dict[int, np.ndarray],
    belief_states_flat: np.ndarray,
    layers: list[int],
    title: str,
    path: Path,
    n_cols: int = 7,
) -> None:
    """PCA of residual stream per layer, colored by belief state. 4x7 grid for 28 layers."""
    n_layers = len(layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    beliefs_rgb = belief_states_flat / (
        belief_states_flat.sum(axis=1, keepdims=True) + 1e-10
    )

    for idx, layer in enumerate(sorted(layers)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        pca = PCA(n_components=6)
        pca_result = pca.fit_transform(all_activations[layer])
        evr = pca.explained_variance_ratio_

        ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=beliefs_rgb,
            s=2,
            alpha=0.4,
            rasterized=True,
        )
        ax.set_xlabel(f"PC0 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC1 ({evr[1]:.1%})")
        ax.set_title(f"Layer {layer}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for idx in range(n_layers, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results: list[ConfigResult], path: Path) -> None:
    """Multi-panel comparison across all configs."""
    # Color palette: group by process
    process_names = sorted(set(r.process_name for r in results))
    process_cmaps = {
        process_names[i]: plt.cm.get_cmap(cmap)
        for i, cmap in enumerate(["Blues", "Oranges", "Greens", "Reds"])
        if i < len(process_names)
    }

    # Assign colors within each process group
    process_counts: dict[str, int] = {}
    result_colors = []
    for r in results:
        idx = process_counts.get(r.process_name, 0)
        process_counts[r.process_name] = idx + 1
        n_in_group = sum(1 for rr in results if rr.process_name == r.process_name)
        frac = 0.4 + 0.5 * (idx / max(1, n_in_group - 1))
        result_colors.append(process_cmaps[r.process_name](frac))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: R² by layer
    ax = axes[0]
    for i, r in enumerate(results):
        layers = sorted(r.r2_per_layer.keys())
        values = [r.r2_per_layer[l] for l in layers]
        ax.plot(layers, values, marker=".", markersize=3, linewidth=1.2,
                label=r.label, color=result_colors[i])
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Belief State Probe R² by Layer — All Configurations")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: KL divergence
    ax = axes[1]
    for i, r in enumerate(results):
        positions = np.arange(len(r.kl_mean))
        ax.plot(positions, r.kl_mean, linewidth=1.0,
                label=r.label, color=result_colors[i])
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("KL(HMM || LLM)")
    ax.set_title("KL Divergence Convergence — All Configurations")
    ax.set_xscale("log")
    ax.set_xlim(1, max(len(r.kl_mean) for r in results))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results_csv(results: list[ConfigResult], out_dir: Path) -> None:
    """Save R² and KL divergence results to CSV files."""
    if not results:
        return

    # Collect all param keys across configs
    all_param_keys = sorted(
        {k for r in results for k in r.process_params}
    )
    layers = sorted(results[0].r2_per_layer.keys())

    # sweep_results.csv: one row per config
    results_path = out_dir / "sweep_results.csv"
    with open(results_path, "w", newline="") as f:
        param_cols = [f"param_{k}" for k in all_param_keys]
        r2_cols = [f"r2_layer_{l}" for l in layers]
        header = (
            ["label", "process_name"]
            + param_cols
            + r2_cols
            + ["best_r2", "best_r2_layer", "mean_kl", "std_kl"]
        )
        writer = csv.writer(f)
        writer.writerow(header)

        for r in results:
            best_layer = max(r.r2_per_layer, key=r.r2_per_layer.get)
            row = [
                r.label,
                r.process_name,
                *[r.process_params.get(k, "") for k in all_param_keys],
                *[r.r2_per_layer[l] for l in layers],
                r.r2_per_layer[best_layer],
                best_layer,
                float(r.kl_mean.mean()),
                float(r.kl_mean.std()),
            ]
            writer.writerow(row)

    # kl_by_position.csv: long format, one row per (config, position)
    kl_path = out_dir / "kl_by_position.csv"
    with open(kl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "position", "kl_mean", "kl_std"])
        for r in results:
            for pos in range(len(r.kl_mean)):
                writer.writerow([
                    r.label,
                    pos,
                    float(r.kl_mean[pos]),
                    float(r.kl_std[pos]),
                ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Belief state geometry sweep")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-user",
        type=str,
        default=None,
        help="Override output_user from the config file",
    )
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    apply_runtime_overrides(config, output_user=args.output_user)
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="belief_state_sweep")

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Device: {device}")

    # Load model once (shared across all configs)
    model = load_model(config.model_name, device, logger, n_ctx=config.n_ctx_override)

    # Enumerate all configs from sweep entries
    all_config_runs: list[tuple[str, dict[str, float], SweepEntry]] = []
    for entry in config.sweeps:
        param_combos = expand_param_grid(entry)
        for params in param_combos:
            all_config_runs.append((entry.process_name, params, entry))

    total_configs = len(all_config_runs)
    logger.info(f"Total configurations to sweep: {total_configs}")
    for i, (pname, params, _) in enumerate(all_config_runs):
        logger.info(f"  [{i + 1}/{total_configs}] {pname}: {params}")

    # Run each configuration
    all_results: list[ConfigResult] = []

    for i, (process_name, params, entry) in enumerate(all_config_runs):
        label = make_config_label(process_name, params)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Config [{i + 1}/{total_configs}]: {label}")
        logger.info(f"{'=' * 60}")

        # Resolve vocab tokens
        hmm_temp = build_hidden_markov_model(
            process_name, process_params=params, device=None,
        )
        vocab_tokens = resolve_vocab_tokens(
            hmm_temp, entry, config.default_vocab_tokens
        )
        logger.info(
            f"Process: {process_name}, params: {params}, "
            f"vocab: {vocab_tokens} ({hmm_temp.vocab_size} symbols, "
            f"{hmm_temp.num_states} states)"
        )

        # Per-config output subdirectory
        config_dir = out_dir / "configs" / label
        (config_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Run pipeline
        result = run_single_config(
            process_name=process_name,
            process_params=params,
            vocab_tokens=vocab_tokens,
            model=model,
            config=config,
            logger=logger,
            pca_plot_path=config_dir / "figures" / "pca_by_layer.png",
            seq_length=entry.seq_length,
            n_sequences=entry.n_sequences,
        )

        # Save per-config plots
        plot_belief_simplex(
            result.belief_states_flat,
            title=f"Ground Truth Beliefs — {label}",
            path=config_dir / "figures" / "belief_simplex.png",
        )
        plot_kl_divergence(
            result.kl_mean,
            result.kl_std,
            title=f"KL(HMM || LLM) — {label}",
            path=config_dir / "figures" / "kl_divergence.png",
        )
        plot_r2_by_layer(
            result.r2_per_layer,
            title=f"Belief State R² by Layer — {label}",
            path=config_dir / "figures" / "r2_by_layer.png",
        )

        # Save per-config metrics
        metrics = {
            "process_name": process_name,
            "process_params": params,
            "vocab_tokens": vocab_tokens,
            "r2_per_layer": {str(k): v for k, v in result.r2_per_layer.items()},
            "mse_per_layer": {str(k): v for k, v in result.mse_per_layer.items()},
            "mean_kl": float(result.kl_mean.mean()),
        }
        with open(config_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_results.append(result)
        logger.info(
            f"Config {label} complete. "
            f"Best R²={max(result.r2_per_layer.values()):.4f}"
        )

    # Cross-config comparison
    logger.info(f"\n{'=' * 60}")
    logger.info("Generating cross-config comparison plots...")
    logger.info(f"{'=' * 60}")

    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(all_results, path=comparison_dir / "comparison.png")

    # Aggregate summary
    summary = {
        "total_configs": total_configs,
        "model_name": config.model_name,
        "seq_length": config.seq_length,
        "n_sequences": config.n_sequences,
        "configs": [
            {
                "label": r.label,
                "process_name": r.process_name,
                "process_params": r.process_params,
                "best_r2": float(max(r.r2_per_layer.values())),
                "best_r2_layer": int(max(r.r2_per_layer, key=r.r2_per_layer.get)),
                "mean_kl": float(r.kl_mean.mean()),
            }
            for r in all_results
        ],
    }
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_results_csv(all_results, out_dir)

    logger.info(f"\nAll outputs written to {out_dir}")
    logger.info("Sweep summary:")
    for entry in summary["configs"]:
        logger.info(
            f"  {entry['label']}: best R²={entry['best_r2']:.4f} "
            f"(layer {entry['best_r2_layer']}), mean KL={entry['mean_kl']:.4f}"
        )


if __name__ == "__main__":
    main()
