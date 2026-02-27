#!/usr/bin/env python3
"""
SPAR-9 Experiment 1 — Token probability diagnostic.

Runs Llama 3.2 3B on Mess3 sequences and records the full next-token
distribution at every position, producing:

  1. top_k_report.json / top_k_report.txt  — top-K token IDs by mean probability
     across all sequences and positions, with decoded surface forms.
  2. figures/token_heatmap.png             — per-position heatmap of the top-K tokens
     for the first sequence.

Usage:
    python experiments/token_probability_diagnostic.py \\
        experiments/configs/token_probability_diagnostic.yaml
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentConfig, load_config, setup_output_dir
from experiment_utils import get_device, load_model, setup_logging
from hmm.hmm import Mess3HMM


@dataclass
class TokenProbDiagnosticConfig(ExperimentConfig):
    n_sequences: int
    seq_length: int
    vocab_mapping: dict[str, int]
    top_k_report: int = 50
    top_k_heatmap: int = 20


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_token_heatmap(
    probs: np.ndarray,
    token_labels: list[str],
    token_indices: list[int],
    path: Path,
) -> None:
    """
    Heatmap of shape (top_k, seq_len).

    probs:         (seq_len, vocab_size) for a single sequence
    token_labels:  decoded surface forms for the selected token IDs
    token_indices: the selected token IDs (column indices into probs)
    """
    import plotly.graph_objects as go

    selected = probs[:, token_indices].T   # (top_k, seq_len)
    n_pos = probs.shape[0]

    fig = go.Figure(go.Heatmap(
        z=selected,
        x=list(range(n_pos)),
        y=token_labels,
        colorscale="Viridis",
        colorbar=dict(title="Probability"),
    ))
    fig.update_layout(
        title=(
            f"Top-{len(token_indices)} tokens by total probability mass "
            f"(sequence 0, {n_pos} positions)"
        ),
        xaxis_title="Sequence position",
        yaxis_title="Token surface form",
        yaxis=dict(tickmode="array", tickvals=list(range(len(token_labels))), ticktext=token_labels),
        height=max(600, 30 * len(token_indices)),
        width=1400,
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_top_tokens_over_sequence(
    probs: np.ndarray,
    decode_fn,
    top_k: int,
    path: Path,
) -> None:
    """
    Line plot of the top-K tokens by total probability mass over the sequence,
    plus a single "rest" line for everything else.

    probs:     (seq_len, vocab_size) for a single sequence
    decode_fn: callable(token_id: int) -> str
    """
    import plotly.graph_objects as go

    seq_len = probs.shape[0]
    positions = list(range(seq_len))

    total_mass = probs.sum(axis=0)                           # (vocab_size,)
    top_ids = np.argsort(total_mass)[::-1][:top_k].tolist()

    top_probs = probs[:, top_ids]                            # (seq_len, top_k)
    rest_probs = 1.0 - top_probs.sum(axis=1)                 # (seq_len,)

    fig = go.Figure()
    for rank, tid in enumerate(top_ids):
        label = repr(decode_fn(tid))
        fig.add_trace(go.Scatter(
            x=positions,
            y=top_probs[:, rank].tolist(),
            mode="lines",
            name=label,
            line=dict(width=1.5),
        ))

    fig.add_trace(go.Scatter(
        x=positions,
        y=rest_probs.tolist(),
        mode="lines",
        name="rest",
        line=dict(color="lightgrey", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title=f"Top-{top_k} tokens by total probability mass over sequence (sequence 0)",
        xaxis_title="Sequence position",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        legend=dict(x=1.01, y=1, xanchor="left"),
        width=1400,
        height=500,
        margin=dict(r=150),
    )
    fig.write_image(str(path.with_suffix(".png")))


def plot_emission_junk_over_sequence(
    emission_per_pos: np.ndarray,
    path: Path,
) -> None:
    """
    Line plot of emission and junk probability mass over sequence positions,
    averaged across sequences with ±1 std shaded areas.

    emission_per_pos: (n_sequences, L)
    """
    import plotly.graph_objects as go

    n_seq, L = emission_per_pos.shape
    junk_per_pos = 1.0 - emission_per_pos

    positions = np.arange(L)

    emission_mean = emission_per_pos.mean(axis=0)
    emission_std = emission_per_pos.std(axis=0)
    junk_mean = junk_per_pos.mean(axis=0)
    junk_std = junk_per_pos.std(axis=0)

    def _band(x: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, name: str):
        upper = (mean + std).tolist()
        lower = (mean - std).tolist()
        return [
            go.Scatter(
                x=x.tolist() + x.tolist()[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=color,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                opacity=0.25,
            ),
            go.Scatter(
                x=x.tolist(),
                y=mean.tolist(),
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            ),
        ]

    fig = go.Figure()
    for trace in _band(positions, emission_mean, emission_std, "royalblue", "Emission mass"):
        fig.add_trace(trace)
    for trace in _band(positions, junk_mean, junk_std, "tomato", "Junk mass"):
        fig.add_trace(trace)

    fig.update_layout(
        title=f"Emission vs. junk probability mass over sequence (mean ± 1 std, n={n_seq} sequences)",
        xaxis_title="Sequence position",
        yaxis_title="Probability mass",
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.75, y=0.95),
        width=1200,
        height=450,
    )
    fig.write_image(str(path.with_suffix(".png")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1], TokenProbDiagnosticConfig)
    device = get_device()

    out_dir = setup_output_dir(config)
    logger = setup_logging(out_dir, name="token_diag")

    logger.info(f"Output dir  : {out_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"n_sequences : {config.n_sequences}")
    logger.info(f"seq_length  : {config.seq_length}")

    seq_seeds: list[int] = [int(torch.randint(2**31, (1,)).item()) for _ in range(config.n_sequences)]

    model = load_model(config.model_name, device, logger)
    vocab_size: int = model.cfg.d_vocab

    hmm = Mess3HMM()
    p = config.hmm.process_params
    if "x" in p and "alpha" in p:
        hmm.create_hmm(p["x"], p["alpha"])
        logger.info(f"Mess3 HMM: x={p['x']}, alpha={p['alpha']}")

    idx_to_token: dict[int, str] = {v: k for k, v in config.vocab_mapping.items()}
    n_hmm_tokens = len(config.vocab_mapping)
    L = config.seq_length

    # Collect both no-space (position 0) and space-prefixed (positions 1+) token IDs
    # for each emission. "A B C" tokenises as ['A',' B',' C'] so mid_tok_ids[0] would
    # wrongly be 'A' (no space) rather than ' A'; we compute them separately instead.
    ordered = [idx_to_token[i] for i in range(n_hmm_tokens)]
    emission_id_set: set[int] = set()
    for em in ordered:
        for text in [em, " " + em]:
            tids = model.to_tokens(text, prepend_bos=False)[0].tolist()
            if len(tids) == 1:
                emission_id_set.add(tids[0])
    emission_ids_arr = np.array(sorted(emission_id_set), dtype=np.int64)
    logger.info(
        f"Emission token IDs: "
        + ", ".join(
            f"{tid} ({repr(model.tokenizer.decode([tid]))})"
            for tid in sorted(emission_id_set)
        )
    )

    # ── Phase 1: Generate all sequences and tokenize ──────────────────────────
    logger.info("Generating sequences ...")
    all_seq_tokens: list[np.ndarray] = []
    all_llm_tokens: list[torch.Tensor] = []

    for seq_idx in range(config.n_sequences):
        torch.manual_seed(seq_seeds[seq_idx])
        tokens_batch, _, _ = hmm.generate_dataset(1, L, return_states=True)
        seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
        text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
        llm_tok = model.to_tokens(text, prepend_bos=True)   # (1, L+1)
        assert llm_tok.shape[1] == L + 1, (
            f"Sequence {seq_idx}: expected {L + 1} LLM tokens, got {llm_tok.shape[1]}. "
            "Ensure every HMM token maps to exactly one LLM token."
        )
        all_seq_tokens.append(seq_tokens)
        all_llm_tokens.append(llm_tok[0])   # (L+1,)

    # ── Phase 2: Single batched forward pass ──────────────────────────────────
    batch_tokens = torch.stack(all_llm_tokens, dim=0)   # (N, L+1)
    logger.info(f"Running batched forward pass: {batch_tokens.shape} ...")

    with torch.no_grad():
        logits = model(batch_tokens, return_type="logits")   # (N, L+1, vocab_size)

    # ── Phase 3: Compute statistics per sequence ──────────────────────────────
    prob_sum = np.zeros(vocab_size, dtype=np.float64)
    total_positions = 0

    seq_emission_mass: list[float] = []
    seq_junk_mass: list[float] = []
    all_emission_mass_per_pos: list[np.ndarray] = []
    first_seq_probs: np.ndarray | None = None

    for seq_idx in range(config.n_sequences):
        # logits[seq_idx, 0:L, :] — positions predicting HMM tokens 0..L-1
        seq_probs = F.softmax(logits[seq_idx, :L, :].float(), dim=-1).cpu().numpy()
        # seq_probs: (L, vocab_size)

        emission_per_pos = seq_probs[:, emission_ids_arr].sum(axis=1)   # (L,)
        emission_mass = float(emission_per_pos.mean())
        junk_mass = 1.0 - emission_mass
        seq_emission_mass.append(emission_mass)
        seq_junk_mass.append(junk_mass)
        all_emission_mass_per_pos.append(emission_per_pos)
        logger.info(
            f"  Sequence {seq_idx + 1}/{config.n_sequences}:"
            f"  emission mass={emission_mass:.4f}  junk mass={junk_mass:.4f}"
        )

        prob_sum += seq_probs.sum(axis=0)
        total_positions += L

        if seq_idx == 0:
            first_seq_probs = seq_probs.copy()

    del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Emission / junk mass summary ──────────────────────────────────────────
    mean_emission = float(np.mean(seq_emission_mass))
    mean_junk = float(np.mean(seq_junk_mass))
    logger.info(
        f"\nAggregate over {config.n_sequences} sequences:"
        f"\n  Mean emission mass : {mean_emission:.4f}"
        f"\n  Mean junk mass     : {mean_junk:.4f}"
    )

    # ── Top-K report ──────────────────────────────────────────────────────────
    mean_probs = prob_sum / total_positions
    top_k = config.top_k_report
    top_ids = np.argsort(mean_probs)[::-1][:top_k]

    report_entries: list[dict] = []
    for rank, tid in enumerate(top_ids):
        surface = model.tokenizer.decode([int(tid)])
        report_entries.append({
            "rank": rank + 1,
            "token_id": int(tid),
            "surface_form": repr(surface),
            "mean_prob": float(mean_probs[tid]),
        })

    with open(out_dir / "top_k_report.json", "w") as f:
        json.dump(
            {
                "n_sequences": config.n_sequences,
                "seq_length": L,
                "total_positions": total_positions,
                "top_k": top_k,
                "seq_seeds": seq_seeds,
                "emission_token_ids": sorted(emission_id_set),
                "mean_emission_mass": mean_emission,
                "mean_junk_mass": mean_junk,
                "per_seq_emission_mass": seq_emission_mass,
                "per_seq_junk_mass": seq_junk_mass,
                "entries": report_entries,
            },
            f,
            indent=2,
        )

    report_lines: list[str] = [
        f"Top-{top_k} tokens by mean probability over {config.n_sequences} sequences × {L} positions\n",
        f"{'Rank':>5}  {'Token ID':>10}  {'Surface form':>25}  {'Mean prob':>12}",
        "-" * 60,
    ]
    for entry in report_entries:
        report_lines.append(
            f"{entry['rank']:>5}  {entry['token_id']:>10}  {entry['surface_form']:>25}  {entry['mean_prob']:>12.6f}"
        )
    report_text = "\n".join(report_lines)

    with open(out_dir / "top_k_report.txt", "w") as f:
        f.write(report_text + "\n")

    logger.info(f"\n{report_text}")

    # ── Per-position heatmap ──────────────────────────────────────────────────
    assert first_seq_probs is not None
    k_heatmap = config.top_k_heatmap

    # Select top-k by total mass across all positions in the first sequence
    total_mass = first_seq_probs.sum(axis=0)
    heatmap_ids = np.argsort(total_mass)[::-1][:k_heatmap].tolist()
    heatmap_labels = [repr(model.tokenizer.decode([int(tid)])) for tid in heatmap_ids]

    plot_token_heatmap(
        first_seq_probs,
        heatmap_labels,
        heatmap_ids,
        out_dir / "figures" / "token_heatmap",
    )

    plot_top_tokens_over_sequence(
        first_seq_probs,
        decode_fn=lambda tid: model.tokenizer.decode([tid]),
        top_k=10,
        path=out_dir / "figures" / "top_tokens_over_sequence",
    )

    plot_emission_junk_over_sequence(
        np.stack(all_emission_mass_per_pos, axis=0),
        out_dir / "figures" / "emission_junk_over_sequence",
    )

    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
