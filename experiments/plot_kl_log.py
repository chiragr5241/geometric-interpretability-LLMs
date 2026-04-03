#!/usr/bin/env python3
"""One-off: KL(optimal ‖ model) over a single short Mess3 sequence, log y-axis.

Finds the KL threshold (first position where smoothed KL drops into the bottom
`fraction` of its max–min range) and annotates it on the plot.

Usage:
    python experiments/plot_kl_log.py [--seq-length 500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_utils import (
    build_emission_matrix,
    compute_optimal_probs,
    get_device,
    get_model_probs_projected,
    load_model,
    resolve_hmm_token_ids,
    setup_logging,
)
from hmm.hmm import Mess3HMM
from metrics.probe_metrics import compute_kl, find_kl_threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-length", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--fraction", type=float, default=0.02)
    p.add_argument("--min-position", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "outputs" / "dani" / f"{timestamp}_kl_log"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, name="kl_log")

    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Seq length: {args.seq_length}  seed: {args.seed}")

    hmm = Mess3HMM()
    hmm.create_hmm(x=0.05, alpha=0.9)
    emit = build_emission_matrix(hmm)

    torch.manual_seed(args.seed)
    tokens_batch, _, _ = hmm.generate_dataset(1, args.seq_length, return_states=True)
    beliefs_batch = hmm.compute_belief_state(tokens_batch)
    seq_tokens: np.ndarray = tokens_batch[0].cpu().numpy()
    seq_beliefs: np.ndarray = beliefs_batch[0].cpu().numpy()

    idx_to_token: dict[int, str] = {0: "A", 1: "B", 2: "C"}
    model = load_model("meta-llama/Llama-3.2-3B", device, logger, n_ctx=args.seq_length + 10)

    first_tok_id, mid_tok_ids = resolve_hmm_token_ids(model, idx_to_token, 3, logger)

    text = " ".join(idx_to_token[int(t)] for t in seq_tokens)
    llm_tokens = model.to_tokens(text, prepend_bos=True, truncate=False)

    with torch.no_grad():
        logits = model(llm_tokens, return_type="logits")

    model_probs = get_model_probs_projected(logits, first_tok_id, mid_tok_ids, args.seq_length)
    optimal_probs = compute_optimal_probs(seq_beliefs, emit)

    del model, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    kl_raw, kl_smooth = compute_kl(model_probs, optimal_probs, smooth_window=args.smooth_window, include_junk=True)

    kl_t, crossed = find_kl_threshold(
        model_probs, optimal_probs,
        fraction=args.fraction,
        smooth_window=args.smooth_window,
        min_position=args.min_position,
        include_junk=True,
        logger=logger,
    )
    status = "threshold crossed" if crossed else "fallback argmin"
    logger.info(f"KL threshold t* = {kl_t}  ({status})")

    kl_min = float(kl_smooth.min())
    kl_max = float(kl_smooth.max())
    threshold_val = kl_min + args.fraction * (kl_max - kl_min)

    positions = np.arange(len(kl_raw))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions, y=kl_raw,
        mode="lines",
        name="KL (raw)",
        line=dict(color="lightblue", width=1),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=positions, y=kl_smooth,
        mode="lines",
        name=f"KL (smoothed, w={args.smooth_window})",
        line=dict(color="royalblue", width=2),
    ))
    fig.add_hline(
        y=threshold_val,
        line=dict(color="red", dash="dash", width=1.5),
        annotation_text=f"threshold (fraction={args.fraction})",
        annotation_position="top right",
    )
    if args.min_position > 0:
        fig.add_vline(
            x=args.min_position,
            line=dict(color="green", dash="dash", width=1.5),
            annotation_text=f"min_position={args.min_position}",
            annotation_position="top right",
        )
    fig.add_vline(
        x=kl_t,
        line=dict(color="orange", dash="dot", width=2),
        annotation_text=f"t*={kl_t}" + ("" if crossed else " (argmin)"),
        annotation_position="top left",
    )
    fig.update_layout(
        title="KL(optimal ‖ model) over sequence — Mess3, Llama 3.2 3B",
        xaxis_title="Sequence position",
        yaxis=dict(title="KL [nats]", type="log"),
        legend=dict(x=0.7, y=0.95),
        height=480,
        width=760,
        margin=dict(t=80, b=60, l=70, r=40),
    )

    out = fig_dir / "kl_log.png"
    fig.write_image(str(out))
    logger.info(f"Saved to {out}")
    logger.info(f"All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
