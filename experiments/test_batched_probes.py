#!/usr/bin/env python3
"""Verify that train_probes_batched produces equivalent results to
training probes individually via train_probe (N=1 batched).

Generates synthetic linear-regression data with varying sequence lengths,
trains probes both ways, and checks that test MSE and learned weights
converge to the same values within tolerance.

Usage:
    python experiments/test_batched_probes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probes import ProbeInput, train_probe, train_probes_batched

D_MODEL = 64
N_STATES = 3
SEQ_LENS = [100, 150, 200, 300, 400, 500]
EPOCHS = 1000
MSE_TOL = 5e-3
WEIGHT_TOL = 5e-3


def make_probe_input(
    rng: np.random.Generator,
    seq_len: int,
) -> ProbeInput:
    """Synthetic data with a ground-truth linear map + small noise,
    so the convex MSE problem has a sharp, unique optimum that both
    training paths should converge to."""
    gt_W = rng.standard_normal((D_MODEL, N_STATES)).astype(np.float32) * 0.1
    gt_b = rng.standard_normal(N_STATES).astype(np.float32) * 0.01
    acts = rng.standard_normal((seq_len, D_MODEL)).astype(np.float32)
    beliefs = (acts @ gt_W + gt_b
               + rng.standard_normal((seq_len, N_STATES)).astype(np.float32) * 0.02)
    tokens = rng.integers(0, N_STATES, size=seq_len).astype(np.int64)
    preds = rng.dirichlet(np.ones(N_STATES), size=seq_len).astype(np.float32)
    return ProbeInput(
        activations=acts,
        gt_belief_states=beliefs,
        tokens=tokens,
        gt_next_token_preds=preds,
        computed_next_token_preds=preds.copy(),
    )


def main() -> None:
    n_probes = len(SEQ_LENS)
    rng = np.random.default_rng(42)
    cpu = torch.device("cpu")

    inputs = [make_probe_input(rng, sl) for sl in SEQ_LENS]

    print(f"Training {n_probes} probes individually ({EPOCHS} epochs each) ...")
    t0 = time.perf_counter()
    individual = [
        train_probe(
            activations=inp.activations,
            gt_belief_states=inp.gt_belief_states,
            tokens=inp.tokens,
            gt_next_token_preds=inp.gt_next_token_preds,
            computed_next_token_preds=inp.computed_next_token_preds,
            epochs=EPOCHS,
            device=cpu,
        )
        for inp in inputs
    ]
    t_ind = time.perf_counter() - t0

    print(f"Training {n_probes} probes batched ({EPOCHS} epochs) ...")
    t0 = time.perf_counter()
    batched = train_probes_batched(inputs, epochs=EPOCHS, device=cpu)
    t_bat = time.perf_counter() - t0

    print(f"\nIndividual: {t_ind:.2f}s  |  Batched: {t_bat:.2f}s  "
          f"({t_ind / t_bat:.1f}x speedup)\n")

    all_pass = True
    for i in range(n_probes):
        ind = individual[i]
        bat = batched[i]

        mse_diff = abs(ind.test_mse - bat.test_mse)
        W_diff = float(np.abs(
            ind.probe.W.detach().numpy() - bat.probe.W.detach().numpy()
        ).max())
        bias_diff = float(np.abs(
            ind.probe.bias.detach().numpy() - bat.probe.bias.detach().numpy()
        ).max())
        beliefs_diff = float(np.abs(
            ind.computed_belief_states - bat.computed_belief_states
        ).max())

        mse_ok = mse_diff < MSE_TOL
        w_ok = W_diff < WEIGHT_TOL
        ok = mse_ok and w_ok
        if not ok:
            all_pass = False

        print(f"  Probe {i} (seq_len={SEQ_LENS[i]}): {'PASS' if ok else 'FAIL'}  "
              f"test_mse_diff={mse_diff:.2e}  "
              f"W_max_diff={W_diff:.2e}  "
              f"bias_diff={bias_diff:.2e}  "
              f"beliefs_diff={beliefs_diff:.2e}")

    # --- dict overload sanity check ---
    print("\nDict-overload test ...")
    dict_inputs = {f"probe_{i}": inp for i, inp in enumerate(inputs)}
    dict_results = train_probes_batched(dict_inputs, epochs=EPOCHS, device=cpu)
    dict_ok = (
        isinstance(dict_results, dict)
        and set(dict_results.keys()) == set(dict_inputs.keys())
    )
    print(f"  Keys preserved: {'PASS' if dict_ok else 'FAIL'}")
    if not dict_ok:
        all_pass = False

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("SOME CHECKS FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
