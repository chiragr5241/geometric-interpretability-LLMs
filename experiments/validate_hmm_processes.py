#!/usr/bin/env python3
"""Validate HMM process tensors used by the final single-sequence experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmm import build_process_hmm


PROCESS_PARAMS = {
    "strata": {"a": 0.96, "t0": 0.38, "t1": 0.54},
    "wing": {"x": 0.96, "y": 0.4},
}


def _validate_process(process_name: str, process_params: dict) -> None:
    hmm = build_process_hmm(process_name, process_params, hmm_device="cpu")
    T = hmm.T_3d_matrix
    assert T.shape == (2, 3, 3), f"{process_name}: expected (2, 3, 3), got {tuple(T.shape)}"
    assert torch.isfinite(T).all(), f"{process_name}: non-finite transition entries"
    assert (T >= 0).all(), f"{process_name}: negative transition entries"

    # Column convention: for each current state, sum over emitted token and next state.
    col_sums = T.sum(dim=(0, 1))
    assert torch.allclose(col_sums, torch.ones(3), atol=1e-6), (
        f"{process_name}: transition columns do not sum to 1: {col_sums.tolist()}"
    )

    tokens, _tokens_y, states = hmm.generate_dataset(4, 12, return_states=True)
    beliefs = hmm.compute_belief_state(tokens)
    assert tokens.shape == (4, 12), f"{process_name}: bad token shape {tuple(tokens.shape)}"
    assert states.shape == (4, 12), f"{process_name}: bad state shape {tuple(states.shape)}"
    assert beliefs.shape == (4, 13, 3), f"{process_name}: bad belief shape {tuple(beliefs.shape)}"
    assert torch.isfinite(beliefs).all(), f"{process_name}: non-finite beliefs"
    assert torch.allclose(beliefs.sum(dim=-1), torch.ones(4, 13), atol=1e-5), (
        f"{process_name}: beliefs are not normalized"
    )

    cont_tokens, cont_beliefs = hmm.sample_continuation(beliefs[0, 0].numpy(), 5)
    assert cont_tokens.shape == (5,), f"{process_name}: bad continuation token shape"
    assert cont_beliefs.shape == (5, 3), f"{process_name}: bad continuation belief shape"


def main() -> None:
    for process_name, process_params in PROCESS_PARAMS.items():
        _validate_process(process_name, process_params)
        print(f"{process_name}: ok")


if __name__ == "__main__":
    main()
