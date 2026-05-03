from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, overload

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

K = TypeVar("K")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Decoder(nn.Module):
    """Affine map from belief states to residual-stream activations.

    Forward: act_hat = eta @ W.T + bias
    where W ∈ R^(d_model × n_states), bias ∈ R^(d_model).
    """

    def __init__(self, d_model: int, n_states: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, n_states) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        return eta @ self.W.T + self.bias


@dataclass
class DecoderInput:
    """Per-decoder training data."""

    activations: np.ndarray    # (seq_len, d_model) — reconstruction targets
    belief_states: np.ndarray  # (seq_len, n_states) — inputs


@dataclass(eq=False)
class DecoderResult:
    decoder: Decoder
    train_loss_curve: list[float]
    eval_loss_curve: list[float]
    train_loss: float
    eval_loss: float
    train_split_idx: int
    normalized_train_loss: float
    normalized_eval_loss: float

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.decoder.state_dict(), path / "decoder.pt")
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "train_loss_curve": self.train_loss_curve,
                    "eval_loss_curve": self.eval_loss_curve,
                    "train_loss": self.train_loss,
                    "eval_loss": self.eval_loss,
                    "train_split_idx": self.train_split_idx,
                    "normalized_train_loss": self.normalized_train_loss,
                    "normalized_eval_loss": self.normalized_eval_loss,
                    "d_model": self.decoder.W.shape[0],
                    "n_states": self.decoder.W.shape[1],
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "DecoderResult":
        path = Path(path)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        decoder = Decoder(meta["d_model"], meta["n_states"])
        decoder.load_state_dict(
            torch.load(path / "decoder.pt", map_location="cpu", weights_only=True)
        )
        decoder.eval()
        return cls(
            decoder=decoder,
            train_loss_curve=meta["train_loss_curve"],
            eval_loss_curve=meta["eval_loss_curve"],
            train_loss=meta["train_loss"],
            eval_loss=meta["eval_loss"],
            train_split_idx=meta["train_split_idx"],
            normalized_train_loss=meta["normalized_train_loss"],
            normalized_eval_loss=meta["normalized_eval_loss"],
        )


@overload
def train_batched(
    inputs: list[DecoderInput],
    split: float = ...,
    lr: float = ...,
    max_epochs: int = ...,
    patience: int = ...,
    min_relative_improvement: float = ...,
    device: torch.device | None = ...,
    max_decoders_per_batch: int | None = ...,
) -> list[DecoderResult]: ...


@overload
def train_batched(
    inputs: dict[K, DecoderInput],
    split: float = ...,
    lr: float = ...,
    max_epochs: int = ...,
    patience: int = ...,
    min_relative_improvement: float = ...,
    device: torch.device | None = ...,
    max_decoders_per_batch: int | None = ...,
) -> dict[K, DecoderResult]: ...


def train_batched(
    inputs: list[DecoderInput] | dict[Any, DecoderInput],
    split: float = 0.8,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 200,
    min_relative_improvement: float = 1e-3,
    device: torch.device | None = None,
    max_decoders_per_batch: int | None = None,
) -> list[DecoderResult] | dict[Any, DecoderResult]:
    """Train multiple decoders in a single batched operation.

    Each decoder learns an independent affine map from belief states to
    residual-stream activations.  Training uses Adam with MSE (mean over
    d_model dimensions) loss and an 80/20 token-position train/eval split.

    Training stops per-decoder when eval loss has not improved by at least
    ``min_relative_improvement`` relative to the running best for ``patience``
    consecutive epochs, or when ``max_epochs`` is reached for all decoders.

    Dict overload: when *inputs* is a ``dict[K, DecoderInput]``, the return
    value is a ``dict[K, DecoderResult]`` with matching keys::

        results = train_batched({layer: DecoderInput(...) for layer in layers})
        dr = results[layer]

    Args:
        inputs: One DecoderInput per decoder.  Accepts list (returns list) or
            dict (returns dict with the same keys).  All entries must share
            the same ``d_model`` and ``n_states``.
        split: Fraction of positions used for training; the rest is eval.
        lr: Adam learning rate.
        max_epochs: Maximum number of gradient-descent steps.
        patience: Number of epochs without sufficient improvement before a
            decoder is considered converged.
        min_relative_improvement: Minimum fractional improvement over the
            running best eval loss required to reset the patience counter,
            i.e. ``(best - new) / best >= min_relative_improvement``.
        device: Training device.  Defaults to CUDA > MPS > CPU.
        max_decoders_per_batch: If set, inputs are split into chunks of at
            most this size and trained sequentially, with results merged.

    Returns:
        DecoderResult objects in the same order / under the same keys as inputs.
    """
    if device is None:
        device = _get_device()

    keys: list[Any] | None = None
    if isinstance(inputs, dict):
        keys = list(inputs.keys())
        input_list = list(inputs.values())
    else:
        input_list = list(inputs)

    if max_decoders_per_batch is not None and len(input_list) > max_decoders_per_batch:
        results_list: list[DecoderResult] = []
        for start in range(0, len(input_list), max_decoders_per_batch):
            chunk = input_list[start : start + max_decoders_per_batch]
            results_list.extend(
                _train_batched_impl(chunk, split, lr, max_epochs, patience, min_relative_improvement, device)
            )
        if keys is not None:
            return dict(zip(keys, results_list))
        return results_list

    results = _train_batched_impl(input_list, split, lr, max_epochs, patience, min_relative_improvement, device)
    if keys is not None:
        return dict(zip(keys, results))
    return results


def _train_batched_impl(
    input_list: list[DecoderInput],
    split: float,
    lr: float,
    max_epochs: int,
    patience: int,
    min_relative_improvement: float,
    device: torch.device,
) -> list[DecoderResult]:
    N = len(input_list)
    d_model = input_list[0].activations.shape[1]
    n_states = input_list[0].belief_states.shape[1]
    seq_lens = [inp.activations.shape[0] for inp in input_list]
    max_len = max(seq_lens)

    beliefs_pad = torch.zeros(N, max_len, n_states, device=device)
    acts_pad = torch.zeros(N, max_len, d_model, device=device)
    train_mask = torch.zeros(N, max_len, dtype=torch.bool, device=device)
    eval_mask = torch.zeros(N, max_len, dtype=torch.bool, device=device)

    split_indices: list[int] = []
    act_norms: list[float] = []
    for i, inp in enumerate(input_list):
        sl = seq_lens[i]
        beliefs_pad[i, :sl] = torch.tensor(inp.belief_states, dtype=torch.float32)
        acts_pad[i, :sl] = torch.tensor(inp.activations, dtype=torch.float32)
        split_idx = sl if split >= 1.0 else int(sl * split)
        split_idx = max(0, min(split_idx, sl))
        split_indices.append(split_idx)
        train_mask[i, :split_idx] = True
        eval_mask[i, split_idx:sl] = True
        act_norms.append(float(np.linalg.norm(inp.activations, axis=-1).mean()))

    # W_batch: (N, n_states, d_model) so that
    #   bmm(beliefs_pad, W_batch) = (N, max_len, d_model)  ← decoded activations
    W_batch = nn.Parameter(torch.randn(N, n_states, d_model, device=device) * 0.01)
    bias_batch = nn.Parameter(torch.zeros(N, d_model, device=device))
    optimizer = Adam([W_batch, bias_batch], lr=lr)

    train_counts = train_mask.sum(dim=1).float()  # (N,)
    eval_counts = eval_mask.sum(dim=1).float()    # (N,)
    use_eval_holdout = bool(eval_counts.min() > 0)

    train_loss_curves: list[list[float]] = [[] for _ in range(N)]
    eval_loss_curves: list[list[float]] = [[] for _ in range(N)]
    convergence_epoch: list[int | None] = [None] * N
    converged = torch.zeros(N, dtype=torch.bool, device=device)
    best_loss = torch.full((N,), float("inf"), device=device)
    epochs_since_best = torch.zeros(N, dtype=torch.long, device=device)

    for epoch in range(max_epochs):
        active = ~converged  # (N,) — decoders still being trained
        if not active.any():
            break

        optimizer.zero_grad()
        preds = torch.bmm(beliefs_pad, W_batch) + bias_batch.unsqueeze(1)  # (N, max_len, d_model)
        sq_err = ((preds - acts_pad) ** 2).mean(dim=-1)                    # (N, max_len)
        per_decoder_train = (sq_err * train_mask).sum(dim=1) / train_counts
        total_loss = (per_decoder_train * active.float()).sum()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if use_eval_holdout:
                per_decoder_eval = (sq_err.detach() * eval_mask).sum(dim=1) / eval_counts
            else:
                per_decoder_eval = per_decoder_train.detach()

            convergence_loss = per_decoder_eval
            improved = convergence_loss < best_loss * (1.0 - min_relative_improvement)
            best_loss[improved] = convergence_loss[improved]
            epochs_since_best[improved] = 0
            epochs_since_best[~improved & active] += 1
            newly_converged = active & (epochs_since_best >= patience)
            if newly_converged.any():
                for i in torch.nonzero(newly_converged, as_tuple=False).flatten().tolist():
                    convergence_epoch[i] = epoch
            converged |= newly_converged

        train_d = per_decoder_train.detach()
        for i in range(N):
            if convergence_epoch[i] is None:
                train_loss_curves[i].append(train_d[i].item())
                eval_loss_curves[i].append(per_decoder_eval[i].item())

    with torch.no_grad():
        final_preds = torch.bmm(beliefs_pad, W_batch) + bias_batch.unsqueeze(1)
        final_sq = ((final_preds - acts_pad) ** 2).mean(dim=-1)
        final_train = (final_sq * train_mask).sum(dim=1) / train_counts
        if use_eval_holdout:
            final_eval = (final_sq * eval_mask).sum(dim=1) / eval_counts
        else:
            final_eval = final_train

    results: list[DecoderResult] = []
    for i in range(N):
        decoder = Decoder(d_model, n_states)
        # W_batch[i] is (n_states, d_model); Decoder.W is (d_model, n_states)
        decoder.W.data = W_batch[i].T.detach().cpu()
        decoder.bias.data = bias_batch[i].detach().cpu()
        decoder.eval()

        train_loss = float(final_train[i].item())
        eval_loss = float(final_eval[i].item())
        norm_sq = act_norms[i] ** 2

        results.append(
            DecoderResult(
                decoder=decoder,
                train_loss_curve=train_loss_curves[i],
                eval_loss_curve=eval_loss_curves[i],
                train_loss=train_loss,
                eval_loss=eval_loss,
                train_split_idx=split_indices[i],
                normalized_train_loss=train_loss / (norm_sq + 1e-10),
                normalized_eval_loss=eval_loss / (norm_sq + 1e-10),
            )
        )

    return results
