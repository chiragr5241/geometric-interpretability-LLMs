from __future__ import annotations

import json
from dataclasses import dataclass, field
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


class Probe(nn.Module):
    def __init__(self, d_model: int, n_states: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, n_states) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_states))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.bias


@dataclass
class ProbeInput:
    """Per-probe inputs for :func:`train_probes_batched`.

    All arrays for a single probe share the same leading dimension
    (``seq_len``), but different ``ProbeInput`` instances in the same
    batch may have different ``seq_len`` values — padding and masking
    are handled internally.

    Attributes:
        activations: ``(seq_len, d_model)`` residual-stream activations.
        gt_belief_states: ``(seq_len, n_states)`` ground-truth belief vectors.
        tokens: Token indices (kept verbatim in the returned
            :class:`ProbeResult`).
        gt_next_token_preds: Ground-truth next-token prediction vectors.
        computed_next_token_preds: Model's next-token predictions.
    """

    activations: np.ndarray
    gt_belief_states: np.ndarray
    tokens: np.ndarray
    gt_next_token_preds: np.ndarray
    computed_next_token_preds: np.ndarray
    split_idx: int | None = None


@dataclass(eq=False)
class ProbeResult:
    probe: Probe
    train_mse_curve: list[float]
    test_mse: float
    test_split_idx: int
    train_tokens: np.ndarray
    activations: np.ndarray
    gt_next_token_preds: np.ndarray
    computed_next_token_preds: np.ndarray
    gt_belief_states: np.ndarray
    computed_belief_states: np.ndarray
    kl_threshold: int | None = None
    test_mse_curve: list[float] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.probe.state_dict(), path / "probe.pt")
        np.savez(
            path / "arrays.npz",
            train_tokens=self.train_tokens,
            activations=self.activations,
            gt_next_token_preds=self.gt_next_token_preds,
            computed_next_token_preds=self.computed_next_token_preds,
            gt_belief_states=self.gt_belief_states,
            computed_belief_states=self.computed_belief_states,
        )
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "train_mse_curve": self.train_mse_curve,
                    "test_mse_curve": self.test_mse_curve,
                    "test_mse": self.test_mse,
                    "test_split_idx": self.test_split_idx,
                    "kl_threshold": self.kl_threshold,
                    "d_model": int(self.activations.shape[1]),
                    "n_states": int(self.gt_belief_states.shape[1]),
                },
                f,
                indent=2,
            )

    def save_weights_only(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.probe.state_dict(), path / "probe.pt")
        with open(path / "probe_metadata.json", "w") as f:
            json.dump(
                {
                    "train_mse_curve": self.train_mse_curve,
                    "test_mse_curve": self.test_mse_curve,
                    "test_mse": self.test_mse,
                    "test_split_idx": self.test_split_idx,
                    "kl_threshold": self.kl_threshold,
                    "d_model": int(self.probe.W.shape[0]),
                    "n_states": int(self.probe.W.shape[1]),
                },
                f,
                indent=2,
            )

    @classmethod
    def load_weights_only(cls, path: Path) -> "ProbeResult":
        path = Path(path)
        with open(path / "probe_metadata.json") as f:
            meta = json.load(f)
        probe = Probe(meta["d_model"], meta["n_states"])
        probe.load_state_dict(
            torch.load(path / "probe.pt", map_location="cpu", weights_only=True)
        )
        probe.eval()
        dummy_acts = np.zeros((0, meta["d_model"]), dtype=np.float32)
        dummy_beliefs = np.zeros((0, meta["n_states"]), dtype=np.float32)
        dummy_tokens = np.zeros((0,), dtype=np.int64)
        return cls(
            probe=probe,
            train_mse_curve=meta["train_mse_curve"],
            test_mse_curve=meta.get("test_mse_curve", []),
            test_mse=meta["test_mse"],
            test_split_idx=meta["test_split_idx"],
            kl_threshold=meta["kl_threshold"],
            train_tokens=dummy_tokens,
            activations=dummy_acts,
            gt_next_token_preds=dummy_acts,
            computed_next_token_preds=dummy_acts,
            gt_belief_states=dummy_beliefs,
            computed_belief_states=dummy_beliefs,
        )

    @classmethod
    def load(cls, path: Path) -> "ProbeResult":
        path = Path(path)
        arrays = np.load(path / "arrays.npz")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        probe = Probe(meta["d_model"], meta["n_states"])
        probe.load_state_dict(
            torch.load(path / "probe.pt", map_location="cpu", weights_only=True)
        )
        probe.eval()
        return cls(
            probe=probe,
            train_mse_curve=meta["train_mse_curve"],
            test_mse_curve=meta.get("test_mse_curve", []),
            test_mse=meta["test_mse"],
            test_split_idx=meta["test_split_idx"],
            kl_threshold=meta["kl_threshold"],
            train_tokens=arrays["train_tokens"],
            activations=arrays["activations"],
            gt_next_token_preds=arrays["gt_next_token_preds"],
            computed_next_token_preds=arrays["computed_next_token_preds"],
            gt_belief_states=arrays["gt_belief_states"],
            computed_belief_states=arrays["computed_belief_states"],
        )


@overload
def train_probes_batched(
    inputs: list[ProbeInput],
    split: float = ...,
    lr: float = ...,
    epochs: int = ...,
    max_epochs: int | None = ...,
    patience: int | None = ...,
    min_relative_improvement: float = ...,
    device: torch.device | None = ...,
    max_probes_per_batch: int | None = ...,
) -> list[ProbeResult]: ...


@overload
def train_probes_batched(
    inputs: dict[K, ProbeInput],
    split: float = ...,
    lr: float = ...,
    epochs: int = ...,
    max_epochs: int | None = ...,
    patience: int | None = ...,
    min_relative_improvement: float = ...,
    device: torch.device | None = ...,
    max_probes_per_batch: int | None = ...,
) -> dict[K, ProbeResult]: ...


def train_probes_batched(
    inputs: list[ProbeInput] | dict[Any, ProbeInput],
    split: float = 0.8,
    lr: float = 1e-3,
    epochs: int = 1000,
    max_epochs: int | None = None,
    patience: int | None = None,
    min_relative_improvement: float = 1e-3,
    device: torch.device | None = None,
    max_probes_per_batch: int | None = None,
) -> list[ProbeResult] | dict[Any, ProbeResult]:
    """Train multiple linear probes in a single batched operation.

    Each probe learns an independent affine map from residual-stream
    activations to belief states.  All probes share the same ``d_model``
    and ``n_states`` dimensions but may have **different sequence lengths**.
    Internally the inputs are zero-padded to the longest sequence and
    boolean masks ensure that each probe's loss is computed only over its
    valid positions.

    Because every probe has its own slice of the weight / bias tensors and
    the total loss is the sum of per-probe mean-MSE terms, the gradients
    (and therefore the Adam updates) for each probe are mathematically
    identical to training it individually — the only difference is that
    ``torch.bmm`` runs all forward passes as a single kernel call.

    **Dict overload:** when *inputs* is a ``dict[K, ProbeInput]``, the
    return value is a ``dict[K, ProbeResult]`` with matching keys, so
    callers never need to track integer indices::

        results = train_probes_batched({
            (layer, "full"): ProbeInput(...),
            (layer, "post"): ProbeInput(...),
        })
        full_pr = results[(layer, "full")]

    Args:
        inputs: One :class:`ProbeInput` per probe.  Accepts either a
            ``list`` (returns a ``list``) or a ``dict`` (returns a
            ``dict`` with the same keys).  All entries must have the
            same ``d_model`` and ``n_states``.
        split: Fraction of each sequence used for training; the remainder
            is held out for the per-probe test MSE.
        lr: Learning rate for Adam.
        epochs: Number of gradient-descent steps when ``max_epochs`` is not set.
        max_epochs: Maximum number of gradient-descent steps when using
            patience-based convergence.
        patience: Number of epochs without sufficient improvement before a
            probe is considered converged. If unset, training runs for exactly
            ``epochs`` steps, preserving historical behavior.
        min_relative_improvement: Fractional improvement over the patience
            window required to keep training an unconverged probe.
        device: Torch device for training.  Defaults to CUDA > MPS > CPU.
        max_probes_per_batch: If set, inputs are split into chunks of at
            most this size and trained sequentially, with results merged
            before returning.  Useful for bounding the size of the padded
            activation tensor ``(N, max_len, d_model)`` on-device.

    Returns:
        :class:`ProbeResult` objects in the same order / under the same
        keys as *inputs*.
    """
    if device is None:
        device = _get_device()
    if max_epochs is None:
        max_epochs = epochs

    keys: list[Any] | None = None
    if isinstance(inputs, dict):
        keys = list(inputs.keys())
        input_list = list(inputs.values())
    else:
        input_list = list(inputs)

    if max_probes_per_batch is not None and len(input_list) > max_probes_per_batch:
        results_list: list[ProbeResult] = []
        for start in range(0, len(input_list), max_probes_per_batch):
            chunk = input_list[start : start + max_probes_per_batch]
            results_list.extend(
                _train_probes_batched_impl(
                    chunk,
                    split,
                    lr,
                    max_epochs,
                    patience,
                    min_relative_improvement,
                    device,
                )
            )
        if keys is not None:
            return dict(zip(keys, results_list))
        return results_list

    if keys is not None:
        results = _train_probes_batched_impl(
            input_list,
            split,
            lr,
            max_epochs,
            patience,
            min_relative_improvement,
            device,
        )
        return dict(zip(keys, results))
    return _train_probes_batched_impl(
        input_list,
        split,
        lr,
        max_epochs,
        patience,
        min_relative_improvement,
        device,
    )


def _train_probes_batched_impl(
    input_list: list[ProbeInput],
    split: float,
    lr: float,
    max_epochs: int,
    patience: int | None,
    min_relative_improvement: float,
    device: torch.device,
) -> list[ProbeResult]:
    N = len(input_list)
    d_model = input_list[0].activations.shape[1]
    n_states = input_list[0].gt_belief_states.shape[1]
    seq_lens = [inp.activations.shape[0] for inp in input_list]
    max_len = max(seq_lens)

    acts_pad = torch.zeros(N, max_len, d_model, device=device)
    tgt_pad = torch.zeros(N, max_len, n_states, device=device)
    train_mask = torch.zeros(N, max_len, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, max_len, dtype=torch.bool, device=device)

    split_indices: list[int] = []
    for i, inp in enumerate(input_list):
        sl = seq_lens[i]
        acts_pad[i, :sl] = torch.tensor(inp.activations, dtype=torch.float32)
        tgt_pad[i, :sl] = torch.tensor(inp.gt_belief_states, dtype=torch.float32)
        split_idx = inp.split_idx if inp.split_idx is not None else (sl if split >= 1.0 else int(sl * split))
        split_idx = max(0, min(split_idx, sl))
        split_indices.append(split_idx)
        train_mask[i, :split_idx] = True
        test_mask[i, split_idx:sl] = True

    W = nn.Parameter(torch.randn(N, d_model, n_states, device=device) * 0.01)
    bias = nn.Parameter(torch.zeros(N, n_states, device=device))
    optimizer = Adam([W, bias], lr=lr)

    train_counts = train_mask.sum(dim=1).float()  # (N,)
    test_counts = test_mask.sum(dim=1).float()
    use_test_holdout = bool(test_counts.min() > 0)

    if patience is not None and patience <= 0:
        raise ValueError(f"patience must be positive when set, got {patience}")

    train_mse_curves: list[list[float]] = [[] for _ in range(N)]
    test_mse_curves: list[list[float]] = [[] for _ in range(N)]
    convergence_epoch: list[int | None] = [None] * N
    converged = torch.zeros(N, dtype=torch.bool, device=device)
    best_loss = torch.full((N,), float("inf"), device=device)
    epochs_since_best = torch.zeros(N, dtype=torch.long, device=device)

    for epoch in range(max_epochs):
        active = ~converged
        if patience is not None and not active.any():
            break

        optimizer.zero_grad()
        preds = torch.bmm(acts_pad, W) + bias.unsqueeze(1)
        sq_err = ((preds - tgt_pad) ** 2).mean(dim=-1)  # (N, max_len)
        per_probe_loss = (sq_err * train_mask).sum(dim=1) / train_counts
        total_loss = (per_probe_loss * active.float()).sum()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds_after = torch.bmm(acts_pad, W) + bias.unsqueeze(1)
            sq_err_after = ((preds_after - tgt_pad) ** 2).mean(dim=-1)
            per_probe_detached = (sq_err_after * train_mask).sum(dim=1) / train_counts
            if use_test_holdout:
                per_probe_test = (sq_err_after * test_mask).sum(dim=1) / test_counts
            else:
                per_probe_test = per_probe_detached
            convergence_loss = per_probe_test if use_test_holdout else per_probe_detached

        for i in range(N):
            if convergence_epoch[i] is None:
                train_mse_curves[i].append(per_probe_detached[i].item())
                test_mse_curves[i].append(per_probe_test[i].item())

        if patience is not None:
            with torch.no_grad():
                improved = convergence_loss < best_loss * (1.0 - min_relative_improvement)
                best_loss[improved] = convergence_loss[improved]
                epochs_since_best[improved] = 0
                epochs_since_best[~improved & active] += 1
                newly_converged = active & (epochs_since_best >= patience)
                if newly_converged.any():
                    for i in torch.nonzero(newly_converged, as_tuple=False).flatten().tolist():
                        convergence_epoch[i] = epoch
                converged |= newly_converged

    with torch.no_grad():
        preds = torch.bmm(acts_pad, W) + bias.unsqueeze(1)
        sq_err = ((preds - tgt_pad) ** 2).mean(dim=-1)
        if use_test_holdout:
            test_mses = (sq_err * test_mask).sum(dim=1) / test_counts
        else:
            train_only = (sq_err * train_mask).sum(dim=1) / train_counts.clamp(min=1e-10)
            test_mses = train_only
        all_computed = preds.cpu().numpy()

    results: list[ProbeResult] = []
    for i, inp in enumerate(input_list):
        sl = seq_lens[i]
        probe = Probe(d_model, n_states)
        probe.W.data = W[i].detach().cpu()
        probe.bias.data = bias[i].detach().cpu()
        probe.eval()

        results.append(ProbeResult(
            probe=probe,
            train_mse_curve=train_mse_curves[i],
            test_mse_curve=test_mse_curves[i],
            test_mse=test_mses[i].item(),
            test_split_idx=split_indices[i],
            train_tokens=inp.tokens,
            activations=inp.activations,
            gt_next_token_preds=inp.gt_next_token_preds,
            computed_next_token_preds=inp.computed_next_token_preds,
            gt_belief_states=inp.gt_belief_states,
            computed_belief_states=all_computed[i, :sl],
        ))

    return results


def train_probe(
    activations: np.ndarray,
    gt_belief_states: np.ndarray,
    tokens: np.ndarray,
    gt_next_token_preds: np.ndarray,
    computed_next_token_preds: np.ndarray,
    split: float = 0.8,
    lr: float = 1e-3,
    epochs: int = 1000,
    max_epochs: int | None = None,
    patience: int | None = None,
    min_relative_improvement: float = 1e-3,
    device: torch.device | None = None,
) -> ProbeResult:
    """Train a single linear probe.  Convenience wrapper around
    :func:`train_probes_batched` with ``N = 1``."""
    return train_probes_batched(
        [ProbeInput(activations, gt_belief_states, tokens,
                    gt_next_token_preds, computed_next_token_preds)],
        split=split,
        lr=lr,
        epochs=epochs,
        max_epochs=max_epochs,
        patience=patience,
        min_relative_improvement=min_relative_improvement,
        device=device,
    )[0]
