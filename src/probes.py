from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


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


@dataclass(eq=False)
class ProbeResult:
    probe: Probe
    train_mse_curve: list[float]
    test_mse: float
    train_tokens: np.ndarray
    activations: np.ndarray
    gt_next_token_preds: np.ndarray
    computed_next_token_preds: np.ndarray
    gt_belief_states: np.ndarray
    computed_belief_states: np.ndarray
    kl_threshold: int | None = None

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
                    "test_mse": self.test_mse,
                    "kl_threshold": self.kl_threshold,
                    "d_model": int(self.activations.shape[1]),
                    "n_states": int(self.gt_belief_states.shape[1]),
                },
                f,
                indent=2,
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
            test_mse=meta["test_mse"],
            kl_threshold=meta["kl_threshold"],
            train_tokens=arrays["train_tokens"],
            activations=arrays["activations"],
            gt_next_token_preds=arrays["gt_next_token_preds"],
            computed_next_token_preds=arrays["computed_next_token_preds"],
            gt_belief_states=arrays["gt_belief_states"],
            computed_belief_states=arrays["computed_belief_states"],
        )


def train_probe(
    activations: np.ndarray,
    gt_belief_states: np.ndarray,
    tokens: np.ndarray,
    gt_next_token_preds: np.ndarray,
    computed_next_token_preds: np.ndarray,
    split: float = 0.8,
    lr: float = 1e-3,
    epochs: int = 1000,
) -> ProbeResult:
    device = _get_device()
    seq_len, d_model = activations.shape
    n_states = gt_belief_states.shape[1]
    split_idx = int(seq_len * split)

    act_t = torch.tensor(activations, dtype=torch.float32, device=device)
    bs_t = torch.tensor(gt_belief_states, dtype=torch.float32, device=device)

    probe = Probe(d_model, n_states).to(device)
    optimizer = Adam(probe.parameters(), lr=lr)

    train_mse_curve: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = probe(act_t[:split_idx])
        loss = nn.functional.mse_loss(preds, bs_t[:split_idx])
        loss.backward()
        optimizer.step()
        train_mse_curve.append(loss.item())

    with torch.no_grad():
        test_mse = nn.functional.mse_loss(probe(act_t[split_idx:]), bs_t[split_idx:]).item()
        computed_belief_states = probe(act_t).cpu().numpy()

    return ProbeResult(
        probe=probe,
        train_mse_curve=train_mse_curve,
        test_mse=test_mse,
        train_tokens=tokens,
        activations=activations,
        gt_next_token_preds=gt_next_token_preds,
        computed_next_token_preds=computed_next_token_preds,
        gt_belief_states=gt_belief_states,
        computed_belief_states=computed_belief_states,
    )
