"""Tuned lens translator model and training."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to (approximately) orthogonalize a 2D matrix.

    Coefficients (a, b, c) from Keller Jordan's Muon implementation.
    """
    assert G.ndim == 2, "Newton-Schulz expects a 2D tensor"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon — Momentum + orthogonalized 2D update via Newton-Schulz.

    Reference: Keller Jordan, 2024. The tuned-lens paper notes Muon as a
    recommended optimizer alternative to Adam. For 1D parameters (e.g. bias)
    we fall back to plain SGD-momentum.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                u = g.add(buf, alpha=momentum) if nesterov else buf
                if u.ndim == 2:
                    u_o = _zeropower_via_newtonschulz5(u, steps=ns_steps)
                    scale = max(1.0, u.size(-2) / u.size(-1)) ** 0.5
                    p.add_(u_o, alpha=-lr * scale)
                else:
                    p.add_(u, alpha=-lr)
        return loss


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "muon":
        # Muon's "natural" LR is larger than Adam's; we still honor user-supplied lr.
        return Muon(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


class TunedLensTranslator(nn.Module):
    """Per-layer affine translator h_l -> h_tilde, identity-initialized.

    Following the tuned lens paper (arXiv:2303.08112), the translator is
    initialized as the identity so that training starts from the logit-lens
    baseline.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_tuned_lens(
    activations_by_layer: dict[int, np.ndarray],
    model,
    layers: list[int],
    target_logits: np.ndarray,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    optimizer_name: str = "adam",
) -> tuple[dict[int, TunedLensTranslator], dict[int, list[float]]]:
    """Train one TunedLensTranslator per layer (faithful full-vocabulary version).

    Trains each translator so that:
        softmax(unembed(ln_final(T_l(h_l))))
    matches the model's full output distribution via KL(p_model || p_lens).

    Parameters
    ----------
    activations_by_layer : dict mapping layer index -> (N, d_model) array
    model : HookedTransformer
    layers : list of layer indices to train
    target_logits : (N, vocab_size) full model logits from the final layer
    n_epochs, lr, batch_size : training hyperparameters

    Returns
    -------
    translators : dict mapping layer -> trained TunedLensTranslator
    loss_curves : dict mapping layer -> list of per-epoch mean KL loss
    """
    device = model.unembed.W_U.device
    d_model = model.cfg.d_model

    W_U = model.unembed.W_U.detach()
    b_U = model.unembed.b_U.detach()

    target_log_probs = F.log_softmax(
        torch.tensor(target_logits, dtype=torch.float32), dim=-1
    )

    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}

    for layer in tqdm(layers, desc="Training tuned lens"):
        acts = torch.tensor(activations_by_layer[layer], dtype=torch.float32)
        n = acts.shape[0]

        translator = TunedLensTranslator(d_model).to(device)
        optimizer = _make_optimizer(optimizer_name, translator.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        epoch_losses = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                acts_batch = acts[idx].to(device)
                target_batch = target_log_probs[idx].to(device)

                optimizer.zero_grad()
                h = translator(acts_batch)
                normed = model.ln_final(h)
                full_logits = normed @ W_U + b_U
                log_probs = F.log_softmax(full_logits, dim=-1)

                loss = F.kl_div(log_probs, target_batch, reduction="batchmean", log_target=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_losses.append(epoch_loss / max(n_batches, 1))

        translators[layer] = translator.cpu().eval()
        for p in translators[layer].parameters():
            p.requires_grad = False
        loss_curves[layer] = epoch_losses
        torch.cuda.empty_cache()

        logger.info(f"  Layer {layer:2d}: final KL = {epoch_losses[-1]:.4f}")

    return translators, loss_curves


def train_tuned_lens_concept(
    activations_by_layer: dict[int, np.ndarray],
    model,
    concept_ids: list[int],
    layers: list[int],
    target_concept_values: np.ndarray,
    target_is_probs: bool = False,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    optimizer_name: str = "adam",
) -> tuple[dict[int, TunedLensTranslator], dict[int, list[float]]]:
    """Train one TunedLensTranslator per layer using concept-token logits only.

    Parameters
    ----------
    activations_by_layer : dict mapping layer index -> (N, d_model) array
    model : HookedTransformer
    concept_ids : LLM token IDs for HMM emission symbols
    layers : list of layer indices to train
    target_concept_values : (N, n_concepts) — logits if target_is_probs=False,
        probabilities if target_is_probs=True
    target_is_probs : if True, target is already probabilities
    n_epochs, lr, batch_size : training hyperparameters
    """
    device = model.unembed.W_U.device
    d_model = model.cfg.d_model

    W_c = model.unembed.W_U[:, concept_ids].detach().to(device)
    b_c = model.unembed.b_U[concept_ids].detach().to(device)

    target = torch.as_tensor(target_concept_values, dtype=torch.float32)
    target_probs = target if target_is_probs else F.softmax(target, dim=-1)

    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}

    for layer in tqdm(layers, desc="Training tuned lens (concept)"):
        acts = torch.tensor(activations_by_layer[layer], dtype=torch.float32)
        n = acts.shape[0]

        translator = TunedLensTranslator(d_model).to(device)
        optimizer = _make_optimizer(optimizer_name, translator.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        epoch_losses = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                acts_batch = acts[idx].to(device)
                target_batch = target_probs[idx].to(device)

                optimizer.zero_grad()
                h = translator(acts_batch)
                normed = model.ln_final(h)
                concept_logit = normed @ W_c + b_c
                log_probs = F.log_softmax(concept_logit, dim=-1)

                loss = F.kl_div(log_probs, target_batch, reduction="batchmean")
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_losses.append(epoch_loss / max(n_batches, 1))

        translators[layer] = translator.cpu().eval()
        for p in translators[layer].parameters():
            p.requires_grad = False
        loss_curves[layer] = epoch_losses
        torch.cuda.empty_cache()

        logger.info(f"  Layer {layer:2d}: final KL = {epoch_losses[-1]:.4f}")

    return translators, loss_curves


def apply_logit_lens(
    activations: np.ndarray,
    model,
    concept_ids: list[int],
    batch_size: int = 1024,
) -> np.ndarray:
    """Apply the raw logit lens (no training): ln_final + unembed -> concept probs.

    Parameters
    ----------
    activations : (N, d_model)
    model : HookedTransformer
    concept_ids : LLM token IDs for HMM emission symbols

    Returns
    -------
    probs : (N, n_concepts) softmax probabilities over concept tokens
    """
    device = model.unembed.W_U.device
    W_concept = model.unembed.W_U[:, concept_ids].detach()
    b_concept = model.unembed.b_U[concept_ids].detach()

    acts_t = torch.tensor(activations, dtype=torch.float32)
    probs_list = []

    with torch.no_grad():
        for start in range(0, acts_t.shape[0], batch_size):
            batch = acts_t[start : start + batch_size].to(device)
            normed = model.ln_final(batch)
            concept_logits = normed @ W_concept + b_concept
            probs_list.append(F.softmax(concept_logits, dim=-1).cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def apply_tuned_lens(
    activations: np.ndarray,
    translator: TunedLensTranslator,
    model,
    concept_ids: list[int],
    batch_size: int = 1024,
) -> np.ndarray:
    """Apply a trained tuned lens translator -> concept probs.

    Parameters
    ----------
    activations : (N, d_model)
    translator : trained TunedLensTranslator
    model : HookedTransformer
    concept_ids : LLM token IDs for HMM emission symbols

    Returns
    -------
    probs : (N, n_concepts)
    """
    device = model.unembed.W_U.device
    W_concept = model.unembed.W_U[:, concept_ids].detach()
    b_concept = model.unembed.b_U[concept_ids].detach()
    translator_dev = translator.to(device)

    acts_t = torch.tensor(activations, dtype=torch.float32)
    probs_list = []

    with torch.no_grad():
        for start in range(0, acts_t.shape[0], batch_size):
            batch = acts_t[start : start + batch_size].to(device)
            h = translator_dev(batch)
            normed = model.ln_final(h)
            concept_logits = normed @ W_concept + b_concept
            probs_list.append(F.softmax(concept_logits, dim=-1).cpu().numpy())

    translator.cpu()
    return np.concatenate(probs_list, axis=0)


def apply_tuned_lens_full_logits(
    activations: np.ndarray,
    translator: TunedLensTranslator,
    model,
    batch_size: int = 512,
) -> np.ndarray:
    """Apply tuned lens and return full-vocabulary logits.

    Returns
    -------
    logits : (N, vocab_size)
    """
    device = model.unembed.W_U.device
    W_U = model.unembed.W_U.detach()
    b_U = model.unembed.b_U.detach()
    translator_dev = translator.to(device)

    acts_t = torch.tensor(activations, dtype=torch.float32)
    logits_list = []

    with torch.no_grad():
        for start in range(0, acts_t.shape[0], batch_size):
            batch = acts_t[start : start + batch_size].to(device)
            h = translator_dev(batch)
            normed = model.ln_final(h)
            full_logits = normed @ W_U + b_U
            logits_list.append(full_logits.cpu().float().numpy())

    translator.cpu()
    return np.concatenate(logits_list, axis=0)