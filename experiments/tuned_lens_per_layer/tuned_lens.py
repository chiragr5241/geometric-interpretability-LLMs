"""Tuned lens translator model and training."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    return f"{int(m)}m{s:04.1f}s"


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


# ──────────────────── eval-time weight extraction ────────────────────────────


@dataclass
class EvalWeights:
    """Concept-column weights extracted from the backbone for evaluation.

    Holds only the parts needed by apply_logit_lens / apply_tuned_lens:
    the concept-token columns of W_U, their bias, and a copy of ln_final.
    Tensors live on the same device as the model at extraction time.

    Call extract_eval_weights() while the backbone is still on GPU, then
    optionally move the backbone to CPU (freeing ~18 GB) before evaluation.
    """

    W_c: torch.Tensor        # [d_model, n_concepts] float32
    b_c: torch.Tensor        # [n_concepts] float32
    ln_final: nn.Module      # LayerNorm (no grad)
    device: torch.device


def extract_eval_weights(model, concept_ids: list[int]) -> EvalWeights:
    """Extract the minimal weights for evaluation from a HookedTransformer.

    Call this while the model is still on GPU. The result stores only the
    concept-column slices of W_U (~KB) rather than the full matrix (~GB),
    so the backbone can be moved to CPU or deleted after this call.
    """
    device = model.unembed.W_U.device
    W_c = model.unembed.W_U[:, concept_ids].detach().to(torch.float32)
    b_c = model.unembed.b_U[concept_ids].detach().to(torch.float32)
    ln_final = copy.deepcopy(model.ln_final)
    for p in ln_final.parameters():
        p.requires_grad_(False)
    return EvalWeights(W_c=W_c, b_c=b_c, ln_final=ln_final, device=device)


def save_translators(translators: dict[int, TunedLensTranslator], save_dir: Path) -> None:
    """Save translator state_dicts to ``save_dir/layer_<L>.pt``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for layer, translator in translators.items():
        torch.save(translator.state_dict(), save_dir / f"layer_{layer}.pt")


def load_translator(save_dir: Path, layer: int) -> TunedLensTranslator:
    """Load a translator from ``save_dir/layer_<layer>.pt``.

    Infers d_model from the saved weight shape — no separate d_model arg needed.
    """
    state = torch.load(save_dir / f"layer_{layer}.pt", weights_only=True, map_location="cpu")
    d_model = state["linear.weight"].shape[0]
    t = TunedLensTranslator(d_model)
    t.load_state_dict(state)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def train_tuned_lens(
    activations_by_layer: dict[int, np.ndarray],
    model,
    layers: list[int],
    target_final_resid: np.ndarray,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    optimizer_name: str = "adam",
    use_bf16: bool = False,
) -> tuple[dict[int, TunedLensTranslator], dict[int, list[float]]]:
    """Train one TunedLensTranslator per layer (faithful full-vocabulary version).

    Trains each translator so that:
        softmax(unembed(ln_final(T_l(h_l))))
    matches the model's full output distribution via KL(p_model || p_lens).

    Memory-optimized: instead of pre-storing the full ``(N, vocab_size)``
    target log-probabilities tensor (≈24 GB for Qwen-9B at 24K positions),
    we cache the final-layer residual ``(N, d_model)`` (≈0.4 GB) and recompute
    the target log_softmax per batch. The unembed matmul is cheap (~3 ms on
    A40 for batch=512, vocab=248K) and removes the dominant memory footprint.

    Parameters
    ----------
    activations_by_layer : dict mapping layer index -> (N, d_model) array
    model : HookedTransformer
    layers : list of layer indices to train
    target_final_resid : (N, d_model) cached final-layer residual whose
        ``ln_final + unembed + softmax`` defines the target distribution
    n_epochs, lr, batch_size : training hyperparameters
    """
    device = model.unembed.W_U.device
    d_model = model.cfg.d_model

    W_U = model.unembed.W_U.detach()
    b_U = model.unembed.b_U.detach()

    # Cache final residual on GPU once (small: N * d_model * 4 bytes).
    target_resid_gpu = torch.from_numpy(target_final_resid).to(
        device=device, dtype=torch.float32
    )
    n = target_resid_gpu.shape[0]

    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}

    for layer in tqdm(layers, desc="Training tuned lens"):
        layer_t0 = time.time()
        acts = torch.from_numpy(activations_by_layer[layer]).to(
            device=device, dtype=torch.float32
        )

        translator = TunedLensTranslator(d_model).to(device)
        optimizer = _make_optimizer(optimizer_name, translator.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        epoch_losses = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                acts_batch = acts[idx]

                # Compute target log-probs for this batch from cached residuals.
                with torch.no_grad():
                    target_resid_batch = target_resid_gpu[idx]
                    normed_t = model.ln_final(target_resid_batch)
                    target_logits = normed_t @ W_U + b_U
                    target_log_probs_batch = F.log_softmax(
                        target_logits.float(), dim=-1
                    )

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    h = translator(acts_batch)
                    normed = model.ln_final(h)
                    full_logits = normed @ W_U + b_U
                # cast to float32 before log_softmax: bfloat16 lacks precision over large vocabs
                log_probs = F.log_softmax(full_logits.float(), dim=-1)

                loss = F.kl_div(
                    log_probs, target_log_probs_batch,
                    reduction="batchmean", log_target=True,
                )
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
        del acts
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(
            f"  Layer {layer:2d}: final KL = {epoch_losses[-1]:.4f}  "
            f"(took {_fmt_secs(time.time() - layer_t0)})"
        )

    del target_resid_gpu
    torch.cuda.empty_cache()
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
    use_bf16: bool = False,
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
    target_probs = (target if target_is_probs else F.softmax(target, dim=-1)).to(device)

    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}

    for layer in tqdm(layers, desc="Training tuned lens (concept)"):
        layer_t0 = time.time()
        acts = torch.from_numpy(activations_by_layer[layer]).to(
            device=device, dtype=torch.float32,
        )
        n = acts.shape[0]

        translator = TunedLensTranslator(d_model).to(device)
        optimizer = _make_optimizer(optimizer_name, translator.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        epoch_losses = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                acts_batch = acts[idx]
                target_batch = target_probs[idx]

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    h = translator(acts_batch)
                    normed = model.ln_final(h)
                    concept_logit = normed @ W_c + b_c
                log_probs = F.log_softmax(concept_logit.float(), dim=-1)

                loss = F.kl_div(log_probs, target_batch.float(), reduction="batchmean")
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
        del acts
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(
            f"  Layer {layer:2d}: final KL = {epoch_losses[-1]:.4f}  "
            f"(took {_fmt_secs(time.time() - layer_t0)})"
        )

    return translators, loss_curves


def _concept_softmax_fp32(
    h: torch.Tensor,
    ln_final: nn.Module,
    W_c_fp32: torch.Tensor,
    b_c_fp32: torch.Tensor,
) -> torch.Tensor:
    """Apply ln_final + concept-only unembed + softmax, all in fp32.

    Explicit fp32 cast prevents the fp16 overflow path that produced
    spurious +inf KL values in early-layer logit-lens in old runs.
    """
    normed = ln_final(h).to(torch.float32)
    concept_logits = normed @ W_c_fp32 + b_c_fp32
    return F.softmax(concept_logits, dim=-1)


def _check_finite(probs: np.ndarray, where: str, layer: int | None = None) -> np.ndarray:
    """Defense in depth: replace any non-finite probs with a uniform row.

    Logs a warning with offending row counts so silent corruption never
    propagates into the metrics.
    """
    if np.isfinite(probs).all():
        return probs
    bad_rows = ~np.isfinite(probs).all(axis=-1)
    n_bad = int(bad_rows.sum())
    n_concepts = probs.shape[-1]
    layer_str = f"layer={layer}" if layer is not None else "?"
    logger.warning(
        f"{where} ({layer_str}): {n_bad}/{len(probs)} rows had non-finite probs; "
        f"replacing with uniform 1/{n_concepts}"
    )
    probs = probs.copy()
    probs[bad_rows] = 1.0 / n_concepts
    return probs


def apply_logit_lens(
    activations: np.ndarray,
    eval_weights: EvalWeights,
    batch_size: int = 1024,
    layer: int | None = None,
) -> np.ndarray:
    """Apply the raw logit lens (no translator): ln_final + concept-only unembed -> probs.

    End-to-end fp32 to prevent the fp16 overflow path that produced spurious
    +inf KL at early layers in old runs.
    """
    device = eval_weights.device
    W_c = eval_weights.W_c.to(device)
    b_c = eval_weights.b_c.to(device)
    ln_final = eval_weights.ln_final.to(device)

    acts_t = torch.from_numpy(np.ascontiguousarray(activations)).to(torch.float32)
    probs_list = []

    with torch.no_grad():
        for start in range(0, acts_t.shape[0], batch_size):
            batch = acts_t[start : start + batch_size].to(device)
            probs_list.append(_concept_softmax_fp32(batch, ln_final, W_c, b_c).cpu().numpy())

    return _check_finite(np.concatenate(probs_list, axis=0), "apply_logit_lens", layer)


def apply_tuned_lens(
    activations: np.ndarray,
    translator: TunedLensTranslator,
    eval_weights: EvalWeights,
    batch_size: int = 1024,
    layer: int | None = None,
) -> np.ndarray:
    """Apply a trained translator then concept-only unembed -> probs (fp32 throughout)."""
    device = eval_weights.device
    W_c = eval_weights.W_c.to(device)
    b_c = eval_weights.b_c.to(device)
    ln_final = eval_weights.ln_final.to(device)
    translator_dev = translator.to(device)

    acts_t = torch.from_numpy(np.ascontiguousarray(activations)).to(torch.float32)
    probs_list = []

    with torch.no_grad():
        for start in range(0, acts_t.shape[0], batch_size):
            batch = acts_t[start : start + batch_size].to(device)
            h = translator_dev(batch)
            probs_list.append(_concept_softmax_fp32(h, ln_final, W_c, b_c).cpu().numpy())

    translator.cpu()
    return _check_finite(np.concatenate(probs_list, axis=0), "apply_tuned_lens", layer)


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

    acts_t = torch.from_numpy(np.ascontiguousarray(activations)).to(torch.float32)
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