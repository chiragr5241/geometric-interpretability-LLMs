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
    lr: float = 1e-5,
    batch_size: int = 512,
    optimizer_name: str = "adam",
    use_bf16: bool = False,
    layer_chunk: int = 4,
    device: torch.device | str | None = None,
) -> tuple[dict[int, TunedLensTranslator], dict[int, list[float]]]:
    """Train one TunedLensTranslator per layer (full-vocabulary KL target).

    Trains each translator so that:
        softmax(unembed(ln_final(T_l(h_l))))
    matches the model's full output distribution via KL(p_model || p_lens).

    All L layers are trained in parallel as a single batched (L, D, D)
    translator. Per training step, the target log-softmax is computed once
    (shared by all layers), and the lens forward is run in chunks of
    ``layer_chunk`` layers to bound peak memory on the (L, B, V) tensor.

    The function deepcopies ``model.ln_final`` and moves it (plus W_U/b_U)
    to ``device``, so it is safe to call after ``model.cpu()`` — that frees
    ~18 GB of backbone weights from GPU for the duration of training.

    For Qwen3.5-9B at N=30K positions / V=248K / L=32 on A40 with backbone
    moved to CPU before training:
      - acts stacked on GPU:       ~15 GB fp32  (~7.5 GB if use_bf16)
      - W_U fp32:                  ~4 GB
      - W (L,D,D) + Adam state:    ~6 GB
      - peak lens_logits/backward: ~4 GB at layer_chunk=4
    Total ≈ 30 GB; comfortable on a 48 GB A40.

    Parameters
    ----------
    activations_by_layer : dict mapping layer index -> (N, d_model) array
    model : HookedTransformer-like (WrappedHFModel works too)
    layers : list of layer indices to train
    target_final_resid : (N, d_model) cached final-layer residual whose
        ``ln_final + unembed + softmax`` defines the target distribution
    n_epochs, lr, batch_size, optimizer_name : training hyperparameters
    use_bf16 : if True, autocast translator forward + store activations in bf16
    layer_chunk : how many layers to forward+backward at once per batch
    device : where to train; defaults to CUDA if available else CPU. Use this
        to train on GPU even when the backbone has been moved to CPU to save
        memory.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    d_model = model.cfg.d_model
    L = len(layers)

    # Unembed + final-layer norm on the training device, in fp32. This is what
    # makes train_tuned_lens robust to model.cpu() being called before training.
    W_U = model.unembed.W_U.detach().to(device=device, dtype=torch.float32)
    b_U = model.unembed.b_U.detach().to(device=device, dtype=torch.float32)
    ln_final = copy.deepcopy(model.ln_final).to(device)
    for p in ln_final.parameters():
        p.requires_grad_(False)

    # Target residual on GPU once, fp32. (N * D * 4 bytes ≈ 0.5 GB.)
    target_resid_gpu = torch.from_numpy(target_final_resid).to(
        device=device, dtype=torch.float32,
    )
    n = target_resid_gpu.shape[0]

    # Stacked activations on GPU once: (L, N, D). bf16 if requested (halves memory).
    act_dtype = torch.bfloat16 if use_bf16 else torch.float32
    acts_stacked = torch.empty((L, n, d_model), device=device, dtype=act_dtype)
    for i, layer in enumerate(layers):
        acts_stacked[i].copy_(
            torch.from_numpy(activations_by_layer[layer]).to(device=device, dtype=act_dtype)
        )

    # Batched translator: identity-init W (L, D, D), zero bias (L, D), fp32.
    W = (
        torch.eye(d_model, device=device, dtype=torch.float32)
        .unsqueeze(0).expand(L, -1, -1).contiguous().clone()
    )
    b = torch.zeros(L, d_model, device=device, dtype=torch.float32)
    W.requires_grad_(True)
    b.requires_grad_(True)

    optimizer = _make_optimizer(optimizer_name, [W, b], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Per-layer epoch losses for the existing loss_curves return shape.
    epoch_losses = torch.zeros(L, n_epochs, device=device, dtype=torch.float32)

    t_train = time.time()
    for epoch in tqdm(range(n_epochs), desc="Training tuned lens (parallel)"):
        perm = torch.randperm(n, device=device)
        epoch_sum = torch.zeros(L, device=device, dtype=torch.float32)
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            B = idx.shape[0]

            # ── target log-softmax: ONCE per batch (no_grad, fp32) ────────────
            with torch.no_grad():
                tgt_resid = target_resid_gpu.index_select(0, idx)        # (B, D)
                tgt_logits = ln_final(tgt_resid).to(torch.float32) @ W_U + b_U  # (B, V)
                target_lp = F.log_softmax(tgt_logits, dim=-1)            # (B, V) fp32
                target_p = target_lp.exp()                               # (B, V) fp32

            optimizer.zero_grad(set_to_none=True)
            batch_kl = torch.zeros(L, device=device, dtype=torch.float32)

            # ── lens forward, chunked across layers ──────────────────────────
            for cs in range(0, L, layer_chunk):
                ce = min(cs + layer_chunk, L)
                Lc = ce - cs

                acts_chunk = acts_stacked[cs:ce].index_select(1, idx)    # (Lc, B, D)

                # Translator forward: y[l] = acts[l] @ W[l].T + b[l]
                # Done in act_dtype (bf16 with autocast if requested), else fp32.
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lens_h = (
                            torch.bmm(acts_chunk, W[cs:ce].to(torch.bfloat16).transpose(-1, -2))
                            + b[cs:ce].to(torch.bfloat16).unsqueeze(1)
                        )                                                # (Lc, B, D) bf16
                        lens_h_normed = ln_final(lens_h)                 # (Lc, B, D)
                else:
                    lens_h = (
                        torch.bmm(acts_chunk, W[cs:ce].transpose(-1, -2))
                        + b[cs:ce].unsqueeze(1)
                    )                                                    # (Lc, B, D) fp32
                    lens_h_normed = ln_final(lens_h)                     # (Lc, B, D)

                # Unembed + log-softmax in fp32 (vocab is huge, precision matters).
                lens_logits = lens_h_normed.to(torch.float32) @ W_U + b_U  # (Lc, B, V)
                lens_lp = F.log_softmax(lens_logits, dim=-1)             # (Lc, B, V) fp32

                # Per-layer batchmean KL(p_model || p_lens):
                # sum_v p_model * (log p_model - log p_lens), mean over B.
                chunk_kl = (
                    target_p.unsqueeze(0) * (target_lp.unsqueeze(0) - lens_lp)
                ).sum(dim=-1).mean(dim=-1)                               # (Lc,)
                chunk_kl.sum().backward()                                # grads into W[cs:ce], b[cs:ce]
                batch_kl[cs:ce] = chunk_kl.detach()

                del acts_chunk, lens_h, lens_h_normed, lens_logits, lens_lp

            optimizer.step()
            epoch_sum += batch_kl
            n_batches += 1

        scheduler.step()
        epoch_losses[:, epoch] = epoch_sum / max(n_batches, 1)

    train_secs = time.time() - t_train
    logger.info(
        f"  Tuned lens trained on {L} layers x {n_epochs} epochs in "
        f"{_fmt_secs(train_secs)} (parallel, layer_chunk={layer_chunk})"
    )

    # ── pack the batched W, b back into per-layer TunedLensTranslator dicts ──
    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}
    W_cpu = W.detach().cpu()
    b_cpu = b.detach().cpu()
    losses_cpu = epoch_losses.detach().cpu().tolist()
    for i, layer in enumerate(layers):
        t = TunedLensTranslator(d_model)
        t.linear.weight.data.copy_(W_cpu[i])
        t.linear.bias.data.copy_(b_cpu[i])
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        translators[layer] = t
        loss_curves[layer] = losses_cpu[i]
        logger.info(f"  Layer {layer:2d}: final KL = {losses_cpu[i][-1]:.4f}")

    del W, b, acts_stacked, target_resid_gpu
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return translators, loss_curves


def train_tuned_lens_concept(
    activations_by_layer: dict[int, np.ndarray],
    model,
    concept_ids: list[int],
    layers: list[int],
    target_concept_values: np.ndarray,
    target_is_probs: bool = False,
    n_epochs: int = 50,
    lr: float = 1e-5,
    batch_size: int = 512,
    optimizer_name: str = "adam",
    use_bf16: bool = False,
    device: torch.device | str | None = None,
) -> tuple[dict[int, TunedLensTranslator], dict[int, list[float]]]:
    """Train one TunedLensTranslator per layer using concept-token logits only.

    Parallel-layers variant: all L translators are trained simultaneously as a
    single batched (L, D, D) tensor. The concept vocab is tiny (V_c≈3 for HMM),
    so no layer chunking is needed. Robust to ``model.cpu()`` having been called
    before training — see ``device`` parameter.

    Parameters
    ----------
    activations_by_layer : dict mapping layer index -> (N, d_model) array
    model : HookedTransformer-like
    concept_ids : LLM token IDs for HMM emission symbols
    layers : list of layer indices to train
    target_concept_values : (N, n_concepts) — logits if target_is_probs=False,
        probabilities if target_is_probs=True
    target_is_probs : if True, target is already probabilities
    n_epochs, lr, batch_size, optimizer_name, use_bf16 : training hyperparameters
    device : where to train; defaults to CUDA if available. Use this to train
        on GPU even when the backbone has been moved to CPU to save memory.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    d_model = model.cfg.d_model
    L = len(layers)

    # ln_final + concept unembed on the training device, in fp32.
    ln_final = copy.deepcopy(model.ln_final).to(device)
    for p in ln_final.parameters():
        p.requires_grad_(False)

    # Concept-vocab columns of the unembed, always fp32.
    W_c = model.unembed.W_U[:, concept_ids].detach().to(device=device, dtype=torch.float32)
    b_c = model.unembed.b_U[concept_ids].detach().to(device=device, dtype=torch.float32)

    # Target probabilities, fp32.
    target = torch.as_tensor(target_concept_values, dtype=torch.float32)
    target_probs = (target if target_is_probs else F.softmax(target, dim=-1)).to(device)
    n = target_probs.shape[0]

    # Stacked activations (L, N, D) on GPU.
    act_dtype = torch.bfloat16 if use_bf16 else torch.float32
    acts_stacked = torch.empty((L, n, d_model), device=device, dtype=act_dtype)
    for i, layer in enumerate(layers):
        acts_stacked[i].copy_(
            torch.from_numpy(activations_by_layer[layer]).to(device=device, dtype=act_dtype)
        )

    # Batched translator: identity W (L, D, D), zero bias (L, D), fp32.
    W = (
        torch.eye(d_model, device=device, dtype=torch.float32)
        .unsqueeze(0).expand(L, -1, -1).contiguous().clone()
    )
    b = torch.zeros(L, d_model, device=device, dtype=torch.float32)
    W.requires_grad_(True)
    b.requires_grad_(True)

    optimizer = _make_optimizer(optimizer_name, [W, b], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    epoch_losses = torch.zeros(L, n_epochs, device=device, dtype=torch.float32)

    t_train = time.time()
    for epoch in tqdm(range(n_epochs), desc="Training tuned lens (concept, parallel)"):
        perm = torch.randperm(n, device=device)
        epoch_sum = torch.zeros(L, device=device, dtype=torch.float32)
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            B = idx.shape[0]

            tgt_p = target_probs.index_select(0, idx)                    # (B, V_c)
            acts_batch = acts_stacked.index_select(1, idx)               # (L, B, D)

            optimizer.zero_grad(set_to_none=True)

            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lens_h = (
                        torch.bmm(acts_batch, W.to(torch.bfloat16).transpose(-1, -2))
                        + b.to(torch.bfloat16).unsqueeze(1)
                    )                                                    # (L, B, D)
                    lens_h_normed = ln_final(lens_h)
            else:
                lens_h = (
                    torch.bmm(acts_batch, W.transpose(-1, -2))
                    + b.unsqueeze(1)
                )                                                        # (L, B, D)
                lens_h_normed = ln_final(lens_h)

            # Unembed and KL in fp32.
            concept_logits = lens_h_normed.to(torch.float32) @ W_c + b_c  # (L, B, V_c)
            log_probs = F.log_softmax(concept_logits, dim=-1)             # (L, B, V_c)

            # Per-layer batchmean KL(target || lens).
            kl_per_layer = F.kl_div(
                log_probs,
                tgt_p.unsqueeze(0).expand(L, -1, -1),
                reduction="none",
            ).sum(dim=-1).mean(dim=-1)                                    # (L,)
            kl_per_layer.sum().backward()
            optimizer.step()

            epoch_sum += kl_per_layer.detach()
            n_batches += 1

        scheduler.step()
        epoch_losses[:, epoch] = epoch_sum / max(n_batches, 1)

    train_secs = time.time() - t_train
    logger.info(
        f"  Concept tuned lens trained on {L} layers x {n_epochs} epochs in "
        f"{_fmt_secs(train_secs)} (parallel)"
    )

    translators: dict[int, TunedLensTranslator] = {}
    loss_curves: dict[int, list[float]] = {}
    W_cpu = W.detach().cpu()
    b_cpu = b.detach().cpu()
    losses_cpu = epoch_losses.detach().cpu().tolist()
    for i, layer in enumerate(layers):
        t = TunedLensTranslator(d_model)
        t.linear.weight.data.copy_(W_cpu[i])
        t.linear.bias.data.copy_(b_cpu[i])
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        translators[layer] = t
        loss_curves[layer] = losses_cpu[i]
        logger.info(f"  Layer {layer:2d}: final KL = {losses_cpu[i][-1]:.4f}")

    del W, b, acts_stacked, target_probs
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
    W_U = model.unembed.W_U.detach().to(torch.float32)
    b_U = model.unembed.b_U.detach().to(torch.float32)
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