"""HMM and belief-state geometry utilities."""
from __future__ import annotations

import torch

from .hmm import (
    BeliefHMM,
    FernHMM,
    Mess3HMM,
    StrataHMM,
    WingHMM,
    belief_to_barycentric,
    belief_to_barycentric_evolution,
    belief_to_hex,
    set_hmm_device,
)


def build_process_hmm(
    process_name: str,
    process_params: dict,
    hmm_device: torch.device | str | None = None,
) -> BeliefHMM:
    """Instantiate and configure an HMM by name.

    Parameters
    ----------
    process_name : str
        ``"mess3"``, ``"fern"``, ``"strata"``, or ``"wing"``.
    process_params : dict
        Keyword arguments forwarded to the HMM's ``create_hmm`` method.
        - mess3: ``{"x": float, "alpha": float}`` or ``{"x": float, "a": float}``
          (``a`` is accepted as an alias for ``alpha``, matching simplexity's convention)
        - fern:  ``{"x": float}``
        - strata: ``{"a": float, "t0": float, "t1": float}``
        - wing: ``{"x": float, "y": float}``
    """
    params = dict(process_params)
    if process_name == "mess3":
        if "a" in params and "alpha" not in params:
            params["alpha"] = params.pop("a")
        hmm = Mess3HMM(hmm_device=hmm_device)
        hmm.create_hmm(**params)
        return hmm
    if process_name == "fern":
        hmm = FernHMM(hmm_device=hmm_device)
        hmm.create_hmm(**params)
        return hmm
    if process_name == "strata":
        hmm = StrataHMM(hmm_device=hmm_device)
        hmm.create_hmm(**params)
        return hmm
    if process_name == "wing":
        hmm = WingHMM(hmm_device=hmm_device)
        hmm.create_hmm(**params)
        return hmm
    expected = "'mess3', 'fern', 'strata', or 'wing'"
    raise ValueError(f"Unknown process_name {process_name!r}. Expected {expected}.")


__all__ = [
    "BeliefHMM",
    "FernHMM",
    "Mess3HMM",
    "StrataHMM",
    "WingHMM",
    "build_process_hmm",
    "set_hmm_device",
    "belief_to_hex",
    "belief_to_barycentric",
    "belief_to_barycentric_evolution",
]
