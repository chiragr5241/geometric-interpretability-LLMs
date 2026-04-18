"""HMM and belief-state geometry utilities (e.g. Mess3, Fern, barycentric visualization)."""
from .hmm import BeliefHMM, FernHMM, Mess3HMM, belief_to_barycentric, belief_to_barycentric_evolution, belief_to_hex


def build_process_hmm(process_name: str, process_params: dict) -> BeliefHMM:
    """Instantiate and configure an HMM by name.

    Parameters
    ----------
    process_name : str
        ``"mess3"`` or ``"fern"``.
    process_params : dict
        Keyword arguments forwarded to the HMM's ``create_hmm`` method.
        - mess3: ``{"x": float, "alpha": float}`` or ``{"x": float, "a": float}``
          (``a`` is accepted as an alias for ``alpha``, matching simplexity's convention)
        - fern:  ``{"x": float}``
    """
    if process_name == "mess3":
        params = dict(process_params)
        if "a" in params and "alpha" not in params:
            params["alpha"] = params.pop("a")
        hmm = Mess3HMM()
        hmm.create_hmm(**params)
        return hmm
    if process_name == "fern":
        hmm = FernHMM()
        hmm.create_hmm(**process_params)
        return hmm
    raise ValueError(f"Unknown process_name {process_name!r}. Expected 'mess3' or 'fern'.")


__all__ = [
    "BeliefHMM",
    "FernHMM",
    "Mess3HMM",
    "build_process_hmm",
    "belief_to_hex",
    "belief_to_barycentric",
    "belief_to_barycentric_evolution",
]
