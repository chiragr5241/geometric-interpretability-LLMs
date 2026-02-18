"""HMM and belief-state geometry utilities (e.g. Mess3, barycentric visualization)."""
from .hmm import Mess3HMM, belief_to_barycentric, belief_to_barycentric_evolution, belief_to_hex

__all__ = [
    "Mess3HMM",
    "belief_to_hex",
    "belief_to_barycentric",
    "belief_to_barycentric_evolution",
]
