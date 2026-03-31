"""Generate the optimal belief geometry figure for the Mess3 process."""

import sys
sys.path.insert(0, "src")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch

from hmm.hmm import Mess3HMM


def to_barycentric(beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2) * beliefs[:, 2]
    return x, y


def generate_figure(output_path: str = "figures/optimal_belief_simplex.png", num_sequences: int = 10, seq_length: int = 5000):
    hmm = Mess3HMM(seed=42)
    hmm.create_hmm(x=0.05, alpha=0.9)

    tokens, _, states = hmm.generate_dataset(
        num_sequences=num_sequences, seq_length=seq_length, return_states=True
    )
    beliefs = hmm.compute_belief_state(tokens)
    beliefs_np = beliefs[:, 1:].detach().cpu().numpy().reshape(-1, 3)
    states_np = states.detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    sqrt3 = np.sqrt(3)
    triangle = Polygon(
        [[0, 0], [1, 0], [0.5, sqrt3 / 2]],
        fill=False, edgecolor="black", linewidth=1.0
    )
    ax.add_patch(triangle)

    for t in (1 / 3, 2 / 3):
        ax.plot([1 - t, 0.5 * (1 - t)], [0, (sqrt3 / 2) * (1 - t)],
                color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.plot([t, t + 0.5 * (1 - t)], [0, (sqrt3 / 2) * (1 - t)],
                color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.plot([0.5 * t, 1 - 0.5 * t], [(sqrt3 / 2) * t, (sqrt3 / 2) * t],
                color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

    bx, by = to_barycentric(beliefs_np)
    colors = np.clip(beliefs_np, 0, 1)

    ax.scatter(bx, by, c=colors, s=1.0, alpha=0.4, rasterized=True)

    label_offset = 0.06
    ax.text(0, -label_offset, r"$S_1$", ha="center", va="top", fontsize=11)
    ax.text(1, -label_offset, r"$S_2$", ha="center", va="top", fontsize=11)
    ax.text(0.5, sqrt3 / 2 + label_offset, r"$S_3$", ha="center", va="bottom", fontsize=11)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, sqrt3 / 2 + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="figures/optimal_belief_simplex.png")
    parser.add_argument("-n", "--num-sequences", type=int, default=10)
    parser.add_argument("-l", "--seq-length", type=int, default=5000)
    args = parser.parse_args()
    generate_figure(output_path=args.output, num_sequences=args.num_sequences, seq_length=args.seq_length)
