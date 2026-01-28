import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters", type=int, nargs="+", default=[1, 2], help="Iterations to plot"
    )
    parser.add_argument(
        "--save", type=str, default="../outputs/training_progress.png", help="Save path"
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, iter_num in enumerate(args.iters):
        history_path = f"../outputs/iter{iter_num}/training_history.npy"

        if not os.path.exists(history_path):
            print(f"No history found for iter{iter_num}, skipping")
            continue

        history = np.load(history_path, allow_pickle=True).item()

        gens = history["generation"]
        best = history["best"]
        mean = history["mean"]
        worst = history["worst"]

        color = colors[i % len(colors)]

        # Left plot: Best scores
        axes[0].plot(
            gens,
            best,
            label=f"Iter {iter_num} (peak: {max(best):.1f})",
            color=color,
            linewidth=2,
        )

        # Right plot: Mean with shaded region
        axes[1].plot(gens, mean, label=f"Iter {iter_num}", color=color, linewidth=2)
        axes[1].fill_between(gens, worst, best, alpha=0.2, color=color)

    # Solved threshold line
    for ax in axes:
        ax.axhline(y=900, color="red", linestyle="--", alpha=0.5, label="Solved (900)")

    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Reward")
    axes[0].set_title("Best Reward per Generation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Mean Reward (shaded: worst to best)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("World Models Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")

    if not args.no_show:
        plt.show()
