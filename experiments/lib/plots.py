"""Shared plotting utilities for VELM experiments."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plots(out_dir, losses_dict, accs, gap_accs, sample_acts):
    """Save all standard experiment plots to out_dir.

    Args:
        out_dir: output directory (created if needed)
        losses_dict: {name: [loss_values]}
        accs: {name: accuracy}
        gap_accs: {name: [(gap, accuracy), ...]}
        sample_acts: {name: 2d_array}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Loss curves
    plt.figure()
    for k, v in losses_dict.items():
        plt.plot(v, label=k)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    # Summary text
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for name, acc in accs.items():
            f.write(f"{name}: accuracy={acc:.4f}\n")

    # Gap accuracy
    plt.figure()
    for name, ga in gap_accs.items():
        gaps = [g for g, a in ga]
        acc = [a for g, a in ga]
        plt.plot(gaps, acc, "-o", label=name)
    plt.xlabel("Gap (positions between memory and query)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs gap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_accuracy.png"))
    plt.close()

    # Activation heatmaps
    for name, mat in sample_acts.items():
        plt.figure(figsize=(6, 4))
        plt.imshow(mat, aspect="auto", cmap="bwr")
        plt.colorbar()
        plt.title(f"Activation heatmap: {name}")
        plt.xlabel("Hidden dim")
        plt.ylabel("Position / Block")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"act_{name}.png"))
        plt.close()
