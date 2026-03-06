"""
n_values.py — Sample-size heatmap for the metrics poster.

Shows how many data points (colour) and how many recordings/organoids
(number printed in each cell) back each drug × metric combination.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from experiments import (
    DECAY50_EXPERIMENTS as FLUO_EXPERIMENTS,
    MECHANICAL_EXPERIMENTS as MECH_EXPERIMENTS,
    IBI_EXPERIMENTS, DRUG_LABELS,
)
from statistical_analysis import (
    collect_decay50_per_beat,
    collect_ibi_per_beat,
    collect_mechanical_per_beat,
)

OUTPUT_PATH = "comparison_of_transients/n_values_heatmap.png"

METRIC_COLS = ["Decay50", "BtB Variability", "Contraction Time", "Relaxation Time"]


def plot_n_values_heatmap(output_path=OUTPUT_PATH):
    """Build and save the sample-size heatmap."""

    # --- collect per-beat counts ---
    decay50 = collect_decay50_per_beat()
    ibi = collect_ibi_per_beat()
    ct, rt = collect_mechanical_per_beat()

    n_points = np.zeros((len(DRUG_LABELS), len(METRIC_COLS)), dtype=int)
    n_recordings = np.zeros_like(n_points)

    for i, drug in enumerate(DRUG_LABELS):
        # data-point counts
        n_points[i, 0] = len(decay50.get(drug, []))
        n_points[i, 1] = len(ibi.get(drug, []))
        n_points[i, 2] = len(ct.get(drug, []))
        n_points[i, 3] = len(rt.get(drug, []))

        # recording / organoid counts
        # Decay50: one CSV per entry in FLUO_EXPERIMENTS
        n_recordings[i, 0] = sum(1 for lbl, _ in FLUO_EXPERIMENTS if lbl == drug)
        # IBI
        n_recordings[i, 1] = len(IBI_EXPERIMENTS.get(drug, []))
        # Mechanical (CT & RT share the same recordings)
        n_recordings[i, 2] = len(MECH_EXPERIMENTS.get(drug, []))
        n_recordings[i, 3] = n_recordings[i, 2]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 4))

    cmap = plt.cm.Purples
    norm = mcolors.LogNorm(vmin=max(1, n_points[n_points > 0].min()), vmax=n_points.max())

    im = ax.imshow(n_points, cmap=cmap, norm=norm, aspect="auto")

    # annotate each cell with the number of recordings
    for i in range(len(DRUG_LABELS)):
        for j in range(len(METRIC_COLS)):
            val = n_points[i, j]
            text_color = "white" if val > n_points.max() / 2 else "black"
            ax.text(j, i, str(n_recordings[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=text_color)

    # axes
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, fontsize=10)
    ax.set_yticks(range(len(DRUG_LABELS)))
    ax.set_yticklabels(DRUG_LABELS, fontsize=10)

    # group headers
    ax.text(0.5, -0.6, "Calcium", ha="center", va="center",
            fontsize=11, fontweight="bold", transform=ax.transData)
    ax.text(2.5, -0.6, "Mechanical", ha="center", va="center",
            fontsize=11, fontweight="bold", transform=ax.transData)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("n data points", fontsize=10)

    ax.set_title("Sample sizes per condition", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    plot_n_values_heatmap()
