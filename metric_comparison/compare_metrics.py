"""
Metric Comparison Module
========================
Compares drug-effect metrics (Control vs QUAN) across fluorescent and
mechanical pipelines using real experimental data from all_results.csv.
Outputs grouped bar charts with error bars into metric_comparison/outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRUGS = ["Control", "QUAN"]
DRUG_COLORS = {
    "Control": "#5B4C9E",   # deep violet
    "QUAN": "#4A90D9",      # soft cornflower
}
VIDEO_TYPES = ["fluorescent", "mechanical"]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "plots",
                         "batch_results", "all_results.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# All possible metrics per video type (col_name -> display label)
# Only columns present in the CSV will be plotted.
FLUOR_METRICS_ALL = {
    "bpm": "Beat Rate (BPM)",
    "amplitude": "Peak Amplitude (dF/F0)",
    "rise_time_ms": "Rise Time (ms)",
    "decay_time_ms": "Decay Time (ms)",
    "beat_regularity_cv": "Beat Regularity (CV%)",
    "transient_duration_ms": "Transient Duration (ms)",
}

MECH_METRICS_ALL = {
    "bpm": "Contraction Rate (BPM)",
    "amplitude": "Contraction Amplitude",
    "contraction_velocity_pct_s": "Contraction Velocity (%/s)",
    "relaxation_velocity_pct_s": "Relaxation Velocity (%/s)",
    "beat_regularity_cv": "Beat Regularity (CV%)",
    "contraction_duration_ms": "Contraction Duration (ms)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _available_metrics(df, metrics_dict):
    """Return subset of metrics_dict where the column exists and has data."""
    available = {}
    for col, label in metrics_dict.items():
        if col in df.columns and df[col].notna().any():
            available[col] = label
    return available


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_metric_bars(df, col_to_label, title, out_path, colors, drugs):
    """Bar charts for each metric with SEM error bars."""
    n_metrics = len(col_to_label)
    if n_metrics == 0:
        return
    ncols = min(n_metrics, 3)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, (col, label) in enumerate(col_to_label.items()):
        ax = axes[idx]
        grouped = df.groupby("drug")[col]
        means = grouped.mean().reindex(drugs)
        sems = grouped.sem().reindex(drugs)

        x = np.arange(len(drugs))
        bar_colors = [colors[d] for d in drugs]
        ax.bar(x, means.values, yerr=sems.values, color=bar_colors,
               capsize=4, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(drugs, fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(len(col_to_label), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_bpm(df, out_path, colors, drugs):
    """Side-by-side fluorescent vs mechanical BPM."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, vtype in zip(axes, VIDEO_TYPES):
        sub = df[df["video_type"] == vtype]
        grouped = sub.groupby("drug")["bpm"]
        means = grouped.mean().reindex(drugs)
        sems = grouped.sem().reindex(drugs)

        x = np.arange(len(drugs))
        bar_colors = [colors[d] for d in drugs]
        ax.bar(x, means.values, yerr=sems.values, color=bar_colors,
               capsize=4, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(drugs, fontsize=10)
        ax.set_ylabel("BPM", fontsize=11)
        label = "Beat Rate (BPM)" if vtype == "fluorescent" else "Contraction Rate (BPM)"
        ax.set_title(f"{vtype.capitalize()} — {label}", fontsize=11,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Combined BPM Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load real data
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Keep only Control and QUAN
    df = df[df["drug"].isin(DRUGS)].copy()
    # Drop rows with 0 bpm (failed detections)
    df = df[df["bpm"] > 0]
    print(f"Loaded {len(df)} traces for {DRUGS}\n")

    # Fluorescent bar charts
    df_fluor = df[df["video_type"] == "fluorescent"]
    fluor_metrics = _available_metrics(df_fluor, FLUOR_METRICS_ALL)
    print(f"Fluorescent metrics available: {list(fluor_metrics.values())}")
    plot_metric_bars(
        df_fluor, fluor_metrics,
        "Fluorescent Pipeline — Control vs QUAN",
        os.path.join(OUT_DIR, "fluorescent_metrics_comparison.png"),
        DRUG_COLORS, DRUGS,
    )

    # Mechanical bar charts
    df_mech = df[df["video_type"] == "mechanical"]
    mech_metrics = _available_metrics(df_mech, MECH_METRICS_ALL)
    print(f"Mechanical metrics available: {list(mech_metrics.values())}")
    plot_metric_bars(
        df_mech, mech_metrics,
        "Mechanical Pipeline — Control vs QUAN",
        os.path.join(OUT_DIR, "mechanical_metrics_comparison.png"),
        DRUG_COLORS, DRUGS,
    )

    # Combined BPM
    plot_combined_bpm(
        df, os.path.join(OUT_DIR, "combined_bpm_comparison.png"),
        DRUG_COLORS, DRUGS,
    )

    print("\nDone. All plots saved to metric_comparison/outputs/")


if __name__ == "__main__":
    main()
