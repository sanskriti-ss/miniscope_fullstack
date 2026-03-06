"""
Compare fluorescence and mechanical metrics across experiments as a bar chart.

Usage:
    python comparison_of_transients/compare_metrics.py

Edit the METRICS_DATA dictionary below with real values once available.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import unified config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import DRUG_LABELS as EXPERIMENTS, BAR_COLORS as COLORS

# ============================================================================
# CONFIGURATION
# ============================================================================

# Metrics to plot — each key is a metric name, value is a list of values per experiment
# Order: [Control, Thar 0.1nM, DOF 100nM, QUAN 30nM]
# Extracted from FF0_roi1_smooth with 0.3s heavy smoothing + valley-depth peak merging
# FWHM uses inter-group trough baseline with interpolated half-max crossings
METRICS_DATA = {
    "FWHM\n(s)":              [0.207, 0.214, 0.338, 0.431],
    "Beat-to-Beat\nVariability (%)": [4.6,   38.3,  25.9,  7.1],
    "Contraction\nTime (s)":  [0.593, 0.718, 0.674, 0.464],    # From mechanical_experiments.py
    "Relaxation\nTime (s)":   [0.475, 0.411, 0.846, 0.568],    # From mechanical_experiments.py
}

# V4 metrics: replace FWHM with Decay50
# Decay50 extracted from FF0_roi1_smooth with 0.3s heavy smoothing
# Order: [Control, Thar 0.1nM, DOF 100nM, QUAN 30nM]
METRICS_DATA_V4 = {
    "Decay50\n(s)":           [0.109, 0.089, 0.200, 0.240],
    "Beat-to-Beat\nVariability (%)": [4.6,   38.3,  25.9,  7.1],
    "Contraction\nTime (s)":  [0.593, 0.718, 0.674, 0.464],
    "Relaxation\nTime (s)":   [0.475, 0.411, 0.846, 0.568],
}

# Error bars (std dev across peaks within each experiment)
METRICS_ERROR = {
    "FWHM\n(s)":              [0.014, 0.081, 0.151, 0.086],
    "Beat-to-Beat\nVariability (%)": [None,  None,  None,  None],   # CV of inter-beat intervals, no error bar
    "Contraction\nTime (s)":  [0.299, 0.365, 0.392, 0.201],    # From mechanical_experiments.py
    "Relaxation\nTime (s)":   [0.202, 0.280, 0.376, 0.164],    # From mechanical_experiments.py
}

METRICS_ERROR_V4 = {
    "Decay50\n(s)":           [0.014, 0.035, 0.115, 0.080],
    "Beat-to-Beat\nVariability (%)": [None,  None,  None,  None],
    "Contraction\nTime (s)":  [0.299, 0.365, 0.392, 0.201],
    "Relaxation\nTime (s)":   [0.202, 0.280, 0.376, 0.164],
}

OUTPUT_PATH = "comparison_of_transients/metrics_poster.png"


# ============================================================================
# IMPLEMENTATION
# ============================================================================

def plot_metrics(experiments, metrics_data, colors, output_path, metrics_error=None,
                 significance=None):
    """
    Create a grouped bar chart comparing metrics across experiments.

    Parameters
    ----------
    experiments : list of str
        Experiment labels
    metrics_data : dict
        {metric_name: [value_per_experiment, ...]}
    colors : list of str
        One color per experiment
    output_path : str
        Where to save the figure
    metrics_error : dict, optional
        {metric_name: [error_per_experiment, ...]}
    significance : dict, optional
        {metric_name: ["", "*", "", "*"]} — "*" above bars with p < 0.05 vs Control
    """
    metric_names = list(metrics_data.keys())
    n_metrics = len(metric_names)
    n_experiments = len(experiments)

    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 10))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_experiments)
    bar_width = 0.6

    for ax, metric_name in zip(axes, metric_names):
        raw_values = metrics_data[metric_name]
        raw_errs = metrics_error.get(metric_name) if metrics_error else None
        # Replace None values with 0 for plotting, track which are missing
        values = [v if v is not None else 0 for v in raw_values]
        has_data = [v is not None for v in raw_values]
        errs = None
        if raw_errs and not all(e is None for e in raw_errs):
            errs = [e if e is not None else 0 for e in raw_errs]
        # Dim colors for missing data
        bar_colors = [c if hd else '#e0e0e0' for c, hd in zip(colors, has_data)]
        bars = ax.bar(x, values, width=bar_width, color=bar_colors, edgecolor='white', linewidth=1.2,
                      yerr=errs, capsize=5, error_kw={'linewidth': 1.5, 'color': '#333333'})

        # Add significance stars
        if significance and metric_name in significance:
            sigs = significance[metric_name]
            for i, star in enumerate(sigs):
                if star:
                    # Position star above the bar (or above error bar if present)
                    bar_top = values[i]
                    if errs:
                        bar_top += errs[i]
                    y_pos = bar_top + ax.get_ylim()[1] * 0.02
                    ax.text(x[i], y_pos, star, ha='center', va='bottom',
                            fontsize=28, fontweight='bold', color='black')
            # Add headroom so stars don't clip
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.15)

        ax.set_title(metric_name, fontsize=24, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=16)
        # Y-axis label from metric unit
        lines = metric_name.split('\n')
        unit = lines[-1].strip() if len(lines) > 1 else ""
        if unit == "(s)":
            ax.set_ylabel("Time (s)", fontsize=18, fontweight='bold')
        elif unit:
            ax.set_ylabel(unit, fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor='white', label=e)
                      for c, e in zip(colors, experiments)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=n_experiments,
               fontsize=22, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved metrics bar chart to {output_path}")


if __name__ == "__main__":
    plot_metrics(EXPERIMENTS, METRICS_DATA, COLORS, OUTPUT_PATH, metrics_error=METRICS_ERROR)

    # V4: Decay50 instead of FWHM, with statistical significance
    from statistical_analysis import get_significance
    sig = get_significance()
    plot_metrics(EXPERIMENTS, METRICS_DATA_V4, COLORS,
                 "comparison_of_transients/metrics_poster_v4.png",
                 metrics_error=METRICS_ERROR_V4,
                 significance=sig)

    from n_values import plot_n_values_heatmap
    plot_n_values_heatmap()
