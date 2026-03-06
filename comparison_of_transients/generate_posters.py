"""
Generate all poster figures from a single script.

Produces:
1. Fluorescent transient comparison posters (stacked, zoomed, grid, grid+decay50)
2. Mechanical transient comparison posters (stacked, zoomed, grid)
3. Metrics bar charts (FWHM version, Decay50/V4 version with stats + n-values heatmap)

Usage:
    python comparison_of_transients/generate_posters.py
"""

import os
import sys

# Ensure this directory is on the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (
    FLUORESCENT_EXPERIMENTS, ROI_INDEX, CLIP_SEC,
    SMOOTH_WINDOW_SEC, SMOOTH_POLYORDER,
)


def generate_fluorescent_posters():
    """Generate all fluorescent transient comparison figures."""
    from compare_transients import plot_comparison

    print("=" * 70)
    print("FLUORESCENT TRANSIENT POSTERS")
    print("=" * 70)

    # Standard stacked version
    plot_comparison(
        FLUORESCENT_EXPERIMENTS,
        clip_sec=CLIP_SEC,
        roi_index=ROI_INDEX,
        smooth_window_sec=SMOOTH_WINDOW_SEC,
        smooth_polyorder=SMOOTH_POLYORDER,
        output_path="comparison_of_transients/comparison_poster.png",
    )

    # Zoomed poster: 5s, taller panels, full drug names
    plot_comparison(
        FLUORESCENT_EXPERIMENTS,
        clip_sec=5,
        roi_index=ROI_INDEX,
        smooth_window_sec=SMOOTH_WINDOW_SEC,
        smooth_polyorder=SMOOTH_POLYORDER,
        output_path="comparison_of_transients/transience_comparison_poster2.png",
        ylim=(0.97, 1.05),
        panel_height=3.5,
        line_width=3.5,
        raw_line_width=1.2,
        font_scale=1.3,
        label_overrides=[
            "Control",
            "Thapsigargin 0.1 nM",
            "Dofetilide 100 nM",
            "Quinidine 30 nM",
        ],
    )

    # 2x2 grid poster
    plot_comparison(
        FLUORESCENT_EXPERIMENTS,
        clip_sec=5,
        roi_index=ROI_INDEX,
        smooth_window_sec=SMOOTH_WINDOW_SEC,
        smooth_polyorder=SMOOTH_POLYORDER,
        output_path="comparison_of_transients/transience_comparison_poster3.png",
        ylim=(0.97, 1.05),
        panel_height=3.5,
        line_width=3.5,
        raw_line_width=1.2,
        font_scale=1.4,
        label_overrides=[
            "Control",
            "Thapsigargin 0.1 nM",
            "Dofetilide 100 nM",
            "Quinidine 30 nM",
        ],
        layout="grid",
    )

    # 2x2 grid with decay50 annotations
    plot_comparison(
        FLUORESCENT_EXPERIMENTS,
        clip_sec=5,
        roi_index=ROI_INDEX,
        smooth_window_sec=SMOOTH_WINDOW_SEC,
        smooth_polyorder=SMOOTH_POLYORDER,
        output_path="comparison_of_transients/transience_comparison_poster_v4.png",
        ylim=(0.97, 1.05),
        panel_height=3.5,
        line_width=3.5,
        raw_line_width=1.2,
        font_scale=1.4,
        label_overrides=[
            "Control",
            "Thapsigargin 0.1 nM",
            "Dofetilide 100 nM",
            "Quinidine 30 nM",
        ],
        layout="grid",
        show_decay50=True,
    )


def generate_mechanical_posters():
    """Generate all mechanical transient comparison figures."""
    from compare_mechanical_transients import plot_comparison

    print("\n" + "=" * 70)
    print("MECHANICAL TRANSIENT POSTERS")
    print("=" * 70)

    # Standard version
    plot_comparison()

    # Zoomed poster: 5s, full drug names
    plot_comparison(
        output_path="comparison_of_transients/mechanical_comparison_poster2.png",
        clip_sec=5,
        ylim=(-0.03, 0.03),
        panel_height=3.5,
        line_width=3.5,
        raw_line_width=1.2,
        font_scale=1.3,
        label_overrides=[
            "Control",
            "Thapsigargin 0.1 nM",
            "Dofetilide 100 nM",
            "Quinidine 30 nM",
        ],
    )

    # 2x2 grid poster
    plot_comparison(
        output_path="comparison_of_transients/mechanical_comparison_poster3.png",
        clip_sec=5,
        ylim=(-0.03, 0.03),
        panel_height=3.5,
        line_width=3.5,
        raw_line_width=1.2,
        font_scale=1.4,
        label_overrides=[
            "Control",
            "Thapsigargin 0.1 nM",
            "Dofetilide 100 nM",
            "Quinidine 30 nM",
        ],
        layout="grid",
    )


def generate_metrics_posters():
    """Generate metrics bar charts with stats and n-values heatmap."""
    from compare_metrics import (
        plot_metrics,
        METRICS_DATA, METRICS_DATA_V4,
        METRICS_ERROR, METRICS_ERROR_V4,
    )
    from experiments import DRUG_LABELS, BAR_COLORS

    print("\n" + "=" * 70)
    print("METRICS BAR CHARTS")
    print("=" * 70)

    # Standard version (with FWHM)
    plot_metrics(DRUG_LABELS, METRICS_DATA, BAR_COLORS,
                 "comparison_of_transients/metrics_poster.png",
                 metrics_error=METRICS_ERROR)

    # V4: Decay50 with statistical significance
    from statistical_analysis import get_significance
    sig = get_significance()
    plot_metrics(DRUG_LABELS, METRICS_DATA_V4, BAR_COLORS,
                 "comparison_of_transients/metrics_poster_v4.png",
                 metrics_error=METRICS_ERROR_V4,
                 significance=sig)

    # n-values heatmap
    from n_values import plot_n_values_heatmap
    plot_n_values_heatmap()


if __name__ == "__main__":
    generate_fluorescent_posters()
    generate_mechanical_posters()
    generate_metrics_posters()
    print("\n" + "=" * 70)
    print("ALL POSTER FIGURES GENERATED")
    print("=" * 70)
