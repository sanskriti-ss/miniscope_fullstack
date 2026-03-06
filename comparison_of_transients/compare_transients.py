"""
Compare calcium transient traces from multiple experiments on a shared time axis.

Usage:
    python comparison_of_transients/compare_transients.py

Reads fluorescence_traces.csv files from each experiment's plot folder,
produces a stacked poster-friendly figure with a shared time axis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# Import unified config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (
    FLUORESCENT_EXPERIMENTS as EXPERIMENTS,
    CLIP_SEC, ROI_INDEX, SMOOTH_WINDOW_SEC, SMOOTH_POLYORDER,
    get_experiment_colors,
)

OUTPUT_PATH = "comparison_of_transients/comparison_poster.png"


# ============================================================================
# IMPLEMENTATION
# ============================================================================

def load_experiment(csv_path, roi_index=None):
    """Load a fluorescence_traces.csv and return time + FF0 arrays."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    # Try relative to repo root first, then absolute
    if not os.path.isabs(csv_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(repo_root, csv_path)

    df = pd.read_csv(csv_path)

    # Pick ROI column
    ff0_cols = [c for c in df.columns if c.startswith("FF0_roi") and c.count("_") == 1]
    if not ff0_cols:
        raise ValueError(f"No FF0_roi columns in {csv_path}")

    if roi_index is not None:
        col = f"FF0_roi{roi_index}"
        if col not in ff0_cols:
            print(f"[Warning] ROI {roi_index} not found in {csv_path}, using {ff0_cols[0]}")
            col = ff0_cols[0]
    else:
        col = ff0_cols[0]

    return df["time_s"].values, df[col].values


def smooth_and_detect(time, raw, fps_est, window_sec=0.15, polyorder=2):
    """Smooth trace and detect peaks/dips. Returns smoothed, peaks, dips."""
    win = int(window_sec * fps_est)
    if win % 2 == 0:
        win += 1
    win = max(5, win)

    mask = ~np.isnan(raw)
    smoothed = raw.copy()
    if np.sum(mask) >= win:
        smoothed[mask] = savgol_filter(raw[mask], win, polyorder)

    signal_range = np.percentile(smoothed[mask], 95) - np.percentile(smoothed[mask], 5)
    prominence = max(0.002, signal_range * 0.25)
    min_distance = max(1, int(0.4 * fps_est))

    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)
    dips, _ = find_peaks(-smoothed, distance=min_distance, prominence=prominence)

    return smoothed, peaks, dips


def _compute_decay50_for_plot(time, smooth, fps):
    """Compute decay50 annotations for a smoothed FF0 trace."""
    from scipy.signal import savgol_filter as _sgf

    # Heavy smooth for detection
    win = int(0.3 * fps)
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(smooth) // 3))
    det = _sgf(smooth, win, 2) if win < len(smooth) else smooth.copy()

    signal_range = np.percentile(det, 95) - np.percentile(det, 5)
    prominence = max(0.003, signal_range * 0.3)
    min_distance = max(1, int(0.6 * fps))

    peaks, _ = find_peaks(det, distance=min_distance, prominence=prominence)
    if len(peaks) == 0:
        return [], det

    annotations = []
    for k, pk in enumerate(peaks):
        right_bound = peaks[k + 1] if k + 1 < len(peaks) else len(det)
        region = det[pk:right_bound]
        if len(region) < 3:
            continue
        trough_idx = pk + np.argmin(region)
        baseline = det[trough_idx]
        peak_val = det[pk]
        amplitude = peak_val - baseline
        if amplitude < 0.003:
            continue
        half_level = baseline + 0.5 * amplitude

        cross_idx = None
        for j in range(1, len(region)):
            if region[j] <= half_level:
                y0 = region[j - 1]
                y1 = region[j]
                frac = (y0 - half_level) / (y0 - y1) if (y0 - y1) != 0 else 0
                cross_idx = pk + (j - 1) + frac
                break
        if cross_idx is not None:
            cross_time = np.interp(cross_idx, np.arange(len(time)), time)
            d50 = cross_time - time[pk]
            if 0.02 < d50 < 2.0:
                annotations.append({
                    'peak_idx': pk, 'peak_val': peak_val,
                    'baseline': baseline, 'half_level': half_level,
                    'cross_time': cross_time, 'd50': d50,
                })
    return annotations, det


def plot_comparison(experiments, clip_sec=10, roi_index=1,
                    smooth_window_sec=0.15, smooth_polyorder=2,
                    output_path="comparison_poster.png",
                    ylim=(0.94, 1.06), panel_height=2.75,
                    line_width=2.5, raw_line_width=1.0,
                    font_scale=1.0, label_overrides=None,
                    layout="stacked", show_decay50=False):
    """
    Create a stacked comparison figure.

    Parameters
    ----------
    experiments : list of (label, csv_path)
    clip_sec : float
        Show first N seconds (0 = all data)
    roi_index : int or None
        Which ROI to plot (1-based)
    output_path : str
        Where to save the figure
    ylim : tuple
        (ymin, ymax) for all panels
    panel_height : float
        Height per panel in inches
    line_width : float
        Width of smooth trend line
    raw_line_width : float
        Width of raw trace line
    font_scale : float
        Multiplier for all font sizes
    layout : str
        "stacked" (Nx1) or "grid" (2x2)
    show_decay50 : bool
        If True, annotate decay50 on each beat
    """
    n = len(experiments)
    if n == 0:
        print("No experiments configured. Edit EXPERIMENTS in compare_transients.py.")
        return

    fs = font_scale

    if layout == "grid":
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes_2d = plt.subplots(nrows, ncols, figsize=(16, panel_height * nrows))
        axes = axes_2d.flatten()[:n]
        # Hide unused subplot slots
        for j in range(n, nrows * ncols):
            axes_2d.flatten()[j].set_visible(False)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(16, panel_height * n), sharex=True)
        if n == 1:
            axes = [axes]

    for i, (ax, (label, csv_path)) in enumerate(zip(axes, experiments)):
        time, raw = load_experiment(csv_path, roi_index=roi_index)

        # Clip
        if clip_sec and clip_sec > 0:
            clip_mask = time <= clip_sec
            time = time[clip_mask]
            raw = raw[clip_mask]

        # Estimate fps
        fps_est = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 25.0

        smoothed, peaks, dips = smooth_and_detect(
            time, raw, fps_est, smooth_window_sec, smooth_polyorder
        )

        # Plot with experiment-specific colors
        trend_color, raw_color = get_experiment_colors(label)
        ax.plot(time, raw, color=raw_color, linewidth=raw_line_width, alpha=0.4)
        ax.plot(time, smoothed, color=trend_color, linewidth=line_width, alpha=0.95)

        # Decay50 annotations
        if show_decay50:
            annots, det = _compute_decay50_for_plot(time, smoothed, fps_est)
            for a in annots:
                pk = a['peak_idx']
                ax.plot(time[pk], a['peak_val'], 'v', color='red', markersize=7 * fs, zorder=5)
                ax.hlines(a['half_level'], time[pk], a['cross_time'],
                          colors='red', linewidth=1.2, linestyles='dashed', alpha=0.7, zorder=4)
                ax.plot(a['cross_time'], a['half_level'], 'o', color='red',
                        markersize=4 * fs, zorder=5)
                ax.annotate(f"{a['d50']*1000:.0f}ms",
                            xy=(a['cross_time'], a['half_level']),
                            xytext=(3, 8), textcoords='offset points',
                            fontsize=8 * fs, color='red', fontweight='bold')

        display_label = label_overrides[i] if label_overrides and i < len(label_overrides) else label
        ax.set_ylabel("F/F0", fontsize=16 * fs)
        ax.set_ylim(*ylim)
        ax.set_title(display_label, fontsize=18 * fs, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14 * fs)
        ax.set_xlabel("Time (s)", fontsize=16 * fs)

    if layout != "grid":
        # Only bottom panel gets x-label in stacked mode
        for ax in axes[:-1]:
            ax.set_xlabel("")
        axes[-1].set_xlabel("Time (s)", fontsize=18 * fs)

    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    plot_comparison(
        EXPERIMENTS,
        clip_sec=CLIP_SEC,
        roi_index=ROI_INDEX,
        smooth_window_sec=SMOOTH_WINDOW_SEC,
        smooth_polyorder=SMOOTH_POLYORDER,
        output_path=OUTPUT_PATH,
    )

    # Zoomed-in poster version: 5s, taller panels, bigger features, full drug names
    plot_comparison(
        EXPERIMENTS,
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

    # 2x2 grid poster version
    plot_comparison(
        EXPERIMENTS,
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

    # V4: 2x2 grid with decay50 annotations
    plot_comparison(
        EXPERIMENTS,
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
