"""
Compare mechanical transient traces from multiple experiments on a shared time axis.

Reads contractility or fluorescence CSVs from each experiment folder (as configured
in experiments.py), produces a stacked poster-friendly figure.

Usage:
    python comparison_of_transients/compare_mechanical_transients.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Import unified config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (
    MECHANICAL_EXPERIMENTS as EXPERIMENTS,
    DRUG_LABELS, DRUG_COLORS, CLIP_SEC,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which experiment index to use per drug (first = 0)
EXPERIMENT_INDEX = {
    "Control": 0,
    "Thar 0.1 nM": 0,
    "DOF 100 nM": 0,    # 16_03_51
    "QUAN 30 nM": 0,    # 17_21_02
}

# Optional: start offset per drug (skip initial seconds with drift/transients)
START_OFFSET_SEC = {
    "Control": 0,
    "Thar 0.1 nM": 10,  # skip first 10s — baseline drift in early portion
    "DOF 100 nM": 0,
    "QUAN 30 nM": 0,
}

EXPERIMENT_COLORS = DRUG_COLORS

OUTPUT_PATH = "comparison_of_transients/mechanical_comparison_poster.png"


# ============================================================================
# IMPLEMENTATION
# ============================================================================

def find_csv(folder_path):
    """Find the best CSV in an experiment folder."""
    if not os.path.isabs(folder_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(repo_root, folder_path)

    for name in ["contractility_windowed.csv", "contractility_traces.csv",
                  "fluorescence_traces.csv"]:
        csv_path = os.path.join(folder_path, name)
        if os.path.exists(csv_path):
            return csv_path
    return None


def load_mechanical_trace(csv_path):
    """
    Load a mechanical trace CSV and return (time, raw_signal, smooth_signal).

    Handles both contractility pipeline (area_smooth) and fluorescence pipeline
    (FF0_roi*_smooth) outputs.
    """
    df = pd.read_csv(csv_path)
    time = df["time_s"].values

    # Try contractility columns first
    if "area_smooth" in df.columns:
        smooth = df["area_smooth"].values
        # Use area_detrend as raw overlay (trend removed but unsmoothed)
        raw = df["area_detrend"].values if "area_detrend" in df.columns else smooth
        return time, raw, smooth

    # Try fluorescence columns (brightfield processed through fluo pipeline)
    ff0_smooth = [c for c in df.columns if c.startswith("FF0_roi") and c.endswith("_smooth")]
    if ff0_smooth:
        smooth_col = ff0_smooth[0]
        raw_col = smooth_col.replace("_smooth", "")
        smooth = df[smooth_col].values
        raw = df[raw_col].values if raw_col in df.columns else smooth
        # Convert FF0 to fractional change (center around 0 like area_smooth)
        baseline = np.median(smooth)
        smooth = (smooth - baseline) / baseline if baseline > 0 else smooth
        raw = (raw - np.median(raw)) / np.median(raw) if np.median(raw) > 0 else raw
        return time, raw, smooth

    raise ValueError(f"No suitable signal columns in {csv_path}")


def smooth_trace(signal, fps, window_sec=0.15, polyorder=2):
    """Additional smoothing if needed."""
    win = int(window_sec * fps)
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(signal) // 3))
    if win >= len(signal):
        return signal
    return savgol_filter(signal, win, polyorder)


def plot_comparison(output_path=None, clip_sec=None, ylim=(-0.04, 0.04),
                    panel_height=2.75, line_width=2.5, raw_line_width=1.0,
                    font_scale=1.0, label_overrides=None, layout="stacked"):
    if output_path is None:
        output_path = OUTPUT_PATH
    if clip_sec is None:
        clip_sec = CLIP_SEC

    n = len(DRUG_LABELS)
    fs = font_scale

    if layout == "grid":
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes_2d = plt.subplots(nrows, ncols, figsize=(16, panel_height * nrows))
        axes = axes_2d.flatten()[:n]
        for j in range(n, nrows * ncols):
            axes_2d.flatten()[j].set_visible(False)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(16, panel_height * n), sharex=True)
        if n == 1:
            axes = [axes]

    for i, (ax, drug_label) in enumerate(zip(axes, DRUG_LABELS)):
        display_label = label_overrides[i] if label_overrides and i < len(label_overrides) else drug_label
        folder_paths = EXPERIMENTS.get(drug_label, [])
        idx = EXPERIMENT_INDEX.get(drug_label, 0)

        if not folder_paths or idx >= len(folder_paths):
            ax.set_title(f"{display_label} — NO DATA", fontsize=18 * fs, fontweight='bold', loc='left')
            ax.set_ylabel("Fractional\nChange", fontsize=16 * fs)
            continue

        csv_path = find_csv(folder_paths[idx])
        if csv_path is None:
            ax.set_title(f"{drug_label} — CSV NOT FOUND", fontsize=18 * fs, fontweight='bold', loc='left')
            continue

        time, raw, smooth = load_mechanical_trace(csv_path)

        # Apply start offset and clip
        start_off = START_OFFSET_SEC.get(drug_label, 0)
        if clip_sec and clip_sec > 0:
            mask = (time >= time[0] + start_off) & (time <= time[0] + start_off + clip_sec)
            time = time[mask]
            raw = raw[mask]
            smooth = smooth[mask]
            # Shift time to start at 0
            time = time - time[0]

        # Local linear detrend to remove any residual baseline drift in the window
        if len(smooth) > 10:
            coeffs = np.polyfit(time, smooth, 1)
            trend = np.polyval(coeffs, time)
            smooth = smooth - trend
            coeffs_r = np.polyfit(time, raw, 1)
            trend_r = np.polyval(coeffs_r, time)
            raw = raw - trend_r

        trend_color, raw_color = EXPERIMENT_COLORS[drug_label]
        ax.plot(time, raw, color=raw_color, linewidth=raw_line_width, alpha=0.4)
        ax.plot(time, smooth, color=trend_color, linewidth=line_width, alpha=0.95)

        ax.set_ylabel("Fractional\nChange", fontsize=16 * fs)
        ax.set_title(display_label, fontsize=18 * fs, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14 * fs)
        if layout == "grid":
            ax.set_xlabel("Time (s)", fontsize=16 * fs)

    # Set shared y-limits across all panels for fair comparison
    for ax in axes:
        ax.set_ylim(*ylim)

    if layout != "grid":
        axes[-1].set_xlabel("Time (s)", fontsize=18 * fs)
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mechanical comparison plot to {output_path}")


if __name__ == "__main__":
    # Standard version
    plot_comparison()

    # Zoomed-in poster version: 5s, taller panels, bigger features, full drug names
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

    # 2x2 grid poster version
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
