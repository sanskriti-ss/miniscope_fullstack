"""
Extract Decay50 (t50%) from calcium fluorescence transients.

Decay50 = time for the FF0 signal to decay from peak to 50% of peak amplitude
(measured from the local baseline before each beat).

Usage:
    python comparison_of_transients/extract_decay50.py
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
    DECAY50_EXPERIMENTS as EXPERIMENTS,
    ROI_INDEX, CLIP_SEC, BAR_COLORS, DRUG_LABELS,
)

# Single-color lookup for annotated plots (trend color per drug)
COLORS = dict(zip(DRUG_LABELS, BAR_COLORS))


# ============================================================================
# IMPLEMENTATION
# ============================================================================

def load_trace(csv_path, roi_index=1):
    """Load FF0_roi{roi_index}_smooth from a fluorescence CSV."""
    if not os.path.isabs(csv_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(repo_root, csv_path)

    df = pd.read_csv(csv_path)
    time = df["time_s"].values
    smooth_col = f"FF0_roi{roi_index}_smooth"
    raw_col = f"FF0_roi{roi_index}"

    if smooth_col not in df.columns:
        # Fallback to first available smooth column
        smooth_cols = [c for c in df.columns if c.startswith("FF0_roi") and c.endswith("_smooth")]
        if not smooth_cols:
            raise ValueError(f"No smooth FF0 columns in {csv_path}")
        smooth_col = smooth_cols[0]
        raw_col = smooth_col.replace("_smooth", "")

    smooth = df[smooth_col].values
    raw = df[raw_col].values if raw_col in df.columns else smooth
    return time, raw, smooth


def heavy_smooth(signal, fps, window_sec=0.3):
    """Heavy Savgol smoothing for robust peak detection."""
    win = int(window_sec * fps)
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(signal) // 3))
    if win >= len(signal):
        return signal.copy()
    return savgol_filter(signal, win, 2)


def compute_decay50(time, smooth, fps):
    """
    Detect peaks in the FF0 smooth signal and measure decay50 for each.

    Uses heavy smoothing for peak detection to avoid noise peaks, then
    measures decay50 on the lighter-smoothed signal.

    For each peak:
    1. Find the local baseline (trough between this peak and next)
    2. Amplitude = peak_value - baseline
    3. half_level = baseline + 0.5 * amplitude
    4. Scan forward from the peak until signal crosses half_level
    5. Interpolate to get precise crossing time
    6. decay50 = crossing_time - peak_time

    Returns: decay50_values, peak_indices, half_cross_indices, baselines, det_smooth
    """
    # Heavy smoothing for robust peak detection
    det = heavy_smooth(smooth, fps, window_sec=0.3)

    signal_range = np.percentile(det, 95) - np.percentile(det, 5)
    prominence = max(0.003, signal_range * 0.3)
    min_distance = max(1, int(0.6 * fps))  # at least 0.6s between peaks

    peaks, _ = find_peaks(det, distance=min_distance, prominence=prominence)

    if len(peaks) == 0:
        return [], [], [], [], det

    # Also find troughs between peaks for baseline estimation
    troughs, _ = find_peaks(-det, distance=min_distance // 2)

    decay50_values = []
    valid_peaks = []
    half_cross_times = []
    baselines = []

    for k, pk in enumerate(peaks):
        # Baseline: find the nearest trough AFTER this peak (before next peak)
        right_bound = peaks[k + 1] if k + 1 < len(peaks) else len(det)

        # Find minimum in the region between this peak and next peak
        region = det[pk:right_bound]
        if len(region) < 3:
            continue
        trough_rel = np.argmin(region)
        trough_idx = pk + trough_rel
        baseline = det[trough_idx]

        peak_val = det[pk]
        amplitude = peak_val - baseline
        if amplitude < 0.003:
            continue

        half_level = baseline + 0.5 * amplitude

        # Scan forward from peak on the detection signal
        decay_region = det[pk:right_bound]

        cross_idx = None
        for j in range(1, len(decay_region)):
            if decay_region[j] <= half_level:
                y0 = decay_region[j - 1]
                y1 = decay_region[j]
                frac = (y0 - half_level) / (y0 - y1) if (y0 - y1) != 0 else 0
                cross_idx = pk + (j - 1) + frac
                break

        if cross_idx is not None:
            cross_time = np.interp(cross_idx, np.arange(len(time)), time)
            peak_time = time[pk]
            d50 = cross_time - peak_time
            if 0.02 < d50 < 2.0:  # sanity bounds
                decay50_values.append(d50)
                valid_peaks.append(pk)
                half_cross_times.append(cross_idx)
                baselines.append(baseline)

    return decay50_values, valid_peaks, half_cross_times, baselines, det


def plot_annotated_transients(output_path="comparison_of_transients/decay50_annotated.png"):
    """Plot each experiment's calcium transient with decay50 annotations."""
    n = len(EXPERIMENTS)
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    all_results = {}

    for ax, (label, csv_path) in zip(axes, EXPERIMENTS):
        time, raw, smooth = load_trace(csv_path, ROI_INDEX)

        # Clip
        if CLIP_SEC > 0:
            mask = time <= CLIP_SEC
            time = time[mask]
            raw = raw[mask]
            smooth = smooth[mask]

        fps = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 25.0

        decay50_vals, valid_peaks, half_crosses, baselines, det = compute_decay50(time, smooth, fps)

        color = COLORS[label]
        ax.plot(time, raw, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(time, smooth, color=color, linewidth=1.5, alpha=0.5, label='smooth')
        ax.plot(time, det, color=color, linewidth=2.5, alpha=0.95, label='detection')

        # Annotate each detected beat
        for k, (pk, hc, bl) in enumerate(zip(valid_peaks, half_crosses, baselines)):
            peak_val = det[pk]
            amplitude = peak_val - bl
            half_level = bl + 0.5 * amplitude

            # Mark peak
            ax.plot(time[pk], peak_val, 'v', color='red', markersize=8, zorder=5)

            # Mark baseline level at peak time
            ax.plot(time[pk], bl, '^', color='blue', markersize=6, zorder=5)

            # Draw horizontal line at 50% level from peak to crossing
            hc_time = np.interp(hc, np.arange(len(time)), time)
            ax.hlines(half_level, time[pk], hc_time, colors='red', linewidth=1.5,
                      linestyles='dashed', zorder=4)

            # Mark the 50% crossing point
            ax.plot(hc_time, half_level, 'o', color='red', markersize=5, zorder=5)

            # Label with decay50 value
            d50 = decay50_vals[k]
            ax.annotate(f"{d50*1000:.0f}ms",
                        xy=(hc_time, half_level),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold')

        mean_d50 = np.mean(decay50_vals) if decay50_vals else float('nan')
        std_d50 = np.std(decay50_vals) if decay50_vals else float('nan')
        n_beats = len(decay50_vals)

        all_results[label] = {
            "mean": mean_d50,
            "std": std_d50,
            "n_beats": n_beats,
            "values": decay50_vals,
        }

        ax.set_title(f"{label}  —  Decay50: {mean_d50*1000:.1f} ± {std_d50*1000:.1f} ms  (n={n_beats})",
                      fontsize=16, fontweight='bold', loc='left')
        ax.set_ylabel("F/F0", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

    axes[-1].set_xlabel("Time (s)", fontsize=14)
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved annotated decay50 plot to {output_path}")

    # Print summary
    print("\n=== Decay50 Summary ===")
    print(f"{'Drug':<15} {'Mean (s)':<12} {'Std (s)':<12} {'n_beats'}")
    for label, res in all_results.items():
        print(f"{label:<15} {res['mean']:.4f}       {res['std']:.4f}       {res['n_beats']}")

    return all_results


if __name__ == "__main__":
    results = plot_annotated_transients()
