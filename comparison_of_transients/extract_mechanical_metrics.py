"""
Extract contraction time and relaxation time from mechanical (brightfield)
contractility traces produced by the batch pipeline.

Reads experiment paths from experiments.py config file,
detects individual beats in the area_smooth signal, and measures
contraction/relaxation timing per beat.

Usage:
    python comparison_of_transients/extract_mechanical_metrics.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Import unified config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import MECHANICAL_EXPERIMENTS as EXPERIMENTS, DRUG_LABELS


# ============================================================================
# CONFIGURATION
# ============================================================================

# Signal column to use for beat detection
SIGNAL_COL = "area_smooth"

# Minimum prominence for peak detection (fractional units)
MIN_PROMINENCE = 0.003

# Minimum distance between peaks in seconds
MIN_PEAK_DISTANCE_SEC = 0.8

OUTPUT_CSV = "comparison_of_transients/mechanical_timing_metrics.csv"


# ============================================================================
# IMPLEMENTATION
# ============================================================================

def detect_beats_area(time, signal, fps, min_prominence=0.003,
                      min_peak_distance_sec=0.8):
    """
    Detect individual beats from area_smooth signal.

    Area signal: peaks = relaxed (max area), troughs = contracted (min area).

    Returns lists of (contraction_time_s, relaxation_time_s) per beat.
    """
    min_dist = max(3, int(fps * min_peak_distance_sec))

    # Find peaks (relaxed states = area maxima)
    peaks, _ = find_peaks(signal, prominence=min_prominence, distance=min_dist)

    # Find troughs (contracted states = area minima)
    troughs, _ = find_peaks(-signal, prominence=min_prominence, distance=min_dist)

    if len(peaks) < 2 or len(troughs) < 1:
        return [], [], peaks, troughs

    contraction_times = []
    relaxation_times = []

    # For each consecutive pair of peaks, find the trough between them
    for i in range(len(peaks) - 1):
        p1 = peaks[i]      # relaxed state (start)
        p2 = peaks[i + 1]  # relaxed state (end)

        # Find trough between these two peaks
        between_troughs = troughs[(troughs > p1) & (troughs < p2)]
        if len(between_troughs) == 0:
            continue

        # Use the deepest trough between the peaks
        trough_idx = between_troughs[np.argmin(signal[between_troughs])]

        t_peak1 = time[p1]
        t_trough = time[trough_idx]
        t_peak2 = time[p2]

        # Contraction time: peak1 (relaxed) -> trough (contracted)
        ct = t_trough - t_peak1
        # Relaxation time: trough (contracted) -> peak2 (relaxed)
        rt = t_peak2 - t_trough

        # Sanity check: both should be positive and reasonable
        if 0.02 < ct < 2.0 and 0.02 < rt < 2.0:
            contraction_times.append(ct)
            relaxation_times.append(rt)

    return contraction_times, relaxation_times, peaks, troughs


def find_csv(folder_path):
    """Find the best CSV in an experiment folder (prefer windowed)."""
    # Resolve relative to repo root
    if not os.path.isabs(folder_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(repo_root, folder_path)

    for name in ["contractility_windowed.csv", "contractility_traces.csv",
                  "fluorescence_traces.csv"]:
        csv_path = os.path.join(folder_path, name)
        if os.path.exists(csv_path):
            return csv_path
    return None


def process_experiment(csv_path, signal_col=SIGNAL_COL):
    """Process one experiment's CSV and return contraction/relaxation times."""
    df = pd.read_csv(csv_path)

    if signal_col not in df.columns:
        # Try mechanical columns first, then fluorescence columns
        fallbacks = ["area_smooth", "area_detrend", "area_norm",
                     "fixed_mask_intensity_smooth"]
        # Also check for FF0_roi*_smooth columns (fluorescence pipeline on brightfield)
        ff0_smooth = [c for c in df.columns if c.startswith("FF0_roi") and c.endswith("_smooth")]
        fallbacks.extend(ff0_smooth)
        for fallback in fallbacks:
            if fallback in df.columns:
                signal_col = fallback
                break
        else:
            print(f"  [WARN] No suitable signal column in {csv_path}")
            return None
    print(f"    Using signal column: {signal_col}")

    time = df["time_s"].values
    signal = df[signal_col].values

    # Estimate fps
    fps = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 30.0

    ct_list, rt_list, peaks, troughs = detect_beats_area(
        time, signal, fps,
        min_prominence=MIN_PROMINENCE,
        min_peak_distance_sec=MIN_PEAK_DISTANCE_SEC,
    )

    if not ct_list:
        print(f"  [WARN] No beats detected in {csv_path}")
        return None

    return {
        "contraction_times": ct_list,
        "relaxation_times": rt_list,
        "n_beats": len(ct_list),
        "mean_ct": np.mean(ct_list),
        "std_ct": np.std(ct_list),
        "mean_rt": np.mean(rt_list),
        "std_rt": np.std(rt_list),
    }


def main():
    print("=" * 70)
    print("MECHANICAL CONTRACTION/RELAXATION TIME EXTRACTION")
    print("=" * 70)
    print(f"Reading experiment paths from experiments.py\n")

    all_rows = []
    ct_values = []
    ct_errors = []
    rt_values = []
    rt_errors = []

    for drug_label in DRUG_LABELS:
        folder_paths = EXPERIMENTS.get(drug_label, [])

        if not folder_paths:
            print(f"  {drug_label}: NO EXPERIMENTS CONFIGURED")
            ct_values.append(None)
            ct_errors.append(None)
            rt_values.append(None)
            rt_errors.append(None)
            continue

        print(f"\n--- {drug_label} ---")

        # Collect all beat-level data across experiments for this condition
        all_ct = []
        all_rt = []
        condition_rows = []

        for folder_path in folder_paths:
            csv_path = find_csv(folder_path)
            if csv_path is None:
                print(f"  [WARN] No CSV found in {folder_path}")
                continue

            folder_name = os.path.basename(folder_path)
            print(f"  Processing: {folder_name}")

            result = process_experiment(csv_path)
            if result is None:
                continue

            all_ct.extend(result["contraction_times"])
            all_rt.extend(result["relaxation_times"])

            condition_rows.append({
                "drug_label": drug_label,
                "folder": folder_name,
                "csv_path": csv_path,
                "n_beats": result["n_beats"],
                "mean_contraction_time_s": result["mean_ct"],
                "std_contraction_time_s": result["std_ct"],
                "mean_relaxation_time_s": result["mean_rt"],
                "std_relaxation_time_s": result["std_rt"],
            })
            print(f"    {result['n_beats']} beats: "
                  f"CT={result['mean_ct']:.3f}±{result['std_ct']:.3f}s, "
                  f"RT={result['mean_rt']:.3f}±{result['std_rt']:.3f}s")

        all_rows.extend(condition_rows)

        if not all_ct:
            print(f"  {drug_label}: NO BEATS DETECTED across all experiments")
            ct_values.append(None)
            ct_errors.append(None)
            rt_values.append(None)
            rt_errors.append(None)
            continue

        # Pool all individual beats for this condition
        mean_ct = np.mean(all_ct)
        std_ct = np.std(all_ct)
        mean_rt = np.mean(all_rt)
        std_rt = np.std(all_rt)

        ct_values.append(round(mean_ct, 3))
        ct_errors.append(round(std_ct, 3))
        rt_values.append(round(mean_rt, 3))
        rt_errors.append(round(std_rt, 3))

        print(f"  POOLED: CT={mean_ct:.3f}±{std_ct:.3f}s, "
              f"RT={mean_rt:.3f}±{std_rt:.3f}s "
              f"({len(all_ct)} total beats from {len(condition_rows)} experiments)")

    # Save per-experiment CSV
    if all_rows:
        df_results = pd.DataFrame(all_rows)
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved per-experiment results to {OUTPUT_CSV}")

    # Print copy-paste ready values
    print("\n" + "=" * 70)
    print("VALUES FOR compare_metrics.py")
    print("=" * 70)

    print("\nMETRICS_DATA:")
    ct_str = [str(v) if v is not None else "None" for v in ct_values]
    rt_str = [str(v) if v is not None else "None" for v in rt_values]
    print(f'    "Contraction\\nTime (s)":  [{", ".join(ct_str)}],')
    print(f'    "Relaxation\\nTime (s)":   [{", ".join(rt_str)}],')

    print("\nMETRICS_ERROR:")
    cte_str = [str(v) if v is not None else "None" for v in ct_errors]
    rte_str = [str(v) if v is not None else "None" for v in rt_errors]
    print(f'    "Contraction\\nTime (s)":  [{", ".join(cte_str)}],')
    print(f'    "Relaxation\\nTime (s)":   [{", ".join(rte_str)}],')


if __name__ == "__main__":
    main()
