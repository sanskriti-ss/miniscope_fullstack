"""
Statistical significance testing for cardiac metrics across drug conditions.

Process:
1. Collect individual beat-level measurements for each metric x condition
2. One-Way ANOVA (scipy.stats.f_oneway) tests if ANY group differs
3. Post-hoc pairwise t-tests (each drug vs Control)
4. Mark "*" if p < 0.05

Metrics:
- Decay50 (fluorescence): time for FF0 signal to decay to 50% of peak amplitude
- Beat-to-Beat Variability: coefficient of variation of inter-beat intervals
- Contraction Time (mechanical): time from relaxed to contracted state
- Relaxation Time (mechanical): time from contracted to relaxed state

Usage:
    python comparison_of_transients/statistical_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Add parent and current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import (
    DECAY50_EXPERIMENTS as FLUO_EXPERIMENTS,
    MECHANICAL_EXPERIMENTS as MECH_EXPERIMENTS,
    IBI_EXPERIMENTS, DRUG_LABELS, ROI_INDEX, CLIP_SEC,
)
from extract_decay50 import load_trace, compute_decay50
from extract_mechanical_metrics import find_csv, process_experiment, detect_beats_area, SIGNAL_COL, MIN_PROMINENCE, MIN_PEAK_DISTANCE_SEC

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_COMPARISONS = 3  # Thar vs Control, DOF vs Control, QUAN vs Control
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / NUM_COMPARISONS

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PER_BEAT_CSV = os.path.join(OUTPUT_DIR, "per_beat_data.csv")
STATS_CSV = os.path.join(OUTPUT_DIR, "statistical_results.csv")


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_decay50_per_beat():
    """Collect per-beat Decay50 values for each drug condition.

    Returns dict: {drug_label: [decay50_value, ...]}
    """
    per_beat = {}
    for label, csv_path in FLUO_EXPERIMENTS:
        time, raw, smooth = load_trace(csv_path, ROI_INDEX)
        if CLIP_SEC > 0:
            mask = time <= CLIP_SEC
            time, smooth = time[mask], smooth[mask]
        fps = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 25.0
        decay50_vals, _, _, _, _ = compute_decay50(time, smooth, fps)
        per_beat[label] = decay50_vals
    return per_beat


def collect_ibi_per_beat():
    """Collect per-beat inter-beat intervals for Beat-to-Beat Variability.

    Uses expanded IBI_EXPERIMENTS config to pool IBIs from multiple
    recordings per condition for greater statistical power.

    Returns dict: {drug_label: [ibi_value, ...]}
    """
    from scipy.signal import find_peaks
    from extract_decay50 import heavy_smooth

    per_beat = {}
    for label in DRUG_LABELS:
        csv_paths = IBI_EXPERIMENTS.get(label, [])
        all_ibis = []
        for csv_path in csv_paths:
            try:
                time, raw, smooth = load_trace(csv_path, ROI_INDEX)
            except Exception as e:
                print(f"    [WARN] Could not load {csv_path}: {e}")
                continue
            # No time clipping for IBI — we only need peak timing, and some
            # recordings beat slower so clipping to 8s loses too many peaks.
            fps = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 25.0

            det = heavy_smooth(smooth, fps, window_sec=0.3)
            signal_range = np.percentile(det, 95) - np.percentile(det, 5)
            prominence = max(0.003, signal_range * 0.3)
            min_distance = max(1, int(0.6 * fps))
            peaks, _ = find_peaks(det, distance=min_distance, prominence=prominence)

            if len(peaks) >= 2:
                ibis = np.diff(time[peaks]).tolist()
                all_ibis.extend(ibis)
                print(f"    {os.path.basename(os.path.dirname(csv_path))}: "
                      f"{len(peaks)} peaks, {len(ibis)} IBIs")
        per_beat[label] = all_ibis
    return per_beat


def collect_mechanical_per_beat():
    """Collect per-beat Contraction Time and Relaxation Time.

    Returns (ct_dict, rt_dict) where each is {drug_label: [values, ...]}
    """
    ct_dict = {}
    rt_dict = {}

    for drug_label in DRUG_LABELS:
        folder_paths = MECH_EXPERIMENTS.get(drug_label, [])
        all_ct = []
        all_rt = []

        for folder_path in folder_paths:
            csv_path = find_csv(folder_path)
            if csv_path is None:
                continue
            result = process_experiment(csv_path)
            if result is None:
                continue
            all_ct.extend(result["contraction_times"])
            all_rt.extend(result["relaxation_times"])

        ct_dict[drug_label] = all_ct
        rt_dict[drug_label] = all_rt

    return ct_dict, rt_dict


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_anova_and_posthoc(per_beat_dict, metric_name, test_variance=False):
    """
    Run One-Way ANOVA across all groups, then post-hoc tests vs Control.

    For most metrics, uses t-tests to compare means.
    For variability metrics (test_variance=True), uses Levene's test to
    compare the spread/variance of values between groups.

    Parameters
    ----------
    per_beat_dict : dict
        {drug_label: [per_beat_values]} in DRUG_LABELS order
    metric_name : str
        Name of the metric for reporting
    test_variance : bool
        If True, use Levene's test (compares variance/spread) instead of
        t-test (compares means). Use for variability metrics.

    Returns
    -------
    dict with keys:
        'metric': metric name
        'f_stat': ANOVA F-statistic (or Levene's statistic)
        'anova_p': omnibus p-value
        'significance': list of "" or "*" per drug (Control always "")
        'pairwise': list of dicts with drug, t_stat, p_raw, p_corrected, significant
        'group_sizes': list of n per drug
        'test_type': 't-test' or 'levene'
    """
    groups = []
    group_sizes = []
    for label in DRUG_LABELS:
        vals = per_beat_dict.get(label, [])
        groups.append(np.array(vals, dtype=float))
        group_sizes.append(len(vals))

    # Need at least 2 values per group for meaningful stats
    valid_groups = [g for g in groups if len(g) >= 2]

    test_type = "levene" if test_variance else "t-test"
    result = {
        'metric': metric_name,
        'f_stat': np.nan,
        'anova_p': np.nan,
        'significance': [""] * len(DRUG_LABELS),
        'pairwise': [],
        'group_sizes': group_sizes,
        'test_type': test_type,
    }

    if len(valid_groups) < 2:
        print(f"  [SKIP] {metric_name}: fewer than 2 groups with >= 2 values")
        return result

    # Omnibus test
    if test_variance:
        # Levene's test across all groups: tests if variances differ
        f_stat, anova_p = stats.levene(*valid_groups)
    else:
        # One-Way ANOVA: tests if means differ
        f_stat, anova_p = stats.f_oneway(*valid_groups)
    result['f_stat'] = f_stat
    result['anova_p'] = anova_p

    # Post-hoc: each drug vs Control (index 0)
    control = groups[0]
    significance = [""]  # Control vs itself = no test

    for i in range(1, len(DRUG_LABELS)):
        drug = groups[i]
        label = DRUG_LABELS[i]

        if len(control) < 2 or len(drug) < 2:
            significance.append("")
            result['pairwise'].append({
                'drug': label, 't_stat': np.nan,
                'p_raw': np.nan, 'p_corrected': np.nan,
                'significant': False,
            })
            continue

        if test_variance:
            # Levene's test: does this drug have different spread than Control?
            t_stat, p_raw = stats.levene(control, drug)
        else:
            t_stat, p_raw = stats.ttest_ind(control, drug)
        p_corrected = p_raw
        sig = p_corrected < ALPHA

        significance.append("*" if sig else "")
        result['pairwise'].append({
            'drug': label,
            't_stat': t_stat,
            'p_raw': p_raw,
            'p_corrected': p_corrected,
            'significant': sig,
        })

    result['significance'] = significance
    return result


# ============================================================================
# MAIN
# ============================================================================

def run_all_stats():
    """Run statistical tests for all metrics. Returns dict of results."""
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)

    # Collect per-beat data
    print("\n--- Collecting Decay50 per-beat data ---")
    decay50_data = collect_decay50_per_beat()
    for label, vals in decay50_data.items():
        print(f"  {label}: {len(vals)} beats")

    print("\n--- Collecting IBI per-beat data ---")
    ibi_data = collect_ibi_per_beat()
    for label, vals in ibi_data.items():
        print(f"  {label}: {len(vals)} intervals")

    print("\n--- Collecting mechanical per-beat data ---")
    ct_data, rt_data = collect_mechanical_per_beat()
    for label in DRUG_LABELS:
        print(f"  {label}: CT={len(ct_data.get(label, []))} beats, "
              f"RT={len(rt_data.get(label, []))} beats")

    # Run stats for each metric
    all_metric_data = {
        "Decay50\n(s)": decay50_data,
        "Beat-to-Beat\nVariability (%)": ibi_data,
        "Contraction\nTime (s)": ct_data,
        "Relaxation\nTime (s)": rt_data,
    }

    results = {}
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Metrics that measure variability (spread) rather than a mean value
    # need Levene's test instead of t-test
    variance_metrics = {"Beat-to-Beat\nVariability (%)"}

    for metric_name, per_beat in all_metric_data.items():
        display_name = metric_name.replace("\n", " ")
        is_variance = metric_name in variance_metrics
        test_label = "Levene" if is_variance else "ANOVA"
        print(f"\n--- {display_name} {'(Levene — testing variance)' if is_variance else ''} ---")
        res = run_anova_and_posthoc(per_beat, metric_name, test_variance=is_variance)
        results[metric_name] = res

        print(f"  {test_label}: F={res['f_stat']:.4f}, p={res['anova_p']:.6f}")
        print(f"  Group sizes: {res['group_sizes']}")
        for pw in res['pairwise']:
            star = " *" if pw['significant'] else ""
            print(f"  {pw['drug']} vs Control: "
                  f"stat={pw['t_stat']:.4f}, p={pw['p_raw']:.6f}{star}")

    # Build SIGNIFICANCE dict for compare_metrics.py
    SIGNIFICANCE = {}
    for metric_name, res in results.items():
        SIGNIFICANCE[metric_name] = res['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DICT (for compare_metrics.py)")
    print("=" * 70)
    for metric, sigs in SIGNIFICANCE.items():
        display = metric.replace("\n", " ")
        print(f"  {display}: {sigs}")

    # Export per-beat data to CSV
    _export_per_beat_csv(all_metric_data)

    # Export stats summary CSV
    _export_stats_csv(results)

    return results, SIGNIFICANCE


def _export_per_beat_csv(all_metric_data):
    """Save per-beat data to CSV for reproducibility."""
    rows = []
    for metric_name, per_beat in all_metric_data.items():
        display_name = metric_name.replace("\n", " ")
        for drug_label, values in per_beat.items():
            for i, val in enumerate(values):
                rows.append({
                    'drug': drug_label,
                    'metric': display_name,
                    'beat_index': i,
                    'value': val,
                })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(PER_BEAT_CSV, index=False)
        print(f"\nSaved per-beat data to {PER_BEAT_CSV} ({len(rows)} rows)")


def _export_stats_csv(results):
    """Save statistical results summary to CSV."""
    rows = []
    for metric_name, res in results.items():
        display_name = metric_name.replace("\n", " ")
        row = {
            'metric': display_name,
            'anova_F': res['f_stat'],
            'anova_p': res['anova_p'],
        }
        for pw in res['pairwise']:
            drug_short = pw['drug'].replace(" ", "_")
            row[f'{drug_short}_t'] = pw['t_stat']
            row[f'{drug_short}_p_raw'] = pw['p_raw']
            row[f'{drug_short}_p_corrected'] = pw['p_corrected']
            row[f'{drug_short}_significant'] = pw['significant']
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(STATS_CSV, index=False)
        print(f"Saved statistical results to {STATS_CSV}")


def get_significance():
    """Run all stats and return just the SIGNIFICANCE dict.

    Returns dict: {metric_name: [""/"*" per drug]}
    """
    _, significance = run_all_stats()
    return significance


if __name__ == "__main__":
    run_all_stats()
