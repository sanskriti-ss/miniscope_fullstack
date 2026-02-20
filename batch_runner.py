"""
Batch Processing Pipeline for Multi-Video Organoid Analysis.
Parses experiment folder metadata, routes to fluorescent or mechanical pipeline,
selects the best analysis window, and generates grouped comparison plots.

Usage:
    python batch_runner.py --input-dir /path/to/experiment_folder --window 20
"""

import os
import re
import sys
import argparse
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Imports from existing miniscope_fullstack modules
# ---------------------------------------------------------------------------
from roi_selection import extract_frame_channel
from roi_detection import detect_rois_dispatcher
from plotting import save_roi_overlay_image, save_spike_trace_plot
from roi_selection import preview_video_and_draw_rois
from helper_functions import (
    load_video_metadata,
    build_reference_image,
    load_timestamps_from_file,
    apply_timestamps_to_traces,
    extract_traces,
    normalize_traces_FF0,
    dilate_roi_masks,
    smooth_traces,
    detrend_traces,
    detect_spikes_and_dips,
)
from main_mech import (
    detect_organoid,
    detect_organoid_cellpose,
    extract_contractility_traces,
    normalize_contractility,
    detect_contractions,
    create_padded_roi,
    save_roi_overlay as save_mech_roi_overlay,
    plot_contractility,
)


# ============================================================================
# Step 1 — Folder Discovery & Metadata Parsing
# ============================================================================

def parse_folder_name(folder_name):
    """Parse experiment metadata from a folder name.

    Expected patterns (with typos):
        HH_MM_SS_Drug[N]_[Conc]_Type_PacingHz_FPSfps[_tags]

    Returns dict with keys:
        drug, concentration, condition_id, pacing_hz, fps, video_type,
        tags, raw_name
    """
    raw = folder_name
    result = {
        "drug": None,
        "concentration": None,
        "condition_id": None,
        "pacing_hz": None,
        "fps": None,
        "video_type": None,
        "tags": [],
        "raw_name": raw,
    }

    # Strip leading timestamp  HH_MM_SS_
    name = re.sub(r"^\d{1,2}_\d{1,2}_\d{1,2}_", "", raw)

    # --- Extra tags ---
    if "_firsthalfonly" in name.lower():
        result["tags"].append("firsthalfonly")

    # --- Video type (fuzzy) ---
    lower = name.lower()
    if re.search(r"fl", lower):           # flo, fluo, fluoresent, fluorescent
        result["video_type"] = "fluorescent"
    elif re.search(r"bright|bf", lower):
        result["video_type"] = "mechanical"

    # --- Drug ---
    drug_map = [
        (r"quan", "QUAN"),
        (r"thar", "THAR"),
        (r"dofe?", "DOF"),
        (r"control", "Control"),
        (r"wash", "Wash"),
        (r"test", "Testrun"),
    ]
    for pattern, label in drug_map:
        m = re.search(pattern + r"(\d+)?", lower)
        if m:
            result["drug"] = label
            if m.group(1):
                result["condition_id"] = int(m.group(1))
            break

    # --- Concentration ---
    m_conc = re.search(r"(\d+)\s*([num])M", name, re.IGNORECASE)
    if m_conc:
        result["concentration"] = m_conc.group(1) + m_conc.group(2).lower() + "M"

    # --- Pacing Hz ---
    m_hz = re.search(r"([\d.]+)\s*hz", name, re.IGNORECASE)
    if m_hz:
        result["pacing_hz"] = float(m_hz.group(1))

    # --- FPS ---
    m_fps = re.search(r"(\d+)\s*fps", name, re.IGNORECASE)
    if m_fps:
        result["fps"] = int(m_fps.group(1))

    return result


def _load_timestamps_miniscope(ts_path):
    """Load timeStamps.csv from a Miniscope folder.

    Returns (timestamps_sec, real_fps) or (None, None).
    """
    if not os.path.exists(ts_path):
        return None, None
    try:
        ts_df = pd.read_csv(ts_path)
        # Column: "Time Stamp (ms)"
        time_col = None
        for c in ts_df.columns:
            if "time" in c.lower() and "stamp" in c.lower():
                time_col = c
                break
        if time_col is None:
            # fallback to second column
            time_col = ts_df.columns[1] if len(ts_df.columns) >= 2 else ts_df.columns[0]

        ts = ts_df[time_col].values.astype(float)

        # Convert ms → s if needed
        # Check last value — if > 500, likely milliseconds
        if len(ts) > 5 and np.max(np.abs(ts)) > 500:
            ts = ts / 1000.0

        # Make relative to first stamp
        ts = ts - ts[0]

        if len(ts) > 1:
            intervals = np.diff(ts)
            real_fps = 1.0 / np.median(intervals[intervals > 0])
        else:
            real_fps = None
        return ts, real_fps
    except Exception:
        return None, None


def _video_duration_from_timestamps(ts_path, n_avi_frames):
    """Return (duration_sec, real_fps) using timestamps file."""
    ts, real_fps = _load_timestamps_miniscope(ts_path)
    if ts is not None and len(ts) > 1:
        n_usable = min(len(ts), n_avi_frames)
        duration = ts[n_usable - 1]
        return duration, real_fps, ts
    return None, None, None


def discover_videos(root_folder, min_duration_sec=2.0):
    """Walk *root_folder* and return a list of VideoEntry dicts.

    Each dict:
        folder_name, video_path, ts_path, avi_name, duration_sec,
        real_fps, timestamps, metadata (from parse_folder_name)
    """
    entries = []
    for subfolder in sorted(os.listdir(root_folder)):
        miniscope_dir = os.path.join(root_folder, subfolder, "My_V4_Miniscope")
        if not os.path.isdir(miniscope_dir):
            continue

        ts_path = os.path.join(miniscope_dir, "timeStamps.csv")
        meta = parse_folder_name(subfolder)

        for avi in sorted(os.listdir(miniscope_dir)):
            if not avi.endswith(".avi"):
                continue
            video_path = os.path.join(miniscope_dir, avi)

            # Get frame count from OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if n_frames < 2:
                continue

            duration, real_fps, ts_arr = _video_duration_from_timestamps(ts_path, n_frames)
            if duration is None:
                # fallback: assume 30 fps
                duration = n_frames / 30.0
                real_fps = 30.0
                ts_arr = None

            if duration < min_duration_sec:
                print(f"  [skip] {subfolder}/{avi}: {duration:.1f}s < {min_duration_sec}s")
                continue

            entries.append({
                "folder_name": subfolder,
                "video_path": video_path,
                "ts_path": ts_path,
                "avi_name": avi,
                "n_frames": n_frames,
                "duration_sec": duration,
                "real_fps": real_fps,
                "timestamps": ts_arr,
                "metadata": meta,
            })

    return entries


# ============================================================================
# Step 2 — Best Window Selection
# ============================================================================

def find_best_window(signal, time, window_sec=20.0, min_window=5.0):
    """Select the best analysis window based on peak regularity.

    Returns (start_idx, end_idx).
    """
    total_duration = time[-1] - time[0]

    if total_duration < min_window:
        return None, None  # too short

    if total_duration <= window_sec:
        # Use full trace, but skip first 1 s if possible
        skip_idx = int(np.searchsorted(time, time[0] + 1.0))
        if (time[-1] - time[skip_idx]) < min_window:
            skip_idx = 0
        return skip_idx, len(signal)

    # Skip first 1 second (transient)
    skip_time = time[0] + 1.0
    skip_idx = int(np.searchsorted(time, skip_time))

    best_score = -1
    best_start = skip_idx
    best_end = len(signal)

    # Slide window in steps of 0.5 s
    step_sec = 0.5
    pos_time = time[skip_idx]
    while pos_time + window_sec <= time[-1] + 0.01:
        i_start = int(np.searchsorted(time, pos_time))
        i_end = int(np.searchsorted(time, pos_time + window_sec))
        i_end = min(i_end, len(signal))

        seg = signal[i_start:i_end]
        if len(seg) < 10:
            pos_time += step_sec
            continue

        # Score: many regular peaks
        try:
            seg_std = np.std(seg)
            prominence = max(0.003, seg_std * 0.3)
            fps_est = len(seg) / (time[i_end - 1] - time[i_start])
            min_dist = max(3, int(fps_est * 0.4))
            peaks, _ = find_peaks(seg, prominence=prominence, distance=min_dist)
            n_peaks = len(peaks)
            if n_peaks >= 2:
                intervals = np.diff(time[i_start:i_end][peaks])
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 999
                score = n_peaks / (1.0 + cv)
            else:
                score = 0
        except Exception:
            score = 0

        if score > best_score:
            best_score = score
            best_start = i_start
            best_end = i_end

        pos_time += step_sec

    return best_start, best_end


# ============================================================================
# Step 3 — Pipeline Runners
# ============================================================================

def run_fluorescent(video_path, ts_path, metadata, out_dir, window_sec=20.0):
    """Run fluorescent pipeline on a single video. Returns result dict."""
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata
    fps_ocv, n_frames, h, w, start_frame, end_frame = load_video_metadata(video_path)

    # Use timestamps for real fps
    ts_arr, real_fps = _load_timestamps_miniscope(ts_path)
    fps = real_fps if real_fps else fps_ocv

    # Manual ROI selection — user draws ROIs on each fluorescent video
    print(f"  Opening manual ROI selection for: {os.path.basename(video_path)}")
    print(f"  Folder: {metadata.get('raw_name', '')}")
    roi_masks, roi_info = preview_video_and_draw_rois(
        video_path, n_preview_frames=150, channel=0,
    )

    if len(roi_masks) == 0:
        print(f"  [WARN] No ROIs drawn — skipping")
        return None

    # Save ROI overlay
    roi_masks_orig = [m.copy() for m in roi_masks]
    roi_masks_dilated = dilate_roi_masks(roi_masks, radius=40)
    save_roi_overlay_image(
        video_path, roi_masks_dilated, roi_info,
        out_path=os.path.join(out_dir, "rois_on_first_frame.png"),
        roi_masks_original=roi_masks_orig,
    )

    # Extract traces
    df, _ = extract_traces(
        video_path, roi_masks_dilated, channel=0,
        start_frame=start_frame, end_frame=end_frame,
        roi_masks_for_f0=roi_masks_orig,
    )

    # Apply timestamps
    if ts_arr is not None:
        df = apply_timestamps_to_traces(df, ts_arr, start_frame=start_frame)

    # Normalize → smooth → detrend → detect spikes
    df = normalize_traces_FF0(df, f0_mode="percentile", f0_percentile=10, f0_first_n=20)
    df = smooth_traces(df, window_length=5, polyorder=3, additional_smoothing=False)
    df = detrend_traces(df, polyorder=2)
    df = detect_spikes_and_dips(df, fps)

    # Save full traces
    df.to_csv(os.path.join(out_dir, "fluorescence_traces.csv"), index=False)

    # Pick best signal column for windowing (prefer smoothed detrended of first ROI)
    smooth_cols = [c for c in df.columns if c.endswith("_smooth") and "detrended" not in c and c.startswith("FF0_roi")]
    if not smooth_cols:
        smooth_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_" not in c.split("FF0_roi")[1]]
    sig_col = smooth_cols[0] if smooth_cols else None

    bpm = 0.0
    amplitude = 0.0
    spike_count = 0
    beat_regularity_cv = np.nan
    rise_time_ms = np.nan
    decay_time_ms = np.nan
    transient_duration_ms = np.nan
    w_start, w_end = 0, len(df)

    if sig_col:
        signal = df[sig_col].values
        time = df["time_s"].values
        ws, we = find_best_window(signal, time, window_sec=window_sec)
        if ws is not None:
            w_start, w_end = ws, we

        # Windowed trace
        df_win = df.iloc[w_start:w_end].copy().reset_index(drop=True)
        df_win.to_csv(os.path.join(out_dir, "fluorescence_windowed.csv"), index=False)

        # BPM from spike intervals in window
        spike_col = sig_col.replace("_smooth", "") + "_spike"
        if spike_col not in df_win.columns:
            # try base column
            base_col = re.sub(r"_smooth$|_detrended$", "", sig_col)
            spike_col = base_col + "_spike"

        spike_intervals = None
        if spike_col in df_win.columns:
            spike_idx = df_win.index[df_win[spike_col] > 0].values
            spike_count = len(spike_idx)
            if spike_count >= 2:
                spike_intervals = np.diff(df_win.loc[spike_idx, "time_s"].values)
                bpm = 60.0 / np.mean(spike_intervals)
                # Beat regularity (CV%)
                if np.mean(spike_intervals) > 0:
                    beat_regularity_cv = float(
                        np.std(spike_intervals) / np.mean(spike_intervals) * 100
                    )

        # Amplitude from peaks in window
        win_sig = df_win[sig_col].values if sig_col in df_win.columns else None
        win_time = df_win["time_s"].values
        if win_sig is not None:
            baseline = np.median(win_sig)
            peaks_idx, props = find_peaks(win_sig - baseline, prominence=0.003)
            if len(peaks_idx) > 0:
                amplitude = float(np.mean(props["prominences"]))

            # Rise time, decay time, transient duration from averaged beat
            if len(peaks_idx) >= 2:
                # Segment signal into individual beats (peak-to-peak)
                beats = []
                beat_durations_s = []
                for bi in range(len(peaks_idx) - 1):
                    seg = win_sig[peaks_idx[bi]:peaks_idx[bi + 1]]
                    seg_t = win_time[peaks_idx[bi]:peaks_idx[bi + 1]]
                    if len(seg) > 4:
                        beat_durations_s.append(seg_t[-1] - seg_t[0])
                        interp_seg = np.interp(
                            np.linspace(0, 1, 50),
                            np.linspace(0, 1, len(seg)), seg,
                        )
                        beats.append(interp_seg)

                if beats:
                    avg_beat = np.mean(beats, axis=0)
                    avg_dur_ms = np.mean(beat_durations_s) * 1000.0
                    peak_idx_avg = np.argmax(avg_beat)
                    beat_min = np.min(avg_beat)
                    beat_max = np.max(avg_beat)
                    r10 = beat_min + 0.1 * (beat_max - beat_min)
                    r90 = beat_min + 0.9 * (beat_max - beat_min)

                    # Rise time: 10% to 90% on upstroke
                    upstroke = avg_beat[:peak_idx_avg + 1] if peak_idx_avg > 0 else avg_beat
                    t10 = np.searchsorted(upstroke, r10)
                    t90 = np.searchsorted(upstroke, r90)
                    rise_time_ms = float((t90 - t10) / 50.0 * avg_dur_ms)

                    # Decay time: 90% to 10% on downstroke
                    downstroke = avg_beat[peak_idx_avg:]
                    d90 = np.searchsorted(-downstroke, -r90)
                    d10 = np.searchsorted(-downstroke, -r10)
                    decay_time_ms = float(abs(d10 - d90) / 50.0 * avg_dur_ms)

                    # Transient duration at 50% height
                    half = beat_min + 0.5 * (beat_max - beat_min)
                    above_half = avg_beat > half
                    if np.any(above_half):
                        first_above = np.argmax(above_half)
                        last_above = len(above_half) - 1 - np.argmax(above_half[::-1])
                        transient_duration_ms = float(
                            (last_above - first_above) / 50.0 * avg_dur_ms
                        )
    else:
        df_win = df

    # Save spike plot
    save_spike_trace_plot(
        df_win,
        out_path=os.path.join(out_dir, "fluorescence_spikes.png"),
        video_name=metadata.get("raw_name", ""),
    )

    return {
        "bpm": bpm,
        "amplitude": amplitude,
        "spike_count": spike_count,
        "beat_regularity_cv": beat_regularity_cv,
        "rise_time_ms": rise_time_ms,
        "decay_time_ms": decay_time_ms,
        "transient_duration_ms": transient_duration_ms,
        "windowed_trace": df_win[sig_col].values if sig_col and sig_col in df_win.columns else None,
        "windowed_time": df_win["time_s"].values,
        "window_start": float(df["time_s"].values[w_start]),
        "window_end": float(df["time_s"].values[min(w_end, len(df)) - 1]),
        "duration": float(df["time_s"].values[-1]),
        "metadata": metadata,
    }


def run_mechanical(video_path, ts_path, metadata, out_dir, window_sec=20.0,
                   allow_manual=True):
    """Run mechanical / brightfield pipeline on a single video. Returns result dict."""
    os.makedirs(out_dir, exist_ok=True)

    # Video info
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Timestamps
    ts_arr, real_fps = _load_timestamps_miniscope(ts_path)
    fps = real_fps if real_fps else 30.0

    # Detect organoid — motion first, then manual/cellpose fallback
    mask, info, ref_img = detect_organoid(
        video_path, n_frames=min(200, n_frames), allow_manual=allow_manual,
    )
    if mask is None:
        print(f"  [WARN] No organoid detected — skipping")
        return None

    # ROI overlay
    save_mech_roi_overlay(video_path, mask, info, os.path.join(out_dir, "mech_roi_overlay.png"))

    # Padded ROI
    roi_mask = create_padded_roi(mask, pad_fraction=0.3)

    # Extract contractility
    df, _ = extract_contractility_traces(
        video_path, organoid_mask=mask, roi_mask=roi_mask,
        start_frame=0, end_frame=None, fps=fps,
    )

    # Apply timestamps to time_s column
    if ts_arr is not None and len(ts_arr) >= len(df):
        df["time_s"] = ts_arr[:len(df)] - ts_arr[0]

    # Normalize
    df = normalize_contractility(df, fps=fps)

    # Save full traces
    df.to_csv(os.path.join(out_dir, "contractility_traces.csv"), index=False)

    # Best window on fixed_mask_intensity_smooth
    sig_col = "fixed_mask_intensity_smooth"
    bpm = 0.0
    amplitude = 0.0
    contraction_count = 0
    beat_regularity_cv = np.nan
    contraction_velocity_pct_s = np.nan
    relaxation_velocity_pct_s = np.nan
    contraction_duration_ms = np.nan
    w_start, w_end = 0, len(df)

    if sig_col in df.columns:
        signal = df[sig_col].values
        time = df["time_s"].values
        ws, we = find_best_window(signal, time, window_sec=window_sec)
        if ws is not None:
            w_start, w_end = ws, we

        df_win = df.iloc[w_start:w_end].copy().reset_index(drop=True)
        df_win.to_csv(os.path.join(out_dir, "contractility_windowed.csv"), index=False)

        # Detect contractions in window
        peaks, props, beat_rate = detect_contractions(df_win, metric="area_smooth", fps=fps)
        contraction_count = len(peaks)
        bpm = beat_rate

        # Amplitude from fixed_mask_intensity_smooth peaks
        int_sig = df_win[sig_col].values
        win_time = df_win["time_s"].values
        int_peaks, int_props = find_peaks(int_sig, prominence=0.002, distance=max(5, int(fps * 1.0)))
        if len(int_peaks) > 0 and "prominences" in int_props:
            amplitude = float(np.mean(int_props["prominences"]))
        if len(int_peaks) > 1:
            int_intervals = np.diff(win_time[int_peaks])
            bpm = 60.0 / np.mean(int_intervals)
            # Beat regularity (CV%)
            if np.mean(int_intervals) > 0:
                beat_regularity_cv = float(
                    np.std(int_intervals) / np.mean(int_intervals) * 100
                )

        # Contraction/relaxation velocity from signal derivative
        baseline = np.percentile(int_sig, 90)
        if baseline > 0:
            dt = np.gradient(win_time)
            dsig = np.gradient(int_sig, win_time)
            # Contraction = signal decreasing (area shrinks), relaxation = signal increasing
            contraction_velocity_pct_s = float(np.abs(np.min(dsig)) / baseline * 100)
            relaxation_velocity_pct_s = float(np.abs(np.max(dsig)) / baseline * 100)

        # Contraction duration at 50% depth from averaged beat
        if len(int_peaks) >= 2:
            beats_mech = []
            beat_durations_s = []
            for bi in range(len(int_peaks) - 1):
                seg = int_sig[int_peaks[bi]:int_peaks[bi + 1]]
                seg_t = win_time[int_peaks[bi]:int_peaks[bi + 1]]
                if len(seg) > 4:
                    beat_durations_s.append(seg_t[-1] - seg_t[0])
                    interp_seg = np.interp(
                        np.linspace(0, 1, 50),
                        np.linspace(0, 1, len(seg)), seg,
                    )
                    beats_mech.append(interp_seg)
            if beats_mech:
                avg_beat = np.mean(beats_mech, axis=0)
                avg_dur_ms = np.mean(beat_durations_s) * 1000.0
                beat_min = np.min(avg_beat)
                beat_max = np.max(avg_beat)
                half = beat_max - 0.5 * (beat_max - beat_min)
                below_half = avg_beat < half
                if np.any(below_half):
                    first_below = np.argmax(below_half)
                    last_below = len(below_half) - 1 - np.argmax(below_half[::-1])
                    contraction_duration_ms = float(
                        (last_below - first_below) / 50.0 * avg_dur_ms
                    )
    else:
        df_win = df

    # Plots
    plot_contractility(df_win, fps, out_dir, video_name=metadata.get("raw_name", ""))

    return {
        "bpm": bpm,
        "amplitude": amplitude,
        "contraction_count": contraction_count,
        "beat_regularity_cv": beat_regularity_cv,
        "contraction_velocity_pct_s": contraction_velocity_pct_s,
        "relaxation_velocity_pct_s": relaxation_velocity_pct_s,
        "contraction_duration_ms": contraction_duration_ms,
        "windowed_trace": df_win[sig_col].values if sig_col in df_win.columns else None,
        "windowed_time": df_win["time_s"].values,
        "window_start": float(df["time_s"].values[w_start]),
        "window_end": float(df["time_s"].values[min(w_end, len(df)) - 1]),
        "duration": float(df["time_s"].values[-1]),
        "metadata": metadata,
    }


# ============================================================================
# Step 4 — Comparison Plots
# ============================================================================

def _condition_label(r):
    """Build a short label from result metadata."""
    m = r["metadata"]
    parts = [m.get("drug") or "?"]
    if m.get("condition_id"):
        parts[0] += str(m["condition_id"])
    if m.get("concentration"):
        parts.append(m["concentration"])
    return " ".join(parts)


def _drug_sort_key(drug):
    order = {"Control": 0, "QUAN": 1, "THAR": 2, "DOF": 3, "Wash": 4, "Testrun": 5}
    return order.get(drug, 9)


def plot_comparison_bpm_amplitude(results, out_path):
    """Bar charts of BPM and amplitude grouped by drug & concentration."""
    if not results:
        return

    drugs = sorted({r["metadata"]["drug"] for r in results if r["metadata"]["drug"]},
                   key=_drug_sort_key)
    n_drugs = max(len(drugs), 1)

    fig, axes = plt.subplots(2, n_drugs, figsize=(5 * n_drugs, 8), squeeze=False)

    for col_idx, drug in enumerate(drugs):
        subset = [r for r in results if r["metadata"]["drug"] == drug]
        labels = [_condition_label(r) for r in subset]
        bpms = [r["bpm"] for r in subset]
        amps = [r["amplitude"] for r in subset]

        x = np.arange(len(labels))
        axes[0, col_idx].bar(x, bpms, color="steelblue", edgecolor="black")
        axes[0, col_idx].set_xticks(x)
        axes[0, col_idx].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[0, col_idx].set_ylabel("BPM")
        axes[0, col_idx].set_title(f"{drug} — BPM", fontweight="bold")
        axes[0, col_idx].grid(axis="y", alpha=0.3)

        axes[1, col_idx].bar(x, amps, color="coral", edgecolor="black")
        axes[1, col_idx].set_xticks(x)
        axes[1, col_idx].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1, col_idx].set_ylabel("Amplitude")
        axes[1, col_idx].set_title(f"{drug} — Amplitude", fontweight="bold")
        axes[1, col_idx].grid(axis="y", alpha=0.3)

    fig.suptitle("BPM & Amplitude by Drug/Concentration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out_path}")


def _plot_trace_grid(results, out_path, title, ylabel):
    """Overlay windowed traces in a grid of (drug, concentration) subplots."""
    if not results:
        return

    # Group by (drug, concentration)
    groups = {}
    for r in results:
        m = r["metadata"]
        key = (m.get("drug", "?"), m.get("concentration") or "—")
        groups.setdefault(key, []).append(r)

    keys = sorted(groups.keys(), key=lambda k: (_drug_sort_key(k[0]), k[1] or ""))
    n = len(keys)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        drug, conc = key
        for r in groups[key]:
            trace = r.get("windowed_trace")
            t = r.get("windowed_time")
            if trace is None or t is None:
                continue
            # Time-align to 0
            t = t - t[0]
            lbl = _condition_label(r)
            pacing = r["metadata"].get("pacing_hz")
            if pacing:
                lbl += f" ({pacing}Hz)"
            ax.plot(t, trace, linewidth=1.2, alpha=0.8, label=lbl)
        ax.set_title(f"{drug} {conc}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_comparison_traces_fluorescent(results, out_path):
    _plot_trace_grid(
        results, out_path,
        title="Fluorescent Traces — Best Window Comparison",
        ylabel="F/F0",
    )


def plot_comparison_traces_mechanical(results, out_path):
    _plot_trace_grid(
        results, out_path,
        title="Mechanical Traces — Best Window Comparison",
        ylabel="Fractional Intensity Change",
    )


def plot_comparison_by_pacing(results, out_path):
    """Overlay traces coloured by drug/concentration, grouped by pacing rate."""
    if not results:
        return

    pacing_groups = {}
    for r in results:
        hz = r["metadata"].get("pacing_hz") or 0
        pacing_groups.setdefault(hz, []).append(r)

    pacing_keys = sorted(pacing_groups.keys())
    n = len(pacing_keys)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    cmap = plt.cm.get_cmap("tab10")
    for col_idx, hz in enumerate(pacing_keys):
        ax = axes[0][col_idx]
        for i, r in enumerate(pacing_groups[hz]):
            trace = r.get("windowed_trace")
            t = r.get("windowed_time")
            if trace is None or t is None:
                continue
            t = t - t[0]
            lbl = _condition_label(r)
            ax.plot(t, trace, linewidth=1.2, alpha=0.8, label=lbl, color=cmap(i % 10))
        ax.set_title(f"Pacing {hz} Hz", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Traces Grouped by Pacing Frequency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out_path}")


# ============================================================================
# Step 5 — Batch Runner
# ============================================================================

def run_batch(root_folder, window_sec=20, out_dir="plots/batch_results",
              allow_manual=True):
    os.makedirs(out_dir, exist_ok=True)

    # Discover
    print("=" * 70)
    print("BATCH DISCOVERY")
    print("=" * 70)
    entries = discover_videos(root_folder)
    print(f"\nFound {len(entries)} usable videos in {root_folder}\n")

    # Summary table
    print(f"{'Folder':<55} {'AVI':<6} {'Type':<13} {'Drug':<10} {'Conc':<8} "
          f"{'Hz':<5} {'FPS':<5} {'Dur(s)':<7} {'Frames':<7}")
    print("-" * 120)
    for e in entries:
        m = e["metadata"]
        print(f"{e['folder_name']:<55} {e['avi_name']:<6} "
              f"{(m.get('video_type') or '?'):<13} "
              f"{(m.get('drug') or '?'):<10} "
              f"{(m.get('concentration') or '—'):<8} "
              f"{str(m.get('pacing_hz') or '?'):<5} "
              f"{str(round(e['real_fps'], 1) if e['real_fps'] else '?'):<5} "
              f"{e['duration_sec']:<7.1f} {e['n_frames']:<7}")
    print()

    # Process
    all_results = []
    fluo_results = []
    mech_results = []

    for i, entry in enumerate(entries):
        meta = entry["metadata"]
        vtype = meta.get("video_type")
        tag = f"[{i+1}/{len(entries)}]"
        safe_name = entry["folder_name"] + "_" + entry["avi_name"].replace(".avi", "")
        print(f"\n{tag} Processing: {entry['folder_name']}/{entry['avi_name']}  ({vtype})")

        try:
            if vtype == "fluorescent":
                vid_out = os.path.join(out_dir, "fluorescent", safe_name)
                result = run_fluorescent(
                    entry["video_path"], entry["ts_path"], meta,
                    vid_out, window_sec=window_sec,
                )
                if result:
                    result["folder"] = entry["folder_name"]
                    result["avi"] = entry["avi_name"]
                    result["video_type"] = "fluorescent"
                    all_results.append(result)
                    fluo_results.append(result)

            elif vtype == "mechanical":
                vid_out = os.path.join(out_dir, "mechanical", safe_name)
                result = run_mechanical(
                    entry["video_path"], entry["ts_path"], meta,
                    vid_out, window_sec=window_sec,
                    allow_manual=allow_manual,
                )
                if result:
                    result["folder"] = entry["folder_name"]
                    result["avi"] = entry["avi_name"]
                    result["video_type"] = "mechanical"
                    all_results.append(result)
                    mech_results.append(result)

            else:
                print(f"  [skip] Unknown video type for {entry['folder_name']}")

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()

    # Save master CSV
    if all_results:
        rows = []
        for r in all_results:
            m = r["metadata"]
            row = {
                "folder": r.get("folder"),
                "avi": r.get("avi"),
                "drug": m.get("drug"),
                "concentration": m.get("concentration"),
                "condition_id": m.get("condition_id"),
                "pacing_hz": m.get("pacing_hz"),
                "video_type": r.get("video_type"),
                "bpm": r.get("bpm"),
                "amplitude": r.get("amplitude"),
                "spike_count": r.get("spike_count", r.get("contraction_count", 0)),
                "duration": r.get("duration"),
                "window_start": r.get("window_start"),
                "window_end": r.get("window_end"),
                "beat_regularity_cv": r.get("beat_regularity_cv"),
            }
            # Fluorescent-specific
            if r.get("video_type") == "fluorescent":
                row["rise_time_ms"] = r.get("rise_time_ms")
                row["decay_time_ms"] = r.get("decay_time_ms")
                row["transient_duration_ms"] = r.get("transient_duration_ms")
            # Mechanical-specific
            if r.get("video_type") == "mechanical":
                row["contraction_velocity_pct_s"] = r.get("contraction_velocity_pct_s")
                row["relaxation_velocity_pct_s"] = r.get("relaxation_velocity_pct_s")
                row["contraction_duration_ms"] = r.get("contraction_duration_ms")
            rows.append(row)
        csv_path = os.path.join(out_dir, "all_results.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n[Data] Saved {csv_path}")

    # Comparison plots
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    if all_results:
        plot_comparison_bpm_amplitude(
            all_results, os.path.join(out_dir, "comparison_bpm_amplitude.png"))

    if fluo_results:
        plot_comparison_traces_fluorescent(
            fluo_results, os.path.join(out_dir, "comparison_traces_fluorescent.png"))

    if mech_results:
        plot_comparison_traces_mechanical(
            mech_results, os.path.join(out_dir, "comparison_traces_mechanical.png"))

    if all_results:
        plot_comparison_by_pacing(
            all_results, os.path.join(out_dir, "comparison_by_pacing.png"))

    # Final summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"  Total videos processed: {len(all_results)}/{len(entries)}")
    print(f"  Fluorescent: {len(fluo_results)}")
    print(f"  Mechanical:  {len(mech_results)}")
    print(f"  Output dir:  {out_dir}/")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch processing pipeline for multi-video organoid analysis.")
    parser.add_argument("--input-dir", required=True,
                        help="Root folder containing experiment subfolders")
    parser.add_argument("--window", type=float, default=20.0,
                        help="Best-window length in seconds (default 20)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: auto-incremented plots/batch_resultsN)")
    parser.add_argument("--no-manual", action="store_true",
                        help="Disable manual ROI fallback (fall through to cellpose)")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        base = "plots/batch_results"
        if not os.path.exists(base):
            out_dir = base
        else:
            n = 2
            while os.path.exists(f"{base}{n}"):
                n += 1
            out_dir = f"{base}{n}"

    run_batch(args.input_dir, window_sec=args.window, out_dir=out_dir,
              allow_manual=not args.no_manual)


if __name__ == "__main__":
    main()
