"""
Plotting Functions
Contains functions for visualizing ROIs and fluorescence traces.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


def save_roi_overlay_image(
    video_path: str,
    roi_masks,
    roi_info,
    out_path: str = "rois_on_first_frame.png",
    roi_masks_original=None,
) -> None:
    """
    Draw ROI outlines and indices on the first frame of the video
    and save as an image.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    roi_masks : list
        List of ROI masks (may be dilated)
    roi_info : list
        List of ROI info dictionaries
    out_path : str
        Output file path
    roi_masks_original : list, optional
        Original ROI masks before dilation (if available, will draw both)
    """
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame for ROI visualization")

    overlay = first_frame.copy()

    # Draw original masks first (if provided) in green
    if roi_masks_original is not None:
        for idx, (mask_orig, info) in enumerate(zip(roi_masks_original, roi_info), start=1):
            contours, _ = cv2.findContours(
                mask_orig.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # Draw original contours in green (dashed style approximated with thinner line)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    # Then draw dilated masks in red
    for idx, (mask, info) in enumerate(zip(roi_masks, roi_info), start=1):
        # Find contours of each ROI mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on overlay (red outline for dilated)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Draw ROI index near the centroid
        cx, cy = info["centroid"]
        cv2.putText(
            overlay,
            str(idx),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    # Add legend to image
    if roi_masks_original is not None:
        cv2.putText(
            overlay,
            "Green=Original ROI, Red=Dilated ROI",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_path, overlay)
    print(f"Saved ROI overlay image to {out_path}")


def save_trace_plot(df, out_path="fluorescence_traces_plot.png"):
    """
    Plots all FF0 traces in the dataframe and saves as a PNG.
    """
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]

    if len(roi_cols) == 0:
        raise ValueError("No FF0 columns found in dataframe")

    plt.figure(figsize=(10, 5))

    for col in roi_cols:
        plt.plot(df["time_s"], df[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel("F/F0")
    
    # Place legend outside if many ROIs
    if len(roi_cols) > 5:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    else:
        plt.legend()
        plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()  # prevents display in some environments

    print(f"Saved trace plot to {out_path}")


def save_smoothed_trace_plot(df, out_path="fluorescence_traces_plot_smoothed.png"):
    """
    Plots all smoothed FF0 traces in the dataframe and saves as a PNG.
    """
    print(f"Available columns: {df.columns.tolist()}")
    roi_cols = [c for c in df.columns if "FF0_roi" in c and "_smooth" in c]
    print(f"Found smoothed columns: {roi_cols}")

    if len(roi_cols) == 0:
        raise ValueError("No smoothed FF0 columns found in dataframe")

    plt.figure(figsize=(10, 5))

    for col in roi_cols:
        label = col.replace("_smooth", "")
        plt.plot(df["time_s"], df[col], label=label, linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("F/F0 (Smoothed)")
    
    # Place legend outside if many ROIs
    if len(roi_cols) > 5:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    else:
        plt.legend()
        plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()  # prevents display in some environments

    print(f"Saved smoothed trace plot to {out_path}")


def save_debug_f0_trace_plot(df, debug_percentile=10, out_path="debug_fluorescence_traces_debug_f0.png"):
    """
    Plots smoothed FF0 traces using a conservative F0 (lower percentile).
    Shows maximum contrast for visualizing subtle activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with fluorescence data
    debug_percentile : float
        Percentile to use for F0 (lower = more contrast, 10 = very conservative)
    out_path : str
        Output file path
    """
    # Extract raw F columns
    roi_cols = [c for c in df.columns if c.startswith("F_roi") and not c.startswith("F0_")]
    
    if len(roi_cols) == 0:
        print("[Warning] No F_roi columns found for debug F0 plot")
        return
    
    plt.figure(figsize=(10, 5))
    
    for col in roi_cols:
        roi_num = col.replace("F_roi", "")
        raw_signal = df[col].values
        
        # Calculate conservative F0 (lowest percentile)
        F0_debug = np.nanpercentile(raw_signal, debug_percentile)
        
        # Normalize with debug F0
        ff0_debug = raw_signal / (F0_debug if F0_debug > 0 else np.nan)
        
        # Smooth the debug trace
        smooth_col = f"FF0_roi{roi_num}_smooth"
        if smooth_col in df.columns:
            smooth_signal = df[smooth_col].values
            # Re-normalize smoothed with debug F0
            ff0_smooth_debug = smooth_signal / (F0_debug if F0_debug > 0 else np.nan)
        else:
            ff0_smooth_debug = None
        
        # Plot
        label = f"ROI {roi_num}"
        plt.plot(df["time_s"], ff0_debug, label=label, linewidth=2, alpha=0.7)
        
        if ff0_smooth_debug is not None:
            plt.plot(df["time_s"], ff0_smooth_debug, linewidth=2.5, alpha=0.9)
    
    plt.xlabel("Time (s)")
    plt.ylabel(f"F/F0 (Conservative F0 = {debug_percentile}th percentile)")
    plt.title("Debug: Traces with Conservative Baseline (Maximum Contrast)")
    
    # Place legend outside if many ROIs
    if len(roi_cols) > 5:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.legend()
        plt.tight_layout()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved debug F0 trace plot to {out_path}")


def estimate_dominant_frequency(trace, fps, method='peak_count'):
    """
    Estimate the dominant frequency of a signal.
    Parameters
    ----------
    trace : np.ndarray
        The input signal (fluorescence trace)
    fps : float
        Sampling rate (frames per second)
    method : str
        'fft': Use FFT (for wave-filtered smooth signals)
        'peak_count': Use peak detection (for raw/smoothed flashing signals)
    Returns
    -------
    float
        Dominant frequency in Hz
    """
    from scipy.signal import find_peaks
    
    n = len(trace)
    trace = np.nan_to_num(trace)  # Replace NaNs with zero
    
    if method == 'peak_count':
        # Count peaks in the signal to estimate flashing frequency
        # Use prominence-based detection (relative to local baseline, not global threshold)
        # This handles drifting baselines much better than absolute height thresholds
        
        signal_std = np.std(trace)
        # Prominence: how much the peak stands out from its local surroundings
        # Set to ~0.3-0.5 of signal std to catch real peaks while ignoring noise
        prominence = max(0.003, signal_std * 0.3)
        
        # Minimum distance between peaks (in samples)
        # For 60 fps, 0.5 seconds = 30 samples minimum spacing
        min_distance = max(10, int(fps * 0.5))
        
        peaks, properties = find_peaks(trace, prominence=prominence, distance=min_distance)
        
        if len(peaks) >= 2:
            # Calculate average time between peaks
            peak_times = peaks / fps
            intervals = np.diff(peak_times)
            mean_interval = np.mean(intervals)
            return 1.0 / mean_interval if mean_interval > 0 else 0.0
        else:
            return 0.0
    else:
        # FFT method (original)
        freqs = np.fft.rfftfreq(n, d=1.0/fps)
        fft_vals = np.abs(np.fft.rfft(trace))
        # Ignore DC component (freq=0)
        fft_vals[0] = 0
        dominant_idx = np.argmax(fft_vals)
        return freqs[dominant_idx]


def save_wave_trace_plot(df, out_path="fluorescence_traces_plot_waves.png", fps=None):
    """
    Plots only base wave-filtered FF0 traces (not smoothed) in the dataframe and saves as a PNG.
    Adds dominant frequency to legend using peak-counting for better accuracy.
    """
    # Only plot columns matching FF0_roi[0-9]+_wave (not _smooth_wave)
    roi_cols = [c for c in df.columns if re.match(r"FF0_roi\d+_wave$", c)]
    if len(roi_cols) == 0:
        raise ValueError("No wave FF0 columns found in dataframe")
    plt.figure(figsize=(10, 5))
    for col in roi_cols:
        label = col.replace("_wave", "")
        freq_label = ""
        if fps is not None:
            # Use peak-counting on smoothed trace for accurate flashing frequency
            # Extract base column name (e.g., "FF0_roi1" from "FF0_roi1_wave")
            base_col = col.replace("_wave", "")
            smooth_col = f"{base_col}_smooth"
            if smooth_col in df.columns:
                freq = estimate_dominant_frequency(df[smooth_col].values, fps, method='peak_count')
            else:
                # Fallback to FFT on wave trace
                freq = estimate_dominant_frequency(df[col].values, fps, method='fft')
            freq_label = f" (flash freq: {freq:.2f} Hz)"
        plt.plot(df["time_s"], df[col], label=label + freq_label, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0 (Wave Component)")
    
    # Place legend outside if many ROIs
    if len(roi_cols) > 5:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    else:
        plt.legend()
        plt.tight_layout()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved wave trace plot to {out_path}")


def save_spike_trace_plot(df, out_path="fluorescence_spikes.png", video_name=None):
    """
    Plots fluorescence traces with spikes and dips highlighted.
    Shows the original FF0 trace with markers for detected spikes (upward) and dips (downward).
    Each ROI gets two subplots: one for the signal and one for the derivative.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with fluorescence traces
    out_path : str
        Output file path
    video_name : str, optional
        Name of the source video file to include in the title
    """
    # Get all base FF0 columns (not derivatives or smoothed)
    roi_cols = [c for c in df.columns if re.match(r"FF0_roi\d+$", c)]
    
    if len(roi_cols) == 0:
        raise ValueError("No FF0 columns found in dataframe")
    
    # Create subplots - two rows per ROI (signal + derivative)
    n_rois = len(roi_cols)
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 3*n_rows))
    
    for idx, col in enumerate(roi_cols):
        roi_num = col.split('_')[-1].replace('roi', '')
        
        # Create subplot for this ROI
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        # Plot the base trace
        ax.plot(df["time_s"], df[col], 'b-', linewidth=1.2, alpha=0.8, label='F/F0')
        
        # Highlight spikes (if detected)
        spike_col = f"{col}_spike"
        dip_col = f"{col}_dip"
        
        if spike_col in df.columns:
            spike_times = df.loc[df[spike_col] > 0, "time_s"]
            spike_vals = df.loc[df[spike_col] > 0, col]
            if len(spike_times) > 0:
                ax.scatter(spike_times, spike_vals, color='red', s=30, 
                          marker='^', alpha=0.9, label=f'Spikes ({len(spike_times)})', zorder=5)
        
        if dip_col in df.columns:
            dip_times = df.loc[df[dip_col] > 0, "time_s"]
            dip_vals = df.loc[df[dip_col] > 0, col]
            if len(dip_times) > 0:
                ax.scatter(dip_times, dip_vals, color='green', s=30, 
                          marker='v', alpha=0.9, label=f'Dips ({len(dip_times)})', zorder=5)
        
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("F/F0", fontsize=9)
        ax.set_title(f"ROI {roi_num} - Fluorescence Transients", fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Add overall figure title with video name if provided
    if video_name:
        fig.suptitle(video_name, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spike trace plot to {out_path}")


def save_detrended_trace_plot(df, out_path="fluorescence_traces_detrended.png"):
    """
    Plots smoothed fluorescence traces with baseline drift removed (detrended).
    Shows the effect of removing slow baseline changes while preserving fast flashing peaks.
    
    The detrending process:
    1. Takes smoothed traces (FF0_roi_smooth)
    2. Fits a low-order polynomial (e.g., quadratic) to extract slow trend
    3. Subtracts the trend to remove photobleaching/focus drift effects
    4. Recenters to original mean to maintain scale
    
    This visualization helps identify:
    - Real activity peaks that may be obscured by drift
    - Whether baseline drop is gradual (photobleaching) or sharp (movement/focus)
    - Stabilized signal for further analysis
    """
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_detrended" in c]
    
    if len(roi_cols) == 0:
        print("[Warning] No detrended FF0 columns found. Detrending not applied yet.")
        return
    
    n_rois = len(roi_cols)
    
    # Create figure with subplots - one per ROI for clarity
    fig, axes = plt.subplots(n_rois, 1, figsize=(12, 3*n_rois), sharex=True)
    
    if n_rois == 1:
        axes = [axes]
    
    for ax, col in zip(axes, roi_cols):
        roi_num = col.replace("FF0_roi", "").replace("_detrended", "")
        smooth_col = f"FF0_roi{roi_num}_smooth"
        
        # Plot both smoothed (original) and detrended versions
        if smooth_col in df.columns:
            ax.plot(df["time_s"], df[smooth_col], 'b-', linewidth=1.5, 
                   label='Original (with drift)', alpha=0.6)
        
        # Plot detrended version
        ax.plot(df["time_s"], df[col], 'r-', linewidth=2, 
               label='Detrended (drift removed)', alpha=0.9)
        
        ax.set_ylabel(f"ROI {roi_num}\nF/F0", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle('Fluorescence Traces: Original vs Detrended (Baseline Drift Removed)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detrended trace plot to {out_path}")
