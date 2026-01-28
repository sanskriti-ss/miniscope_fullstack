"""
Plotting Functions
Contains functions for visualizing ROIs and fluorescence traces.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


def save_roi_overlay_image(
    video_path: str,
    roi_masks,
    roi_info,
    out_path: str = "rois_on_first_frame.png",
) -> None:
    """
    Draw ROI outlines and indices on the first frame of the video
    and save as an image.
    """
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame for ROI visualization")

    overlay = first_frame.copy()

    for idx, (mask, info) in enumerate(zip(roi_masks, roi_info), start=1):
        # Find contours of each ROI mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on overlay (red outline)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

        # Draw ROI index near the centroid
        cx, cy = info["centroid"]
        cv2.putText(
            overlay,
            str(idx),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()  # prevents display in some environments

    print(f"Saved smoothed trace plot to {out_path}")


def estimate_dominant_frequency(trace, fps):
    """
    Estimate the dominant frequency of a signal using FFT.
    Parameters
    ----------
    trace : np.ndarray
        The input signal (wave-filtered fluorescence trace)
    fps : float
        Sampling rate (frames per second)
    Returns
    -------
    float
        Dominant frequency in Hz
    """
    n = len(trace)
    trace = np.nan_to_num(trace)  # Replace NaNs with zero for FFT
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft_vals = np.abs(np.fft.rfft(trace))
    # Ignore DC component (freq=0)
    fft_vals[0] = 0
    dominant_idx = np.argmax(fft_vals)
    return freqs[dominant_idx]


def save_wave_trace_plot(df, out_path="fluorescence_traces_plot_waves.png", fps=None):
    """
    Plots only base wave-filtered FF0 traces (not smoothed) in the dataframe and saves as a PNG.
    Adds dominant frequency to legend.
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
            freq = estimate_dominant_frequency(df[col].values, fps)
            freq_label = f" (main freq: {freq:.2f} Hz)"
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
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved wave trace plot to {out_path}")


def save_spike_trace_plot(df, out_path="fluorescence_spikes.png"):
    """
    Plots fluorescence traces with spikes and dips highlighted.
    Shows the original FF0 trace with markers for detected spikes (upward) and dips (downward).
    Each ROI gets two subplots: one for the signal and one for the derivative.
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
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spike trace plot to {out_path}")
