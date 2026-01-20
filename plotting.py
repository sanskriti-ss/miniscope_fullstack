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
