"""
Trace Extraction Functions
Extract fluorescence traces from video using ROI masks.
"""

import cv2
import numpy as np
import pandas as pd

from roi_selection import extract_frame_channel
from .signal_processing import compute_f0


def dilate_roi_masks(roi_masks, radius: int):
    """
    Dilate ROI masks to handle movement.
    
    Parameters
    ----------
    roi_masks : list
        List of binary ROI masks
    radius : int
        Dilation radius in pixels (0 = no dilation)
    
    Returns
    -------
    list
        List of dilated ROI masks
    """
    if radius <= 0:
        return roi_masks
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = []
    for m in roi_masks:
        dilated.append(cv2.dilate(m.astype(np.uint8), kernel))
    return dilated


def extract_traces(path, roi_masks, channel=1, start_frame=0, end_frame=None, roi_masks_for_f0=None):
    """
    Extract fluorescence intensity traces from video for each ROI.
    
    Parameters
    ----------
    path : str
        Path to video file
    roi_masks : list
        List of binary ROI masks (may be dilated)
    channel : int
        Color channel to extract (0=Blue, 1=Green, 2=Red)
    start_frame : int
        Frame to start extraction
    end_frame : int or None
        Frame to end extraction (None = end of video)
    roi_masks_for_f0 : list or None
        Original (non-dilated) masks for F0 calculation
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: frame, time_s, F_roi1, F_roi2, ...
    fps : float
        Frames per second
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for trace extraction")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_rois = len(roi_masks)
    
    # If no separate F0 masks provided, use the trace masks for F0 as well
    if roi_masks_for_f0 is None:
        roi_masks_for_f0 = roi_masks
    
    rows = []
    frame_idx = start_frame
    relative_idx = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        ret, frame = cap.read()
        if not ret:
            break

        img = extract_frame_channel(frame, channel)

        t = relative_idx / fps
        F_vals = []

        # Extract signal from potentially dilated masks
        for trace_mask in roi_masks:
            vals = img[trace_mask.astype(bool)]
            if vals.size == 0:
                F_vals.append(np.nan)
            else:
                F_vals.append(float(np.mean(vals)))

        rows.append([relative_idx, t] + F_vals)
        frame_idx += 1
        relative_idx += 1

    cap.release()

    cols = ["frame", "time_s"] + [f"F_roi{i+1}" for i in range(n_rois)]
    df = pd.DataFrame(rows, columns=cols)
    
    # Store the F0 masks for later use in normalization
    df._f0_masks = roi_masks_for_f0
    
    return df, fps


def normalize_traces_FF0(df, f0_mode, f0_percentile, f0_first_n):
    """
    Normalize fluorescence traces to F/F0.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw F_roi columns
    f0_mode : str
        Mode for F0 calculation ("percentile" or "mean_first_n")
    f0_percentile : float
        Percentile for F0 (if mode="percentile")
    f0_first_n : int
        Number of frames for F0 (if mode="mean_first_n")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added F0_roi and FF0_roi columns
    """
    n_rois = sum(col.startswith("F_roi") for col in df.columns)
    for i in range(n_rois):
        col = f"F_roi{i+1}"
        Fi = df[col].values
        F0 = compute_f0(
            Fi,
            mode=f0_mode,
            percentile=f0_percentile,
            first_n=f0_first_n,
        )
        df[f"F0_roi{i+1}"] = F0
        df[f"FF0_roi{i+1}"] = df[col] / (F0 if F0 != 0 else np.nan)
    return df


def normalize_traces_FF0_custom(df, f0_percentile):
    """
    Normalize traces using a custom F0 percentile.
    Useful for debug visualization with different baseline settings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw F_roi columns
    f0_percentile : float
        Percentile for F0 calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added FF0_roi_debug columns
    """
    df_custom = df.copy()
    n_rois = sum(col.startswith("F_roi") for col in df.columns)
    for i in range(n_rois):
        col = f"F_roi{i+1}"
        Fi = df[col].values
        F0 = compute_f0(Fi, mode="percentile", percentile=f0_percentile)
        df_custom[f"FF0_roi{i+1}_debug"] = df[col] / (F0 if F0 != 0 else np.nan)
    return df_custom
