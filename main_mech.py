"""
Mechanical Movement Tracking for Organoids
Detects and quantifies beating/pulsation by tracking organoid border fluctuations.
Uses high-pass filtering to capture the border between organoid and background.
"""


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter
import argparse

# Import configuration variables
from vars import *
# Import ROI detection strategies
from roi_detection import detect_rois_dispatcher
# Import plotting functions
from plotting import save_roi_overlay_image
# Import shared ROI selection functions
from roi_selection import preview_video_and_draw_rois, extract_frame_channel


def extract_frame_channel(frame, channel=0):
    """
    Extract channel data from frame, handling both grayscale and color videos.
    """
    if len(frame.shape) == 2:
        return frame.astype(np.float32)
    elif len(frame.shape) == 3:
        if frame.shape[2] == 1:
            return frame[:, :, 0].astype(np.float32)
        else:
            return frame[:, :, channel].astype(np.float32)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


def load_video_metadata(path, start_time_sec=0, end_time_sec=0, skip_first_frames=0):
    """Load video metadata and calculate frame range after clipping."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Calculate frame range
    start_frame = max(skip_first_frames, int(start_time_sec * fps))
    end_frame = int(end_time_sec * fps) if end_time_sec > 0 else total_frames
    end_frame = min(end_frame, total_frames)
    n_frames = end_frame - start_frame
    
    if n_frames <= 0:
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")
    
    if start_frame > 0 or end_frame < total_frames:
        print(f"[Video Clipping] Processing frames {start_frame} to {end_frame} "
              f"(time: {start_frame/fps:.1f}s to {end_frame/fps:.1f}s)")
    
    return fps, n_frames, height, width, start_frame, end_frame


def build_reference_image(path, n_ref_frames, use_max_projection=True, channel=1, start_frame=0):
    """Build reference image from video frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reference image")
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    count = 0
    while count < n_ref_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_channel(frame, channel)
        frames.append(gray)
        count += 1

    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames read for reference image")

    stack = np.stack(frames, axis=0)
    if use_max_projection:
        ref = np.max(stack, axis=0)
    else:
        ref = np.mean(stack, axis=0)
    ref = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ref


def apply_highpass_filter(image, kernel_size=31):
    """
    Apply high-pass filter to emphasize edges/borders.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int
        Size of the Gaussian kernel for low-pass (odd number)
    
    Returns
    -------
    np.ndarray
        High-pass filtered image (edges emphasized)
    """
    # Apply Gaussian blur to get low-frequency component
    low_freq = cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), 0)
    
    # Subtract low-freq from original to get high-freq (edges)
    high_freq = image.astype(np.float32) - low_freq
    
    # Normalize to 0-255 range
    high_freq = cv2.normalize(high_freq, None, 0, 255, cv2.NORM_MINMAX)
    
    return high_freq.astype(np.uint8)


def detect_organoid_border(image, roi_mask, threshold_percentile=75):
    """
    Detect the border of an organoid within an ROI mask.
    
    Parameters
    ----------
    image : np.ndarray
        High-pass filtered image emphasizing edges
    roi_mask : np.ndarray
        Binary mask indicating the ROI region
    threshold_percentile : float
        Percentile threshold for edge detection
    
    Returns
    -------
    border_mask : np.ndarray
        Binary mask of the detected border
    border_pixels : int
        Number of pixels in the border
    perimeter_length : float
        Estimated perimeter length
    """
    # Apply ROI mask to the high-pass filtered image
    roi_edges = image.copy()
    roi_edges[roi_mask == 0] = 0
    
    # Threshold to keep only strong edges
    if np.sum(roi_mask > 0) > 0:
        roi_values = roi_edges[roi_mask > 0]
        threshold = np.percentile(roi_values, threshold_percentile)
    else:
        threshold = 128
    
    border_mask = (roi_edges > threshold).astype(np.uint8)
    
    # Clean up border with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, kernel)
    
    # Count border pixels
    border_pixels = np.sum(border_mask)
    
    # Estimate perimeter length using contours
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_length = sum([cv2.arcLength(cnt, closed=True) for cnt in contours])
    
    return border_mask, border_pixels, perimeter_length


def compute_organoid_size_metrics(image, roi_mask):
    """
    Compute multiple size metrics for an organoid.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale frame
    roi_mask : np.ndarray
        Binary mask indicating the ROI region
    
    Returns
    -------
    dict
        Dictionary containing area, perimeter, equivalent_diameter, extent
    """
    # Threshold the image within the ROI to get organoid boundary
    roi_region = image.copy()
    roi_region[roi_mask == 0] = 0
    
    # Adaptive threshold within ROI
    if np.sum(roi_mask > 0) > 0:
        roi_values = roi_region[roi_mask > 0]
        threshold = np.percentile(roi_values, 50)  # Median threshold
    else:
        threshold = 128
    
    binary = (roi_region > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'area': 0,
            'perimeter': 0,
            'equivalent_diameter': 0,
            'extent': 0
        }
    
    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    
    # Equivalent diameter (diameter of circle with same area)
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
    
    # Extent (ratio of contour area to bounding box area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'equivalent_diameter': equivalent_diameter,
        'extent': extent
    }


def extract_mechanical_traces(path, roi_masks, roi_info, channel=1, 
                              start_frame=0, end_frame=None, 
                              highpass_kernel=31, border_threshold=75):
    """
    Extract mechanical movement traces by tracking organoid border and size changes.
    
    Parameters
    ----------
    path : str
        Path to video file
    roi_masks : list
        List of ROI masks
    roi_info : list
        List of ROI information dictionaries
    channel : int
        Video channel to analyze
    start_frame : int
        Starting frame
    end_frame : int
        Ending frame (None for end of video)
    highpass_kernel : int
        Kernel size for high-pass filter
    border_threshold : float
        Percentile threshold for border detection
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing mechanical metrics over time
    float
        Frames per second
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for trace extraction")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_rois = len(roi_masks)
    rows = []
    frame_idx = start_frame
    relative_idx = 0

    print(f"[Mechanical Tracking] Processing {n_rois} ROIs...")
    
    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        ret, frame = cap.read()
        if not ret:
            break

        # Extract grayscale channel
        img = extract_frame_channel(frame, channel)
        
        # Apply high-pass filter to emphasize borders
        img_highpass = apply_highpass_filter(img, kernel_size=highpass_kernel)
        
        t = relative_idx / fps
        
        # Collect metrics for each ROI
        row_data = [relative_idx, t]
        
        for roi_idx, (mask, info) in enumerate(zip(roi_masks, roi_info)):
            # Detect border using high-pass filtered image
            border_mask, border_pixels, perimeter = detect_organoid_border(
                img_highpass, mask, threshold_percentile=border_threshold
            )
            
            # Compute size metrics from original image
            size_metrics = compute_organoid_size_metrics(img, mask)
            
            # Store metrics
            row_data.extend([
                border_pixels,
                perimeter,
                size_metrics['area'],
                size_metrics['perimeter'],
                size_metrics['equivalent_diameter'],
                size_metrics['extent']
            ])
        
        rows.append(row_data)
        frame_idx += 1
        relative_idx += 1
        
        # Progress indicator
        if relative_idx % 100 == 0:
            print(f"  Processed {relative_idx} frames...")

    cap.release()
    
    # Build column names
    cols = ["frame", "time_s"]
    for i in range(n_rois):
        cols.extend([
            f"roi{i+1}_border_pixels",
            f"roi{i+1}_border_perimeter",
            f"roi{i+1}_area",
            f"roi{i+1}_contour_perimeter",
            f"roi{i+1}_diameter",
            f"roi{i+1}_extent"
        ])
    
    df = pd.DataFrame(rows, columns=cols)
    print(f"[Mechanical Tracking] Extracted traces for {len(df)} frames")
    
    return df, fps


def normalize_and_smooth_traces(df, window_size=21):
    """
    Normalize mechanical traces to baseline and smooth.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mechanical metrics
    window_size : int
        Window size for moving average smoothing
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized and smoothed traces
    """
    df_norm = df.copy()
    
    # Find all metric columns (exclude frame and time_s)
    metric_cols = [c for c in df.columns if c not in ['frame', 'time_s']]
    
    for col in metric_cols:
        data = df[col].values
        
        # Normalize to baseline (first 10% of frames)
        baseline_length = max(10, len(data) // 10)
        baseline = np.mean(data[:baseline_length])
        
        if baseline > 0:
            normalized = (data / baseline) - 1.0  # Convert to fractional change
        else:
            normalized = np.zeros_like(data)
        
        # Smooth with moving average
        smoothed = pd.Series(normalized).rolling(window=window_size, center=True, min_periods=1).mean().values
        
        df_norm[f"{col}_norm"] = normalized
        df_norm[f"{col}_smooth"] = smoothed
    
    return df_norm


def detect_beats(df, roi_idx=1, metric='diameter', prominence_threshold=0.01, fps=30):
    """
    Detect beating events from mechanical traces.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mechanical metrics
    roi_idx : int
        ROI index (1-based)
    metric : str
        Which metric to use ('diameter', 'area', 'border_perimeter', etc.)
    prominence_threshold : float
        Minimum prominence for peak detection
    fps : float
        Frames per second
    
    Returns
    -------
    peaks : np.ndarray
        Frame indices of detected peaks
    properties : dict
        Properties of detected peaks
    beat_rate : float
        Estimated beat rate in beats per minute
    """
    col_name = f"roi{roi_idx}_{metric}_smooth"
    
    if col_name not in df.columns:
        print(f"[Warning] Column {col_name} not found in DataFrame")
        return np.array([]), {}, 0.0
    
    signal = df[col_name].values
    
    # Find peaks (expansions) in the signal
    peaks, properties = find_peaks(signal, prominence=prominence_threshold, distance=int(fps*0.2))
    
    # Calculate beat rate
    if len(peaks) > 1:
        # Average time between beats
        peak_times = df.loc[peaks, 'time_s'].values
        intervals = np.diff(peak_times)
        mean_interval = np.mean(intervals)
        beat_rate = 60.0 / mean_interval if mean_interval > 0 else 0.0
    else:
        beat_rate = 0.0
    
    print(f"[Beat Detection] ROI {roi_idx}: Found {len(peaks)} beats, rate = {beat_rate:.1f} BPM")
    
    return peaks, properties, beat_rate


def plot_mechanical_traces(df, roi_idx=1, fps=30, out_path="mechanical_traces.png"):
    """
    Plot all mechanical metrics for a single ROI.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mechanical metrics
    roi_idx : int
        ROI index (1-based)
    fps : float
        Frames per second
    out_path : str
        Output file path
    """
    # Find all columns for this ROI
    roi_cols = [c for c in df.columns if c.startswith(f"roi{roi_idx}_") and "_smooth" in c]
    
    if len(roi_cols) == 0:
        print(f"[Warning] No data found for ROI {roi_idx}")
        return
    
    # Create subplots
    n_metrics = len(roi_cols)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, col in zip(axes, roi_cols):
        ax.plot(df['time_s'], df[col], linewidth=1.5)
        
        # Detect and mark beats for diameter metric
        if 'diameter' in col:
            base_col = col.replace('_smooth', '')
            metric_name = base_col.replace(f'roi{roi_idx}_', '')
            peaks, _, beat_rate = detect_beats(df, roi_idx=roi_idx, metric=metric_name, fps=fps)
            
            if len(peaks) > 0:
                ax.plot(df.loc[peaks, 'time_s'], df.loc[peaks, col], 'ro', markersize=8, label=f'{beat_rate:.1f} BPM')
                ax.legend()
        
        # Clean up column name for label
        label = col.replace(f'roi{roi_idx}_', '').replace('_smooth', '').replace('_', ' ').title()
        ax.set_ylabel(label + '\n(Fractional Change)')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Mechanical Movement Traces - ROI {roi_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Plotting] Saved mechanical traces to {out_path}")
    plt.close()


def plot_beat_summary(df, fps=30, out_path="beat_summary.png"):
    """
    Create a summary plot showing beating for all ROIs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mechanical metrics
    fps : float
        Frames per second
    out_path : str
        Output file path
    """
    # Find all ROIs
    roi_indices = []
    for col in df.columns:
        if col.startswith('roi') and '_diameter_smooth' in col:
            roi_num = int(col.split('_')[0].replace('roi', ''))
            if roi_num not in roi_indices:
                roi_indices.append(roi_num)
    
    roi_indices.sort()
    n_rois = len(roi_indices)
    
    if n_rois == 0:
        print("[Warning] No ROI data found for beat summary")
        return
    
    fig, axes = plt.subplots(n_rois, 1, figsize=(14, 3 * n_rois), sharex=True)
    
    if n_rois == 1:
        axes = [axes]
    
    for ax, roi_idx in zip(axes, roi_indices):
        col = f'roi{roi_idx}_diameter_smooth'
        
        if col not in df.columns:
            continue
        
        # Plot diameter trace
        ax.plot(df['time_s'], df[col], linewidth=1.5, color='steelblue', label='Diameter')
        
        # Detect and mark beats
        peaks, _, beat_rate = detect_beats(df, roi_idx=roi_idx, metric='diameter', fps=fps)
        
        if len(peaks) > 0:
            ax.plot(df.loc[peaks, 'time_s'], df.loc[peaks, col], 'ro', 
                   markersize=10, label=f'Beats ({beat_rate:.1f} BPM)')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right')
        
        ax.set_ylabel(f'ROI {roi_idx}\nDiameter Change')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'ROI {roi_idx} - Beat Rate: {beat_rate:.1f} BPM', fontweight='bold')
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Organoid Beating Analysis - All ROIs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Plotting] Saved beat summary to {out_path}")
    plt.close()


def detect_single_organoid_roi(video_path, channel=0, n_frames=50, start_frame=0):
    """
    Detect a single large organoid ROI by finding the region with strongest borders/edges.
    Uses high-pass filtering to emphasize organoid borders over smooth bright backgrounds.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    channel : int
        Channel to analyze
    n_frames : int
        Number of frames to average for detection
    start_frame : int
        Starting frame
    
    Returns
    -------
    mask : np.ndarray
        Binary mask of the organoid ROI
    info : dict
        Dictionary with ROI information
    """
    print("[Organoid Detection] Creating averaged reference image...")
    
    # Build reference image
    ref_img = build_reference_image(video_path, n_frames, use_max_projection=False, 
                                    channel=channel, start_frame=start_frame)
    
    # Save original for debugging
    cv2.imwrite("debug_organoid_original.png", ref_img)
    print("[Debug] Saved original image to debug_organoid_original.png")
    
    # STEP 1: Apply high-pass filter to emphasize edges/borders
    print("[Organoid Detection] Applying high-pass filter to detect borders...")
    highpass = apply_highpass_filter(ref_img, kernel_size=51)
    
    # Save high-pass filtered image for debugging
    cv2.imwrite("debug_organoid_highpass.png", highpass)
    print("[Debug] Saved high-pass filtered image to debug_organoid_highpass.png")
    
    # STEP 2: Threshold highpass image to get strong edges
    print("[Organoid Detection] Finding strongest edges in highpass image...")
    h, w = ref_img.shape
    
    # Use high percentile threshold to keep only the brightest edges (organoid border)
    threshold_val = np.percentile(highpass, 93)  # Top 7% of pixel intensities
    print(f"[Debug] Using threshold={threshold_val:.1f} (93rd percentile)")
    
    _, strong_edges = cv2.threshold(highpass, threshold_val, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug_organoid_edges_strong.png", strong_edges)
    print(f"[Debug] Strong edges: {np.count_nonzero(strong_edges)} pixels")
    
    # STEP 3: Find connected components of edges (to find the organoid edge cluster)
    print("[Organoid Detection] Finding edge clusters...")
    # LIGHTLY dilate to connect only very close edge fragments (not everything!)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_connected = cv2.dilate(strong_edges, kernel, iterations=1)
    cv2.imwrite("debug_organoid_edges_connected.png", edges_connected)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges_connected, connectivity=8, ltype=cv2.CV_32S
    )
    
    print(f"[Debug] Found {num_labels - 1} edge clusters")
    
    # Find clusters and prioritize those in bottom-right area (x > w/2, y > h/2)
    candidate_clusters = []
    
    for label in range(1, num_labels):  # Skip background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        
        # Filter: only consider clusters with reasonable size
        if area < 200 or area > h * w * 0.3:  # Too small or too large
            continue
        
        # Check if in bottom-right quadrant (x > w/2, y > h/2)
        in_bottom_right = (cx > w / 2) and (cy > h / 2)
        
        candidate_clusters.append({
            'label': label,
            'area': area,
            'center': (cx, cy),
            'in_bottom_right': in_bottom_right
        })
    
    # Sort: bottom-right first, then by area
    candidate_clusters.sort(key=lambda x: (not x['in_bottom_right'], -x['area']))
    
    print(f"[Debug] Top 15 edge clusters (prioritizing bottom-right quadrant x>{w/2:.0f}, y>{h/2:.0f}):")
    for i, cluster in enumerate(candidate_clusters[:15]):
        loc = "BOTTOM-RIGHT" if cluster['in_bottom_right'] else "other"
        print(f"  {i+1}. label={cluster['label']}, area={cluster['area']}, center=({cluster['center'][0]:.0f}, {cluster['center'][1]:.0f}) [{loc}]")
    
    if len(candidate_clusters) == 0:
        print("[ERROR] No significant edge cluster found!")
        return None, None
    
    # Select the first bottom-right cluster, or fall back to largest if none
    best_cluster = candidate_clusters[0]
    best_label = best_cluster['label']
    
    print(f"[Debug] Selected edge cluster: label={best_label}, area={best_cluster['area']}, center=({best_cluster['center'][0]:.0f}, {best_cluster['center'][1]:.0f})")
    
    # STEP 4: Get the organoid center and size from this cluster
    cluster_mask = (labels == best_label).astype(np.uint8)
    edge_points = np.where(cluster_mask > 0)
    
    center_y = np.mean(edge_points[0])
    center_x = np.mean(edge_points[1])
    
    # Calculate standard deviation to estimate radius
    std_y = np.std(edge_points[0])
    std_x = np.std(edge_points[1])
    
    # Radius is approximately 1.5 * std (to encompass the organoid)
    radius_y = std_y * 1.8
    radius_x = std_x * 1.8
    
    print(f"[Debug] Organoid center: ({center_x:.0f}, {center_y:.0f})")
    print(f"[Debug] Estimated radius: x={radius_x:.0f}, y={radius_y:.0f}")
    
    # STEP 5: Create elliptical mask around the detected center
    print("[Organoid Detection] Creating organoid mask...")
    binary_filled = np.zeros((h, w), dtype=np.uint8)
    
    # Draw filled ellipse
    cv2.ellipse(binary_filled, 
                (int(center_x), int(center_y)),
                (int(radius_x), int(radius_y)),
                0, 0, 360, 255, -1)
    
    cv2.imwrite("debug_organoid_binary.png", binary_filled)
    filled_pixels = np.count_nonzero(binary_filled)
    print(f"[Debug] Organoid mask: {filled_pixels} pixels")
    
    # STEP 6: Verify the mask makes sense
    if filled_pixels < 1000 or filled_pixels > h * w * 0.7:
        print(f"[ERROR] Mask size is unreasonable: {filled_pixels} pixels")
        return None, None
    
    # STEP 7: Extract organoid info from the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_filled, connectivity=8, ltype=cv2.CV_32S
    )
    
    if num_labels <= 1:
        print("[ERROR] Mask is empty!")
        return None, None
    
    # Get the main component (should be label 1)
    best_label = 1
    mask = (labels == best_label).astype(np.uint8)
    
    # Get info
    area = stats[best_label, cv2.CC_STAT_AREA]
    cx, cy = centroids[best_label]
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    w_bbox = stats[best_label, cv2.CC_STAT_WIDTH]
    h_bbox = stats[best_label, cv2.CC_STAT_HEIGHT]
    
    info = {
        "label": 1,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": (int(x), int(y), int(w_bbox), int(h_bbox)),
    }
    
    print(f"[Organoid Detection] Found organoid: area={area:.0f} pixels, center=({cx:.0f}, {cy:.0f})")
    
    # Save final mask for debugging
    cv2.imwrite("debug_organoid_mask.png", mask * 255)
    print("[Debug] Saved organoid mask to debug_organoid_mask.png")
    
    return mask, info


def manual_roi_selection(video_path, channel=0, start_frame=0, n_frames=10):
    """
    Allow user to manually draw a polygon ROI on the video frame.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    channel : int
        Channel to display
    start_frame : int
        Starting frame
    n_frames : int
        Number of frames to average for display
    
    Returns
    -------
    mask : np.ndarray
        Binary mask of the manually selected ROI
    info : dict
        Dictionary with ROI information
    """
    print("[Manual ROI Selection] Loading reference image...")
    
    # Build averaged reference image for display
    ref_img = build_reference_image(video_path, n_frames, use_max_projection=False,
                                   channel=channel, start_frame=start_frame)
    
    # Apply highpass filter to show organoid border more clearly
    highpass = apply_highpass_filter(ref_img, kernel_size=51)
    
    # Create display image (blend original and highpass for better visibility)
    display = cv2.addWeighted(ref_img, 0.6, highpass, 0.4, 0)
    display_color = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    
    h, w = ref_img.shape
    
    # Variables for polygon drawing
    points = []
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, drawing, display_color
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start/continue drawing
            points.append((x, y))
            drawing = True
            
            # Draw the point
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            # Draw line back to first point if more than 2 points
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
            
            cv2.imshow("Manual ROI Selection", temp_display)
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Show preview line
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            if len(points) > 0:
                cv2.line(temp_display, points[-1], (x, y), (0, 255, 0), 1)
            
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
            
            cv2.imshow("Manual ROI Selection", temp_display)
    
    # Create window and set mouse callback
    cv2.namedWindow("Manual ROI Selection")
    cv2.setMouseCallback("Manual ROI Selection", mouse_callback)
    cv2.imshow("Manual ROI Selection", display_color)
    
    print("\n" + "="*70)
    print("MANUAL ROI SELECTION INSTRUCTIONS:")
    print("="*70)
    print("  - Click to add points around the organoid")
    print("  - Press ENTER when done (will auto-close the polygon)")
    print("  - Press 'r' to reset and start over")
    print("  - Press 'ESC' to cancel")
    print("="*70 + "\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER - finish
            if len(points) >= 3:
                break
            else:
                print("[Warning] Need at least 3 points to create a polygon")
        
        elif key == ord('r'):  # Reset
            points = []
            drawing = False
            cv2.imshow("Manual ROI Selection", display_color)
            print("[Manual ROI] Reset - start drawing again")
        
        elif key == 27:  # ESC - cancel
            print("[Manual ROI] Cancelled by user")
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print("[ERROR] Need at least 3 points to create ROI")
        return None, None
    
    # Create mask from polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    
    # Calculate ROI info
    M = cv2.moments(points_array)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = np.mean(points_array[:, 0]), np.mean(points_array[:, 1])
    
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    bbox = (int(np.min(x_coords)), int(np.min(y_coords)),
            int(np.max(x_coords) - np.min(x_coords)),
            int(np.max(y_coords) - np.min(y_coords)))
    
    area = np.sum(mask > 0)
    
    info = {
        "label": 1,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": bbox,
    }
    
    print(f"[Manual ROI] Created ROI: area={area} pixels, center=({cx:.0f}, {cy:.0f})")
    
    # Save mask for debugging
    cv2.imwrite("debug_organoid_mask.png", mask)
    cv2.imwrite("debug_organoid_original.png", ref_img)
    cv2.imwrite("debug_organoid_highpass.png", highpass)
    
    return mask.astype(np.uint8), info


def main():
    """Main pipeline for mechanical movement tracking."""
    
    parser = argparse.ArgumentParser(description="Run the mechanical movement tracking pipeline.")
    parser.add_argument("--manual", action="store_true", help="Enable manual ROI selection mode.")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ORGANOID MECHANICAL MOVEMENT TRACKING")
    print("=" * 70)
    

    # --- Output directory setup ---
    avi_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    out_dir = os.path.join("plots", avi_base)
    os.makedirs(out_dir, exist_ok=True)

    # Load video metadata
    fps, n_frames, h, w, start_frame, end_frame = load_video_metadata(
        VIDEO_PATH,
        start_time_sec=START_TIME_SEC,
        end_time_sec=END_TIME_SEC,
        skip_first_frames=SKIP_FIRST_FRAMES
    )
    print(f"\n[Video] {VIDEO_PATH}")
    print(f"  FPS: {fps}, Frames: {n_frames} (after clipping), Size: {w}x{h}")

    # 1. Detect or manually select organoid ROI
    if args.manual or MANUAL_ROI_SELECTION:
        print(f"\n[ROI Selection] Manual ROI selection mode...")
        roi_masks, roi_info = preview_video_and_draw_rois(
            video_path=VIDEO_PATH,
            channel=CHANNEL,
            n_preview_frames=min(N_REF_FRAMES, n_frames)
        )
        
        if len(roi_masks) == 0:
            print("\n[ERROR] No ROIs selected! Cannot track mechanical movement.")
            return
        
        # Use first ROI for mechanical analysis
        mask, info = roi_masks[0], roi_info[0]
    else:
        print(f"\n[ROI Detection] Automatic organoid detection...")
        mask, info = detect_single_organoid_roi(
            video_path=VIDEO_PATH,
            channel=CHANNEL,
            n_frames=min(N_REF_FRAMES, n_frames),
            start_frame=start_frame
        )
    
    if mask is None or info is None:
        print("\n[ERROR] No organoid ROI created! Cannot track mechanical movement.")
        if not MANUAL_ROI_SELECTION:
            print("Try using MANUAL_ROI_SELECTION=True in vars.py to draw your own ROI.")
        return
    
    roi_masks = [mask]
    roi_info = [info]
    print(f"  Organoid ROI: {info['area']} pixels")
    
    # Save ROI overlay for reference
    save_roi_overlay_image(
        video_path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        out_path=os.path.join(out_dir, "mechanical_rois_overlay.png"),
    )
    print(f"  Saved ROI overlay to {os.path.join(out_dir, 'mechanical_rois_overlay.png')}")

    # 2. Extract mechanical traces
    print(f"\n[Mechanical Analysis] Extracting movement traces...")
    df, fps = extract_mechanical_traces(
        path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        channel=CHANNEL,
        start_frame=start_frame,
        end_frame=end_frame,
        highpass_kernel=31,  # High-pass filter kernel size
        border_threshold=75   # Border detection threshold percentile
    )
    
    # 3. Normalize and smooth traces
    print(f"\n[Processing] Normalizing and smoothing traces...")
    df = normalize_and_smooth_traces(df, window_size=21)
    
    # 4. Save raw data
    csv_path = os.path.join(out_dir, "mechanical_traces.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved mechanical traces to {csv_path}")
    
    # 5. Plot individual ROI traces (now just one organoid)
    print(f"\n[Plotting] Generating plots...")
    plot_mechanical_traces(df, roi_idx=1, fps=fps, 
                          out_path=os.path.join(out_dir, "mechanical_traces_organoid.png"))
    
    # 6. Create beat summary
    plot_beat_summary(df, fps=fps, out_path=os.path.join(out_dir, "mechanical_beat_summary.png"))
    
    # 7. Print summary statistics
    print(f"\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    _, _, beat_rate = detect_beats(df, roi_idx=1, metric='diameter', fps=fps)
    print(f"  Organoid Beat Rate: {beat_rate:.1f} BPM")
    print(f"  Organoid Size: {roi_info[0]['area']} pixels")
    print(f"  Analysis Duration: {df['time_s'].iloc[-1]:.1f} seconds")
    
    print("\n[Complete] Mechanical movement analysis finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()
