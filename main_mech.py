"""
Mechanical Movement Tracking for Organoids
Detects and quantifies beating/pulsation by tracking organoid border fluctuations.
Uses high-pass filtering to capture the border between organoid and background.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter

# Import configuration variables
from vars import *
# Import ROI detection strategies
from roi_detection import detect_rois_dispatcher
# Import plotting functions
from plotting import save_roi_overlay_image


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
    
    # STEP 2: Create center ROI mask (organoid is centered, ~1/3 width and height)
    h, w = ref_img.shape
    center_y, center_x = h // 2, w // 2
    roi_radius = int(min(h, w) * 0.22)  # Slightly smaller to focus on organoid
    
    # Create circular ROI around center
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (center_x, center_y), roi_radius, 255, -1)
    cv2.imwrite("debug_organoid_roi_circle.png", roi_mask)
    print(f"[Organoid Detection] Created center ROI: radius={roi_radius} at center=({center_x}, {center_y})")
    
    # STEP 3: Threshold high-pass image with lower threshold to capture border
    print("[Organoid Detection] Creating strong edge map from high-pass...")
    _, edge_strong = cv2.threshold(highpass, 80, 255, cv2.THRESH_BINARY)  # Lower threshold
    
    # Apply ROI mask
    edge_strong_roi = edge_strong.copy()
    edge_strong_roi[roi_mask == 0] = 0
    cv2.imwrite("debug_organoid_edges_strong.png", edge_strong_roi)
    strong_edge_pixels = np.count_nonzero(edge_strong_roi)
    print(f"[Debug] Strong edges in ROI: {strong_edge_pixels} pixels")
    
    # STEP 4: Minimally dilate just enough to close small gaps in the edge ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_dilated = cv2.dilate(edge_strong_roi, kernel, iterations=3)
    cv2.imwrite("debug_organoid_edges_dilated.png", edges_dilated)
    print(f"[Debug] Dilated edges: {np.count_nonzero(edges_dilated)} pixels")
    
    # STEP 5: Light morphological closing to form continuous ring
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_ring = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cv2.imwrite("debug_organoid_binary_ring.png", binary_ring)
    print(f"[Debug] After closing: {np.count_nonzero(binary_ring)} pixels")
    
    # STEP 6: Use the edge ring itself as the organoid mask (don't fill interior)
    # This keeps the mask following the actual detected border
    print("[Organoid Detection] Using edge ring directly as organoid boundary...")
    binary_filled = binary_ring.copy()
    
    # Optionally fill small interior holes, but keep it tight
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_filled = cv2.morphologyEx(binary_filled, cv2.MORPH_CLOSE, kernel_fill, iterations=1)
    
    # Keep only pixels in ROI
    binary_filled[roi_mask == 0] = 0
    cv2.imwrite("debug_organoid_binary.png", binary_filled)
    filled_pixels = np.count_nonzero(binary_filled)
    print(f"[Debug] Final mask: {filled_pixels} pixels")
    
    # STEP 7: Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_filled, connectivity=8, ltype=cv2.CV_32S
    )
    
    if num_labels <= 1:
        print("[ERROR] No organoid detected!")
        return None, None
    
    # Debug: print all detected components
    print(f"[Debug] Found {num_labels - 1} components:")
    
    # Find the largest component
    all_contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_label = None
    best_area = 0
    
    for contour in all_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        
        if area < 100:  # Very permissive minimum
            continue
        
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0
        
        print(f"  Contour: area={area:.0f}, perimeter={perimeter:.1f}, circularity={circularity:.3f}")
        
        # Select the largest component
        if area > best_area:
            best_area = area
            # Find which label
            temp_mask = np.zeros_like(binary_filled)
            cv2.drawContours(temp_mask, [contour], -1, 255, -1)
            
            for label in range(1, num_labels):
                if np.any((labels == label) & (temp_mask > 0)):
                    best_label = label
                    break
    
    if best_label is None:
        print(f"[ERROR] No suitable organoid detected!")
        return None, None
    
    # Create final mask
    mask = (labels == best_label).astype(np.uint8)
    
    # Get info
    area = stats[best_label, cv2.CC_STAT_AREA]
    cx, cy = centroids[best_label]
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    w = stats[best_label, cv2.CC_STAT_WIDTH]
    h = stats[best_label, cv2.CC_STAT_HEIGHT]
    
    info = {
        "label": 1,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": (int(x), int(y), int(w), int(h)),
    }
    
    print(f"[Organoid Detection] Found organoid: area={area:.0f} pixels, center=({cx:.0f}, {cy:.0f})")
    
    # Save final mask for debugging
    cv2.imwrite("debug_organoid_mask.png", mask * 255)
    print("[Debug] Saved organoid mask to debug_organoid_mask.png")
    
    return mask, info


def main():
    """Main pipeline for mechanical movement tracking."""
    
    print("=" * 70)
    print("ORGANOID MECHANICAL MOVEMENT TRACKING")
    print("=" * 70)
    
    # Load video metadata
    fps, n_frames, h, w, start_frame, end_frame = load_video_metadata(
        VIDEO_PATH,
        start_time_sec=START_TIME_SEC,
        end_time_sec=END_TIME_SEC,
        skip_first_frames=SKIP_FIRST_FRAMES
    )
    print(f"\n[Video] {VIDEO_PATH}")
    print(f"  FPS: {fps}, Frames: {n_frames} (after clipping), Size: {w}x{h}")

    # 1. Detect single organoid ROI
    print(f"\n[ROI Detection] Detecting single large organoid ROI...")
    mask, info = detect_single_organoid_roi(
        video_path=VIDEO_PATH,
        channel=CHANNEL,
        n_frames=min(N_REF_FRAMES, n_frames),
        start_frame=start_frame
    )
    
    if mask is None or info is None:
        print("\n[ERROR] No organoid detected! Cannot track mechanical movement.")
        print("Try adjusting the video or check if organoid is visible.")
        return
    
    roi_masks = [mask]
    roi_info = [info]
    print(f"  Organoid ROI: {info['area']} pixels")
    
    # Save ROI overlay for reference
    save_roi_overlay_image(
        video_path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        out_path="mechanical_rois_overlay.png",
    )
    print("  Saved ROI overlay to mechanical_rois_overlay.png")

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
    csv_path = "mechanical_traces.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved mechanical traces to {csv_path}")
    
    # 5. Plot individual ROI traces (now just one organoid)
    print(f"\n[Plotting] Generating plots...")
    plot_mechanical_traces(df, roi_idx=1, fps=fps, 
                          out_path=f"mechanical_traces_organoid.png")
    
    # 6. Create beat summary
    plot_beat_summary(df, fps=fps, out_path="mechanical_beat_summary.png")
    
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
