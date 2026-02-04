"""
Main Pipeline for Fluorescence Trace Extraction
Detects ROIs and extracts fluorescence traces from microscopy videos.
"""


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import argparse

# Import configuration variables
from vars import *
# Import ROI detection strategies
from roi_detection import detect_rois_dispatcher
# Import plotting functions
from plotting import (save_roi_overlay_image, save_trace_plot, 
                     save_smoothed_trace_plot, save_wave_trace_plot, 
                     save_spike_trace_plot)


def extract_frame_channel(frame, channel=0):
    """
    Extract channel data from frame, handling both grayscale and color videos.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame from cv2.VideoCapture (can be grayscale or BGR)
    channel : int
        Channel to extract (0=B, 1=G, 2=R). Ignored for grayscale.
    
    Returns
    -------
    np.ndarray (2D, float32)
        Extracted channel data
    """
    if len(frame.shape) == 2:
        # Grayscale video (single channel)
        return frame.astype(np.float32)
    elif len(frame.shape) == 3:
        # Color video (BGR)
        if frame.shape[2] == 1:
            # Single channel stored as 3D array
            return frame[:, :, 0].astype(np.float32)
        else:
            # Multi-channel BGR
            return frame[:, :, channel].astype(np.float32)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


def load_video_metadata(path, start_time_sec=0, end_time_sec=0, skip_first_frames=0):
    """
    Load video metadata and calculate frame range after clipping.
    
    Parameters
    ----------
    path : str
        Path to video file
    start_time_sec : float
        Start processing at this time (seconds)
    end_time_sec : float
        End processing at this time (seconds, 0 = end of video)
    skip_first_frames : int
        Skip this many frames at the start (alternative to start_time_sec)
    
    Returns
    -------
    fps, n_frames, height, width, start_frame, end_frame
    """
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
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reference image")
    
    # Skip to start frame
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


def compute_f0(F_series, mode="percentile", percentile=20, first_n=50):
    F = np.asarray(F_series, dtype=float)
    F = F[np.isfinite(F)]
    if F.size == 0:
        return np.nan
    if mode == "mean_first_n":
        n = min(first_n, F.size)
        return float(np.mean(F[:n]))
    return float(np.percentile(F, percentile))


def dilate_roi_masks(roi_masks, radius: int):
    """Dilate ROI masks to handle movement."""
    if radius <= 0:
        return roi_masks
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = []
    for m in roi_masks:
        dilated.append(cv2.dilate(m.astype(np.uint8), kernel))
    return dilated


def extract_traces(path, roi_masks, channel=1, start_frame=0, end_frame=None):
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

        for m in roi_masks:
            vals = img[m.astype(bool)]
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
    return df, fps


def normalize_traces_FF0(df, f0_mode, f0_percentile, f0_first_n):
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


def smooth_traces(df, window_length=21, polyorder=3, additional_smoothing=True):
    """
    Smooth fluorescence traces using Savitzky-Golay filter with enhanced noise reduction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing time series data with FF0_roi columns
    window_length : int
        Length of the filter window (must be odd and >= polyorder + 1)
        Increased default for noisy data
    polyorder : int
        Order of the polynomial used to fit the samples
    additional_smoothing : bool
        If True, applies additional moving average for very noisy data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added smoothed columns (FF0_roi_smooth)
    """
    df_smoothed = df.copy()
    # Look for FF0_roi columns that don't already have "_smooth" suffix
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" not in c]
    
    print(f"Found columns to smooth: {roi_cols}")
    
    for col in roi_cols:
        data = df[col].values
        # Handle NaN values
        mask = ~np.isnan(data)
        
        if np.sum(mask) < window_length:
            # Not enough valid points, reduce window length
            reduced_window = min(window_length, np.sum(mask))
            if reduced_window < 3:
                df_smoothed[f"{col}_smooth"] = data
                continue
            # Make sure window length is odd
            if reduced_window % 2 == 0:
                reduced_window -= 1
            window_length = max(3, reduced_window)
        
        # Apply Savitzky-Golay filter (only on valid points)
        smoothed = data.copy()
        
        if np.sum(mask) >= window_length:
            # First pass: Savitzky-Golay filter
            smoothed[mask] = savgol_filter(data[mask], window_length, polyorder)
            
            # Second pass: Additional smoothing for very noisy data
            if additional_smoothing:
                # Apply a light moving average to further reduce noise
                smoothed[mask] = uniform_filter1d(smoothed[mask], size=5, mode='nearest')
        
        df_smoothed[f"{col}_smooth"] = smoothed
    
    return df_smoothed


def extract_wave_component(df, fps, low_freq=0.1, high_freq=2.0, order=3):
    """
    Extract sinusoidal-like wave components from fluorescence traces using a Butterworth bandpass filter.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with FF0_roi columns
    fps : float
        Frames per second (sampling rate)
    low_freq : float
        Low cutoff frequency (Hz)
    high_freq : float
        High cutoff frequency (Hz)
    order : int
        Order of the Butterworth filter
    Returns
    -------
    pd.DataFrame
        DataFrame with added wave columns (FF0_roiX_wave)
    """
    df_wave = df.copy()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_wave" not in c]
    nyq = 0.5 * fps
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    for col in roi_cols:
        data = df[col].values
        mask = ~np.isnan(data)
        filtered = np.full_like(data, np.nan)
        if np.sum(mask) > order * 2:
            filtered[mask] = filtfilt(b, a, data[mask])
        df_wave[f"{col}_wave"] = filtered
    return df_wave


def detect_spikes_and_dips(df, fps, spike_threshold_std=1.5, dip_threshold_std=1.5, 
                          spike_prominence=0.02, window_size=10):
    """
    Detect sharp spikes (rapid increases) and dips (rapid decreases) in fluorescence traces.
    Uses derivative-based detection to capture transient events like calcium transients.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with FF0_roi columns
    fps : float
        Frames per second (sampling rate)
    spike_threshold_std : float
        Number of standard deviations above mean derivative to detect spikes
    dip_threshold_std : float
        Number of standard deviations below mean derivative to detect dips
    spike_prominence : float
        Minimum change in F/F0 to be considered a significant transient
    window_size : int
        Window for computing local baseline
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added spike/dip detection columns
    """
    df_spikes = df.copy()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" not in c and "_wave" not in c and "_spike" not in c]
    
    for col in roi_cols:
        data = df[col].values
        mask = ~np.isnan(data)
        
        # Calculate first derivative (rate of change)
        derivative = np.zeros_like(data)
        derivative[1:] = np.diff(data)
        
        # Calculate statistics on the derivative
        valid_deriv = derivative[mask]
        if len(valid_deriv) > 0:
            deriv_mean = np.mean(valid_deriv)
            deriv_std = np.std(valid_deriv)
            
            # Adaptive thresholds based on the signal's variability
            spike_threshold = deriv_mean + spike_threshold_std * deriv_std
            dip_threshold = deriv_mean - dip_threshold_std * deriv_std
        else:
            spike_threshold = 0.03
            dip_threshold = -0.03
        
        # Detect spikes (rapid increases)
        spike_mask = (derivative > spike_threshold) & (derivative > spike_prominence)
        
        # Detect dips (rapid decreases)  
        dip_mask = (derivative < dip_threshold) & (derivative < -spike_prominence)
        
        # Store spike and dip information
        df_spikes[f"{col}_spike"] = spike_mask.astype(float)
        df_spikes[f"{col}_dip"] = dip_mask.astype(float)
        
        # Create derivative trace for visualization
        df_spikes[f"{col}_derivative"] = derivative
    
    return df_spikes



def preview_video_and_draw_rois(video_path, n_preview_frames=150, channel=0, preview_fps=10):
    """
    Display a video preview and allow the user to draw ROIs manually.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    n_preview_frames : int
        Number of frames to use for preview and ROI drawing.
    channel : int
        Channel to extract (0=B, 1=G, 2=R).
    preview_fps : int
        FPS for video preview playback.

    Returns
    -------
    list
        List of manually drawn ROI masks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    # Read frames for preview
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    for _ in range(n_preview_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_channel(frame, channel)
        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames available for preview.")

    print(f"\n{'='*70}")
    print("VIDEO PREVIEW AND MANUAL ROI SELECTION")
    print(f"{'='*70}")
    print("Instructions:")
    print("  - Press SPACE to start/pause video preview")
    print("  - Press 'd' to enter drawing mode")
    print("  - Press 'q' to quit preview and proceed to ROI drawing")
    print("  - Press ESC to cancel")
    print(f"{'='*70}\n")

    # Show video preview
    frame_idx = 0
    playing = False
    delay = max(1, int(1000 / preview_fps))  # milliseconds
    
    cv2.namedWindow("Video Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Preview", 800, 600)
    
    while True:
        if playing:
            display_frame = frames[frame_idx]
            frame_idx = (frame_idx + 1) % len(frames)
        else:
            display_frame = frames[frame_idx]
        
        # Normalize and enhance contrast for better visibility
        normalized_frame = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(normalized_frame)
        
        # Add text overlay
        display_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
        status = "PLAYING" if playing else "PAUSED"
        cv2.putText(display_bgr, f"Frame {frame_idx+1}/{len(frames)} - {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_bgr, "SPACE: play/pause, 'd': draw ROIs, 'q': proceed, ESC: cancel", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Video Preview", display_bgr)
        
        key = cv2.waitKey(delay if playing else 1) & 0xFF
        
        if key == ord(' '):  # SPACE - toggle play/pause
            playing = not playing
        elif key == ord('d'):  # 'd' - enter drawing mode
            break
        elif key == ord('q'):  # 'q' - proceed to drawing
            break
        elif key == 27:  # ESC - cancel
            cv2.destroyAllWindows()
            return [], []
    
    cv2.destroyAllWindows()
    
    # Create reference frame for ROI drawing
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
    # Allow multiple ROI drawing
    all_masks = []
    all_info = []
    roi_counter = 1
    
    while True:
        print(f"\nDraw ROI #{roi_counter} (or press ESC to finish)...")
        mask, info = manual_roi_selection(avg_frame, roi_number=roi_counter)
        
        if mask is None:
            break
            
        all_masks.append(mask)
        all_info.append(info)
        roi_counter += 1
        
        # Ask if user wants to draw another ROI
        print(f"ROI #{roi_counter-1} created. Press 'n' for another ROI, any other key to finish.")
        response = input().strip().lower()
        if response != 'n':
            break
    
    return all_masks, all_info


def manual_roi_selection(avg_frame, roi_number=1):
    """
    Allow the user to manually draw a polygon ROI on the video frame.
    Enhanced version with better UI and instructions.

    Parameters
    ----------
    avg_frame : np.ndarray
        Averaged frame for ROI drawing.
    roi_number : int
        ROI number for labeling.

    Returns
    -------
    mask : np.ndarray
        Binary mask of the manually selected ROI.
    info : dict
        Dictionary with ROI information.
    """
    h, w = avg_frame.shape
    
    # Enhance contrast for better visibility
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(avg_frame)
    display_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Variables for polygon drawing
    points = []
    drawing = False
    polygon_complete = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, drawing, display_color, polygon_complete
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near first point to close polygon (if we have at least 3 points)
            if len(points) >= 3:
                first_pt = points[0]
                distance = np.sqrt((x - first_pt[0])**2 + (y - first_pt[1])**2)
                if distance < 15:  # Close polygon if within 15 pixels of first point
                    print(f"[Manual ROI #{roi_number}] Polygon closed by clicking near first point")
                    polygon_complete = True
                    return
            
            # Start/continue drawing
            points.append((x, y))
            drawing = True
            
            # Draw the point
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 6, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            # Draw line back to first point if more than 2 points
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
                # Draw larger circle around first point to show where to click to close
                cv2.circle(temp_display, points[0], 15, (0, 0, 255), 2)
            
            cv2.imshow(f"Manual ROI Selection - ROI #{roi_number}", temp_display)
        
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # Double-click to finish polygon
            if len(points) >= 3:
                print(f"[Manual ROI #{roi_number}] Polygon finished with double-click")
                polygon_complete = True
                return
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Show preview line
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 6, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            if len(points) > 0:
                cv2.line(temp_display, points[-1], (x, y), (0, 255, 0), 1)
            
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
                # Show red circle around first point for closing
                cv2.circle(temp_display, points[0], 15, (0, 0, 255), 2)
                # Check if mouse is near first point
                first_pt = points[0]
                distance = np.sqrt((x - first_pt[0])**2 + (y - first_pt[1])**2)
                if distance < 15:
                    cv2.circle(temp_display, points[0], 15, (0, 0, 255), -1, 8)  # Filled red circle when hovering
            
            cv2.imshow(f"Manual ROI Selection - ROI #{roi_number}", temp_display)
    
    # Create window and set mouse callback
    window_name = f"Manual ROI Selection - ROI #{roi_number}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_color)
    
    print(f"\n" + "="*70)
    print(f"MANUAL ROI #{roi_number} SELECTION INSTRUCTIONS:")
    print("="*70)
    print("  - Click to add points around the organoid")
    print("  - Click near the FIRST point (red circle) to close polygon")
    print("  - Or DOUBLE-CLICK to finish polygon")
    print("  - Or press ENTER when done (will auto-close the polygon)")
    print("  - Press 'r' to reset and start over")
    print("  - Press 'ESC' to cancel this ROI")
    print("="*70 + "\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Check if polygon was completed via mouse interaction
        if polygon_complete:
            break
        
        if key == 13:  # ENTER - finish
            if len(points) >= 3:
                break
            else:
                print("[Warning] Need at least 3 points to create a polygon")
        
        elif key == ord('r'):  # Reset
            points = []
            drawing = False
            polygon_complete = False
            cv2.imshow(window_name, display_color)
            print(f"[Manual ROI #{roi_number}] Reset - start drawing again")
        
        elif key == 27:  # ESC - cancel
            print(f"[Manual ROI #{roi_number}] Cancelled by user")
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
        "label": roi_number,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": bbox,
    }
    
    print(f"[Manual ROI #{roi_number}] Created: area={area} pixels, center=({cx:.0f}, {cy:.0f})")
    
    return mask.astype(np.uint8), info


def main():
    parser = argparse.ArgumentParser(description="Run the fluorescence trace extraction pipeline.")
    parser.add_argument("--manual", action="store_true", help="Enable manual ROI selection mode.")
    args = parser.parse_args()

    # --- Output directory setup ---
    avi_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    out_dir = avi_base
    os.makedirs(out_dir, exist_ok=True)

    fps, n_frames, h, w, start_frame, end_frame = load_video_metadata(
        VIDEO_PATH,
        start_time_sec=START_TIME_SEC,
        end_time_sec=END_TIME_SEC,
        skip_first_frames=SKIP_FIRST_FRAMES
    )
    print(f"Video: {VIDEO_PATH}, fps={fps}, frames={n_frames} (after clipping), size={w}x{h}")

    if args.manual:
        print("\n[Manual ROI Selection] Previewing video and allowing manual ROI drawing...")
        roi_masks, roi_info = preview_video_and_draw_rois(VIDEO_PATH, n_preview_frames=50, channel=CHANNEL)
    else:
        # 1. Detect ROIs using selected method
        roi_masks, roi_info, debug_img = detect_rois_dispatcher(
            method=ROI_DETECTION_METHOD,
            video_path=VIDEO_PATH,
            build_ref_fn=build_reference_image,
            extract_frame_fn=extract_frame_channel,
            channel=CHANNEL,
            start_frame=start_frame,
            end_frame=end_frame,
            n_ref_frames=min(N_REF_FRAMES, n_frames),
        )
        print(f"Detected {len(roi_masks)} ROIs using method: {ROI_DETECTION_METHOD}")

    if len(roi_masks) == 0:
        print("\n[WARNING] No ROIs detected! Exiting without further processing.")
        print("Try adjusting detection parameters in vars.py:")
        print("  - Lower TEMPORAL_THRESHOLD_PERCENTILE (more sensitive)")
        print("  - Increase TEMPORAL_SAMPLE_FRAMES (more data)")
        print("  - Reduce MIN_AREA (smaller ROIs)")
        print("  - Try different ROI_DETECTION_METHOD")
        return

    # 2. Dilate ROIs to handle small motion
    roi_masks = dilate_roi_masks(roi_masks, ROI_DILATION_RADIUS)

    # 3. Save ROI outlines on first frame
    save_roi_overlay_image(
        video_path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        out_path=os.path.join(out_dir, "rois_on_first_frame.png"),
    )

    # 4. Extract F(t) for each ROI
    df, fps = extract_traces(VIDEO_PATH, roi_masks, channel=CHANNEL, 
                            start_frame=start_frame, end_frame=end_frame)

    # 5. Compute F/F0
    df = normalize_traces_FF0(df, F0_MODE, F0_PERCENTILE, F0_FIRST_N)
    
    # 5b. Smooth the traces with enhanced parameters for noisy data
    df = smooth_traces(df, window_length=SMOOTH_WINDOW_LENGTH, 
                      polyorder=SMOOTH_POLYORDER, 
                      additional_smoothing=ADDITIONAL_SMOOTHING)
    
    # and save both original and smoothed images
    save_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_traces_plot.png"))
    save_smoothed_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_traces_plot_smoothed.png"))

    # 5c. Extract and plot wave components
    df_wave = extract_wave_component(df, fps, low_freq=WAVE_LOW_FREQ, 
                                    high_freq=WAVE_HIGH_FREQ, order=WAVE_FILTER_ORDER)
    save_wave_trace_plot(df_wave, out_path=os.path.join(out_dir, "fluorescence_traces_plot_waves.png"), fps=fps)
    
    # 5d. Detect and plot spikes and dips
    df = detect_spikes_and_dips(df, fps)
    save_spike_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_spikes.png"))


    # 6. Plot F/F0 vs time
    plt.figure()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]
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
    # plt.show() removed to prevent blocking; plot is saved to file instead

    # 7. Save traces
    df.to_csv(os.path.join(out_dir, "fluorescence_traces.csv"), index=False)
    print(f"Saved traces to {os.path.join(out_dir, 'fluorescence_traces.csv')}")


#### calling main function

if __name__ == "__main__":
    print("Inside main()")
    main()
