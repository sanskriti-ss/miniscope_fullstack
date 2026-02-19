"""
Mechanical Contractility Tracking for Brightfield Organoid Videos
Uses Cellpose for organoid segmentation and tracks area/perimeter changes
to quantify contraction dynamics.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter
import argparse

from cellpose import models as cellpose_models

# Import configuration variables
from vars import *
from helper_functions.timestamps import load_timestamps_from_file


def extract_frame_gray(frame):
    """Convert frame to grayscale float32."""
    if len(frame.shape) == 2:
        return frame.astype(np.float32)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def build_mean_image(video_path, n_frames=50, start_frame=0):
    """Build mean reference image from first n_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_frame_gray(frame))
    cap.release()

    if not frames:
        raise RuntimeError("No frames read")
    stack = np.stack(frames, axis=0)
    mean_img = np.mean(stack, axis=0)
    return cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_organoid_cellpose(video_path, n_frames=30, diameter=None):
    """
    Use Cellpose to segment the organoid in a brightfield reference image.
    Returns a binary mask (uint8, 0/1) of the organoid region and ROI info.
    """
    print("[Cellpose] Building reference image...")
    ref_img = build_mean_image(video_path, n_frames=n_frames)

    # Enhance contrast with CLAHE for better cellpose performance
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(ref_img)

    # For brightfield: organoid is DARK on lighter background.
    # Cellpose expects bright objects on dark background, so invert.
    inverted = cv2.bitwise_not(enhanced)

    cv2.imwrite("debug_mech_cellpose_input.png", enhanced)
    cv2.imwrite("debug_mech_cellpose_input_inverted.png", inverted)
    print("[Debug] Saved cellpose input (original + inverted)")

    # Try multiple cellpose approaches for robustness
    print(f"[Cellpose] Running segmentation (diameter={diameter})...")
    model = cellpose_models.CellposeModel()

    best_mask = None
    best_area = 0
    h_img, w_img = enhanced.shape
    min_organoid_area = h_img * w_img * 0.02  # At least 2% of image

    # Try inverted image first (bright organoid on dark background)
    for img_label, img_input in [("inverted", inverted), ("original", enhanced)]:
        if best_mask is not None:
            break
        for d in [diameter, 250, 300, 200, 150, 0]:
            print(f"[Cellpose] Trying {img_label} image, diameter={d}...")
            masks_cp, flows, styles = model.eval(img_input, diameter=d if d > 0 else None)
            n_objects = masks_cp.max()
            if n_objects == 0:
                continue

            print(f"[Cellpose] Found {n_objects} objects with {img_label}, diameter={d}")

            # Find the largest object that meets minimum size
            for obj_id in range(1, n_objects + 1):
                obj_area = np.sum(masks_cp == obj_id)
                print(f"  Object {obj_id}: area={obj_area} px")
                if obj_area > best_area and obj_area >= min_organoid_area:
                    best_area = obj_area
                    best_mask = (masks_cp == obj_id).astype(np.uint8)

            if best_mask is not None:
                print(f"[Cellpose] Selected organoid: area={best_area} px (from {img_label}, d={d})")
                break

    if best_mask is None:
        print("[WARNING] Cellpose found no large objects! Falling back to threshold-based detection...")
        mask_fb, info_fb = _fallback_threshold_detection(enhanced, ref_img)
        if mask_fb is not None:
            return mask_fb, info_fb, ref_img
        print("[ERROR] Fallback also failed!")
        return None, None, ref_img

    mask = best_mask

    # Get contour info
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
    cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)

    info = {
        "label": best_id,
        "area": int(best_area),
        "centroid": (float(cx), float(cy)),
        "bbox": (x, y, w, h),
    }

    print(f"[Cellpose] Organoid detected: area={info['area']} px, center=({cx},{cy})")

    # Debug visualization
    debug_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.imwrite("debug_mech_cellpose_detection.png", debug_img)

    return mask, info, ref_img


def _fallback_threshold_detection(enhanced, ref_img):
    """
    Fallback organoid detection for brightfield images.
    The organoid is a large, smooth, dark, circular region.
    Uses edge detection + Hough circles, with contour-based backup.
    """
    print("[Fallback] Detecting organoid using edge-based circle detection...")
    h, w = enhanced.shape

    # Step 1: Blur to reduce texture noise, then detect edges
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 80)
    cv2.imwrite("debug_mech_fallback_edges.png", edges)

    # Step 2: Try Hough circle detection
    # Expected organoid radius: ~10-25% of image dimension (not the whole FOV)
    min_r = int(min(h, w) * 0.1)
    max_r = int(min(h, w) * 0.25)

    print(f"[Fallback] Searching for circles r={min_r}-{max_r}...")

    best_circle = None
    for param2 in [30, 25, 20, 15, 10]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.5, minDist=min(h, w) // 3,
            param1=80, param2=param2,
            minRadius=min_r, maxRadius=max_r
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Pick circle closest to image center with good radius
            img_cx, img_cy = w / 2, h / 2
            best_score = -1
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                center_score = 1.0 - dist / (np.sqrt(img_cx**2 + img_cy**2))
                size_score = r / max_r
                score = center_score * 0.6 + size_score * 0.4
                print(f"  Circle: center=({cx},{cy}), r={r}, score={score:.2f} (param2={param2})")
                if score > best_score:
                    best_score = score
                    best_circle = (cx, cy, r)
            if best_circle is not None:
                break

    if best_circle is not None:
        cx, cy, r = best_circle
        print(f"[Fallback] Hough circle: center=({cx},{cy}), radius={r}")

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        area = int(np.pi * r * r)

        info = {
            "label": 1,
            "area": area,
            "centroid": (float(cx), float(cy)),
            "bbox": (cx - r, cy - r, 2 * r, 2 * r),
        }

        debug = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite("debug_mech_fallback_detection.png", debug)
        return mask, info

    # Step 3: Hough failed - use smoothness + darkness thresholding
    print("[Fallback] Hough failed, trying smoothness-based detection...")
    blur_small = cv2.GaussianBlur(enhanced.astype(np.float32), (5, 5), 0)
    blur_sq = cv2.GaussianBlur((enhanced.astype(np.float32))**2, (5, 5), 0)
    local_var = np.clip(blur_sq - blur_small**2, 0, None)
    local_std = np.sqrt(local_var)
    local_std_smooth = cv2.GaussianBlur(local_std, (31, 31), 0)

    inv_std = cv2.bitwise_not(
        cv2.normalize(local_std_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )
    dark_score = cv2.bitwise_not(enhanced)
    combined = cv2.addWeighted(inv_std, 0.5, dark_score, 0.5, 0)
    combined = cv2.GaussianBlur(combined, (15, 15), 0)

    # Use a high threshold to get only the darkest smoothest region
    thresh_val = np.percentile(combined, 85)
    _, binary = cv2.threshold(combined, thresh_val, 255, cv2.THRESH_BINARY)

    # Vignette mask
    vignette = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(vignette, (w // 2, h // 2), int(min(w, h) * 0.45), 255, -1)
    binary = cv2.bitwise_and(binary, vignette)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imwrite("debug_mech_fallback_binary.png", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Pick the most circular contour
    best_cnt = None
    best_circ = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter ** 2)
        print(f"  Contour: area={area:.0f}, circ={circ:.2f}")
        if circ > best_circ:
            best_circ = circ
            best_cnt = cnt

    if best_cnt is None:
        return None, None

    area = cv2.contourArea(best_cnt)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best_cnt], -1, 1, -1)

    M = cv2.moments(best_cnt)
    cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else w // 2
    cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else h // 2
    x, y, bw, bh = cv2.boundingRect(best_cnt)

    info = {
        "label": 1, "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": (x, y, bw, bh),
    }
    print(f"[Fallback] Detected organoid: area={area:.0f} px, center=({cx},{cy})")

    debug = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug, [best_cnt], -1, (0, 255, 0), 2)
    cv2.imwrite("debug_mech_fallback_detection.png", debug)
    return mask, info


def create_padded_roi(mask, pad_fraction=0.3):
    """
    Create a padded ROI around the cellpose mask.
    The pad allows tracking of area changes beyond the initial segmentation boundary.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    pad_x = int(w * pad_fraction)
    pad_y = int(h * pad_fraction)

    H, W = mask.shape
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    padded = np.zeros_like(mask)
    padded[y1:y2, x1:x2] = 1
    return padded


def extract_contractility_traces(video_path, organoid_mask, roi_mask,
                                  start_frame=0, end_frame=None, fps=30.0):
    """
    Extract frame-by-frame contractility metrics from brightfield video.

    For each frame within the ROI region:
    - Re-threshold to find organoid boundary (dark blob on lighter background)
    - Track area, perimeter, equivalent diameter
    - Compute mean intensity within organoid vs background

    Returns DataFrame with contractility metrics over time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # Establish a FIXED threshold from the first few frames for consistent tracking
    n_ref = min(10, end_frame - start_frame)
    ref_thresholds = []
    for _ in range(n_ref):
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_gray(frame)
        roi_vals = gray[roi_mask > 0]
        ref_thresholds.append(np.median(roi_vals))
    fixed_threshold = np.mean(ref_thresholds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Precompute organoid reference area from cellpose mask
    ref_area = np.sum(organoid_mask > 0)
    ref_contours, _ = cv2.findContours(organoid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_cnt = max(ref_contours, key=cv2.contourArea)
    ref_perimeter = cv2.arcLength(ref_cnt, True)

    print(f"[Contractility] Reference area={ref_area}, perimeter={ref_perimeter:.1f}")
    print(f"[Contractility] Fixed threshold={fixed_threshold:.1f}")
    print(f"[Contractility] Processing frames {start_frame} to {end_frame}...")

    rows = []
    frame_idx = start_frame
    relative_idx = 0

    # Morphological kernel for cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        gray = extract_frame_gray(frame)

        # Use FIXED threshold for consistent area measurement
        # Organoid = pixels BELOW threshold (darker)
        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[(gray < fixed_threshold) & (roi_mask > 0)] = 255

        # Clean up morphologically
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find largest contour (the organoid)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = 0, 0

            # Mean intensity inside the organoid
            contour_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            organoid_intensity = np.mean(gray[contour_mask > 0])

            # Background intensity
            bg_mask = (roi_mask > 0) & (contour_mask == 0)
            bg_intensity = np.mean(gray[bg_mask]) if np.any(bg_mask) else 0
        else:
            area = 0
            perimeter = 0
            equiv_diameter = 0
            cx, cy = 0, 0
            organoid_intensity = 0
            bg_intensity = 0

        t = relative_idx / fps

        # Fixed-mask mean intensity (most robust contractility metric)
        # When organoid contracts: it shrinks within the fixed mask,
        # so more background (lighter) is exposed -> mean intensity increases
        fixed_mask_intensity = np.mean(gray[organoid_mask > 0])

        rows.append({
            'frame': relative_idx,
            'time_s': t,
            'area': area,
            'perimeter': perimeter,
            'equiv_diameter': equiv_diameter,
            'centroid_x': cx,
            'centroid_y': cy,
            'organoid_intensity': organoid_intensity,
            'bg_intensity': bg_intensity,
            'contrast': bg_intensity - organoid_intensity,
            'fixed_mask_intensity': fixed_mask_intensity,
        })

        frame_idx += 1
        relative_idx += 1

        if relative_idx % 100 == 0:
            print(f"  Processed {relative_idx} frames...")

    cap.release()

    df = pd.DataFrame(rows)
    print(f"[Contractility] Extracted traces for {len(df)} frames")
    return df, fps


def normalize_contractility(df, fps=30.0):
    """
    Normalize contractility metrics to fractional change from baseline.
    Baseline = median of full trace (robust to outliers).
    Includes outlier rejection, detrending, and smoothing.
    """
    df_out = df.copy()
    metrics = ['area', 'perimeter', 'equiv_diameter', 'contrast', 'fixed_mask_intensity']

    for col in metrics:
        data = df[col].values.astype(np.float64)

        # Use median as baseline (robust to outliers)
        baseline = np.median(data)
        if baseline > 0:
            norm_data = (data - baseline) / baseline
        else:
            norm_data = np.zeros_like(data)

        # Outlier rejection: clip extreme values (>3 sigma from median)
        med = np.median(norm_data)
        mad = np.median(np.abs(norm_data - med))  # median absolute deviation
        sigma_est = mad * 1.4826  # MAD to std conversion
        clip_thresh = 4.0 * sigma_est
        clipped = np.clip(norm_data, med - clip_thresh, med + clip_thresh)
        df_out[f'{col}_norm'] = clipped

        # Detrend: remove slow baseline drift using polynomial fit
        if len(clipped) > 20:
            x = np.arange(len(clipped))
            poly_coeffs = np.polyfit(x, clipped, 3)
            trend = np.polyval(poly_coeffs, x)
            detrended = clipped - trend
        else:
            detrended = clipped
        df_out[f'{col}_detrend'] = detrended

        # Smooth: use window of ~0.3s
        win = min(max(5, int(fps * 0.3)), len(data) // 3)
        if win % 2 == 0:
            win += 1
        win = max(5, win)
        if win < len(data):
            df_out[f'{col}_smooth'] = savgol_filter(
                detrended, window_length=win, polyorder=2
            )
        else:
            df_out[f'{col}_smooth'] = detrended

    return df_out


def detect_contractions(df, metric='area_smooth', fps=30.0, prominence=0.005):
    """Detect contraction events (dips in area = contractions)."""
    if metric not in df.columns:
        return np.array([]), {}, 0.0

    signal = -df[metric].values  # Invert: contractions are area decreases (dips)

    # Minimum distance between contractions: ~1 second (for 0.5Hz pacing, beats are 2s apart)
    min_distance = max(5, int(fps * 1.0))
    peaks, props = find_peaks(signal, prominence=prominence, distance=min_distance)

    if len(peaks) > 1:
        intervals = np.diff(df.loc[peaks, 'time_s'].values)
        beat_rate = 60.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
    else:
        beat_rate = 0.0

    print(f"[Contraction Detection] Found {len(peaks)} contractions, rate={beat_rate:.1f} BPM")
    return peaks, props, beat_rate


def plot_contractility(df, fps, out_dir, video_name=""):
    """Generate comprehensive contractility plots."""

    time = df['time_s'].values

    # --- Plot 1: Raw metrics ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(time, df['area'], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Area (px)')
    axes[0].set_title('Organoid Area')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, df['perimeter'], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Perimeter (px)')
    axes[1].set_title('Organoid Perimeter')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, df['equiv_diameter'], 'r-', linewidth=1.5)
    axes[2].set_ylabel('Equiv. Diameter (px)')
    axes[2].set_title('Equivalent Diameter')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time, df['contrast'], 'm-', linewidth=1.5)
    axes[3].set_ylabel('Contrast (intensity)')
    axes[3].set_title('Organoid-Background Contrast')
    axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Raw Contractility Metrics\n{video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    raw_path = os.path.join(out_dir, "contractility_raw.png")
    plt.savefig(raw_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {raw_path}")
    plt.close()

    # --- Plot 2: Normalized + smoothed with contraction detection ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Area (primary contractility metric)
    if 'area_smooth' in df.columns:
        axes[0].plot(time, df['area_norm'], 'b-', alpha=0.3, linewidth=1, label='Raw')
        axes[0].plot(time, df['area_smooth'], 'b-', linewidth=2, label='Smoothed')

        peaks, _, beat_rate = detect_contractions(df, metric='area_smooth', fps=fps)
        if len(peaks) > 0:
            axes[0].plot(time[peaks], df['area_smooth'].values[peaks], 'rv',
                        markersize=10, label=f'Contractions ({beat_rate:.1f} BPM)')
        axes[0].legend(loc='upper right')
    else:
        axes[0].plot(time, df['area_norm'], 'b-', linewidth=1.5)

    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Area\n(Fractional Change)')
    axes[0].set_title('Area Contractility (contraction = negative)')
    axes[0].grid(True, alpha=0.3)

    # Diameter
    if 'equiv_diameter_smooth' in df.columns:
        axes[1].plot(time, df['equiv_diameter_norm'], 'r-', alpha=0.3, linewidth=1, label='Raw')
        axes[1].plot(time, df['equiv_diameter_smooth'], 'r-', linewidth=2, label='Smoothed')
        axes[1].legend(loc='upper right')
    else:
        axes[1].plot(time, df['equiv_diameter_norm'], 'r-', linewidth=1.5)

    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Diameter\n(Fractional Change)')
    axes[1].set_title('Diameter Change')
    axes[1].grid(True, alpha=0.3)

    # Contrast
    if 'contrast_smooth' in df.columns:
        axes[2].plot(time, df['contrast_norm'], 'm-', alpha=0.3, linewidth=1, label='Raw')
        axes[2].plot(time, df['contrast_smooth'], 'm-', linewidth=2, label='Smoothed')
        axes[2].legend(loc='upper right')
    else:
        axes[2].plot(time, df['contrast_norm'], 'm-', linewidth=1.5)

    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Contrast\n(Fractional Change)')
    axes[2].set_title('Contrast Change')
    axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Normalized Contractility\n{video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    norm_path = os.path.join(out_dir, "contractility_normalized.png")
    plt.savefig(norm_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {norm_path}")
    plt.close()

    # --- Plot 3: Summary with both area and fixed-mask intensity ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Area metric
    metric_col = 'area_smooth' if 'area_smooth' in df.columns else 'area_norm'
    axes[0].plot(time, df[metric_col], 'b-', linewidth=2, label='Area (smoothed)')

    peaks, _, beat_rate = detect_contractions(df, metric='area_smooth', fps=fps)
    if len(peaks) > 0:
        axes[0].plot(time[peaks], df[metric_col].values[peaks], 'rv',
               markersize=10, label=f'Contractions ({beat_rate:.1f} BPM)')

    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Fractional Area Change')
    axes[0].set_title('Area-Based Contractility (contraction = negative)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Fixed-mask intensity metric (contraction = intensity increases as organoid shrinks)
    if 'fixed_mask_intensity_smooth' in df.columns:
        axes[1].plot(time, df['fixed_mask_intensity_smooth'], 'darkgreen', linewidth=2,
                    label='Mean Intensity (smoothed)')
        # For intensity, contraction = peaks (not dips)
        int_signal = df['fixed_mask_intensity_smooth'].values
        min_dist = max(5, int(fps * 1.0))
        int_peaks, _ = find_peaks(int_signal, prominence=0.002, distance=min_dist)
        if len(int_peaks) > 1:
            int_intervals = np.diff(time[int_peaks])
            int_bpm = 60.0 / np.mean(int_intervals)
        else:
            int_bpm = 0
        if len(int_peaks) > 0:
            axes[1].plot(time[int_peaks], int_signal[int_peaks], 'rv',
                        markersize=10, label=f'Contractions ({int_bpm:.1f} BPM)')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Fractional Intensity Change')
    axes[1].set_title('Intensity-Based Contractility (contraction = positive peak)', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Organoid Contractility - {video_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(out_dir, "contractility_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {summary_path}")
    plt.close()

    return raw_path, norm_path, summary_path


def save_roi_overlay(video_path, mask, info, out_path):
    """Save ROI overlay on first frame."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cx, cy = info['centroid']
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"Area: {info['area']} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(out_path, frame)
    print(f"[Overlay] Saved ROI overlay to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Brightfield organoid contractility analysis using Cellpose.")
    parser.add_argument("--diameter", type=int, default=CELLPOSE_DIAMETER,
                        help="Cellpose diameter parameter")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Override FPS (useful when metadata is wrong)")
    args = parser.parse_args()

    print("=" * 70)
    print("BRIGHTFIELD ORGANOID CONTRACTILITY ANALYSIS (Cellpose)")
    print("=" * 70)

    video_path = VIDEO_PATH
    video_name = os.path.basename(video_path)

    # Output directory
    avi_base = os.path.splitext(video_name)[0]
    out_dir = os.path.join("plots", avi_base)
    os.makedirs(out_dir, exist_ok=True)

    # Get video info
    cap = cv2.VideoCapture(video_path)
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Use timestamps file for accurate FPS if available
    ts_data = load_timestamps_from_file(video_path)
    if ts_data is not None and len(ts_data) > 1:
        # ts_data is already in seconds (load_timestamps_from_file converts ms->s)
        ts_intervals = np.diff(ts_data)
        actual_fps = 1.0 / np.mean(ts_intervals)
        total_frames_ts = len(ts_data)
        print(f"[Video] Using timestamps: {total_frames_ts} frames, actual FPS={actual_fps:.1f}")
        fps = actual_fps
        # The video file may have fewer frames than timestamps due to codec issues
        # Use what we can read
    else:
        fps = args.fps if args.fps != 30.0 else (meta_fps if meta_fps > 0 else 30.0)

    print(f"\n[Video] {video_path}")
    print(f"  Readable frames: {total_frames}, Size: {w}x{h}")
    print(f"  FPS: {fps:.1f}")

    # 1. Detect organoid with Cellpose
    print(f"\n[Step 1] Detecting organoid with Cellpose...")
    mask, info, ref_img = detect_organoid_cellpose(
        video_path, n_frames=min(30, total_frames), diameter=args.diameter
    )

    if mask is None:
        print("[ERROR] Could not detect organoid. Exiting.")
        return

    # Save ROI overlay
    save_roi_overlay(video_path, mask, info, os.path.join(out_dir, "mech_roi_overlay.png"))

    # 2. Create padded ROI for tracking
    roi_mask = create_padded_roi(mask, pad_fraction=0.3)

    # 3. Extract contractility traces
    print(f"\n[Step 2] Extracting contractility traces...")
    df, fps = extract_contractility_traces(
        video_path, organoid_mask=mask, roi_mask=roi_mask,
        start_frame=0, end_frame=None, fps=fps
    )

    # 4. Normalize and smooth
    print(f"\n[Step 3] Normalizing and smoothing...")
    df = normalize_contractility(df, fps=fps)

    # 5. Save CSV
    csv_path = os.path.join(out_dir, "contractility_traces.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Data] Saved traces to {csv_path}")

    # 6. Plot
    print(f"\n[Step 4] Generating plots...")
    raw_path, norm_path, summary_path = plot_contractility(df, fps, out_dir, video_name)

    # 7. Summary - use both metrics
    area_peaks, _, area_bpm = detect_contractions(df, metric='area_smooth', fps=fps)

    # Intensity-based: contractions are peaks (intensity increases when organoid shrinks)
    int_signal = df.get('fixed_mask_intensity_smooth')
    if int_signal is not None:
        int_vals = int_signal.values
        min_dist = max(5, int(fps * 1.0))
        int_peaks, _ = find_peaks(int_vals, prominence=0.002, distance=min_dist)
        if len(int_peaks) > 1:
            int_intervals = np.diff(df['time_s'].values[int_peaks])
            int_bpm = 60.0 / np.mean(int_intervals)
        else:
            int_bpm = 0
    else:
        int_peaks = np.array([])
        int_bpm = 0

    print(f"\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Video: {video_name}")
    print(f"  Frames analyzed: {len(df)}")
    print(f"  Duration: {df['time_s'].iloc[-1]:.2f} s")
    print(f"  Organoid area (ref): {info['area']} px")
    print(f"  --- Area-based metric ---")
    print(f"    Contractions: {len(area_peaks)}, Rate: {area_bpm:.1f} BPM")
    print(f"  --- Intensity-based metric (more robust) ---")
    print(f"    Contractions: {len(int_peaks)}, Rate: {int_bpm:.1f} BPM")
    print(f"  Expected rate (0.5Hz pacing): 30.0 BPM")
    print(f"\n  Outputs in: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
