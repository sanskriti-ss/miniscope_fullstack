"""
ROI Detection Strategies
Contains different methods for detecting regions of interest (ROIs) in fluorescence videos.
"""

import cv2
import numpy as np
from vars import *


def detect_rois_from_reference(ref_img, min_area=30, max_area=1000):
    """
    Segment bright green blobs on dark background and return one mask per blob.
    Also filters out grid lines (long skinny components) and tiny specks.
    """
    # Stronger blur to merge specks into blobs
    blurred = cv2.GaussianBlur(ref_img, (11, 11), 0)

    # Compute image stats for diagnostics
    mu, sigma = cv2.meanStdDev(blurred)
    mu = float(mu[0][0])
    sigma = float(sigma[0][0])
    
    print(f"[ROI Detection] Image stats: mean={mu:.1f}, std={sigma:.1f}")
    
    # For low contrast images, enhance contrast first using CLAHE
    # This is crucial for detecting dim organoids
    if sigma < 50:  # Low contrast detected
        print("[ROI Detection] Low contrast detected, applying CLAHE enhancement")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Save enhanced image for debugging
        cv2.imwrite("debug_enhanced.png", enhanced)
        print("[Debug] Saved enhanced image to debug_enhanced.png")
        
        # Recompute stats on enhanced image
        mu_e, sigma_e = cv2.meanStdDev(enhanced)
        mu_e = float(mu_e[0][0])
        sigma_e = float(sigma_e[0][0])
        print(f"[ROI Detection] Enhanced image stats: mean={mu_e:.1f}, std={sigma_e:.1f}")
        
        # Use Otsu's method on enhanced image for automatic threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"[ROI Detection] Using Otsu threshold on enhanced image")
    else:
        # Standard threshold for higher contrast images
        THRESH_MULTIPLIER = 0.2
        thresh_val = mu + THRESH_MULTIPLIER * sigma
        thresh_val = max(0, min(255, int(thresh_val)))
        print(f"[ROI Detection] Using global threshold={thresh_val}")
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean up: remove tiny holes, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )

    masks = []
    roi_info = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Filter out grid lines: very elongated components
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5.0:
            # likely a vertical/horizontal line instead of a blob
            continue

        # Create mask for this component
        mask = (labels == label).astype(np.uint8)
        masks.append(mask)

        cx, cy = centroids[label]
        roi_info.append(
            {
                "label": label,
                "area": int(area),
                "centroid": (float(cx), float(cy)),
                "bbox": (int(x), int(y), int(w), int(h)),
            }
        )

    print(f"connectedComponents found {num_labels - 1} components, "
          f"kept {len(masks)} after area/aspect filtering")
    
    # Save threshold image for debugging
    cv2.imwrite("debug_threshold.png", thresh)
    print(f"[Debug] Saved threshold image to debug_threshold.png")

    return masks, roi_info, thresh


def detect_rois_temporal_std(video_path, extract_frame_fn, channel=1, sample_frames=300, downsample=2, 
                              smooth_sigma=2.0, min_area=30, max_area=1000, start_frame=0, end_frame=None):
    """
    Detect ROIs based on temporal standard deviation (activity).
    Finds regions that vary the most over time, indicating flashing organoids.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    extract_frame_fn : callable
        Function to extract channel from frame
    channel : int
        Color channel to analyze
    sample_frames : int
        Number of frames to analyze (0 = all frames)
    downsample : int
        Analyze every Nth frame for speed
    smooth_sigma : float
        Gaussian smoothing sigma for variation map
    min_area, max_area : int
        Area filtering in pixels
        
    Returns
    -------
    masks, roi_info, std_map
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Apply frame clipping
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    available_frames = (end_frame if end_frame else total_frames) - start_frame
    if sample_frames == 0 or sample_frames > available_frames:
        sample_frames = available_frames
    
    print(f"[Temporal STD] Analyzing {sample_frames} frames (every {downsample} frames)...")
    
    # Collect frames
    frames = []
    frame_idx = start_frame
    collected = 0
    
    while collected < sample_frames:
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx - start_frame) % downsample == 0:
            gray = extract_frame_fn(frame, channel)
            frames.append(gray)
            collected += 1
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) < 10:
        raise RuntimeError("Too few frames collected for temporal analysis")
    
    print(f"[Temporal STD] Collected {len(frames)} frames, computing temporal std...")
    
    # Calculate temporal standard deviation for each pixel
    stack = np.stack(frames, axis=0)  # shape: (n_frames, height, width)
    std_map = np.std(stack, axis=0).astype(np.float32)
    
    # Smooth the variation map to merge nearby active regions
    std_map_smooth = cv2.GaussianBlur(std_map, (0, 0), smooth_sigma)
    
    # Print statistics
    print(f"[Temporal STD] Variation map stats: min={std_map_smooth.min():.2f}, max={std_map_smooth.max():.2f}, "
          f"mean={std_map_smooth.mean():.2f}, std={std_map_smooth.std():.2f}")
    
    # Save std map for debugging
    std_vis = cv2.normalize(std_map_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("debug_temporal_std_map.png", std_vis)
    print(f"[Debug] Saved temporal std map to debug_temporal_std_map.png")
    
    # Threshold: use percentile-based threshold (more sensitive than Otsu for low contrast)
    thresh_val = np.percentile(std_vis, TEMPORAL_THRESHOLD_PERCENTILE)
    print(f"[Temporal STD] Using threshold={thresh_val:.1f} (percentile={TEMPORAL_THRESHOLD_PERCENTILE})")
    _, thresh = cv2.threshold(std_vis, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )
    
    masks = []
    roi_info = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        
        # Filter elongated components (likely noise/edges)
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5.0:
            continue
        
        mask = (labels == label).astype(np.uint8)
        masks.append(mask)
        
        cx, cy = centroids[label]
        roi_info.append({
            "label": label,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
        })
    
    print(f"[Temporal STD] Found {len(masks)} ROIs based on temporal variation")
    
    return masks, roi_info, std_vis


def detect_rois_temporal_cv(video_path, extract_frame_fn, channel=1, sample_frames=300, downsample=2,
                             smooth_sigma=2.0, min_area=30, max_area=1000, start_frame=0, end_frame=None):
    """
    Detect ROIs based on coefficient of variation (CV = std/mean).
    Normalizes variation by baseline brightness, good for dim but active organoids.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Apply frame clipping
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    available_frames = (end_frame if end_frame else total_frames) - start_frame
    if sample_frames == 0 or sample_frames > available_frames:
        sample_frames = available_frames
    
    print(f"[Temporal CV] Analyzing {sample_frames} frames (every {downsample} frames)...")
    
    # Collect frames
    frames = []
    frame_idx = start_frame
    collected = 0
    
    while collected < sample_frames:
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx - start_frame) % downsample == 0:
            gray = extract_frame_fn(frame, channel)
            frames.append(gray)
            collected += 1
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) < 10:
        raise RuntimeError("Too few frames collected for temporal analysis")
    
    print(f"[Temporal CV] Collected {len(frames)} frames, computing CV...")
    
    # Calculate mean and std for each pixel
    stack = np.stack(frames, axis=0)
    mean_map = np.mean(stack, axis=0).astype(np.float32)
    std_map = np.std(stack, axis=0).astype(np.float32)
    
    # Coefficient of variation: std / mean (avoid division by zero)
    cv_map = np.divide(std_map, mean_map + 1e-6)
    
    # Smooth the CV map
    cv_map_smooth = cv2.GaussianBlur(cv_map, (0, 0), smooth_sigma)
    
    # Print statistics
    print(f"[Temporal CV] CV map stats: min={cv_map_smooth.min():.4f}, max={cv_map_smooth.max():.4f}, "
          f"mean={cv_map_smooth.mean():.4f}")
    
    # Save CV map for debugging
    cv_vis = cv2.normalize(cv_map_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("debug_temporal_cv_map.png", cv_vis)
    print(f"[Debug] Saved temporal CV map to debug_temporal_cv_map.png")
    
    # Threshold using percentile (more sensitive than Otsu)
    thresh_val = np.percentile(cv_vis, TEMPORAL_THRESHOLD_PERCENTILE)
    print(f"[Temporal CV] Using threshold={thresh_val:.1f} (percentile={TEMPORAL_THRESHOLD_PERCENTILE})")
    _, thresh = cv2.threshold(cv_vis, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )
    
    masks = []
    roi_info = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5.0:
            continue
        
        mask = (labels == label).astype(np.uint8)
        masks.append(mask)
        
        cx, cy = centroids[label]
        roi_info.append({
            "label": label,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
        })
    
    print(f"[Temporal CV] Found {len(masks)} ROIs based on coefficient of variation")
    
    return masks, roi_info, cv_vis


def detect_rois_peak_frequency(video_path, extract_frame_fn, channel=1, sample_frames=300, downsample=2,
                                smooth_sigma=2.0, min_area=30, max_area=1000, threshold_percentile=75,
                                start_frame=0, end_frame=None):
    """
    Detect ROIs based on peak/flash frequency.
    Counts how often each pixel exceeds a threshold, good for detecting flashing regions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Apply frame clipping
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    available_frames = (end_frame if end_frame else total_frames) - start_frame
    if sample_frames == 0 or sample_frames > available_frames:
        sample_frames = available_frames
    
    print(f"[Peak Frequency] Analyzing {sample_frames} frames (every {downsample} frames)...")
    
    # Collect frames
    frames = []
    frame_idx = start_frame
    collected = 0
    
    while collected < sample_frames:
        if end_frame is not None and frame_idx >= end_frame:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx - start_frame) % downsample == 0:
            gray = extract_frame_fn(frame, channel)
            frames.append(gray)
            collected += 1
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) < 10:
        raise RuntimeError("Too few frames collected for temporal analysis")
    
    print(f"[Peak Frequency] Collected {len(frames)} frames, detecting peaks...")
    
    # Stack frames
    stack = np.stack(frames, axis=0)  # shape: (n_frames, height, width)
    
    # For each pixel, compute a threshold (e.g., 75th percentile of its time series)
    # and count how many times it exceeds this threshold
    threshold_map = np.percentile(stack, threshold_percentile, axis=0)
    
    # Count exceedances for each pixel
    peak_count = np.sum(stack > threshold_map[np.newaxis, :, :], axis=0).astype(np.float32)
    
    # Smooth the peak count map
    peak_smooth = cv2.GaussianBlur(peak_count, (0, 0), smooth_sigma)
    
    # Print statistics
    print(f"[Peak Frequency] Peak count stats: min={peak_smooth.min():.1f}, max={peak_smooth.max():.1f}, "
          f"mean={peak_smooth.mean():.1f}")
    
    # Save peak map for debugging
    peak_vis = cv2.normalize(peak_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("debug_peak_frequency_map.png", peak_vis)
    print(f"[Debug] Saved peak frequency map to debug_peak_frequency_map.png")
    
    # Threshold using percentile
    thresh_val = np.percentile(peak_vis, TEMPORAL_THRESHOLD_PERCENTILE)
    print(f"[Peak Frequency] Using threshold={thresh_val:.1f} (percentile={TEMPORAL_THRESHOLD_PERCENTILE})")
    _, thresh = cv2.threshold(peak_vis, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )
    
    masks = []
    roi_info = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5.0:
            continue
        
        mask = (labels == label).astype(np.uint8)
        masks.append(mask)
        
        cx, cy = centroids[label]
        roi_info.append({
            "label": label,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
        })
    
    print(f"[Peak Frequency] Found {len(masks)} ROIs based on peak frequency")
    
    return masks, roi_info, peak_vis


def detect_rois_dispatcher(method, video_path, build_ref_fn, extract_frame_fn, channel=1, 
                           start_frame=0, end_frame=None, **kwargs):
    """
    Dispatcher function to call the appropriate ROI detection method.
    
    Parameters
    ----------
    method : str
        One of: "static_reference", "temporal_std", "temporal_cv", "peak_frequency"
    video_path : str
        Path to video file
    build_ref_fn : callable
        Function to build reference image
    extract_frame_fn : callable
        Function to extract channel from frame
    channel : int
        Color channel to analyze
    **kwargs : dict
        Additional parameters for specific methods
        
    Returns
    -------
    roi_masks, roi_info, debug_image
    """
    print(f"\n=== ROI Detection Method: {method} ===")
    
    if method == "static_reference":
        # Use the original reference-based method
        ref_img = build_ref_fn(
            video_path,
            n_ref_frames=kwargs.get('n_ref_frames', N_REF_FRAMES),
            use_max_projection=kwargs.get('use_max_projection', USE_MAX_PROJECTION),
            channel=channel,
            start_frame=start_frame,
        )
        return detect_rois_from_reference(
            ref_img,
            min_area=kwargs.get('min_area', MIN_AREA),
            max_area=kwargs.get('max_area', MAX_AREA),
        )
    
    elif method == "temporal_std":
        return detect_rois_temporal_std(
            video_path,
            extract_frame_fn,
            channel=channel,
            sample_frames=kwargs.get('sample_frames', TEMPORAL_SAMPLE_FRAMES),
            downsample=kwargs.get('downsample', TEMPORAL_DOWNSAMPLE),
            smooth_sigma=kwargs.get('smooth_sigma', TEMPORAL_SMOOTH_SIGMA),
            min_area=kwargs.get('min_area', MIN_AREA),
            max_area=kwargs.get('max_area', MAX_AREA),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    elif method == "temporal_cv":
        return detect_rois_temporal_cv(
            video_path,
            extract_frame_fn,
            channel=channel,
            sample_frames=kwargs.get('sample_frames', TEMPORAL_SAMPLE_FRAMES),
            downsample=kwargs.get('downsample', TEMPORAL_DOWNSAMPLE),
            smooth_sigma=kwargs.get('smooth_sigma', TEMPORAL_SMOOTH_SIGMA),
            min_area=kwargs.get('min_area', MIN_AREA),
            max_area=kwargs.get('max_area', MAX_AREA),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    elif method == "peak_frequency":
        return detect_rois_peak_frequency(
            video_path,
            extract_frame_fn,
            channel=channel,
            sample_frames=kwargs.get('sample_frames', TEMPORAL_SAMPLE_FRAMES),
            downsample=kwargs.get('downsample', TEMPORAL_DOWNSAMPLE),
            smooth_sigma=kwargs.get('smooth_sigma', TEMPORAL_SMOOTH_SIGMA),
            min_area=kwargs.get('min_area', MIN_AREA),
            max_area=kwargs.get('max_area', MAX_AREA),
            threshold_percentile=kwargs.get('threshold_percentile', 75),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    else:
        raise ValueError(f"Unknown ROI detection method: {method}. "
                        f"Choose from: static_reference, temporal_std, temporal_cv, peak_frequency")
