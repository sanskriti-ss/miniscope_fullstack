"""
ROI Detection Strategies
Contains different methods for detecting regions of interest (ROIs) in fluorescence videos.
"""

import cv2
import numpy as np
from vars import *

# Try to import Cellpose (optional dependency)
try:
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("[Warning] Cellpose not installed. Install with: pip install cellpose")


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
                              smooth_sigma=2.0, min_area=30, max_area=1000, start_frame=0, end_frame=None,
                              threshold_percentile=None):
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
    _thresh_pct = threshold_percentile if threshold_percentile is not None else TEMPORAL_THRESHOLD_PERCENTILE
    thresh_val = np.percentile(std_vis, _thresh_pct)
    print(f"[Temporal STD] Using threshold={thresh_val:.1f} (percentile={_thresh_pct})")
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
                             smooth_sigma=2.0, min_area=30, max_area=1000, start_frame=0, end_frame=None,
                             threshold_percentile=None):
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
    _thresh_pct = threshold_percentile if threshold_percentile is not None else TEMPORAL_THRESHOLD_PERCENTILE
    thresh_val = np.percentile(cv_vis, _thresh_pct)
    print(f"[Temporal CV] Using threshold={thresh_val:.1f} (percentile={_thresh_pct})")
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
    thresh_val = np.percentile(peak_vis, threshold_percentile)
    print(f"[Peak Frequency] Using threshold={thresh_val:.1f} (percentile={threshold_percentile})")
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


def detect_rois_single_organoid(video_path, extract_frame_fn, channel=1, sample_frames=100, 
                                 min_area=5000, max_area=500000, edge_only=True, 
                                 edge_width=0.3, min_circularity=0.4,
                                 start_frame=0, end_frame=None):
    """
    Detect a single large circular-ish organoid ROI using edge detection and circle fitting.
    Uses high-pass filtering to find organoid borders, then fits a circle.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    extract_frame_fn : callable
        Function to extract channel from frame
    channel : int
        Color channel to analyze
    sample_frames : int
        Number of frames to average for detection
    min_area : int
        Minimum organoid area in pixels
    max_area : int
        Maximum organoid area in pixels  
    edge_only : bool
        If True, create annular (ring) mask focusing on edges
    edge_width : float
        Edge width as fraction of radius (0.3 = outer 30% of organoid)
    min_circularity : float
        Minimum circularity threshold (0-1, where 1 is perfect circle)
    start_frame : int
        Starting frame for analysis
    end_frame : int or None
        Ending frame
        
    Returns
    -------
    masks, roi_info, debug_image
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    available_frames = (end_frame if end_frame else total_frames) - start_frame
    n_frames = min(sample_frames, available_frames)
    
    print(f"[Single Organoid] Analyzing {n_frames} frames to find large organoid...")
    
    # Collect frames and compute temporal activity map (std over time)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_fn(frame, channel)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) < 5:
        raise RuntimeError("Too few frames collected")
    
    # Build activity map based on temporal variation
    stack = np.stack(frames, axis=0)
    mean_img = np.mean(stack, axis=0).astype(np.float32)
    std_img = np.std(stack, axis=0).astype(np.float32)
    
    # Save debug images for diagnosis
    cv2.imwrite("debug_single_organoid_mean.png", cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite("debug_single_organoid_std.png", cv2.normalize(std_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    print("[Debug] Saved mean and std images for diagnosis")
    
    # === HIGH-PASS FILTER APPROACH ===
    # Apply high-pass filter to find edges/borders
    print("[Single Organoid] Applying high-pass filter to detect circular border...")
    
    # Use mean image for edge detection (more stable than std)
    mean_norm = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE for contrast enhancement first
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mean_enhanced = clahe.apply(mean_norm)
    
    # Use Sobel gradient magnitude to detect edges on the ENHANCED image
    # This will capture the clear circular border visible in the enhanced image
    sobelx = cv2.Sobel(mean_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mean_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Smooth the gradient to connect edge fragments
    gradient_smooth = cv2.GaussianBlur(gradient_mag, (5, 5), 0)
    
    # High-pass filter (for comparison/debugging)
    blur_large = cv2.GaussianBlur(mean_enhanced, (31, 31), 0)
    highpass_raw = mean_enhanced.astype(np.float32) - blur_large.astype(np.float32)
    highpass_raw = highpass_raw - highpass_raw.min()
    highpass = cv2.normalize(highpass_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Also compute absolute difference
    highpass_abs = np.abs(mean_enhanced.astype(np.float32) - blur_large.astype(np.float32))
    highpass_abs = cv2.normalize(highpass_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Use Canny edge detection with LOW thresholds for faint edges
    # The enhanced image shows the circle clearly, so use gentle thresholds
    canny_edges = cv2.Canny(mean_enhanced, 15, 50)  # Lower thresholds for faint edges
    
    # Also try on blurred version to reduce noise
    mean_blur = cv2.GaussianBlur(mean_enhanced, (5, 5), 0)
    canny_blur = cv2.Canny(mean_blur, 10, 40)
    
    # Combine both Canny results
    canny_combined = cv2.bitwise_or(canny_edges, canny_blur)
    
    # Also try Laplacian for edge detection
    laplacian = cv2.Laplacian(mean_enhanced, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine high-pass and laplacian for visualization
    edges_combined = cv2.addWeighted(highpass, 0.5, laplacian, 0.5, 0)
    
    # Save edge detection results
    cv2.imwrite("debug_single_organoid_edges.png", edges_combined)
    cv2.imwrite("debug_single_organoid_canny.png", canny_combined)
    cv2.imwrite("debug_single_organoid_enhanced.png", mean_enhanced)
    cv2.imwrite("debug_single_organoid_highpass.png", highpass)
    cv2.imwrite("debug_single_organoid_highpass_abs.png", highpass_abs)
    cv2.imwrite("debug_single_organoid_laplacian.png", laplacian)
    cv2.imwrite("debug_single_organoid_gradient.png", gradient_smooth)
    print("[Debug] Saved edge detection images: edges.png, canny.png, enhanced.png, highpass.png, gradient.png, laplacian.png")
    
    # === APPROACH 1: Find organoid by thresholding the GRADIENT image ===
    # The gradient magnitude shows the border of the organoid clearly
    print("[Single Organoid] Trying to find organoid border from gradient image...")
    
    best_contour = None
    best_score = 0
    
    # Try different threshold levels on the gradient image
    for thresh_pct in [85, 80, 75, 70, 65]:
        thresh_val = np.percentile(gradient_smooth, thresh_pct)
        _, thresh = cv2.threshold(gradient_smooth, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Save threshold image for debugging
        if thresh_pct == 93:
            cv2.imwrite("debug_single_organoid_thresh.png", thresh)
        
        # Dilate to connect border fragments, then close to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Fill holes inside the contour
        # Find contours and fill them
        contours_temp, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(thresh)
        cv2.drawContours(filled, contours_temp, -1, 255, -1)
        
        # Find contours on the filled image
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Skip contours that are too large (more than 40% of image)
            max_reasonable_area = width * height * 0.4
            if area < min_area or area > max_reasonable_area:
                continue
            
            # Calculate circularity (how round it is)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                continue
            
            # Get bounding rect for position check
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            
            # Check aspect ratio (should be roughly square-ish)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            # Skip if too elongated
            if aspect < 0.5:
                continue
            
            # Skip if at edge of image
            if cx < width * 0.1 or cx > width * 0.9 or cy < height * 0.1 or cy > height * 0.9:
                continue
            
            # Score based on circularity and centrality
            dist_from_center = np.sqrt((cx - width/2)**2 + (cy - height/2)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            center_score = 1.0 - (dist_from_center / max_dist)
            
            score = circularity * 0.5 + center_score * 0.3 + aspect * 0.2
            
            print(f"  Contour: area={area:.0f}, circ={circularity:.2f}, center=({cx}, {cy}), "
                  f"aspect={aspect:.2f}, score={score:.2f} [thresh={thresh_pct}%]")
            
            if score > best_score and circularity >= min_circularity:
                best_score = score
                best_contour = cnt
        
        if best_contour is not None:
            break
    
    # If we found a good contour, use it directly
    if best_contour is not None:
        area = cv2.contourArea(best_contour)
        M = cv2.moments(best_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, w, h = cv2.boundingRect(best_contour)
            cx, cy = x + w // 2, y + h // 2
        
        equiv_radius = np.sqrt(area / np.pi)
        
        print(f"[Single Organoid] Found contour: area={area:.0f}, center=({cx}, {cy}), "
              f"equiv_radius={equiv_radius:.0f}")
        
        # Create mask from the actual contour shape
        full_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(full_mask, [best_contour], -1, 255, -1)
        
        # Create edge-only mask if requested
        if edge_only:
            print(f"[Single Organoid] Creating edge-focused mask (outer {edge_width*100:.0f}%)")
            
            # Erode to get inner region
            erosion_pixels = int(equiv_radius * edge_width)
            if erosion_pixels > 2:
                kernel_erode = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (erosion_pixels * 2 + 1, erosion_pixels * 2 + 1)
                )
                inner_mask = cv2.erode(full_mask, kernel_erode, iterations=1)
                edge_mask = cv2.subtract(full_mask, inner_mask)
                final_mask = edge_mask
                print(f"[Single Organoid] Edge mask: {np.sum(edge_mask > 0)} pixels (full: {np.sum(full_mask > 0)})")
            else:
                final_mask = full_mask
        else:
            final_mask = full_mask
        
        # Convert to binary (0/1)
        final_mask = (final_mask > 0).astype(np.uint8)
        
        # Build ROI info
        x, y, w, h = cv2.boundingRect(best_contour)
        roi_info = [{
            "label": 1,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
            "equiv_radius": float(equiv_radius),
            "edge_only": edge_only,
        }]
        
        # Create debug visualization
        debug_img = cv2.cvtColor(mean_enhanced, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [best_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.imwrite("debug_single_organoid.png", debug_img)
        print("[Debug] Saved single organoid detection to debug_single_organoid.png")
        
        return [final_mask], roi_info, debug_img
    
    # === APPROACH 2: HOUGH CIRCLE DETECTION (fallback) ===
    print("[Single Organoid] Contour approach failed, trying Hough Circle detection...")
    
    best_circle = None
    
    # Estimate radius range based on image size and min/max area
    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = int(np.sqrt(max_area / np.pi))
    # Clamp to reasonable values - allow larger circles
    min_radius = max(20, min_radius)
    max_radius = min(int(min(width, height) * 0.8), max_radius)
    
    print(f"[Single Organoid] Searching for circles with radius {min_radius}-{max_radius} pixels")
    
    # Try multiple input images for Hough detection
    # Use the ENHANCED image directly - it shows the circle most clearly
    images_to_try = [
        ("enhanced_blur", cv2.GaussianBlur(mean_enhanced, (9, 9), 0)),
        ("canny", canny_combined),
        ("highpass", cv2.GaussianBlur(highpass, (5, 5), 0)),
    ]
    
    for img_name, img_for_hough in images_to_try:
        if best_circle is not None:
            break
        
        # Try different param2 values (accumulator threshold) - go lower for faint edges
        for param2 in [25, 20, 15, 12, 10, 8, 5]:
            circles = cv2.HoughCircles(
                img_for_hough,
                cv2.HOUGH_GRADIENT,
                dp=1.2,  # Accumulator resolution
                minDist=min(width, height) // 3,  # Only one circle expected
                param1=50,  # Canny high threshold (used internally)
                param2=param2,  # Accumulator threshold - lower = more sensitive
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                print(f"[Single Organoid] Found {len(circles[0])} circles with param2={param2} on {img_name}")
                
                # Score each circle - prefer centered circles with good size
                best_score = -1
                for circle in circles[0]:
                    cx, cy, r = circle
                    # Validate the circle center is within the image
                    if not (0 <= cx < width and 0 <= cy < height):
                        continue
                    
                    # Score: penalize circles too close to edges (likely artifacts)
                    # Distance from center of image
                    img_cx, img_cy = width / 2, height / 2
                    dist_from_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                    max_dist = np.sqrt(img_cx**2 + img_cy**2)
                    center_score = 1.0 - (dist_from_center / max_dist)  # 1.0 = at center, 0 = at corner
                    
                    # Penalize circles at image edges (organoid should be more centered)
                    edge_margin = min(cx, cy, width - cx, height - cy)
                    edge_score = min(1.0, edge_margin / (min(width, height) * 0.2))  # Penalize if within 20% of edge
                    
                    # Prefer larger circles (but not too large)
                    size_score = r / max_radius
                    
                    # Combined score
                    score = center_score * 0.4 + edge_score * 0.4 + size_score * 0.2
                    
                    print(f"  Circle: center=({cx}, {cy}), r={r}, center_score={center_score:.2f}, "
                          f"edge_score={edge_score:.2f}, total={score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_circle = (int(cx), int(cy), int(r))
                
                if best_circle is not None:
                    print(f"[Single Organoid] Selected circle: center=({best_circle[0]}, {best_circle[1]}), radius={best_circle[2]}")
                    break  # Found a good circle
    
    # If Hough failed, fall back to contour-based detection
    if best_circle is None:
        print("[Single Organoid] Hough detection failed, falling back to contour fitting...")
        
        # Try multiple threshold approaches
        for thresh_method in ["otsu", "adaptive", "percentile"]:
            if best_circle is not None:
                break
                
            if thresh_method == "otsu":
                _, edge_thresh = cv2.threshold(edges_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif thresh_method == "adaptive":
                edge_thresh = cv2.adaptiveThreshold(edges_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 21, 2)
            else:  # percentile
                thresh_val = np.percentile(edges_combined, 80)
                _, edge_thresh = cv2.threshold(edges_combined, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect edge fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edge_thresh = cv2.dilate(edge_thresh, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edge_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # Find the contour that best fits a circle
            best_fit_score = 0
            
            for cnt in contours:
                if len(cnt) < 10:  # Need enough points
                    continue
                
                # Fit minimum enclosing circle
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                
                # Check if radius is in valid range
                if radius < min_radius or radius > max_radius:
                    continue
                
                # Check if center is in image
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                
                # Calculate circularity of contour
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    continue
                
                # Score based on size and circularity
                score = radius * circularity
                
                if score > best_fit_score:
                    best_fit_score = score
                    best_circle = (int(cx), int(cy), int(radius))
            
            if best_circle is not None:
                print(f"[Single Organoid] Fitted circle via {thresh_method}: center=({best_circle[0]}, {best_circle[1]}), "
                      f"radius={best_circle[2]}, score={best_fit_score:.1f}")
    
    if best_circle is None:
        print("[Single Organoid] No circular organoid detected!")
        print("  TIP: Check debug_single_organoid_edges.png to see if organoid border is visible")
        combined = cv2.addWeighted(mean_norm, 0.5, edges_combined, 0.5, 0)
        return [], [], combined
    
    cx, cy, radius = best_circle
    area = np.pi * radius * radius
    
    print(f"[Single Organoid] Detected circular organoid:")
    print(f"  Center: ({cx}, {cy})")
    print(f"  Radius: {radius} pixels")
    print(f"  Area: {area:.0f} pixels")
    
    # Create circular mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(full_mask, (cx, cy), radius, 255, -1)
    
    # Create edge-only (annular) mask if requested
    erosion_pixels = 0
    if edge_only:
        print(f"[Single Organoid] Creating edge-focused annular mask (outer {edge_width*100:.0f}%)")
        
        inner_radius = int(radius * (1.0 - edge_width))
        erosion_pixels = radius - inner_radius
        
        if inner_radius > 2:
            inner_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(inner_mask, (cx, cy), inner_radius, 255, -1)
            
            # Edge mask = full circle - inner circle
            edge_mask = cv2.subtract(full_mask, inner_mask)
            final_mask = edge_mask
            
            edge_pixels = np.sum(edge_mask > 0)
            full_pixels = np.sum(full_mask > 0)
            print(f"[Single Organoid] Annular mask: {edge_pixels} pixels (full circle: {full_pixels})")
        else:
            print("[Single Organoid] Radius too small for annular mask, using full circle")
            final_mask = full_mask
    else:
        final_mask = full_mask
    
    # Convert to binary (0/1) mask
    final_mask = (final_mask > 0).astype(np.uint8)
    
    # Build ROI info
    roi_info = [{
        "label": 1,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": (int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)),
        "equiv_radius": float(radius),
        "edge_only": edge_only,
    }]
    
    # Create debug visualization
    combined = cv2.addWeighted(mean_norm, 0.7, edges_combined, 0.3, 0)
    debug_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # Draw the detected circle
    cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 2)  # Outer circle (green)
    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)  # Center point (red)
    
    if edge_only and erosion_pixels > 0:
        inner_radius = radius - erosion_pixels
        cv2.circle(debug_img, (cx, cy), inner_radius, (255, 0, 0), 2)  # Inner circle (blue)
    
    cv2.imwrite("debug_single_organoid.png", debug_img)
    print("[Debug] Saved single organoid detection to debug_single_organoid.png")
    
    return [final_mask], roi_info, debug_img


def detect_rois_cellpose(video_path, extract_frame_fn, channel=1, sample_frames=100,
                         diameter=200, edge_only=True, edge_width=0.1,
                         start_frame=0, end_frame=None):
    """
    Detect organoid ROIs using Cellpose deep learning model.
    
    Cellpose is specifically designed for cell/organoid segmentation and works
    well on microscopy images with varying contrast.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    extract_frame_fn : callable
        Function to extract channel from frame
    channel : int
        Color channel to analyze
    sample_frames : int
        Number of frames to average for detection
    diameter : int
        Expected diameter of organoid in pixels (100-300 typical)
    edge_only : bool
        If True, create annular mask focusing on edges
    edge_width : float
        Edge width as fraction of equivalent radius (0.1 = outer 10%)
    start_frame : int
        Starting frame for analysis
    end_frame : int or None
        Ending frame
        
    Returns
    -------
    masks, roi_info, debug_image
    """
    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("Cellpose is not installed. Install with: pip install cellpose")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    available_frames = (end_frame if end_frame else total_frames) - start_frame
    n_frames = min(sample_frames, available_frames)
    
    print(f"[Cellpose] Analyzing {n_frames} frames to build reference image...")
    
    # Collect frames and compute mean
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_fn(frame, channel)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) < 5:
        raise RuntimeError("Too few frames collected")
    
    # Build mean image
    stack = np.stack(frames, axis=0)
    mean_img = np.mean(stack, axis=0).astype(np.float32)
    mean_norm = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE for better contrast (Cellpose works better with enhanced images)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(mean_norm)
    
    # Save enhanced image for debugging
    cv2.imwrite("debug_cellpose_input.png", enhanced)
    print("[Debug] Saved Cellpose input image to debug_cellpose_input.png")
    
    # Run Cellpose
    print(f"[Cellpose] Running segmentation with diameter={diameter}...")
    model = cellpose_models.CellposeModel()
    masks, flows, styles = model.eval(enhanced, diameter=diameter)
    
    n_objects = masks.max()
    print(f"[Cellpose] Detected {n_objects} objects")
    
    if n_objects == 0:
        print("[Cellpose] No objects detected! Try adjusting CELLPOSE_DIAMETER in vars.py")
        return [], [], enhanced
    
    # Process each detected object
    roi_masks = []
    roi_info = []
    
    for obj_id in range(1, n_objects + 1):
        obj_mask = (masks == obj_id).astype(np.uint8)
        
        # Find contour and properties
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        # Get center and bounding box
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
        
        x, y, w, h = cv2.boundingRect(cnt)
        equiv_radius = np.sqrt(area / np.pi)
        
        print(f"  Object {obj_id}: area={area:.0f}, center=({cx}, {cy}), equiv_radius={equiv_radius:.0f}")
        
        # Create edge-only mask if requested
        if edge_only and equiv_radius > 10:
            # Erode to get inner region
            erosion_pixels = int(equiv_radius * edge_width)
            if erosion_pixels > 2:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (erosion_pixels * 2 + 1, erosion_pixels * 2 + 1)
                )
                inner_mask = cv2.erode(obj_mask, kernel, iterations=1)
                edge_mask = cv2.subtract(obj_mask, inner_mask)
                final_mask = edge_mask
                print(f"  Created edge mask: {np.sum(edge_mask > 0)} pixels (full: {np.sum(obj_mask > 0)})")
            else:
                final_mask = obj_mask
        else:
            final_mask = obj_mask
        
        roi_masks.append(final_mask)
        roi_info.append({
            "label": obj_id,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
            "equiv_radius": float(equiv_radius),
            "edge_only": edge_only,
        })
    
    # Create debug visualization
    debug_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for obj_id in range(1, n_objects + 1):
        obj_mask = (masks == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        
        # Draw center
        if obj_id <= len(roi_info):
            cx, cy = roi_info[obj_id - 1]["centroid"]
            cv2.circle(debug_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    
    cv2.imwrite("debug_cellpose_detection.png", debug_img)
    print("[Debug] Saved Cellpose detection to debug_cellpose_detection.png")
    
    return roi_masks, roi_info, debug_img


def detect_rois_dispatcher(method, video_path, build_ref_fn, extract_frame_fn, channel=1, 
                           start_frame=0, end_frame=None, **kwargs):
    """
    Dispatcher function to call the appropriate ROI detection method.
    
    Parameters
    ----------
    method : str
        One of: "static_reference", "temporal_std", "temporal_cv", "peak_frequency", "single_organoid"
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
            threshold_percentile=kwargs.get('threshold_percentile', None),
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
            threshold_percentile=kwargs.get('threshold_percentile', None),
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
            threshold_percentile=kwargs.get('threshold_percentile', TEMPORAL_THRESHOLD_PERCENTILE),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    elif method == "single_organoid":
        return detect_rois_single_organoid(
            video_path,
            extract_frame_fn,
            channel=channel,
            sample_frames=kwargs.get('sample_frames', TEMPORAL_SAMPLE_FRAMES),
            min_area=kwargs.get('min_area', SINGLE_ORGANOID_MIN_AREA),
            max_area=kwargs.get('max_area', SINGLE_ORGANOID_MAX_AREA),
            edge_only=kwargs.get('edge_only', SINGLE_ORGANOID_EDGE_ONLY),
            edge_width=kwargs.get('edge_width', SINGLE_ORGANOID_EDGE_WIDTH),
            min_circularity=kwargs.get('min_circularity', SINGLE_ORGANOID_CIRCULARITY),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    elif method == "cellpose":
        return detect_rois_cellpose(
            video_path,
            extract_frame_fn,
            channel=channel,
            sample_frames=kwargs.get('sample_frames', TEMPORAL_SAMPLE_FRAMES),
            diameter=kwargs.get('diameter', CELLPOSE_DIAMETER),
            edge_only=kwargs.get('edge_only', SINGLE_ORGANOID_EDGE_ONLY),
            edge_width=kwargs.get('edge_width', SINGLE_ORGANOID_EDGE_WIDTH),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    else:
        raise ValueError(f"Unknown ROI detection method: {method}. "
                        f"Choose from: static_reference, temporal_std, temporal_cv, peak_frequency, single_organoid, cellpose")
