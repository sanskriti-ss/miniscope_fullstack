import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import re

VIDEO_PATH = "commpacer_jan16/3.avi"

# Reference projection settings
N_REF_FRAMES = 20          # how many initial frames to use for ROI detection
USE_MAX_PROJECTION = False   # if False, uses mean projection

# ROI filtering params (pixels)
MIN_AREA = 30   # Reduced for smaller/dimmer organoids
MAX_AREA = 1000 # Increased to catch larger blobs

# for moving blobs
ROI_DILATION_RADIUS = 40  # in pixels, adjust as needed


# F0 settings
F0_MODE = "percentile"      # "percentile" or "mean_first_n"
F0_PERCENTILE = 20
F0_FIRST_N = 50             # used only if F0_MODE == "mean_first_n"

# Choose which channel to use (0 B, 1 G, 2 R); often 1 for fluorescence
CHANNEL = 1


def load_video_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return fps, n_frames, height, width


def build_reference_image(path, n_ref_frames, use_max_projection=True, channel=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reference image")

    frames = []
    count = 0
    while count < n_ref_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = frame[:, :, channel].astype(np.float32)
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


def compute_f0(F_series, mode="percentile", percentile=20, first_n=50):
    F = np.asarray(F_series, dtype=float)
    F = F[np.isfinite(F)]
    if F.size == 0:
        return np.nan
    if mode == "mean_first_n":
        n = min(first_n, F.size)
        return float(np.mean(F[:n]))
    return float(np.percentile(F, percentile))

# for moving blobs
def dilate_roi_masks(roi_masks, radius: int):
    if radius <= 0:
        return roi_masks
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = []
    for m in roi_masks:
        dilated.append(cv2.dilate(m.astype(np.uint8), kernel))
    return dilated


def extract_traces(path, roi_masks, channel=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for trace extraction")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    n_rois = len(roi_masks)
    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame[:, :, channel].astype(np.float32)

        t = frame_idx / fps
        F_vals = []

        for m in roi_masks:
            vals = img[m.astype(bool)]
            if vals.size == 0:
                F_vals.append(np.nan)
            else:
                F_vals.append(float(np.mean(vals)))

        rows.append([frame_idx, t] + F_vals)
        frame_idx += 1

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


### these will help us see the ROIS
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


# for saving the peaks image
def save_trace_plot(df, out_path="fluorescence_traces_plot.png"):
    """
    Plots all FF0 traces in the dataframe and saves as a PNG.
    """
    import matplotlib.pyplot as plt

    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]

    if len(roi_cols) == 0:
        raise ValueError("No FF0 columns found in dataframe")

    plt.figure(figsize=(10, 5))

    for col in roi_cols:
        plt.plot(df["time_s"], df[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel("F/F0")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
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
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()  # prevents display in some environments

    print(f"Saved smoothed trace plot to {out_path}")


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
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved wave trace plot to {out_path}")


################################################
### main function
################################################ 

def main():
    fps, n_frames, h, w = load_video_metadata(VIDEO_PATH)
    print(f"Video: {VIDEO_PATH}, fps={fps}, frames={n_frames}, size={w}x{h}")

    # 1. Build reference image from first N frames
    ref_img = build_reference_image(
        VIDEO_PATH,
        n_ref_frames=min(N_REF_FRAMES, n_frames),
        use_max_projection=USE_MAX_PROJECTION,
        channel=CHANNEL,
    )

    # 2. Detect ROIs automatically
    roi_masks, roi_info, thresh = detect_rois_from_reference(
        ref_img,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
    )
    print(f"Detected {len(roi_masks)} ROIs")
    
    # Dilate ROIs to handle small motion
    roi_masks = dilate_roi_masks(roi_masks, ROI_DILATION_RADIUS)

    # 3. Save ROI outlines on first frame
    save_roi_overlay_image(
        video_path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        out_path="rois_on_first_frame.png",
    )

    # 4. Extract F(t) for each ROI
    df, fps = extract_traces(VIDEO_PATH, roi_masks, channel=CHANNEL)

    # 5. Compute F/F0
    df = normalize_traces_FF0(df, F0_MODE, F0_PERCENTILE, F0_FIRST_N)
    
    # 5b. Smooth the traces with enhanced parameters for noisy data
    df = smooth_traces(df, window_length=21, polyorder=3, additional_smoothing=True)
    
    # and save both original and smoothed images
    save_trace_plot(df, out_path="fluorescence_traces_plot.png")
    save_smoothed_trace_plot(df, out_path="fluorescence_traces_plot_smoothed.png")

    # 5c. Extract and plot wave components
    df_wave = extract_wave_component(df, fps, low_freq=0.1, high_freq=2.0, order=3)
    save_wave_trace_plot(df_wave, out_path="fluorescence_traces_plot_waves.png", fps=fps)


    # 6. Plot F/F0 vs time
    plt.figure()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]
    for col in roi_cols:
        plt.plot(df["time_s"], df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0")
    plt.legend()
    plt.tight_layout()
    # plt.show() removed to prevent blocking; plot is saved to file instead

    # 7. Save traces
    df.to_csv("fluorescence_traces.csv", index=False)
    print("Saved traces to fluorescence_traces.csv")


#### calling main function

if __name__ == "__main__":
    print("Inside main()")
    main()
