import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIDEO_PATH = "input.mp4"

# Reference projection settings
N_REF_FRAMES = 20          # how many initial frames to use for ROI detection
USE_MAX_PROJECTION = False   # if False, uses mean projection

# ROI filtering params (pixels)
MIN_AREA = 600
MAX_AREA = 50_000

# for moving blobs
ROI_DILATION_RADIUS = 100  # in pixels, adjust as needed


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

def detect_rois_from_reference(ref_img, min_area=600, max_area=50_000):
    """
    Segment bright green blobs on dark background and return one mask per blob.
    Also filters out grid lines (long skinny components) and tiny specks.
    """
    # Stronger blur to merge specks into blobs
    blurred = cv2.GaussianBlur(ref_img, (11, 11), 0)

    # Compute a threshold: mean + 0.5 * std as a simple heuristic
    mu, sigma = cv2.meanStdDev(blurred)
    mu = float(mu[0][0])
    sigma = float(sigma[0][0])
    thresh_val = mu + 0.5 * sigma

    # Clamp threshold into [0, 255]
    thresh_val = max(0, min(255, int(thresh_val)))

    _, thresh = cv2.threshold(
        blurred,
        thresh_val,
        255,
        cv2.THRESH_BINARY,
    )

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
    
    # and save the image
    save_trace_plot(df, out_path="fluorescence_traces_plot.png")


    # 6. Plot F/F0 vs time
    plt.figure()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]
    for col in roi_cols:
        plt.plot(df["time_s"], df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Save traces
    df.to_csv("fluorescence_traces.csv", index=False)
    print("Saved traces to fluorescence_traces.csv")


#### calling main function

if __name__ == "__main__":
    print("Inside main()")
    main()
