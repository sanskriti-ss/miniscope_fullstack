import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIDEO_PATH = "input.mp4"

# Reference projection settings
N_REF_FRAMES = 200          # how many initial frames to use for ROI detection
USE_MAX_PROJECTION = True   # if False, uses mean projection

# ROI filtering params (pixels)
MIN_AREA = 20
MAX_AREA = 10_000

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


def detect_rois_from_reference(ref_img, min_area=20, max_area=10_000):
    # Optional smoothing to clean noise
    blurred = cv2.GaussianBlur(ref_img, (3, 3), 0)

    # Otsu thresholding to separate bright ROIs from background
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove small specks, close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components with stats (labels, areas, centroids) :contentReference[oaicite:1]{index=1}
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )

    masks = []
    roi_info = []  # keep area and centroid

    # Label 0 is background
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        # Create binary mask for this component
        mask = (labels == label).astype(np.uint8)
        masks.append(mask)
        cx, cy = centroids[label]
        roi_info.append(
            {"label": label, "area": int(area), "centroid": (float(cx), float(cy))}
        )

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

    # Optional quick sanity check image
    # color_ref = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    # for info in roi_info:
    #     cx, cy = info["centroid"]
    #     cv2.circle(color_ref, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    # cv2.imshow("ROIs on reference", color_ref)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3. Extract F(t) for each ROI
    df, fps = extract_traces(VIDEO_PATH, roi_masks, channel=CHANNEL)

    # 4. Compute F/F0
    df = normalize_traces_FF0(df, F0_MODE, F0_PERCENTILE, F0_FIRST_N)

    # 5. Plot F/F0 versus time
    plt.figure()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi")]
    for col in roi_cols:
        plt.plot(df["time_s"], df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("F/F0")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Save traces
    df.to_csv("fluorescence_traces.csv", index=False)
    print("Saved traces to fluorescence_traces.csv")


if __name__ == "__main__":
    print("Inside main()")
    main()
