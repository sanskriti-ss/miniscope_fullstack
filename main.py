"""
Main Pipeline for Fluorescence Trace Extraction
Detects ROIs and extracts fluorescence traces from microscopy videos.
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import configuration variables
from vars import *
# Import ROI detection strategies
from roi_detection import detect_rois_dispatcher
# Import plotting functions
from plotting import (save_roi_overlay_image, save_trace_plot, 
                     save_smoothed_trace_plot, save_wave_trace_plot, 
                     save_spike_trace_plot, save_debug_f0_trace_plot,
                     save_detrended_trace_plot)
# Import shared ROI selection functions
from roi_selection import preview_video_and_draw_rois, extract_frame_channel
# Import helper functions
from helper_functions import (
    load_timestamps_from_file,
    apply_timestamps_to_traces,
    load_video_metadata,
    build_reference_image,
    smooth_traces,
    detrend_traces,
    extract_wave_component,
    detect_spikes_and_dips,
    dilate_roi_masks,
    extract_traces,
    normalize_traces_FF0,
)


def main():
    parser = argparse.ArgumentParser(description="Run the fluorescence trace extraction pipeline.")
    parser.add_argument("--manual", action="store_true", help="Enable manual ROI selection mode.")
    args = parser.parse_args()

    # --- Output directory setup ---
    avi_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    out_dir = os.path.join("plots", avi_base)
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

    # 2. Save original masks before dilation for visualization
    roi_masks_original = [m.copy() for m in roi_masks]
    
    # 3. Dilate ROIs to handle small motion
    roi_masks = dilate_roi_masks(roi_masks, ROI_DILATION_RADIUS)

    # 4. Save ROI outlines on first frame (showing both original and dilated)
    save_roi_overlay_image(
        video_path=VIDEO_PATH,
        roi_masks=roi_masks,
        roi_info=roi_info,
        out_path=os.path.join(out_dir, "rois_on_first_frame.png"),
        roi_masks_original=roi_masks_original,
    )

    # 5. Extract F(t) for each ROI
    # Pass original masks for accurate F0 calculation (not dilated)
    f0_masks = roi_masks_original if USE_ORIGINAL_MASK_FOR_F0 else None
    df, fps = extract_traces(VIDEO_PATH, roi_masks, channel=CHANNEL, 
                            start_frame=start_frame, end_frame=end_frame,
                            roi_masks_for_f0=f0_masks)

    # 5b. Apply accurate timestamps if available
    timestamps = load_timestamps_from_file(VIDEO_PATH)
    if timestamps is not None:
        df = apply_timestamps_to_traces(df, timestamps, start_frame=start_frame)

    # 6. Compute F/F0
    df = normalize_traces_FF0(df, F0_MODE, F0_PERCENTILE, F0_FIRST_N)
    
    # 6b. Smooth the traces with enhanced parameters for noisy data
    df = smooth_traces(df, window_length=SMOOTH_WINDOW_LENGTH, 
                      polyorder=SMOOTH_POLYORDER, 
                      additional_smoothing=ADDITIONAL_SMOOTHING)
    
    # and save both original and smoothed images

    # --- DEBUG MODES ---
    # 1. Raw (unsmoothed) traces
    save_trace_plot(df, out_path=os.path.join(out_dir, "debug_fluorescence_traces_raw.png"))

    # 2. Normal smoothed traces
    df_smoothed = smooth_traces(df, window_length=SMOOTH_WINDOW_LENGTH, 
                               polyorder=SMOOTH_POLYORDER, 
                               additional_smoothing=ADDITIONAL_SMOOTHING)
    save_smoothed_trace_plot(df_smoothed, out_path=os.path.join(out_dir, "debug_fluorescence_traces_smoothed.png"))

    # 2b. Debug F0 traces (very conservative baseline = 10th percentile)
    save_debug_f0_trace_plot(df_smoothed, debug_percentile=DEBUG_F0_PERCENTILE, 
                            out_path=os.path.join(out_dir, "debug_fluorescence_traces_conservative_f0.png"))

    # 2c. Detrend traces to remove baseline drift (photobleaching, focus drift, etc.)
    if APPLY_DETRENDING:
        df_detrended = detrend_traces(df_smoothed, polyorder=DETREND_POLYORDER)
        save_detrended_trace_plot(df_detrended, out_path=os.path.join(out_dir, "debug_fluorescence_traces_detrended.png"))
    else:
        df_detrended = df_smoothed.copy()

    # 3. Peak-emphasized traces (no wave extraction, baseline smoothing, peaks highlighted)
    df_peaks = detect_spikes_and_dips(df_smoothed, fps)
    save_spike_trace_plot(df_peaks, out_path=os.path.join(out_dir, "debug_fluorescence_peaks_only.png"),
                          video_name=os.path.basename(VIDEO_PATH))

    # --- NORMAL PIPELINE ---
    save_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_traces_plot.png"))
    save_smoothed_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_traces_plot_smoothed.png"))

    # 6c. Extract and plot wave components
    df_wave = extract_wave_component(df, fps, low_freq=WAVE_LOW_FREQ, 
                                    high_freq=WAVE_HIGH_FREQ, order=WAVE_FILTER_ORDER)
    save_wave_trace_plot(df_wave, out_path=os.path.join(out_dir, "fluorescence_traces_plot_waves.png"), fps=fps)
    
    # 6d. Detect and plot spikes and dips
    df = detect_spikes_and_dips(df, fps)
    save_spike_trace_plot(df, out_path=os.path.join(out_dir, "fluorescence_spikes.png"), 
                          video_name=os.path.basename(VIDEO_PATH))


    # 7. Plot F/F0 vs time
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

    # 8. Save traces
    df.to_csv(os.path.join(out_dir, "fluorescence_traces.csv"), index=False)
    print(f"Saved traces to {os.path.join(out_dir, 'fluorescence_traces.csv')}")


#### calling main function

if __name__ == "__main__":
    print("Inside main()")
    main()
