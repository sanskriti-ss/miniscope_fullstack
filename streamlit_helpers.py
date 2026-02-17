"""
Streamlit helper functions — wraps the main pipeline and group visualization
so they can be called programmatically from the Streamlit UI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_fluorescence_pipeline(video_path, config, progress_callback=None):
    """
    Run the full fluorescence trace-extraction pipeline.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the .avi file.
    config : dict
        Keys matching vars.py variable names.  Any key present here will
        override the corresponding module-level variable at runtime.
    progress_callback : callable, optional
        ``progress_callback(fraction, message)`` called after each major step.

    Returns
    -------
    dict  with keys:
        output_dir, plots (list of paths), traces_csv, traces_df,
        n_rois, roi_overlay_path
    """

    def _progress(frac, msg):
        if progress_callback is not None:
            progress_callback(frac, msg)

    # ------------------------------------------------------------------
    # 0.  Monkey-patch vars module with user config
    # ------------------------------------------------------------------
    import vars as vars_mod
    _original_values = {}
    for key, value in config.items():
        if hasattr(vars_mod, key):
            _original_values[key] = getattr(vars_mod, key)
            setattr(vars_mod, key, value)

    # Also make sure VIDEO_PATH is set
    vars_mod.VIDEO_PATH = video_path

    try:
        # Re-import after patching so that ``from vars import *`` picks up
        # the new values in modules that were already loaded.  We force a
        # reimport of roi_detection so its module-level globals refresh.
        import importlib
        import roi_detection as roi_mod
        importlib.reload(roi_mod)

        from roi_detection import detect_rois_dispatcher
        from plotting import (
            save_roi_overlay_image, save_trace_plot,
            save_smoothed_trace_plot, save_wave_trace_plot,
            save_spike_trace_plot, save_debug_f0_trace_plot,
            save_detrended_trace_plot,
        )
        from roi_selection import extract_frame_channel
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

        # Collect generated plot paths
        plot_paths = []

        # ------------------------------------------------------------------
        # 1.  Output directory
        # ------------------------------------------------------------------
        _progress(0.0, "Setting up output directory...")
        avi_base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("plots", avi_base)
        os.makedirs(out_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 2.  Load video metadata
        # ------------------------------------------------------------------
        _progress(0.05, "Loading video metadata...")
        start_time = config.get("START_TIME_SEC", vars_mod.START_TIME_SEC)
        end_time = config.get("END_TIME_SEC", vars_mod.END_TIME_SEC)
        skip_first = config.get("SKIP_FIRST_FRAMES", vars_mod.SKIP_FIRST_FRAMES)

        fps, n_frames, h, w, start_frame, end_frame = load_video_metadata(
            video_path,
            start_time_sec=start_time,
            end_time_sec=end_time,
            skip_first_frames=skip_first,
        )

        # ------------------------------------------------------------------
        # 3.  Detect ROIs
        # ------------------------------------------------------------------
        _progress(0.10, "Detecting ROIs...")
        channel = config.get("CHANNEL", vars_mod.CHANNEL)
        method = config.get("ROI_DETECTION_METHOD", vars_mod.ROI_DETECTION_METHOD)

        dispatcher_kwargs = {
            "n_ref_frames": min(config.get("N_REF_FRAMES", vars_mod.N_REF_FRAMES), n_frames),
        }

        # Pass threshold_percentile for temporal methods
        if "TEMPORAL_THRESHOLD_PERCENTILE" in config:
            dispatcher_kwargs["threshold_percentile"] = config["TEMPORAL_THRESHOLD_PERCENTILE"]

        # Pass cellpose diameter
        if "CELLPOSE_DIAMETER" in config:
            dispatcher_kwargs["diameter"] = config["CELLPOSE_DIAMETER"]

        # Pass area filters
        dispatcher_kwargs["min_area"] = config.get("MIN_AREA", vars_mod.MIN_AREA)
        dispatcher_kwargs["max_area"] = config.get("MAX_AREA", vars_mod.MAX_AREA)

        # Single organoid params
        if method == "single_organoid":
            dispatcher_kwargs["min_area"] = config.get("SINGLE_ORGANOID_MIN_AREA", vars_mod.SINGLE_ORGANOID_MIN_AREA)
            dispatcher_kwargs["max_area"] = config.get("SINGLE_ORGANOID_MAX_AREA", vars_mod.SINGLE_ORGANOID_MAX_AREA)
            dispatcher_kwargs["edge_only"] = config.get("SINGLE_ORGANOID_EDGE_ONLY", vars_mod.SINGLE_ORGANOID_EDGE_ONLY)
            dispatcher_kwargs["edge_width"] = config.get("SINGLE_ORGANOID_EDGE_WIDTH", vars_mod.SINGLE_ORGANOID_EDGE_WIDTH)
            dispatcher_kwargs["min_circularity"] = config.get("SINGLE_ORGANOID_CIRCULARITY", vars_mod.SINGLE_ORGANOID_CIRCULARITY)

        roi_masks, roi_info, debug_img = detect_rois_dispatcher(
            method=method,
            video_path=video_path,
            build_ref_fn=build_reference_image,
            extract_frame_fn=extract_frame_channel,
            channel=channel,
            start_frame=start_frame,
            end_frame=end_frame,
            **dispatcher_kwargs,
        )

        if len(roi_masks) == 0:
            return {
                "output_dir": out_dir,
                "plots": [],
                "traces_csv": None,
                "traces_df": None,
                "n_rois": 0,
                "roi_overlay_path": None,
                "error": "No ROIs detected. Try adjusting detection parameters.",
            }

        # ------------------------------------------------------------------
        # 4.  Dilate & overlay
        # ------------------------------------------------------------------
        _progress(0.30, f"Found {len(roi_masks)} ROIs – dilating & saving overlay...")
        roi_masks_original = [m.copy() for m in roi_masks]
        dilation = config.get("ROI_DILATION_RADIUS", vars_mod.ROI_DILATION_RADIUS)
        roi_masks = dilate_roi_masks(roi_masks, dilation)

        overlay_path = os.path.join(out_dir, "rois_on_first_frame.png")
        save_roi_overlay_image(
            video_path=video_path,
            roi_masks=roi_masks,
            roi_info=roi_info,
            out_path=overlay_path,
            roi_masks_original=roi_masks_original,
        )
        plot_paths.append(overlay_path)

        # ------------------------------------------------------------------
        # 5.  Extract traces
        # ------------------------------------------------------------------
        _progress(0.40, "Extracting fluorescence traces...")
        use_orig = config.get("USE_ORIGINAL_MASK_FOR_F0", vars_mod.USE_ORIGINAL_MASK_FOR_F0)
        f0_masks = roi_masks_original if use_orig else None
        df, fps = extract_traces(
            video_path, roi_masks, channel=channel,
            start_frame=start_frame, end_frame=end_frame,
            roi_masks_for_f0=f0_masks,
        )

        timestamps = load_timestamps_from_file(video_path)
        if timestamps is not None:
            df = apply_timestamps_to_traces(df, timestamps, start_frame=start_frame)

        # ------------------------------------------------------------------
        # 6.  Normalize (F/F0) & smooth
        # ------------------------------------------------------------------
        _progress(0.55, "Normalizing & smoothing traces...")
        f0_mode = config.get("F0_MODE", vars_mod.F0_MODE)
        f0_pct = config.get("F0_PERCENTILE", vars_mod.F0_PERCENTILE)
        f0_n = config.get("F0_FIRST_N", vars_mod.F0_FIRST_N)
        df = normalize_traces_FF0(df, f0_mode, f0_pct, f0_n)

        win_len = config.get("SMOOTH_WINDOW_LENGTH", vars_mod.SMOOTH_WINDOW_LENGTH)
        poly = config.get("SMOOTH_POLYORDER", vars_mod.SMOOTH_POLYORDER)
        add_smooth = config.get("ADDITIONAL_SMOOTHING", vars_mod.ADDITIONAL_SMOOTHING)
        df = smooth_traces(df, window_length=win_len, polyorder=poly,
                           additional_smoothing=add_smooth)

        # ------------------------------------------------------------------
        # 7.  Generate all plots
        # ------------------------------------------------------------------
        _progress(0.65, "Generating plots...")

        # Raw
        p = os.path.join(out_dir, "fluorescence_traces_plot.png")
        save_trace_plot(df, out_path=p)
        plot_paths.append(p)

        # Smoothed
        df_smoothed = smooth_traces(df, window_length=win_len, polyorder=poly,
                                    additional_smoothing=add_smooth)
        p = os.path.join(out_dir, "fluorescence_traces_plot_smoothed.png")
        save_smoothed_trace_plot(df_smoothed, out_path=p)
        plot_paths.append(p)

        # Debug conservative F0
        debug_pct = config.get("DEBUG_F0_PERCENTILE", vars_mod.DEBUG_F0_PERCENTILE)
        p = os.path.join(out_dir, "debug_fluorescence_traces_conservative_f0.png")
        save_debug_f0_trace_plot(df_smoothed, debug_percentile=debug_pct, out_path=p)
        plot_paths.append(p)

        # Detrended
        apply_detrend = config.get("APPLY_DETRENDING", vars_mod.APPLY_DETRENDING)
        detrend_poly = config.get("DETREND_POLYORDER", vars_mod.DETREND_POLYORDER)
        if apply_detrend:
            _progress(0.75, "Detrending traces...")
            df_detrended = detrend_traces(df_smoothed, polyorder=detrend_poly)
            p = os.path.join(out_dir, "fluorescence_traces_detrended.png")
            save_detrended_trace_plot(df_detrended, out_path=p)
            plot_paths.append(p)

        # Spikes
        _progress(0.80, "Detecting spikes...")
        df_peaks = detect_spikes_and_dips(df_smoothed, fps)
        p = os.path.join(out_dir, "fluorescence_spikes.png")
        save_spike_trace_plot(df_peaks, out_path=p,
                              video_name=os.path.basename(video_path))
        plot_paths.append(p)

        # Waves
        _progress(0.85, "Extracting wave component...")
        low_f = config.get("WAVE_LOW_FREQ", vars_mod.WAVE_LOW_FREQ)
        high_f = config.get("WAVE_HIGH_FREQ", vars_mod.WAVE_HIGH_FREQ)
        order = config.get("WAVE_FILTER_ORDER", vars_mod.WAVE_FILTER_ORDER)
        df_wave = extract_wave_component(df, fps, low_freq=low_f,
                                          high_freq=high_f, order=order)
        p = os.path.join(out_dir, "fluorescence_traces_plot_waves.png")
        save_wave_trace_plot(df_wave, out_path=p, fps=fps)
        plot_paths.append(p)

        # ------------------------------------------------------------------
        # 8.  Save CSV
        # ------------------------------------------------------------------
        _progress(0.95, "Saving CSV...")
        csv_path = os.path.join(out_dir, "fluorescence_traces.csv")
        df.to_csv(csv_path, index=False)

        _progress(1.0, "Done!")

        return {
            "output_dir": out_dir,
            "plots": plot_paths,
            "traces_csv": csv_path,
            "traces_df": df,
            "n_rois": len(roi_masks),
            "roi_overlay_path": overlay_path,
        }

    finally:
        # Restore original vars values
        for key, value in _original_values.items():
            setattr(vars_mod, key, value)


def run_group_visualization(plots_dir, segment_duration=4.0):
    """
    Wraps group_vis.py logic and returns result paths.

    Parameters
    ----------
    plots_dir : str
        Directory containing experiment folders with fluorescence_traces.csv files.
    segment_duration : float
        Duration (seconds) of the most-variable segment to extract per ROI.

    Returns
    -------
    dict  with keys: html_path, normalized_html_path, csv_path, summary (str)
    """
    from group_vis import (
        process_experiment_folder,
        print_summary,
        save_plot_data_csv,
        create_bokeh_plot,
        create_normalized_bokeh_plot,
    )

    drug_data = process_experiment_folder(plots_dir, segment_duration=segment_duration)

    if not drug_data:
        return {"html_path": None, "normalized_html_path": None,
                "csv_path": None, "summary": "No data found to process."}

    # Build a short text summary
    lines = []
    total = 0
    for drug_type, experiments in drug_data.items():
        n = len(experiments)
        total += n
        wave_scores = [e["wave_score"] for e in experiments]
        freqs = [e["beat_frequency"] for e in experiments]
        lines.append(
            f"  {drug_type}: {n} traces, "
            f"wave_score={np.mean(wave_scores):.4f}, "
            f"beat_freq={np.mean(freqs):.2f} Hz"
        )
    summary = f"Total traces: {total}\n" + "\n".join(lines)

    csv_path = os.path.join(plots_dir, "group_traces_data.csv")
    save_plot_data_csv(drug_data, csv_path)

    html_path = os.path.join(plots_dir, "group_fluorescence_traces.html")
    create_bokeh_plot(drug_data, html_path)

    norm_html = os.path.join(plots_dir, "group_fluorescence_traces_normalized.html")
    create_normalized_bokeh_plot(drug_data, norm_html)

    return {
        "html_path": html_path,
        "normalized_html_path": norm_html,
        "csv_path": csv_path,
        "summary": summary,
    }
