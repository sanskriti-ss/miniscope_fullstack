"""
Streamlit GUI for Miniscope Fluorescence Pipeline
Run with:  streamlit run app.py
"""

import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before any pyplot import

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so local imports work
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Check for optional cellpose
# ---------------------------------------------------------------------------
try:
    import cellpose  # noqa: F401
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Miniscope Fluorescence Pipeline",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Miniscope Fluorescence Pipeline")

# ---------------------------------------------------------------------------
# Sidebar — mode selector
# ---------------------------------------------------------------------------
mode = st.sidebar.radio("Mode", ["Fluorescence Analysis", "Group Visualization"])


# ===================================================================
# MODE 1: Fluorescence Analysis
# ===================================================================
if mode == "Fluorescence Analysis":

    # ---- Video Settings (always visible) ----
    st.sidebar.header("Video Settings")

    input_dir = os.path.join(REPO_ROOT, "input_data")
    avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi"))) if os.path.isdir(input_dir) else []
    avi_labels = [os.path.basename(f) for f in avi_files]

    video_choice = st.sidebar.selectbox(
        "Select video from input_data/",
        options=["(custom path)"] + avi_labels,
    )
    if video_choice == "(custom path)":
        video_path = st.sidebar.text_input("Video path", value="")
    else:
        video_path = os.path.join(input_dir, video_choice)

    col_start, col_end = st.sidebar.columns(2)
    start_time = col_start.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
    end_time = col_end.number_input("End time (s, 0=end)", min_value=0.0, value=0.0, step=1.0)

    # Videos are grayscale — channel selection not needed, default to 0
    channel = 0

    # ---- ROI Detection ----
    with st.sidebar.expander("ROI Detection", expanded=True):
        all_methods = ["cellpose", "temporal_std", "temporal_cv",
                       "peak_frequency", "single_organoid", "static_reference"]

        if not CELLPOSE_AVAILABLE:
            all_methods.remove("cellpose")
            st.warning("Cellpose not installed (`pip install cellpose`). Deep-learning method unavailable.")

        roi_method = st.selectbox("Detection method", all_methods)

        # Method-specific params
        cellpose_diameter = 200
        temporal_thresh_pct = 50
        single_min_area = 5000
        single_max_area = 500000
        single_edge_only = True
        single_edge_width = 0.1
        single_circularity = 0.4

        if roi_method == "cellpose":
            cellpose_diameter = st.number_input("Cellpose diameter (px)", 50, 600, 200, step=25)
        elif roi_method in ("temporal_std", "temporal_cv", "peak_frequency"):
            temporal_thresh_pct = st.slider("Threshold percentile", 10, 95, 50)
        elif roi_method == "single_organoid":
            single_min_area = st.number_input("Min area (px)", 100, 1000000, 5000, step=500)
            single_max_area = st.number_input("Max area (px)", 1000, 2000000, 500000, step=5000)
            single_edge_only = st.checkbox("Edge-only mask", value=True)
            single_edge_width = st.slider("Edge width (fraction)", 0.05, 0.5, 0.1, step=0.05)
            single_circularity = st.slider("Min circularity", 0.1, 1.0, 0.4, step=0.05)

        min_area = st.number_input("Min ROI area", 1, 10000, 30, step=10)
        max_area = st.number_input("Max ROI area", 10, 100000, 1000, step=100)
        dilation_radius = st.number_input("Dilation radius (px)", 0, 200, 40, step=5)

    # ---- Signal Processing ----
    with st.sidebar.expander("Signal Processing"):
        f0_mode = st.selectbox("F0 mode", ["percentile", "mean_first_n"])
        f0_percentile = st.slider("F0 percentile", 1, 50, 10)
        f0_first_n = st.number_input("F0 first N frames", 5, 200, 20, step=5)

        smooth_window = st.number_input("Smooth window length (odd)", 3, 51, 5, step=2)
        smooth_poly = st.number_input("Smooth poly order", 1, 5, 3)
        additional_smoothing = st.checkbox("Additional moving-average smoothing", value=False)

        apply_detrending = st.checkbox("Apply detrending", value=True)
        detrend_poly = st.number_input("Detrend poly order", 1, 5, 2)

    # ---- Wave Extraction ----
    with st.sidebar.expander("Wave Extraction"):
        wave_low = st.number_input("Low freq (Hz)", 0.01, 10.0, 0.1, step=0.05, format="%.2f")
        wave_high = st.number_input("High freq (Hz)", 0.1, 20.0, 2.0, step=0.1, format="%.1f")
        wave_order = st.number_input("Filter order", 1, 10, 3)

    # ---- RUN button ----
    run_clicked = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

    # ---- Main area ----
    if run_clicked:
        if not video_path or not os.path.isfile(video_path):
            st.error(f"Video file not found: `{video_path}`")
        else:
            # Build config dict
            config = {
                "VIDEO_PATH": video_path,
                "START_TIME_SEC": start_time,
                "END_TIME_SEC": end_time,
                "CHANNEL": channel,
                "ROI_DETECTION_METHOD": roi_method,
                "CELLPOSE_DIAMETER": cellpose_diameter,
                "TEMPORAL_THRESHOLD_PERCENTILE": temporal_thresh_pct,
                "MIN_AREA": min_area,
                "MAX_AREA": max_area,
                "ROI_DILATION_RADIUS": dilation_radius,
                "SINGLE_ORGANOID_MIN_AREA": single_min_area,
                "SINGLE_ORGANOID_MAX_AREA": single_max_area,
                "SINGLE_ORGANOID_EDGE_ONLY": single_edge_only,
                "SINGLE_ORGANOID_EDGE_WIDTH": single_edge_width,
                "SINGLE_ORGANOID_CIRCULARITY": single_circularity,
                "F0_MODE": f0_mode,
                "F0_PERCENTILE": f0_percentile,
                "F0_FIRST_N": f0_first_n,
                "SMOOTH_WINDOW_LENGTH": smooth_window,
                "SMOOTH_POLYORDER": smooth_poly,
                "ADDITIONAL_SMOOTHING": additional_smoothing,
                "APPLY_DETRENDING": apply_detrending,
                "DETREND_POLYORDER": detrend_poly,
                "WAVE_LOW_FREQ": wave_low,
                "WAVE_HIGH_FREQ": wave_high,
                "WAVE_FILTER_ORDER": wave_order,
            }

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress(frac, msg):
                progress_bar.progress(min(frac, 1.0))
                status_text.text(msg)

            from streamlit_helpers import run_fluorescence_pipeline

            with st.spinner("Running pipeline..."):
                result = run_fluorescence_pipeline(video_path, config, progress_callback=_progress)

            progress_bar.empty()
            status_text.empty()

            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Pipeline complete — {result['n_rois']} ROIs detected")

                # ROI overlay
                if result["roi_overlay_path"] and os.path.isfile(result["roi_overlay_path"]):
                    st.subheader("ROI Overlay")
                    st.image(result["roi_overlay_path"], use_container_width=True)

                # Tabbed plot viewer
                plot_files = [p for p in result["plots"] if os.path.isfile(p) and p != result["roi_overlay_path"]]
                if plot_files:
                    st.subheader("Plots")
                    tab_names = [os.path.splitext(os.path.basename(p))[0] for p in plot_files]
                    tabs = st.tabs(tab_names)
                    for tab, path in zip(tabs, plot_files):
                        with tab:
                            st.image(path, use_container_width=True)

                # DataFrame preview & download
                if result["traces_df"] is not None:
                    st.subheader("Traces Preview")
                    st.dataframe(result["traces_df"].head(200), use_container_width=True)

                if result["traces_csv"] and os.path.isfile(result["traces_csv"]):
                    with open(result["traces_csv"], "rb") as f:
                        st.download_button(
                            "Download traces CSV",
                            data=f,
                            file_name="fluorescence_traces.csv",
                            mime="text/csv",
                        )

    else:
        st.info("Configure parameters in the sidebar, then click **Run Pipeline**.")


# ===================================================================
# MODE 2: Group Visualization
# ===================================================================
elif mode == "Group Visualization":
    st.sidebar.header("Group Visualization Settings")

    default_plots_dir = os.path.join(REPO_ROOT, "plots")
    plots_dir = st.sidebar.text_input("Plots directory", value=default_plots_dir)
    segment_duration = st.sidebar.slider("Segment duration (s)", 1.0, 20.0, 4.0, step=0.5)

    run_group = st.sidebar.button("Run Group Visualization", type="primary", use_container_width=True)

    if run_group:
        if not os.path.isdir(plots_dir):
            st.error(f"Directory not found: `{plots_dir}`")
        else:
            from streamlit_helpers import run_group_visualization

            with st.spinner("Processing group visualization..."):
                result = run_group_visualization(plots_dir, segment_duration)

            if result["html_path"] is None:
                st.warning(result["summary"])
            else:
                st.success("Group visualization complete!")

                st.subheader("Summary")
                st.text(result["summary"])

                # Embed Bokeh HTML
                for label, key in [("Grouped Traces", "html_path"),
                                   ("Normalized Traces", "normalized_html_path")]:
                    html_file = result.get(key)
                    if html_file and os.path.isfile(html_file):
                        st.subheader(label)
                        with open(html_file, "r") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=700, scrolling=True)

                # CSV download
                csv_file = result.get("csv_path")
                if csv_file and os.path.isfile(csv_file):
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            "Download group traces CSV",
                            data=f,
                            file_name="group_traces_data.csv",
                            mime="text/csv",
                        )
    else:
        st.info("Set the plots directory and segment duration, then click **Run Group Visualization**.")
