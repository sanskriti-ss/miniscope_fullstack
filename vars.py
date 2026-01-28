"""
Configuration file for trace_extraction.py
Modify these variables to adjust ROI detection and analysis parameters.
"""

# ==============================================================================
# VIDEO SETTINGS
# ==============================================================================

VIDEO_PATH = "01_16_15-32_vid2.avi"

# Video clipping (to ignore portions of video)
START_TIME_SEC = 3      # Start processing at this time (seconds)
END_TIME_SEC = 13        # End processing at this time (seconds, 0 = end of video)
SKIP_FIRST_FRAMES = 0   # Skip this many frames at the start (alternative to START_TIME_SEC)

# Choose which channel to use (0 = Blue, 1 = Green, 2 = Red)
# NOTE: If your video is grayscale (black/white), channel selection has no effect
#       All channels will be identical for grayscale videos
CHANNEL = 0  # Use 0 for grayscale or any channel for color videos


# ==============================================================================
# ROI DETECTION METHOD
# ==============================================================================

# Select ROI detection method
# Options: "static_reference", "temporal_std", "temporal_cv", "peak_frequency"
# 
# - "static_reference": Brightness-based (original method, not recommended for low contrast)
# - "temporal_std": Standard deviation over time (good for flashing organoids)
# - "temporal_cv": Coefficient of variation (best for dim but active organoids)
# - "peak_frequency": Peak counting (good for periodic flashing)
ROI_DETECTION_METHOD = "peak_frequency"


# ==============================================================================
# STATIC REFERENCE METHOD SETTINGS (for "static_reference" method)
# ==============================================================================

N_REF_FRAMES = 20           # Number of initial frames to use for ROI detection
USE_MAX_PROJECTION = False  # If False, uses mean projection; if True, uses max projection


# ==============================================================================
# TEMPORAL METHOD SETTINGS (for temporal_std, temporal_cv, peak_frequency)
# ==============================================================================

TEMPORAL_SAMPLE_FRAMES = 500        # Number of frames to analyze (0 = all frames)
TEMPORAL_DOWNSAMPLE = 1             # Analyze every Nth frame for speed (1 = all frames, 2 = every other frame)
TEMPORAL_SMOOTH_SIGMA = 3.0         # Gaussian smoothing sigma for variation map (higher = smoother)
TEMPORAL_THRESHOLD_PERCENTILE = 50  # Threshold percentile (lower = more sensitive, detects more ROIs)


# ==============================================================================
# ROI FILTERING PARAMETERS
# ==============================================================================

MIN_AREA = 30    # Minimum ROI area in pixels (smaller organoids)
MAX_AREA = 1000  # Maximum ROI area in pixels (larger organoids)

# ROI dilation to handle movement
ROI_DILATION_RADIUS = 40  # Expand ROIs by this many pixels to handle motion (0 = no dilation)


# ==============================================================================
# F0 BASELINE SETTINGS
# ==============================================================================

# F0 computation mode
# "percentile": Use percentile of trace as baseline (robust to activity)
# "mean_first_n": Use mean of first N frames as baseline
F0_MODE = "percentile"

F0_PERCENTILE = 20  # Percentile for baseline (used if F0_MODE = "percentile")
F0_FIRST_N = 50     # Number of frames for baseline (used if F0_MODE = "mean_first_n")


# ==============================================================================
# TRACE SMOOTHING SETTINGS
# ==============================================================================

SMOOTH_WINDOW_LENGTH = 21    # Window length for Savitzky-Golay filter (must be odd)
SMOOTH_POLYORDER = 3         # Polynomial order for smoothing
ADDITIONAL_SMOOTHING = True  # Apply additional moving average smoothing for noisy data


# ==============================================================================
# WAVE EXTRACTION SETTINGS
# ==============================================================================

WAVE_LOW_FREQ = 0.1   # Low cutoff frequency (Hz) for bandpass filter
WAVE_HIGH_FREQ = 2.0  # High cutoff frequency (Hz) for bandpass filter
WAVE_FILTER_ORDER = 3 # Butterworth filter order
