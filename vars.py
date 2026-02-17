"""
Configuration file for trace_extraction.py
Modify these variables to adjust ROI detection and analysis parameters.
"""

# ==============================================================================
# VIDEO SETTINGS
# ==============================================================================

### "15_50_30_yuqi_pacing_Organoid3_Working_vid1.avi" ## this is good

VIDEO_PATH = "input_data/17_17_59_Cont_Organoid1_Fluoresent_Pacing_0.5Hz_20fps.avi"  # Path to input video file

# Video clipping (to ignore portions of video)
START_TIME_SEC = 0      # Start processing at this time (seconds)
END_TIME_SEC = 0        # End processing at this time (seconds, 0 = end of video)
SKIP_FIRST_FRAMES = 0   # Skip this many frames at the start (alternative to START_TIME_SEC)

# Choose which channel to use (0 = Blue, 1 = Green, 2 = Red)
# NOTE: If your video is grayscale (black/white), channel selection has no effect
#       All channels will be identical for grayscale videos
CHANNEL = 0  # Use 0 for grayscale or any channel for color videos

# Manual ROI selection (for mechanical analysis)
MANUAL_ROI_SELECTION = True  # If True, allows user to draw ROI manually; if False, uses automatic detection


# ==============================================================================
# ROI DETECTION METHOD
# ==============================================================================

# Select ROI detection method
# Options: "static_reference", "temporal_std", "temporal_cv", "peak_frequency", "single_organoid", "cellpose"
# 
# - "static_reference": Brightness-based (original method, not recommended for low contrast)
# - "temporal_std": Standard deviation over time (good for flashing organoids)
# - "temporal_cv": Coefficient of variation (best for dim but active organoids)
# - "peak_frequency": Peak counting (good for periodic flashing)
# - "single_organoid": Detects ONE large circular organoid with edge-focused activity
# - "cellpose": Deep learning based detection (RECOMMENDED - best accuracy)
ROI_DETECTION_METHOD = "cellpose"


# ==============================================================================
# CELLPOSE DETECTION SETTINGS (for "cellpose" method - RECOMMENDED)
# ==============================================================================

# Cellpose uses a pre-trained deep learning model for cell/organoid segmentation
# It automatically finds organoid boundaries with high accuracy
CELLPOSE_DIAMETER = 200  # Expected organoid diameter in pixels (100-300 typical)


# ==============================================================================
# SINGLE ORGANOID DETECTION SETTINGS (for "single_organoid" method)
# ==============================================================================

# Detect a single large circular-ish organoid (for whole-organoid analysis)
SINGLE_ORGANOID_MIN_AREA = 5000      # Minimum area in pixels for the organoid
SINGLE_ORGANOID_MAX_AREA = 500000    # Maximum area in pixels
SINGLE_ORGANOID_EDGE_ONLY = True     # If True, create annular mask focusing on edges
SINGLE_ORGANOID_EDGE_WIDTH = 0.1     # Edge width as fraction of radius (0.3 = outer 30%)
SINGLE_ORGANOID_CIRCULARITY = 0.4    # Minimum circularity (0-1, circle=1)


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

# F0 Calculation Strategy
# If True: Calculate F0 from original (non-dilated) ROI mask for accurate baseline
# If False: Calculate F0 from dilated ROI mask (may include background pixels)
# Recommended: True (especially for stationary organoids)
USE_ORIGINAL_MASK_FOR_F0 = True


# ==============================================================================
# F0 BASELINE SETTINGS
# ==============================================================================

# F0 computation mode
# "percentile": Use percentile of trace as baseline (robust to activity)
# "mean_first_n": Use mean of first N frames as baseline
F0_MODE = "percentile"

F0_PERCENTILE = 10  # Percentile for baseline (used if F0_MODE = "percentile")
F0_FIRST_N = 20     # Number of frames for baseline (used if F0_MODE = "mean_first_n")

# Debug mode: Generate additional plot with very conservative F0 (10th percentile)
# This shows maximum contrast for visualization purposes
DEBUG_F0_PERCENTILE = 2  # Very conservative baseline for debug visualization


# ==============================================================================
# TRACE SMOOTHING SETTINGS
# ==============================================================================

SMOOTH_WINDOW_LENGTH = 5    # Window length for Savitzky-Golay filter (must be odd)
SMOOTH_POLYORDER = 3         # Polynomial order for smoothing
ADDITIONAL_SMOOTHING = False  # Apply additional moving average smoothing for noisy data

# Detrending: Remove slow baseline drift while preserving fast flashing
APPLY_DETRENDING = True      # Remove baseline drift (photobleaching, focus drift, etc)
DETREND_POLYORDER = 2        # Polynomial order for detrending (1=linear, 2=quadratic, 3=cubic)


# ==============================================================================
# WAVE EXTRACTION SETTINGS
# ==============================================================================

WAVE_LOW_FREQ = 0.1   # Low cutoff frequency (Hz) for bandpass filter
WAVE_HIGH_FREQ = 2.0  # High cutoff frequency (Hz) for bandpass filter
WAVE_FILTER_ORDER = 3 # Butterworth filter order
