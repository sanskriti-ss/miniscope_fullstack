"""
Helper Functions for Fluorescence Trace Extraction Pipeline
"""

from .timestamps import load_timestamps_from_file, apply_timestamps_to_traces
from .video_utils import load_video_metadata, build_reference_image
from .signal_processing import (
    compute_f0,
    smooth_traces,
    detrend_traces,
    extract_wave_component,
    detect_spikes_and_dips,
)
from .trace_extraction import (
    dilate_roi_masks,
    extract_traces,
    normalize_traces_FF0,
    normalize_traces_FF0_custom,
)

__all__ = [
    # Timestamps
    'load_timestamps_from_file',
    'apply_timestamps_to_traces',
    # Video utilities
    'load_video_metadata',
    'build_reference_image',
    # Signal processing
    'compute_f0',
    'smooth_traces',
    'detrend_traces',
    'extract_wave_component',
    'detect_spikes_and_dips',
    # Trace extraction
    'dilate_roi_masks',
    'extract_traces',
    'normalize_traces_FF0',
    'normalize_traces_FF0_custom',
]
