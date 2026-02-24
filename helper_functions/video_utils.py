"""
Video Utility Functions
Load video metadata and build reference images.
"""

import cv2
import numpy as np

from roi_selection import extract_frame_channel
from helper_functions.timestamps import load_timestamps_from_file


def _seconds_to_frame(time_sec, timestamps, fps):
    """
    Convert a time in seconds to a frame index using timestamps if available,
    otherwise fall back to fps-based calculation.

    Parameters
    ----------
    time_sec : float
        Time in seconds to convert
    timestamps : np.ndarray or None
        Array of timestamps (in seconds) indexed by frame number
    fps : float
        Frames per second (fallback if no timestamps)

    Returns
    -------
    int
        Frame index corresponding to the given time
    """
    if timestamps is not None and len(timestamps) > 1:
        # Find the frame whose timestamp is closest to the requested time
        idx = int(np.searchsorted(timestamps, time_sec, side='right'))
        return idx
    else:
        return int(time_sec * fps)


def load_video_metadata(path, start_time_sec=0, end_time_sec=0, skip_first_frames=0):
    """
    Load video metadata and calculate frame range after clipping.
    Uses timestamps file for accurate seconds-to-frame conversion when available.

    Parameters
    ----------
    path : str
        Path to video file
    start_time_sec : float
        Start processing at this time (seconds)
    end_time_sec : float
        End processing at this time (seconds, 0 = end of video)
    skip_first_frames : int
        Skip this many frames at the start (alternative to start_time_sec)

    Returns
    -------
    fps, n_frames, height, width, start_frame, end_frame
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Load timestamps for accurate time-to-frame conversion
    timestamps = load_timestamps_from_file(path)

    # Calculate frame range using timestamps when available
    start_frame_from_time = _seconds_to_frame(start_time_sec, timestamps, fps) if start_time_sec > 0 else 0
    start_frame = max(skip_first_frames, start_frame_from_time)

    if end_time_sec > 0:
        end_frame = _seconds_to_frame(end_time_sec, timestamps, fps)
    else:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    n_frames = end_frame - start_frame

    if n_frames <= 0:
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")

    # Compute actual times for the print message
    if timestamps is not None and len(timestamps) > 1:
        start_time_actual = timestamps[min(start_frame, len(timestamps)-1)]
        end_time_actual = timestamps[min(end_frame-1, len(timestamps)-1)]
        timing_source = "timestamps"
    else:
        start_time_actual = start_frame / fps
        end_time_actual = end_frame / fps
        timing_source = f"video fps={fps:.1f}"

    if start_frame > 0 or end_frame < total_frames:
        print(f"[Video Clipping] Processing frames {start_frame} to {end_frame} "
              f"(time: {start_time_actual:.1f}s to {end_time_actual:.1f}s, using {timing_source})")

    return fps, n_frames, height, width, start_frame, end_frame


def build_reference_image(path, n_ref_frames, use_max_projection=True, channel=1, start_frame=0):
    """
    Build a reference image from the first N frames of a video.
    
    Parameters
    ----------
    path : str
        Path to video file
    n_ref_frames : int
        Number of frames to use for building the reference
    use_max_projection : bool
        If True, use max projection; if False, use mean projection
    channel : int
        Color channel to extract (0=Blue, 1=Green, 2=Red)
    start_frame : int
        Frame to start from (for clipped videos)
    
    Returns
    -------
    ref : np.ndarray
        Reference image (uint8, normalized to 0-255)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reference image")
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    count = 0
    while count < n_ref_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_channel(frame, channel)
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
