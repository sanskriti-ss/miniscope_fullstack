"""
Timestamp Handling Functions
Load and apply accurate timestamps from external CSV files.
"""

import os
import numpy as np
import pandas as pd


def load_timestamps_from_file(video_path):
    """
    Load timestamps from a corresponding timestamp CSV file if it exists.
    
    Looks for a file named `{video_basename}_timeStamps.csv` in the same
    directory as the video file.
    
    Parameters
    ----------
    video_path : str
        Path to the video file (e.g., "input_data/16_35_20_Fluo_Organoid2_Pacing_0.5Hz.avi")
    
    Returns
    -------
    timestamps : np.ndarray or None
        Array of timestamps (in seconds) indexed by frame number, or None if file not found
    """
    # Construct expected timestamp file path
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_file = os.path.join(video_dir, f"{video_basename}_timeStamps.csv")
    
    if not os.path.exists(timestamp_file):
        return None
    
    print(f"[Timestamps] Found timestamp file: {timestamp_file}")
    
    try:
        # Read the timestamp CSV
        ts_df = pd.read_csv(timestamp_file)
        
        # Try to find the time column (common names)
        time_col = None
        for col_name in ['time', 'Time', 'time_s', 'Time_s', 'timestamp', 'Timestamp', 
                         'time_sec', 'Time_Sec', 'timeStamp', 'TimeStamp']:
            if col_name in ts_df.columns:
                time_col = col_name
                break
        
        # If no named column found, assume first column after frame index or just first column
        if time_col is None:
            if len(ts_df.columns) >= 2:
                # Assume format: frame, time
                time_col = ts_df.columns[1]
            elif len(ts_df.columns) == 1:
                # Single column = just timestamps
                time_col = ts_df.columns[0]
            else:
                print(f"[Timestamps] Warning: Could not identify time column in {timestamp_file}")
                return None
        
        timestamps = ts_df[time_col].values
        
        # Convert to seconds if values seem to be in milliseconds (values > 1000 for first few frames)
        if len(timestamps) > 10 and np.mean(timestamps[:10]) > 100:
            print(f"[Timestamps] Converting from milliseconds to seconds")
            timestamps = timestamps / 1000.0
        
        print(f"[Timestamps] Loaded {len(timestamps)} timestamps (range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s)")
        return timestamps
        
    except Exception as e:
        print(f"[Timestamps] Warning: Error reading timestamp file: {e}")
        return None


def apply_timestamps_to_traces(df, timestamps, start_frame=0):
    """
    Replace calculated time_s column with actual timestamps from file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with fluorescence traces (must have 'frame' and 'time_s' columns)
    timestamps : np.ndarray
        Array of timestamps indexed by frame number
    start_frame : int
        The starting frame index (for clipped videos)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with updated time_s column
    """
    if timestamps is None:
        return df
    
    df = df.copy()
    
    # Get frame indices from the dataframe
    frames = df['frame'].values + start_frame  # Convert relative to absolute frame indices
    
    # Check if we have enough timestamps
    max_frame = int(np.max(frames))
    if max_frame >= len(timestamps):
        print(f"[Timestamps] Warning: Not enough timestamps for all frames "
              f"(need {max_frame+1}, have {len(timestamps)}). Using calculated times for overflow.")
        # Use timestamps where available, fall back to calculated for rest
        new_times = []
        fps_estimate = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 1/30
        for f in frames:
            if f < len(timestamps):
                new_times.append(timestamps[int(f)])
            else:
                # Extrapolate based on average frame interval
                new_times.append(timestamps[-1] + (f - len(timestamps) + 1) * fps_estimate)
        df['time_s'] = new_times
    else:
        # All frames have timestamps - direct lookup
        df['time_s'] = [timestamps[int(f)] for f in frames]
    
    # Normalize to start from 0 if needed
    if df['time_s'].iloc[0] > 0:
        time_offset = df['time_s'].iloc[0]
        df['time_s'] = df['time_s'] - time_offset
        print(f"[Timestamps] Normalized time axis (subtracted {time_offset:.3f}s offset)")
    
    return df
