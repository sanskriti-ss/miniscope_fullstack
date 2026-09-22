"""
Signal Processing Functions
F0 computation, smoothing, detrending, wave extraction, spike detection.
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d


def compute_f0(F_series, mode="percentile", percentile=20, first_n=50):
    """
    Compute the baseline fluorescence (F0) for normalization.
    
    Parameters
    ----------
    F_series : array-like
        Raw fluorescence time series
    mode : str
        "percentile" - use percentile of trace as baseline (robust to activity)
        "mean_first_n" - use mean of first N frames as baseline
    percentile : float
        Percentile for baseline (used if mode="percentile")
    first_n : int
        Number of frames for baseline (used if mode="mean_first_n")
    
    Returns
    -------
    float
        Baseline fluorescence value
    """
    F = np.asarray(F_series, dtype=float)
    F = F[np.isfinite(F)]
    if F.size == 0:
        return np.nan
    if mode == "mean_first_n":
        n = min(first_n, F.size)
        return float(np.mean(F[:n]))
    return float(np.percentile(F, percentile))


def smooth_traces(df, window_length=21, polyorder=3, additional_smoothing=True):
    """
    Smooth fluorescence traces using Savitzky-Golay filter with enhanced noise reduction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing time series data with FF0_roi columns
    window_length : int
        Length of the filter window (must be odd and >= polyorder + 1)
        Increased default for noisy data
    polyorder : int
        Order of the polynomial used to fit the samples
    additional_smoothing : bool
        If True, applies additional moving average for very noisy data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added smoothed columns (FF0_roi_smooth)
    """
    df_smoothed = df.copy()
    # Look for FF0_roi columns that don't already have "_smooth" suffix
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" not in c]
    
    print(f"Found columns to smooth: {roi_cols}")
    
    for col in roi_cols:
        data = df[col].values
        # Handle NaN values
        mask = ~np.isnan(data)
        
        current_window = window_length
        if np.sum(mask) < current_window:
            # Not enough valid points, reduce window length
            reduced_window = min(current_window, np.sum(mask))
            if reduced_window < 3:
                df_smoothed[f"{col}_smooth"] = data
                continue
            # Make sure window length is odd
            if reduced_window % 2 == 0:
                reduced_window -= 1
            current_window = max(3, reduced_window)
        
        # Apply Savitzky-Golay filter (only on valid points)
        smoothed = data.copy()
        
        if np.sum(mask) >= current_window:
            # First pass: Savitzky-Golay filter
            smoothed[mask] = savgol_filter(data[mask], current_window, polyorder)
            
            # Second pass: Additional smoothing for very noisy data
            if additional_smoothing:
                # Apply a light moving average to further reduce noise
                smoothed[mask] = uniform_filter1d(smoothed[mask], size=5, mode='nearest')
        
        df_smoothed[f"{col}_smooth"] = smoothed
    
    return df_smoothed


def detrend_traces(df, polyorder=2):
    """
    Remove slow baseline drift from fluorescence traces while preserving fast flashing.
    
    Fits a polynomial to the smoothed signal and subtracts it to remove:
    - Photobleaching
    - Focus drift
    - Slow baseline changes
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing smoothed traces (FF0_roi_smooth columns)
    polyorder : int
        Polynomial order for detrending (1=linear, 2=quadratic, 3=cubic)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added detrended columns (FF0_roi_detrended)
    """
    df_detrended = df.copy()
    
    # Find smoothed columns to detrend
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" in c and "_detrended" not in c]
    
    if not roi_cols:
        print("[Warning] No smoothed traces found for detrending")
        return df
    
    for col in roi_cols:
        smoothed_signal = df[col].values
        mask = ~np.isnan(smoothed_signal)
        
        if np.sum(mask) < polyorder + 1:
            # Not enough points to fit polynomial
            df_detrended[f"{col}_detrended"] = smoothed_signal
            continue
        
        # Fit polynomial to smoothed signal
        time_indices = np.where(mask)[0]
        coeffs = np.polyfit(time_indices, smoothed_signal[mask], polyorder)
        baseline = np.polyval(coeffs, np.arange(len(smoothed_signal)))
        
        # Subtract baseline to remove drift
        detrended = smoothed_signal - baseline + np.nanmean(smoothed_signal)
        
        df_detrended[f"{col}_detrended"] = detrended
    
    return df_detrended


def extract_wave_component(df, fps, low_freq=0.1, high_freq=2.0, order=3):
    """
    Extract sinusoidal-like wave components from fluorescence traces using a Butterworth bandpass filter.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with FF0_roi columns
    fps : float
        Frames per second (sampling rate)
    low_freq : float
        Low cutoff frequency (Hz)
    high_freq : float
        High cutoff frequency (Hz)
    order : int
        Order of the Butterworth filter
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added wave columns (FF0_roiX_wave)
    """
    df_wave = df.copy()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_wave" not in c]
    nyq = 0.5 * fps
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    for col in roi_cols:
        data = df[col].values
        mask = ~np.isnan(data)
        filtered = np.full_like(data, np.nan)
        if np.sum(mask) > order * 2:
            filtered[mask] = filtfilt(b, a, data[mask])
        df_wave[f"{col}_wave"] = filtered
    return df_wave


def detect_spikes_and_dips(df, fps, min_peak_distance_sec=0.4, prominence_fraction=0.3):
    """
    Detect peaks (local maxima) and dips (local minima) in fluorescence traces.
    Uses scipy.signal.find_peaks with prominence-based detection for robust results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with FF0_roi columns
    fps : float
        Frames per second (sampling rate)
    min_peak_distance_sec : float
        Minimum time between consecutive peaks/dips in seconds
    prominence_fraction : float
        Minimum prominence as a fraction of the signal's peak-to-peak range
        (e.g. 0.3 = peak must stand out by at least 30% of the signal range)

    Returns
    -------
    pd.DataFrame
        DataFrame with added spike/dip detection columns
    """
    from scipy.signal import find_peaks

    df_spikes = df.copy()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" not in c
                and "_wave" not in c and "_spike" not in c and "_dip" not in c
                and "_derivative" not in c and "_detrended" not in c]

    min_distance = max(1, int(min_peak_distance_sec * fps))

    for col in roi_cols:
        data = df[col].values
        valid = data[~np.isnan(data)]

        if len(valid) < 3:
            df_spikes[f"{col}_spike"] = 0.0
            df_spikes[f"{col}_dip"] = 0.0
            continue

        # Adaptive prominence based on signal range
        signal_range = np.percentile(valid, 95) - np.percentile(valid, 5)
        prominence = max(0.002, signal_range * prominence_fraction)

        # Detect peaks (spikes) — local maxima
        peaks, _ = find_peaks(data, distance=min_distance, prominence=prominence)

        # Detect dips (troughs) — local minima (find peaks on inverted signal)
        dips, _ = find_peaks(-data, distance=min_distance, prominence=prominence)

        spike_mask = np.zeros(len(data))
        dip_mask = np.zeros(len(data))
        spike_mask[peaks] = 1.0
        dip_mask[dips] = 1.0

        df_spikes[f"{col}_spike"] = spike_mask
        df_spikes[f"{col}_dip"] = dip_mask

    return df_spikes
