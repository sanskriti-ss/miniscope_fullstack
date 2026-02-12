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


def detect_spikes_and_dips(df, fps, spike_threshold_std=1.5, dip_threshold_std=1.5, 
                          spike_prominence=0.02, window_size=10):
    """
    Detect sharp spikes (rapid increases) and dips (rapid decreases) in fluorescence traces.
    Uses derivative-based detection to capture transient events like calcium transients.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with FF0_roi columns
    fps : float
        Frames per second (sampling rate)
    spike_threshold_std : float
        Number of standard deviations above mean derivative to detect spikes
    dip_threshold_std : float
        Number of standard deviations below mean derivative to detect dips
    spike_prominence : float
        Minimum change in F/F0 to be considered a significant transient
    window_size : int
        Window for computing local baseline
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added spike/dip detection columns
    """
    df_spikes = df.copy()
    roi_cols = [c for c in df.columns if c.startswith("FF0_roi") and "_smooth" not in c and "_wave" not in c and "_spike" not in c]
    
    for col in roi_cols:
        data = df[col].values
        mask = ~np.isnan(data)
        
        # Calculate first derivative (rate of change)
        derivative = np.zeros_like(data)
        derivative[1:] = np.diff(data)
        
        # Calculate statistics on the derivative
        valid_deriv = derivative[mask]
        if len(valid_deriv) > 0:
            deriv_mean = np.mean(valid_deriv)
            deriv_std = np.std(valid_deriv)
            
            # Adaptive thresholds based on the signal's variability
            spike_threshold = deriv_mean + spike_threshold_std * deriv_std
            dip_threshold = deriv_mean - dip_threshold_std * deriv_std
        else:
            spike_threshold = 0.03
            dip_threshold = -0.03
        
        # Detect spikes (rapid increases)
        spike_mask = (derivative > spike_threshold) & (derivative > spike_prominence)
        
        # Detect dips (rapid decreases)  
        dip_mask = (derivative < dip_threshold) & (derivative < -spike_prominence)
        
        # Store spike and dip information
        df_spikes[f"{col}_spike"] = spike_mask.astype(float)
        df_spikes[f"{col}_dip"] = dip_mask.astype(float)
        
        # Create derivative trace for visualization
        df_spikes[f"{col}_derivative"] = derivative
    
    return df_spikes
