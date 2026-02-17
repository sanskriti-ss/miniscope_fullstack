"""
Group Visualization Script for Fluorescent Traces
Combines fluorescent traces from different drug treatments and plots the most "beat-like" sections.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import warnings
import argparse
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

# Import bokeh for plotting
try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import Legend
    from bokeh.layouts import column
    from bokeh.io import curdoc
    from bokeh.palettes import Category10
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    print("Warning: Bokeh not available. Please install with: pip install bokeh")


def extract_drug_type(folder_name: str) -> str:
    """
    Extract drug type from folder name.
    
    Parameters
    ----------
    folder_name : str
        The folder name to parse
        
    Returns
    -------
    str
        Drug type ('DOF', 'QUAN', 'THAR', 'NORMAL', 'NODRUG', or 'UNKNOWN')
    """
    folder_upper = folder_name.upper()
    
    if 'DOF' in folder_upper:
        return 'DOF'
    elif 'QUAN' in folder_upper:
        return 'QUAN'
    elif 'THAR' in folder_upper:
        return 'THAR'
    elif 'NORMAL' in folder_upper:
        return 'NORMAL'
    elif 'NODRUG' in folder_upper:
        return 'NODRUG'
    else:
        return 'UNKNOWN'


def find_most_variable_segment(signal: np.ndarray, time: np.ndarray, 
                               segment_duration: float = 4.0) -> Tuple[int, int]:
    """
    Find the most variable (wave-like) segment in a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        The fluorescence signal
    time : np.ndarray
        Time array corresponding to signal
    segment_duration : float
        Duration of segment to extract in seconds
        
    Returns
    -------
    Tuple[int, int]
        Start and end indices of the most variable segment
    """
    if len(signal) < 10 or len(time) < 10:
        return 0, len(signal)
    
    # Calculate time step
    dt = np.median(np.diff(time))
    segment_frames = int(segment_duration / dt)
    
    if segment_frames >= len(signal):
        return 0, len(signal)
    
    # Calculate rolling variance for segments with finer step size
    max_variance = 0
    best_start = 0
    step_size = max(1, segment_frames // 8)  # Smaller steps for better coverage
    
    print(f"[Segment Selection] Signal length: {len(signal)}, segment_frames: {segment_frames}, step_size: {step_size}")
    
    variances = []
    start_indices = []
    
    for start_idx in range(0, len(signal) - segment_frames + 1, step_size):
        end_idx = start_idx + segment_frames
        segment = signal[start_idx:end_idx]
        
        # Remove NaNs for variance calculation
        valid_segment = segment[~np.isnan(segment)]
        if len(valid_segment) > 10:
            variance = np.var(valid_segment)
            variances.append(variance)
            start_indices.append(start_idx)
            
            if variance > max_variance:
                max_variance = variance
                best_start = start_idx
    
    print(f"[Segment Selection] Found {len(variances)} segments, max variance: {max_variance:.8f} at frame {best_start} (time {time[best_start]:.1f}s)")
    
    # Debug: show top segments
    if len(variances) > 0:
        sorted_indices = np.argsort(variances)[::-1]
        print(f"[Debug] Top 3 segments:")
        for i, idx in enumerate(sorted_indices[:3]):
            start_frame = start_indices[idx]
            var = variances[idx]
            print(f"  {i+1}. Frame {start_frame} (t={time[start_frame]:.1f}s): variance={var:.8f}")
    
    best_end = min(best_start + segment_frames, len(signal))
    return best_start, best_end


def calculate_wave_character(signal: np.ndarray) -> float:
    """
    Calculate the "wave character" of a signal using variance.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
        
    Returns
    -------
    float
        Wave character score (higher = more wave-like)
    """
    valid_signal = signal[~np.isnan(signal)]
    if len(valid_signal) < 3:
        return 0.0
    
    # Use variance as a proxy for wave character
    return np.var(valid_signal)


def estimate_dominant_frequency(signal: np.ndarray, fps: float = 30.0) -> float:
    """
    Estimate the dominant frequency of a signal using FFT.
    
    Parameters
    ----------
    signal : np.ndarray
        The input signal
    fps : float
        Sampling rate (frames per second)
        
    Returns
    -------
    float
        Dominant frequency in Hz
    """
    valid_signal = signal[~np.isnan(signal)]
    if len(valid_signal) < 10:
        return 0.0
    
    # Remove DC component by subtracting mean
    signal_centered = valid_signal - np.mean(valid_signal)
    
    # Calculate FFT
    n = len(signal_centered)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft_vals = np.abs(np.fft.rfft(signal_centered))
    
    # Ignore DC component and very low frequencies
    min_freq_idx = np.argmin(np.abs(freqs - 0.1))  # Ignore below 0.1 Hz
    fft_vals[:min_freq_idx] = 0
    
    # Find dominant frequency
    if np.max(fft_vals) > 0:
        dominant_idx = np.argmax(fft_vals)
        return freqs[dominant_idx]
    
    return 0.0


def estimate_beat_frequency_peaks(signal: np.ndarray, time: np.ndarray) -> float:
    """
    Estimate beat frequency by counting peaks in the signal.
    
    Parameters
    ----------
    signal : np.ndarray
        The input signal
    time : np.ndarray
        Time array
        
    Returns
    -------
    float
        Beat frequency in Hz
    """
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < 10:
        return 0.0
    
    signal_clean = signal[valid_mask]
    time_clean = time[valid_mask]
    
    # Find peaks with some prominence
    signal_std = np.std(signal_clean)
    prominence = max(0.01, signal_std * 0.3)
    
    peaks, _ = find_peaks(signal_clean, prominence=prominence)
    
    if len(peaks) < 2:
        return 0.0
    
    # Calculate time between peaks
    peak_times = time_clean[peaks]
    intervals = np.diff(peak_times)
    
    if len(intervals) == 0:
        return 0.0
    
    # Average interval between peaks
    mean_interval = np.mean(intervals)
    if mean_interval > 0:
        return 1.0 / mean_interval
    
    return 0.0


def load_trace_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load fluorescence trace data from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
        
    Returns
    -------
    Optional[pd.DataFrame]
        Loaded DataFrame or None if loading failed
    """
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['time_s']
        
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Required columns not found in {csv_path}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def process_experiment_folder(plots_dir: str, segment_duration: float = 4.0) -> Dict[str, List[Dict]]:
    """
    Process all experiment folders and extract the most beat-like segments.
    
    Parameters
    ----------
    plots_dir : str
        Path to plots directory
        
    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary with drug types as keys and list of experiment data as values
    """
    drug_data = {}
    
    if not os.path.exists(plots_dir):
        print(f"Plots directory not found: {plots_dir}")
        return drug_data
    
    # Iterate through experiment folders
    for folder_name in os.listdir(plots_dir):
        folder_path = os.path.join(plots_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        csv_path = os.path.join(folder_path, 'fluorescence_traces.csv')
        
        if not os.path.exists(csv_path):
            print(f"No fluorescence_traces.csv found in {folder_path}")
            continue
        
        # Extract drug type
        drug_type = extract_drug_type(folder_name)
        
        # Load trace data
        df = load_trace_data(csv_path)
        if df is None:
            continue
        
        # Find ROI columns (FF0_roi columns, excluding derivatives and spikes)
        roi_cols = [col for col in df.columns if 
                   col.startswith('FF0_roi') and 
                   not any(suffix in col for suffix in ['_spike', '_dip', '_derivative'])]
        
        if len(roi_cols) == 0:
            print(f"No ROI columns found in {csv_path}")
            continue
        
        # Separate raw and smoothed columns
        raw_cols = [col for col in roi_cols if '_smooth' not in col]
        smooth_cols = [col for col in roi_cols if '_smooth' in col]
        
        print(f"[Processing] {folder_name}: {len(raw_cols)} raw ROIs, {len(smooth_cols)} smoothed ROIs")
        
        # Process each raw ROI
        for roi_col in raw_cols:
            signal = df[roi_col].values
            time = df['time_s'].values
            
            # Find most variable segment
            start_idx, end_idx = find_most_variable_segment(signal, time, segment_duration=segment_duration)
            
            # Extract segment
            segment_time = time[start_idx:end_idx] - time[start_idx]  # Normalize to start at 0
            segment_signal = signal[start_idx:end_idx]
            
            # Calculate wave character score
            wave_score = calculate_wave_character(segment_signal)
            
            # Calculate beat frequency using both methods
            freq_fft = estimate_dominant_frequency(segment_signal, fps=30.0)
            freq_peaks = estimate_beat_frequency_peaks(segment_signal, segment_time)
            
            # Use the more reasonable frequency (prefer peaks method for biological signals)
            beat_frequency = freq_peaks if 0.1 <= freq_peaks <= 5.0 else freq_fft
            if not (0.1 <= beat_frequency <= 5.0):
                beat_frequency = 0.0
            
            # Check for corresponding smoothed data
            smooth_signal = None
            smooth_col = roi_col + '_smooth'
            if smooth_col in df.columns:
                smooth_signal = df[smooth_col].values[start_idx:end_idx]
            
            # Store data
            if drug_type not in drug_data:
                drug_data[drug_type] = []
            
            drug_data[drug_type].append({
                'folder': folder_name,
                'roi': roi_col,
                'time': segment_time,
                'signal': segment_signal,
                'smooth_signal': smooth_signal,
                'wave_score': wave_score,
                'beat_frequency': beat_frequency,
                'original_start_time': time[start_idx]
            })
    
    return drug_data


def save_plot_data_csv(drug_data: Dict[str, List[Dict]], output_path: str = "group_traces_data.csv"):
    """
    Save the plot data as a CSV file for further analysis.
    
    Parameters
    ----------
    drug_data : Dict[str, List[Dict]]
        Processed drug data
    output_path : str
        Output CSV file path
    """
    all_data = []
    
    for drug_type, experiments in drug_data.items():
        for exp_idx, exp in enumerate(experiments):
            time = exp['time']
            signal = exp['signal']
            smooth_signal = exp.get('smooth_signal', None)
            
            # Remove NaN values
            valid_mask = ~np.isnan(signal)
            time_clean = time[valid_mask]
            signal_clean = signal[valid_mask]
            
            for t, s in zip(time_clean, signal_clean):
                all_data.append({
                    'drug_type': drug_type,
                    'experiment': f"{drug_type}_{exp_idx+1}",
                    'folder': exp['folder'],
                    'roi': exp['roi'],
                    'time_s': t,
                    'FF0_raw': s,
                    'wave_score': exp['wave_score'],
                    'beat_frequency': exp['beat_frequency'],
                    'original_start_time': exp['original_start_time'],
                    'data_type': 'raw'
                })
            
            # Add smoothed data if available
            if smooth_signal is not None:
                smooth_clean = smooth_signal[valid_mask]
                for t, s in zip(time_clean, smooth_clean):
                    all_data.append({
                        'drug_type': drug_type,
                        'experiment': f"{drug_type}_{exp_idx+1}",
                        'folder': exp['folder'],
                        'roi': exp['roi'],
                        'time_s': t,
                        'FF0_raw': s,
                        'wave_score': exp['wave_score'],
                        'beat_frequency': exp['beat_frequency'],
                        'original_start_time': exp['original_start_time'],
                        'data_type': 'smooth'
                    })
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print(f"Plot data saved to: {output_path}")


def create_bokeh_plot(drug_data: Dict[str, List[Dict]], output_path: str = "group_fluorescence_traces.html"):
    """
    Create Bokeh plot of grouped fluorescence traces.
    
    Parameters
    ----------
    drug_data : Dict[str, List[Dict]]
        Processed drug data
    output_path : str
        Output HTML file path
    """
    if not BOKEH_AVAILABLE:
        print("Bokeh not available. Cannot create plot.")
        return
    
    # Create figure
    p = figure(
        width=900, 
        height=600,
        title="Grouped Fluorescence Traces - Most Beat-like Segments (dotted=raw, solid=smoothed)",
        x_axis_label="Time (s)",
        y_axis_label="F/F0",
        toolbar_location="above"
    )
    
    # Color palette for different drugs
    colors = Category10[10] if len(drug_data) <= 10 else Category10[max(len(drug_data), 3)]
    color_map = {}
    
    # Sort drug data by wave score for consistent legend ordering
    sorted_drugs = sorted(drug_data.items(), key=lambda x: np.mean([exp['wave_score'] for exp in x[1]]), reverse=True)
    
    # Plot traces for each drug type
    for drug_idx, (drug_type, experiments) in enumerate(sorted_drugs):
        color = colors[drug_idx % len(colors)]
        color_map[drug_type] = color
        
        traces_plotted = 0
        mean_wave_score = np.mean([exp['wave_score'] for exp in experiments])
        mean_frequency = np.mean([exp['beat_frequency'] for exp in experiments])
        legend_added = False
        
        for exp in experiments:
            time = exp['time']
            signal = exp['signal']
            smooth_signal = exp.get('smooth_signal', None)
            
            # Remove NaN values from raw signal
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) < 3:
                continue
            
            time_clean = time[valid_mask]
            signal_clean = signal[valid_mask]
            
            # Plot raw data as dotted line
            if not legend_added:
                freq_str = f"{mean_frequency:.1f} Hz" if mean_frequency > 0 else "no beats"
                p.line(
                    time_clean, 
                    signal_clean,
                    line_width=1.5,
                    color=color,
                    alpha=0.6,
                    line_dash='dotted',
                    legend_label=f"{drug_type} ({freq_str}, score: {mean_wave_score:.3f})"
                )
                legend_added = True
            else:
                p.line(
                    time_clean, 
                    signal_clean,
                    line_width=1.5,
                    color=color,
                    alpha=0.6,
                    line_dash='dotted'
                )
            
            # Plot smoothed data as solid line if available
            if smooth_signal is not None:
                smooth_clean = smooth_signal[valid_mask]
                if np.sum(~np.isnan(smooth_clean)) > 3:
                    p.line(
                        time_clean, 
                        smooth_clean,
                        line_width=2,
                        color=color,
                        alpha=0.8
                    )
            
            traces_plotted += 1
        
        print(f"{drug_type}: Plotted {traces_plotted} traces")
    
    # Customize plot
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Output to HTML file
    output_file(output_path)
    save(p)
    print(f"Plot saved to: {output_path}")


def compute_asymmetric_envelope(signal: np.ndarray, rise_window: int = 3, fall_window: int = 12) -> np.ndarray:
    """
    Compute an asymmetric smoothing that captures sharp depolarization (fast rise) 
    and slower repolarization (slow fall).
    
    Uses a one-sided moving average approach: quickly follows upstrokes, slowly follows downstrokes.
    This mimics calcium transient physiology: rapid uptake and slower return to baseline.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (fluorescence trace)
    rise_window : int
        Window size for fast upstroke response (smaller = faster rise)
    fall_window : int
        Window size for slow downstroke response (larger = slower fall)
        
    Returns
    -------
    np.ndarray
        Asymmetric smoothed trace anchored to actual signal values
    """
    # Handle NaN values
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < 3:
        return signal
    
    # Interpolate NaN values for processing
    signal_clean = signal.copy()
    if np.any(~valid_mask):
        indices = np.arange(len(signal))
        signal_clean = np.interp(indices, indices[valid_mask], signal[valid_mask])
    
    # Initialize output with first valid value
    envelope = np.zeros_like(signal_clean)
    first_valid_idx = np.where(valid_mask)[0][0] if np.any(valid_mask) else 0
    envelope[first_valid_idx] = signal_clean[first_valid_idx]
    
    # One-sided adaptive moving average
    for i in range(first_valid_idx + 1, len(signal_clean)):
        current_val = signal_clean[i]
        prev_envelope = envelope[i - 1]
        
        # If signal is rising, use short window (fast response)
        if current_val > prev_envelope:
            # Weighted average biased toward current value (fast rise)
            envelope[i] = 0.4 * prev_envelope + 0.6 * current_val
        else:
            # If signal is falling, use longer window (slow response)
            # Look back and average more history
            start_idx = max(first_valid_idx, i - fall_window)
            envelope[i] = np.mean(signal_clean[start_idx:i+1])
    
    # Restore NaN values
    envelope[~valid_mask] = np.nan
    
    return envelope


def create_normalized_bokeh_plot(drug_data: Dict[str, List[Dict]], output_path: str = "group_fluorescence_traces_normalized.html"):
    """
    Create Bokeh plot with normalized y-axes per drug type for better visibility of variation.
    
    Parameters
    ----------
    drug_data : Dict[str, List[Dict]]
        Processed drug data
    output_path : str
        Output HTML file path
    """
    if not BOKEH_AVAILABLE:
        print("Bokeh not available. Cannot create plot.")
        return
    
    # Color palette for different drugs
    colors = Category10[10] if len(drug_data) <= 10 else Category10[max(len(drug_data), 3)]
    
    # Sort drug data by wave score for consistent legend ordering
    sorted_drugs = sorted(drug_data.items(), key=lambda x: np.mean([exp['wave_score'] for exp in x[1]]), reverse=True)
    
    # Create a subplot for each drug type
    plots = []
    
    for drug_idx, (drug_type, experiments) in enumerate(sorted_drugs):
        color = colors[drug_idx % len(colors)]
        
        # Create figure for this drug
        p = figure(
            width=900,
            height=400,
            title=f"{drug_type} - Normalized Traces (dotted=raw, colored=smoothed, black=asymmetric envelope)",
            x_axis_label="Time (s)",
            y_axis_label="Normalized F/F0",
            toolbar_location="above"
        )
        
        traces_plotted = 0
        mean_wave_score = np.mean([exp['wave_score'] for exp in experiments])
        mean_frequency = np.mean([exp['beat_frequency'] for exp in experiments])
        legend_added = False
        
        for exp in experiments:
            time = exp['time']
            signal = exp['signal']
            smooth_signal = exp.get('smooth_signal', None)
            
            # Remove NaN values from raw signal
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) < 3:
                continue
            
            time_clean = time[valid_mask]
            signal_clean = signal[valid_mask]
            
            # Normalize signal to zero mean and unit scale per experiment
            signal_mean = np.mean(signal_clean)
            signal_std = np.std(signal_clean)
            if signal_std > 0:
                signal_normalized = (signal_clean - signal_mean) / signal_std
            else:
                signal_normalized = signal_clean - signal_mean
            
            # Plot raw data as dotted line
            if not legend_added:
                freq_str = f"{mean_frequency:.1f} Hz" if mean_frequency > 0 else "no beats"
                p.line(
                    time_clean,
                    signal_normalized,
                    line_width=1.5,
                    color=color,
                    alpha=0.6,
                    line_dash='dotted',
                    legend_label=f"{drug_type} ({freq_str}, score: {mean_wave_score:.3f})"
                )
                legend_added = True
            else:
                p.line(
                    time_clean,
                    signal_normalized,
                    line_width=1.5,
                    color=color,
                    alpha=0.6,
                    line_dash='dotted'
                )
            
            # Plot smoothed data as solid line if available
            if smooth_signal is not None:
                smooth_clean = smooth_signal[valid_mask]
                if np.sum(~np.isnan(smooth_clean)) > 3:
                    # Normalize smoothed signal using same parameters as raw
                    smooth_normalized = (smooth_clean - signal_mean) / (signal_std if signal_std > 0 else 1)
                    p.line(
                        time_clean,
                        smooth_normalized,
                        line_width=2,
                        color=color,
                        alpha=0.8
                    )
                    
                    # Add asymmetric envelope (black line) that captures sharp rise and slow fall
                    envelope = compute_asymmetric_envelope(smooth_clean, rise_window=5, fall_window=15)
                    envelope_valid = ~np.isnan(envelope)
                    if np.sum(envelope_valid) > 3:
                        # Normalize envelope using same parameters as raw signal
                        envelope_normalized = (envelope[envelope_valid] - signal_mean) / (signal_std if signal_std > 0 else 1)
                        p.line(
                            time_clean[envelope_valid],
                            envelope_normalized,
                            line_width=2.5,
                            color='black',
                            alpha=0.6,
                            line_dash='solid'
                        )
            
            traces_plotted += 1
        
        print(f"{drug_type}: Plotted {traces_plotted} normalized traces")
        
        # Customize plot
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        plots.append(p)
    
    # Stack plots vertically
    layout = column(*plots)
    
    # Output to HTML file
    output_file(output_path)
    save(layout)
    print(f"Normalized plot saved to: {output_path}")


def print_summary(drug_data: Dict[str, List[Dict]]):
    """
    Print summary of processed data.
    
    Parameters
    ----------
    drug_data : Dict[str, List[Dict]]
        Processed drug data
    """
    print("\n=== Data Summary ===")
    
    total_traces = 0
    
    for drug_type, experiments in drug_data.items():
        print(f"\n{drug_type}:")
        print(f"  Number of traces: {len(experiments)}")
        total_traces += len(experiments)
        
        if experiments:
            wave_scores = [exp['wave_score'] for exp in experiments]
            durations = [len(exp['time']) / 30.0 for exp in experiments]  # Assume ~30fps
            frequencies = [exp['beat_frequency'] for exp in experiments]
            
            print(f"  Wave scores: min={np.min(wave_scores):.4f}, "
                  f"max={np.max(wave_scores):.4f}, mean={np.mean(wave_scores):.4f}")
            print(f"  Beat frequencies: min={np.min(frequencies):.2f}, "
                  f"max={np.max(frequencies):.2f}, mean={np.mean(frequencies):.2f} Hz")
            print(f"  Segment durations: {np.mean(durations):.1f}±{np.std(durations):.1f}s")
            
            folders = list(set(exp['folder'] for exp in experiments))
            rois = list(set(exp['roi'] for exp in experiments))
            print(f"  Folders: {folders}")
            print(f"  ROIs: {rois}")
    
    print(f"\nTotal traces across all drugs: {total_traces}")


def main():
    """Main function to run the group visualization."""
    parser = argparse.ArgumentParser(description="Group fluorescence visualization tool")
    parser.add_argument("--plots-dir", "-d", 
                       default="/Users/sanskriti/Documents/GitHub/miniscope_fullstack/plots",
                       help="Directory containing experiment folders (default: ./plots)")
    parser.add_argument("--segment-duration", "-t", type=float, default=4.0,
                       help="Duration of segments to extract in seconds (default: 4.0)")
    parser.add_argument("--output", "-o", default="group_fluorescence_traces.html",
                       help="Output HTML file name (default: group_fluorescence_traces.html)")
    parser.add_argument("--csv-output", default="group_traces_data.csv",
                       help="Output CSV file name (default: group_traces_data.csv)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip creating the Bokeh plot (only generate CSV)")
    
    args = parser.parse_args()
    
    print("=== Group Fluorescence Visualization ===")
    print(f"Processing data from: {args.plots_dir}")
    print(f"Segment duration: {args.segment_duration}s")
    
    # Process experiment data
    drug_data = process_experiment_folder(args.plots_dir, segment_duration=args.segment_duration)
    
    if not drug_data:
        print("No data found to process!")
        return
    
    # Print summary
    print_summary(drug_data)
    
    # Save plot data as CSV
    save_plot_data_csv(drug_data, args.csv_output)
    
    # Create Bokeh plots (unless disabled)
    if not args.no_plot:
        create_bokeh_plot(drug_data, args.output)
        
        # Also create normalized plot for better visibility
        normalized_output = args.output.replace('.html', '_normalized.html')
        create_normalized_bokeh_plot(drug_data, normalized_output)
        
        print(f"\nVisualization complete!")
        print(f"  - Interactive plot: {args.output}")
        print(f"  - Normalized plot: {normalized_output}")
        print(f"  - Data CSV: {args.csv_output}")
        print(f"Open the HTML files in your browser to view the interactive plots.")
    else:
        print(f"\nData processing complete!")
        print(f"  - Data CSV: {args.csv_output}")


if __name__ == "__main__":
    main()