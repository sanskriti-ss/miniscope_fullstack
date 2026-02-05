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
        Drug type ('DOF', 'QUAN', 'THAR', 'NORMAL', or 'UNKNOWN')
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
    
    # Calculate rolling variance for segments
    max_variance = 0
    best_start = 0
    
    for start_idx in range(0, len(signal) - segment_frames + 1, max(1, segment_frames // 4)):
        end_idx = start_idx + segment_frames
        segment = signal[start_idx:end_idx]
        
        # Remove NaNs for variance calculation
        valid_segment = segment[~np.isnan(segment)]
        if len(valid_segment) > 10:
            variance = np.var(valid_segment)
            if variance > max_variance:
                max_variance = variance
                best_start = start_idx
    
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
        
        # Process each ROI
        for roi_col in roi_cols:
            signal = df[roi_col].values
            time = df['time_s'].values
            
            # Find most variable segment
            start_idx, end_idx = find_most_variable_segment(signal, time, segment_duration=segment_duration)
            
            # Extract segment
            segment_time = time[start_idx:end_idx] - time[start_idx]  # Normalize to start at 0
            segment_signal = signal[start_idx:end_idx]
            
            # Calculate wave character score
            wave_score = calculate_wave_character(segment_signal)
            
            # Store data
            if drug_type not in drug_data:
                drug_data[drug_type] = []
            
            drug_data[drug_type].append({
                'folder': folder_name,
                'roi': roi_col,
                'time': segment_time,
                'signal': segment_signal,
                'wave_score': wave_score,
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
                    'FF0': s,
                    'wave_score': exp['wave_score'],
                    'original_start_time': exp['original_start_time']
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
        title="Grouped Fluorescence Traces - Most Beat-like Segments (4s each)",
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
        
        for exp in experiments:
            time = exp['time']
            signal = exp['signal']
            
            # Remove NaN values
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) < 3:
                continue
            
            time_clean = time[valid_mask]
            signal_clean = signal[valid_mask]
            
            # Plot the trace - add legend only for first trace of each drug type
            if traces_plotted == 0:
                p.line(
                    time_clean, 
                    signal_clean,
                    line_width=2,
                    color=color,
                    alpha=0.7,
                    legend_label=f"{drug_type} (wave score: {mean_wave_score:.3f})"
                )
            else:
                p.line(
                    time_clean, 
                    signal_clean,
                    line_width=2,
                    color=color,
                    alpha=0.7
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
            
            print(f"  Wave scores: min={np.min(wave_scores):.4f}, "
                  f"max={np.max(wave_scores):.4f}, mean={np.mean(wave_scores):.4f}")
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
    
    # Create Bokeh plot (unless disabled)
    if not args.no_plot:
        create_bokeh_plot(drug_data, args.output)
        print(f"\nVisualization complete!")
        print(f"  - Interactive plot: {args.output}")
        print(f"  - Data CSV: {args.csv_output}")
        print(f"Open the HTML file in your browser to view the interactive plot.")
    else:
        print(f"\nData processing complete!")
        print(f"  - Data CSV: {args.csv_output}")


if __name__ == "__main__":
    main()