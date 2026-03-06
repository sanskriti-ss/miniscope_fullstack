"""
Diagnostic script to identify where sharp flashes are being lost in the pipeline.
Compares different processing steps to find the culprit.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vars import *
from helper_functions.roi_selection import extract_frame_channel


def analyze_roi_signal(video_path, roi_mask, channel=0, start_frame=0, end_frame=None, n_frames=None):
    """
    Extract and analyze a single ROI signal through the pipeline.
    
    Returns diagnostics at each step.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract raw signal
    raw_signal = []
    time_data = []
    frame_idx = start_frame
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = extract_frame_channel(frame, channel)
        vals = img[roi_mask.astype(bool)]
        
        if vals.size > 0:
            raw_signal.append(np.mean(vals))
        else:
            raw_signal.append(np.nan)
        
        time_data.append((frame_idx - start_frame) / fps)
        frame_idx += 1
        
        if n_frames and len(raw_signal) >= n_frames:
            break
    
    cap.release()
    
    raw_signal = np.array(raw_signal)
    time_data = np.array(time_data)
    
    # Step 1: Normalize to F/F0
    F0 = np.nanpercentile(raw_signal, F0_PERCENTILE)
    ff0_signal = raw_signal / (F0 if F0 > 0 else np.nan)
    
    # Step 2: Smooth
    mask = ~np.isnan(ff0_signal)
    smoothed = ff0_signal.copy()
    if np.sum(mask) >= SMOOTH_WINDOW_LENGTH:
        smoothed[mask] = savgol_filter(ff0_signal[mask], SMOOTH_WINDOW_LENGTH, SMOOTH_POLYORDER)
        if ADDITIONAL_SMOOTHING:
            smoothed[mask] = uniform_filter1d(smoothed[mask], size=5, mode='nearest')
    
    # Step 3: Light smoothing only
    light_smooth = ff0_signal.copy()
    if np.sum(mask) >= 11:
        light_smooth[mask] = savgol_filter(ff0_signal[mask], 11, 3)  # Much lighter
    
    # Step 4: No smoothing, just raw F/F0
    no_smooth = ff0_signal.copy()
    
    # Calculate statistics
    stats = {
        'raw_mean': np.nanmean(raw_signal),
        'raw_std': np.nanstd(raw_signal),
        'raw_min': np.nanmin(raw_signal),
        'raw_max': np.nanmax(raw_signal),
        'ff0_mean': np.nanmean(ff0_signal),
        'ff0_std': np.nanstd(ff0_signal),
        'ff0_min': np.nanmin(ff0_signal),
        'ff0_max': np.nanmax(ff0_signal),
        'f0_value': F0,
        'n_peaks_raw': len([x for x in np.diff(raw_signal) > 0.5 * np.nanstd(raw_signal) if x]),
        'n_peaks_ff0': len([x for x in np.diff(ff0_signal) > 0.5 * np.nanstd(ff0_signal) if x]),
    }
    
    return {
        'time': time_data,
        'raw': raw_signal,
        'ff0': ff0_signal,
        'smoothed': smoothed,
        'light_smooth': light_smooth,
        'no_smooth': no_smooth,
        'stats': stats,
        'F0': F0,
    }


def plot_diagnostic_comparison(results, out_path="diagnostic_flash_analysis.png"):
    """
    Create a 4-panel comparison plot showing where flashes are lost.
    """
    time = results['time']
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Panel 1: Raw signal
    ax = axes[0]
    ax.plot(time, results['raw'], 'b-', linewidth=1, label='Raw F')
    ax.axhline(results['F0'], color='r', linestyle='--', linewidth=2, label=f'F0 (percentile={F0_PERCENTILE})')
    ax.set_ylabel('Raw Intensity')
    ax.set_title('Step 1: Raw Signal (before normalization)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: F/F0 without smoothing
    ax = axes[1]
    ax.plot(time, results['no_smooth'], 'g-', linewidth=1, label='F/F0 (no smoothing)')
    ax.set_ylabel('F/F0')
    ax.set_title('Step 2: Normalized (F/F0) - No Smoothing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: F/F0 with light smoothing
    ax = axes[2]
    ax.plot(time, results['light_smooth'], 'orange', linewidth=1.5, label='F/F0 (light smoothing: window=11)')
    ax.plot(time, results['no_smooth'], 'g--', linewidth=0.5, alpha=0.5, label='No smoothing (reference)')
    ax.set_ylabel('F/F0')
    ax.set_title('Step 3: Light Smoothing (preserves sharp features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: F/F0 with full smoothing
    ax = axes[3]
    ax.plot(time, results['smoothed'], 'r-', linewidth=1.5, label=f'F/F0 (full smoothing: window={SMOOTH_WINDOW_LENGTH})')
    ax.plot(time, results['no_smooth'], 'g--', linewidth=0.5, alpha=0.5, label='No smoothing (reference)')
    ax.set_ylabel('F/F0')
    ax.set_xlabel('Time (s)')
    ax.set_title('Step 4: Full Smoothing (current pipeline) - May flatten sharp peaks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plot saved to: {out_path}")


def print_diagnostics(results):
    """Print detailed diagnostics."""
    stats = results['stats']
    
    print("\n" + "="*70)
    print("FLASH DETECTION DIAGNOSTICS")
    print("="*70)
    
    print("\n[Raw Signal]")
    print(f"  Mean: {stats['raw_mean']:.1f}")
    print(f"  Std Dev: {stats['raw_std']:.1f}")
    print(f"  Range: [{stats['raw_min']:.1f}, {stats['raw_max']:.1f}]")
    print(f"  Dynamic Range: {(stats['raw_max'] - stats['raw_min']) / stats['raw_std']:.2f}σ")
    
    print("\n[Normalized (F/F0)]")
    print(f"  F0 (percentile={F0_PERCENTILE}): {results['F0']:.1f}")
    print(f"  Mean: {stats['ff0_mean']:.3f}")
    print(f"  Std Dev: {stats['ff0_std']:.3f}")
    print(f"  Range: [{stats['ff0_min']:.3f}, {stats['ff0_max']:.3f}]")
    
    print("\n[Peak Detection]")
    print(f"  Peaks in raw signal: {stats['n_peaks_raw']}")
    print(f"  Peaks in F/F0 signal: {stats['n_peaks_ff0']}")
    
    print("\n[Recommendations]")
    recommendations = []
    
    # Check if F0 is too high
    if stats['raw_min'] < results['F0'] * 0.5:
        recommendations.append("⚠️  F0 is very high compared to minimum signal")
        recommendations.append("    → Consider lowering F0_PERCENTILE (e.g., 10 instead of 20)")
    
    # Check if smoothing window is too large
    if SMOOTH_WINDOW_LENGTH > 21:
        recommendations.append("⚠️  Smoothing window is large and may flatten sharp peaks")
        recommendations.append(f"    → Consider reducing SMOOTH_WINDOW_LENGTH (current: {SMOOTH_WINDOW_LENGTH})")
    
    # Check if signal is noisy
    if stats['ff0_std'] > 0.1:
        recommendations.append("⚠️  Signal has high noise even after normalization")
        recommendations.append("    → Sharp flashes may be obscured by noise")
        recommendations.append("    → Verify ROI placement covers the flashing region")
    
    # Check if dynamic range is small
    if (stats['raw_max'] - stats['raw_min']) < 2 * stats['raw_std']:
        recommendations.append("⚠️  Signal has low dynamic range (flashes are subtle)")
        recommendations.append("    → May be normal for dim organoids")
        recommendations.append("    → Use lighter smoothing or no smoothing to preserve small changes")
    
    if not recommendations:
        recommendations.append("✓ Signal looks good for peak detection")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose where flashes are being lost")
    parser.add_argument("--video", "-v", default=VIDEO_PATH,
                       help="Video file to analyze")
    parser.add_argument("--roi-file", "-r", 
                       help="Path to saved ROI mask numpy file (for reproducibility)")
    parser.add_argument("--output", "-o", default="diagnostic_flash_analysis.png",
                       help="Output diagnostic plot file")
    parser.add_argument("--n-frames", "-n", type=int, default=None,
                       help="Number of frames to analyze (default: all)")
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.video}")
    print(f"ROI Detection Method: {ROI_DETECTION_METHOD}")
    print(f"F0 Mode: {F0_MODE} (percentile={F0_PERCENTILE})")
    print(f"Smoothing: window={SMOOTH_WINDOW_LENGTH}, polyorder={SMOOTH_POLYORDER}, additional={ADDITIONAL_SMOOTHING}")
    
    # Create a simple test ROI (center of image)
    cap = cv2.VideoCapture(args.video)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Create circular ROI in center
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = 40
    cv2.circle(roi_mask, center, radius, 255, -1)
    
    print(f"\nUsing test ROI: circle at center ({center[0]}, {center[1]}), radius={radius}")
    
    # Analyze
    results = analyze_roi_signal(args.video, roi_mask, channel=CHANNEL, n_frames=args.n_frames)
    
    # Print diagnostics
    print_diagnostics(results)
    
    # Create comparison plot
    plot_diagnostic_comparison(results, args.output)
    
    print(f"\nOpen '{args.output}' to see where flashes are being lost in the pipeline")


if __name__ == "__main__":
    main()
