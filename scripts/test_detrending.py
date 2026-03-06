"""
Quick test of detrending functionality on existing CSV data.
Shows the effect of removing baseline drift.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detrend_traces(df, polyorder=2):
    """
    Remove slow baseline drift from fluorescence traces while preserving fast flashing.
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


# Load CSV
df = pd.read_csv('plots/01_27_15-18_DOF_Dark1/fluorescence_traces.csv')

# Apply detrending
df_detrended = detrend_traces(df, polyorder=2)

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top plot: Original smoothed trace
ax = axes[0]
if 'FF0_roi1_smooth' in df.columns:
    ax.plot(df['time_s'], df['FF0_roi1_smooth'], 'b-', linewidth=2, label='Original (with drift)')
    ax.fill_between(df['time_s'], df['FF0_roi1_smooth'], alpha=0.2)
    ax.set_ylabel('F/F0 (Original)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('Baseline Drift Problem: Visible Drop in Middle of Recording', fontsize=13, fontweight='bold')

# Bottom plot: Detrended trace
ax = axes[1]
if 'FF0_roi1_smooth_detrended' in df_detrended.columns:
    ax.plot(df_detrended['time_s'], df_detrended['FF0_roi1_smooth_detrended'], 'r-', linewidth=2, label='Detrended (drift removed)')
    ax.fill_between(df_detrended['time_s'], df_detrended['FF0_roi1_smooth_detrended'], alpha=0.2, color='red')
    ax.set_ylabel('F/F0 (Detrended)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('After Detrending: Peaks/Lows Realigned (Drift Removed)', fontsize=13, fontweight='bold')

plt.suptitle('Detrending Effect: Removing Baseline Drift While Preserving Peaks', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('plots/01_27_15-18_DOF_Dark1/detrending_demonstration.png', dpi=150, bbox_inches='tight')
print("Saved detrending demonstration to plots/01_27_15-18_DOF_Dark1/detrending_demonstration.png")

# Print statistics
print("\n=== DETRENDING ANALYSIS ===")
print(f"Original trace (with drift):")
print(f"  Min: {df['FF0_roi1_smooth'].min():.6f}")
print(f"  Max: {df['FF0_roi1_smooth'].max():.6f}")
print(f"  Range: {df['FF0_roi1_smooth'].max() - df['FF0_roi1_smooth'].min():.6f}")
print(f"  Mean: {df['FF0_roi1_smooth'].mean():.6f}")

print(f"\nDetrended trace (drift removed):")
print(f"  Min: {df_detrended['FF0_roi1_smooth_detrended'].min():.6f}")
print(f"  Max: {df_detrended['FF0_roi1_smooth_detrended'].max():.6f}")
print(f"  Range: {df_detrended['FF0_roi1_smooth_detrended'].max() - df_detrended['FF0_roi1_smooth_detrended'].min():.6f}")
print(f"  Mean: {df_detrended['FF0_roi1_smooth_detrended'].mean():.6f}")

print("\n✓ Detrending removes the gradual baseline drop while preserving sharp peaks/flashes!")
