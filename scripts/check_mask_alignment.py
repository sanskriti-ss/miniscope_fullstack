import cv2
import numpy as np

# Load what we have
original = cv2.imread('debug_organoid_original.png', 0)
highpass = cv2.imread('debug_organoid_highpass.png', 0)
mask = cv2.imread('debug_organoid_mask.png', 0)

print(f"Original: min={original.min()}, max={original.max()}, mean={original.mean():.1f}")
print(f"High-pass: min={highpass.min()}, max={highpass.max()}, mean={highpass.mean():.1f}")
print(f"Mask: non-zero pixels = {np.count_nonzero(mask)}")

# Check where the mask is relative to the organoid center
# The organoid should be brightest in the original and have bright edges in the highpass
mask_y, mask_x = np.where(mask > 0)
if len(mask_x) > 0:
    mask_center_y = (mask_y.min() + mask_y.max()) // 2
    mask_center_x = (mask_x.min() + mask_x.max()) // 2
    mask_radius_x = (mask_x.max() - mask_x.min()) // 2
    mask_radius_y = (mask_y.max() - mask_y.min()) // 2
    
    print(f"\nMask bounds:")
    print(f"  Center: ({mask_center_x}, {mask_center_y})")
    print(f"  Radius X: {mask_radius_x}, Radius Y: {mask_radius_y}")
    print(f"  Bounds: X=[{mask_x.min()}, {mask_x.max()}], Y=[{mask_y.min()}, {mask_y.max()}]")

# Check what brightness values are in the mask region
mask_original_values = original[mask > 0]
print(f"\nOriginal image values inside mask:")
print(f"  Min: {mask_original_values.min()}, Max: {mask_original_values.max()}, Mean: {mask_original_values.mean():.1f}, Median: {np.median(mask_original_values):.1f}")

# Check what values are in the highpass in the mask region
mask_highpass_values = highpass[mask > 0]
print(f"\nHigh-pass image values inside mask:")
print(f"  Min: {mask_highpass_values.min()}, Max: {mask_highpass_values.max()}, Mean: {mask_highpass_values.mean():.1f}, Median: {np.median(mask_highpass_values):.1f}")

# Compare to values outside the mask (in the ROI but not in mask)
roi_mask = np.zeros_like(mask)
cv2.circle(roi_mask, (300, 300), 132, 255, -1)
outside_mask = np.logical_and(roi_mask > 0, mask == 0)

if np.count_nonzero(outside_mask) > 0:
    outside_original = original[outside_mask]
    outside_highpass = highpass[outside_mask]
    print(f"\nValues OUTSIDE mask but in ROI (original): mean={outside_original.mean():.1f}")
    print(f"Values OUTSIDE mask but in ROI (highpass): mean={outside_highpass.mean():.1f}")

# Most importantly: check if the bright border in highpass is at the edge of the mask
# Sample a circle at the mask boundary
h, w = original.shape
cy, cx = 300, 300
# Get points on the mask boundary
boundary_y, boundary_x = np.where(cv2.Canny(mask, 50, 150) > 0)

if len(boundary_x) > 0:
    boundary_highpass = highpass[boundary_y, boundary_x]
    print(f"\nHigh-pass values AT the mask boundary:")
    print(f"  Mean: {boundary_highpass.mean():.1f}, Max: {boundary_highpass.max()}")
else:
    print("No clear boundary found in mask")

# Check sampling around the center
print(f"\nSampling the organoid from center outward (0=center, ~80=outer edge of detected mask):")
for r in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
    y = int(cy - r)
    if 0 <= y < h:
        orig_val = original[y, cx]
        hp_val = highpass[y, cx]
        mask_val = mask[y, cx]
        print(f"  r={r:2d}: original={orig_val:3d}, highpass={hp_val:3d}, in_mask={mask_val > 0}")
