import cv2
import numpy as np

# Load images
highpass = cv2.imread('debug_organoid_highpass.png', 0)
edges_thresh = cv2.imread('debug_organoid_edges_threshold.png', 0)
edges_in_roi = cv2.imread('debug_organoid_edges_in_roi.png', 0)
dilated = cv2.imread('debug_organoid_dilated_edges.png', 0)
binary = cv2.imread('debug_organoid_binary.png', 0)
mask = cv2.imread('debug_organoid_mask.png', 0)
original = cv2.imread('debug_organoid_original.png', 0)

# Check what we got
print(f"highpass max: {highpass.max()}, min: {highpass.min()}, mean: {highpass.mean():.1f}")
print(f"edges_thresh non-zero: {np.count_nonzero(edges_thresh)}")
print(f"edges_in_roi non-zero: {np.count_nonzero(edges_in_roi)}")
print(f"dilated non-zero: {np.count_nonzero(dilated)}")
print(f"binary non-zero: {np.count_nonzero(binary)}")
print(f"mask non-zero: {np.count_nonzero(mask)}")
print(f"original shape: {original.shape}")

# Check if mask covers the organoid border
overlay = original.copy()
overlay[mask > 0] = 255
cv2.imwrite('debug_overlay_mask_on_original.png', overlay)
print("Saved overlay_mask_on_original.png")

# Also check dilated edges on original
overlay2 = original.copy()
overlay2[dilated > 0] = 255
cv2.imwrite('debug_overlay_dilated_on_original.png', overlay2)
print("Saved overlay_dilated_on_original.png")

# Check edges in ROI
overlay3 = original.copy()
overlay3[edges_in_roi > 0] = 255
cv2.imwrite('debug_overlay_edges_in_roi_on_original.png', overlay3)
print("Saved overlay_edges_in_roi_on_original.png")
