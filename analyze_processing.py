import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all the intermediate stages
original = cv2.imread('debug_organoid_original.png', 0)
highpass = cv2.imread('debug_organoid_highpass.png', 0)
edge_map = cv2.imread('debug_organoid_edge_map.png', 0)
edge_map_roi = cv2.imread('debug_organoid_edge_map_roi.png', 0)
edge_dilated = cv2.imread('debug_organoid_edge_dilated.png', 0)
binary_closed = cv2.imread('debug_organoid_binary_closed.png', 0)
binary = cv2.imread('debug_organoid_binary.png', 0)
mask = cv2.imread('debug_organoid_mask.png', 0)

print("Checking intermediate processing steps:")
print(f"original: shape={original.shape}, mean={original.mean():.1f}, max={original.max()}")
print(f"highpass: shape={highpass.shape}, mean={highpass.mean():.1f}, max={highpass.max()}, non-zero={np.count_nonzero(highpass)}")
print(f"edge_map: non-zero={np.count_nonzero(edge_map)}")
print(f"edge_map_roi: non-zero={np.count_nonzero(edge_map_roi)}")
print(f"edge_dilated: non-zero={np.count_nonzero(edge_dilated)}")
print(f"binary_closed: non-zero={np.count_nonzero(binary_closed)}")
print(f"binary: non-zero={np.count_nonzero(binary)}")
print(f"mask: non-zero={np.count_nonzero(mask)}")

# Create detailed visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Row 1: Original and high-pass
axes[0,0].imshow(original, cmap='gray')
axes[0,0].set_title('Original')
axes[0,0].axis('off')

axes[0,1].imshow(highpass, cmap='gray')
axes[0,1].set_title('High-Pass Filter')
axes[0,1].axis('off')

axes[0,2].imshow(edge_map, cmap='gray')
axes[0,2].set_title(f'Edge Map (threshold >50)')
axes[0,2].axis('off')

# Row 2: Processing steps
axes[1,0].imshow(edge_map_roi, cmap='gray')
axes[1,0].set_title('Edge Map in ROI')
axes[1,0].axis('off')

axes[1,1].imshow(edge_dilated, cmap='gray')
axes[1,1].set_title('Dilated Edges')
axes[1,1].axis('off')

axes[1,2].imshow(binary_closed, cmap='gray')
axes[1,2].set_title('After Morphological Close')
axes[1,2].axis('off')

# Row 3: Final steps
axes[2,0].imshow(binary, cmap='gray')
axes[2,0].set_title('After FloodFill')
axes[2,0].axis('off')

axes[2,1].imshow(mask, cmap='gray')
axes[2,1].set_title('Final Mask')
axes[2,1].axis('off')

# Overlay mask on original
overlay = original.copy()
overlay[mask > 0] = 255
axes[2,2].imshow(overlay, cmap='gray')
axes[2,2].set_title('Mask Overlay')
axes[2,2].axis('off')

plt.tight_layout()
plt.savefig('debug_processing_steps.png', dpi=100, bbox_inches='tight')
print("Saved processing steps to debug_processing_steps.png")

# Check if the mask actually overlays well with the organoid border
# by computing how well it aligns with the high-pass edges
highpass_edges = (highpass > 50).astype(np.uint8)
mask_edges = cv2.Canny(mask, 50, 150)

intersection = np.logical_and(highpass_edges, mask_edges > 0)
union = np.logical_or(highpass_edges, mask_edges > 0)

if np.count_nonzero(union) > 0:
    iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    print(f"\nMask-to-Organoid-Border Alignment (IoU): {iou:.3f}")
    if iou < 0.3:
        print("WARNING: Mask does not align well with organoid border!")
    else:
        print("OK: Mask aligns well with organoid border")
