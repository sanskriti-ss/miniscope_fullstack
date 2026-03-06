import cv2
import numpy as np

# Test Cellpose on enhanced image
print("Testing Cellpose on ENHANCED image...")

from cellpose import models

# Load the enhanced image (better contrast)
img = cv2.imread('debug_single_organoid_enhanced.png', cv2.IMREAD_GRAYSCALE)
print('Image shape:', img.shape)

# Run Cellpose - model should now be cached
print("Loading Cellpose model (should be cached)...")
model = models.CellposeModel()

# Try different diameter hints
for diameter in [100, 150, 200, 250, 300]:
    print("\nTrying diameter=%d..." % diameter)
    masks, flows, styles = model.eval(img, diameter=diameter)
    
    n_objects = masks.max()
    print('  Detected %d objects' % n_objects)
    
    if n_objects > 0:
        # Save results for this diameter
        cv2.imwrite('debug_cellpose_mask_d%d.png' % diameter, (masks > 0).astype(np.uint8) * 255)
        
        # Analyze detected objects
        for obj_id in range(1, min(n_objects + 1, 5)):
            obj_mask = (masks == obj_id).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
                print('    Object %d: area=%d, center=(%d,%d), size=(%d,%d)' % (obj_id, area, cx, cy, w, h))
        
        # Save overlay
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for obj_id in range(1, n_objects + 1):
            obj_mask = (masks == obj_id).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        cv2.imwrite('debug_cellpose_overlay_d%d.png' % diameter, vis)
        print('  Saved to debug_cellpose_overlay_d%d.png' % diameter)

print("\nDone!")

