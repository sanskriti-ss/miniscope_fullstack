"""
Manual ROI Selection Module
Contains shared functions for manual ROI selection used by both fluorescence and mechanical analysis.
"""

import cv2
import numpy as np
from vars import *


def extract_frame_channel(frame, channel=0):
    """
    Extract channel data from frame, handling both grayscale and color videos.
    """
    if len(frame.shape) == 2:
        return frame.astype(np.float32)
    elif len(frame.shape) == 3:
        if frame.shape[2] == 1:
            return frame[:, :, 0].astype(np.float32)
        else:
            return frame[:, :, channel].astype(np.float32)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


def preview_video_and_draw_rois(video_path, n_preview_frames=150, channel=0, preview_fps=10):
    """
    Display a video preview and allow the user to draw ROIs manually.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    n_preview_frames : int
        Number of frames to use for preview and ROI drawing.
    channel : int
        Channel to extract (0=B, 1=G, 2=R).
    preview_fps : int
        FPS for video preview playback.

    Returns
    -------
    list
        List of manually drawn ROI masks.
    list
        List of ROI info dictionaries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    # Read frames for preview
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    for _ in range(n_preview_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = extract_frame_channel(frame, channel)
        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames available for preview.")

    print(f"\n{'='*70}")
    print("VIDEO PREVIEW AND MANUAL ROI SELECTION")
    print(f"{'='*70}")
    print("Instructions:")
    print("  - Press SPACE to start/pause video preview")
    print("  - Press 'd' to enter drawing mode")
    print("  - Press 'q' to quit preview and proceed to ROI drawing")
    print("  - Press ESC to cancel")
    print(f"{'='*70}\n")

    # Show video preview
    frame_idx = 0
    playing = False
    delay = max(1, int(1000 / preview_fps))  # milliseconds
    
    cv2.namedWindow("Video Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Preview", 800, 600)
    
    while True:
        if playing:
            display_frame = frames[frame_idx]
            frame_idx = (frame_idx + 1) % len(frames)
        else:
            display_frame = frames[frame_idx]
        
        # Normalize and enhance contrast for better visibility
        normalized_frame = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(normalized_frame)
        
        # Add text overlay
        display_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
        status = "PLAYING" if playing else "PAUSED"
        cv2.putText(display_bgr, f"Frame {frame_idx+1}/{len(frames)} - {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_bgr, "SPACE: play/pause, 'd': draw ROIs, 'q': proceed, ESC: cancel", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Video Preview", display_bgr)
        
        key = cv2.waitKey(delay if playing else 1) & 0xFF
        
        if key == ord(' '):  # SPACE - toggle play/pause
            playing = not playing
        elif key == ord('d'):  # 'd' - enter drawing mode
            break
        elif key == ord('q'):  # 'q' - proceed to drawing
            break
        elif key == 27:  # ESC - cancel
            cv2.destroyAllWindows()
            return [], []
    
    cv2.destroyAllWindows()
    
    # Create reference frame for ROI drawing
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
    # Allow multiple ROI drawing
    all_masks = []
    all_info = []
    roi_counter = 1
    
    while True:
        print(f"\nDraw ROI #{roi_counter} (or press ESC to finish)...")
        mask, info = manual_roi_selection(avg_frame, roi_number=roi_counter)
        
        if mask is None:
            break
            
        all_masks.append(mask)
        all_info.append(info)
        roi_counter += 1
        
        # Ask if user wants to draw another ROI
        print(f"ROI #{roi_counter-1} created. Press 'n' for another ROI, any other key to finish.")
        response = input().strip().lower()
        if response != 'n':
            break
    
    return all_masks, all_info


def manual_roi_selection(avg_frame, roi_number=1):
    """
    Allow the user to manually draw a polygon ROI on the video frame.
    Enhanced version with better UI and instructions.

    Parameters
    ----------
    avg_frame : np.ndarray
        Averaged frame for ROI drawing.
    roi_number : int
        ROI number for labeling.

    Returns
    -------
    mask : np.ndarray
        Binary mask of the manually selected ROI.
    info : dict
        Dictionary with ROI information.
    """
    h, w = avg_frame.shape
    
    # Enhance contrast for better visibility
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(avg_frame)
    display_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Variables for polygon drawing
    points = []
    drawing = False
    polygon_complete = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, drawing, display_color, polygon_complete
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near first point to close polygon (if we have at least 3 points)
            if len(points) >= 3:
                first_pt = points[0]
                distance = np.sqrt((x - first_pt[0])**2 + (y - first_pt[1])**2)
                if distance < 15:  # Close polygon if within 15 pixels of first point
                    print(f"[Manual ROI #{roi_number}] Polygon closed by clicking near first point")
                    polygon_complete = True
                    return
            
            # Start/continue drawing
            points.append((x, y))
            drawing = True
            
            # Draw the point
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 6, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            # Draw line back to first point if more than 2 points
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
                # Draw larger circle around first point to show where to click to close
                cv2.circle(temp_display, points[0], 15, (0, 0, 255), 2)
            
            cv2.imshow(f"Manual ROI Selection - ROI #{roi_number}", temp_display)
        
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # Double-click to finish polygon
            if len(points) >= 3:
                print(f"[Manual ROI #{roi_number}] Polygon finished with double-click")
                polygon_complete = True
                return
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Show preview line
            temp_display = display_color.copy()
            for i, pt in enumerate(points):
                cv2.circle(temp_display, pt, 6, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_display, points[i-1], pt, (0, 255, 0), 2)
            
            if len(points) > 0:
                cv2.line(temp_display, points[-1], (x, y), (0, 255, 0), 1)
            
            if len(points) > 2:
                cv2.line(temp_display, points[-1], points[0], (0, 255, 0), 1)
                # Show red circle around first point for closing
                cv2.circle(temp_display, points[0], 15, (0, 0, 255), 2)
                # Check if mouse is near first point
                first_pt = points[0]
                distance = np.sqrt((x - first_pt[0])**2 + (y - first_pt[1])**2)
                if distance < 15:
                    cv2.circle(temp_display, points[0], 15, (0, 0, 255), -1, 8)  # Filled red circle when hovering
            
            cv2.imshow(f"Manual ROI Selection - ROI #{roi_number}", temp_display)
    
    # Create window and set mouse callback
    window_name = f"Manual ROI Selection - ROI #{roi_number}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_color)
    
    print(f"\n" + "="*70)
    print(f"MANUAL ROI #{roi_number} SELECTION INSTRUCTIONS:")
    print("="*70)
    print("  - Click to add points around the organoid")
    print("  - Click near the FIRST point (red circle) to close polygon")
    print("  - Or DOUBLE-CLICK to finish polygon")
    print("  - Or press ENTER when done (will auto-close the polygon)")
    print("  - Press 'r' to reset and start over")
    print("  - Press 'ESC' to cancel this ROI")
    print("="*70 + "\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Check if polygon was completed via mouse interaction
        if polygon_complete:
            break
        
        if key == 13:  # ENTER - finish
            if len(points) >= 3:
                break
            else:
                print("[Warning] Need at least 3 points to create a polygon")
        
        elif key == ord('r'):  # Reset
            points = []
            drawing = False
            polygon_complete = False
            cv2.imshow(window_name, display_color)
            print(f"[Manual ROI #{roi_number}] Reset - start drawing again")
        
        elif key == 27:  # ESC - cancel
            print(f"[Manual ROI #{roi_number}] Cancelled by user")
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print("[ERROR] Need at least 3 points to create ROI")
        return None, None
    
    # Create mask from polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    
    # Calculate ROI info
    M = cv2.moments(points_array)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = np.mean(points_array[:, 0]), np.mean(points_array[:, 1])
    
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    bbox = (int(np.min(x_coords)), int(np.min(y_coords)),
            int(np.max(x_coords) - np.min(x_coords)),
            int(np.max(y_coords) - np.min(y_coords)))
    
    area = np.sum(mask > 0)
    
    info = {
        "label": roi_number,
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "bbox": bbox,
    }
    
    print(f"[Manual ROI #{roi_number}] Created: area={area} pixels, center=({cx:.0f}, {cy:.0f})")
    
    return mask.astype(np.uint8), info