import cv2
import numpy as np
import json
import os
import glob
import re

"""
Rectifies each input video using per-camera calibration (mtx, dist)
from JSON by precomputing an undistortion map and remapping all frames, then saves the rectified MP4s.

USAGE:
> python 02_rectified_videos.py

"""

def load_calibration(calib_path):
    # Load camera calibration parameters (intrinsics and distortion) from a JSON file.
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def process_video(video_path, calib_path, output_path):
    mtx, dist = load_calibration(calib_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configure MP4 writer matching the input video resolution and FPS.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Build a dense pixel grid to precompute the undistortion mapping once.
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    
    # Compute undistorted pixel coordinates using the original intrinsics as the output projection (P=mtx).
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply the undistortion map to each frame.
        rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        out.write(rectified_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames for {video_path}")
    
    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

def main():
    video_files = glob.glob("mocap_7_videos/out*.mp4") # Path to input video files
    output_dir = "rectified_videos" # Target folder where rectified videos are saved
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_path in video_files:
        # Extract the camera index from filenames like out2.mp4, out5.mp4, etc.
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            # Per-camera calibration JSON expected at camera_data/cam_{index}/calib/camera_calib.json
            calib_path = os.path.join("camera_data", f"cam_{cam_index}", "calib", "camera_calib.json")
        else:
            print("Could not extract camera index from filename:", video_path)
            continue
        
        ## Create one folder for each sample e.g. tracking_01, mocap_1, hpe_1
        output_path = os.path.join(output_dir, '', basename)
        if not os.path.exists(os.path.join(output_dir, '')):
            os.makedirs(os.path.join(output_dir, ''))
            
        print(f"Processing {video_path} using calibration file {calib_path}...")
        process_video(video_path, calib_path, output_path)

if __name__ == "__main__":
    main()
