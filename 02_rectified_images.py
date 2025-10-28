#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rectifies images using per-camera calibration (mtx, dist) by precomputing undistortion remap grids and saving rectified outputs.

USAGE:
> python 02_rectified_images.py
"""

import cv2
import numpy as np
import json
import os
import glob
import re
from pathlib import Path

# Calibration utilities
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

# Cache for rectification maps (per camera and resolution)
_rectify_map_cache = {}  # key: (cam_idx, width, height) -> (map_x, map_y)

def get_rectify_maps(cam_idx, width, height, calib_path):
    # Create (or fetch from cache) remap grids for a given camera
    # and a given resolution. Keeps the same logic as your video script.

    key = (cam_idx, width, height)
    if key in _rectify_map_cache:
        return _rectify_map_cache[key]

    mtx, dist = load_calibration(calib_path)

    # Pixel grid (as in the original script)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)

    # Same rectification operation: undistortPoints with P=mtx
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]

    _rectify_map_cache[key] = (map_x, map_y)
    return map_x, map_y


# Single-image rectification
def rectify_image(image_path, output_path, cam_idx, calib_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Unreadable image: {image_path}")
        return False

    height, width = img.shape[:2]

    try:
        map_x, map_y = get_rectify_maps(cam_idx, width, height, calib_path)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file not found: {calib_path}")
        return False
    except KeyError as e:
        print(f"[ERROR] Missing keys in calibration {calib_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Problem computing maps for {image_path}: {e}")
        return False

    # Same remap logic as the video script
    rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Save with the same original filename
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(output_path, rectified)
    if not ok:
        print(f"[ERROR] Save failed: {output_path}")
        return False

    return True


# Main: batch over images folder
def main():
    input_dir = "train"
    output_dir = "images_rectified"
    os.makedirs(output_dir, exist_ok=True)

    # Filename pattern:
    # e.g. out8_frame_0004_png.rf.345f188c7aba71764ede395be914924c
    name_re = re.compile(r'^out(?P<cam>\d+)_frame_(?P<frame>\d+)_png\.rf\..+$')

    # Grab all files in the images folder (jpg/png etc.)
    image_paths = sorted(
        [p for p in glob.glob(os.path.join(input_dir, '*')) if os.path.isfile(p)]
    )

    if not image_paths:
        print(f"[INFO] No images found in: {input_dir}")
        return

    processed = 0
    skipped = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        m = name_re.match(fname)
        if not m:
            # Does not match the expected pattern: skip
            skipped += 1
            # Optional: print a warning occasionally
            if skipped <= 10:
                print(f"[WARN] Non-matching filename, skipping: {fname}")
            continue

        cam_idx = m.group('cam')  # '2','5','8','13', etc.
        calib_path = os.path.join("camera_data", f"cam_{cam_idx}", "calib", "camera_calib.json")

        out_path = os.path.join(output_dir, fname)

        ok = rectify_image(img_path, out_path, cam_idx, calib_path)
        if ok:
            processed += 1
            if processed % 50 == 0:
                print(f"[INFO] Rectified {processed} images...")
        else:
            skipped += 1

    print(f"[DONE] Completed. Rectified: {processed}, skipped: {skipped}")

if __name__ == "__main__":
    main()
