"""
Reprojects a 3D skeleton into each camera using calibration parameters and generates COCO-style 2D keypoint annotations per image.
Reads rectified images metadata, triangulated 3D joints, and camera calibrations, then saves a new annotations JSON.

USAGE:
> python 02_generate_reprojected_annotations.py

"""

import os
import re
import json
import numpy as np
import cv2

# === CONFIGURATION ===
CALIB_BASE_DIR      = "camera_data"                    # folder containing cam_2, cam_5, ...
CAMERA_IDS          = [2,5,8,13]                       # IDs of the 4 cameras
RECTIFIED_JSON_PATH = "temp/02_temp/02_annotations.coco.rectified.json"
SKELETON3D_PATH     = "temp/02_temp/02_triangulated_3d_skeleton.json"
OUTPUT_JSON_PATH    = "temp/02_temp/02_reprojected_annotations.json"

# === UTILITY FUNCTIONS ===

def load_camera_calib(calib_path):
    # Return (K, dist, rvec, tvec) loaded from camera_calib.json.
    data = json.load(open(calib_path))
    K    = np.array(data['mtx'], dtype=float)
    dist = np.array(data.get('dist', [0,0,0,0,0]), dtype=float)
    rvec = np.array(data['rvecs'], dtype=float).reshape(3,1)
    tvec = np.array(data['tvecs'], dtype=float).reshape(3,1)
    return K, dist, rvec, tvec

def parse_image_name(name):
    # Extract cam_id and frame_idx from the filename: e.g., 'out2_frame_0001.png' → (2, 1)
    # Returns (cam_id, frame_idx) or (None, None) if no match.

    base = os.path.basename(name)
    m = re.match(r"out(\d+)_frame_(\d+)", base)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

# === MAIN ===

def main():
    # 1) Load original JSON for info, licenses, categories, images
    orig = json.load(open(RECTIFIED_JSON_PATH, 'r'))
    info       = orig.get('info', {})
    licenses   = orig.get('licenses', [])
    categories = orig['categories']
    images     = orig['images']

    # 2) Load 3D skeleton
    sk3d = json.load(open(SKELETON3D_PATH, 'r'))['skeleton_3d']
    # e.g., sk3d['frame_0001'] = [[X1,Y1,Z1],[X2,Y2,Z2],...]

    # 3) Load calibrations
    cams = {}
    for cam_id in CAMERA_IDS:
        calib_path = os.path.join(CALIB_BASE_DIR, f"cam_{cam_id}", "calib", "camera_calib.json")
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Cannot find calibration: {calib_path}")
        cams[cam_id] = load_camera_calib(calib_path)

    # 4) Generate new annotations
    annotations = []
    ann_id = 0
    cat_id = categories[0]['id']  # assume a single category: 'person'

    for img in images:
        img_id   = img['id']
        fname    = img.get('extra',{}).get('name', img['file_name'])
        cam_id, frame_idx = parse_image_name(fname)
        if cam_id not in cams or frame_idx is None:
            continue

        # Load 3D points for this frame
        frm_key = f"frame_{frame_idx:04d}"
        if frm_key not in sk3d:
            continue
        pts3d = np.array(sk3d[frm_key], dtype=float)    # (N_joints,3)

        # Project to 2D
        K, dist, rvec, tvec = cams[cam_id]
        imgpts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        pts2d = imgpts.reshape(-1,2)  # (N_joints,2)

        # Build COCO keypoints: [x1,y1,v1, x2,y2,v2, ...]
        # visibility v=2 (visible) for all
        flat_kp = []
        for (x,y) in pts2d:
            flat_kp.extend([float(x), float(y), 2])

        # Compute bbox and area
        xs = pts2d[:,0]; ys = pts2d[:,1]
        x_min, y_min = float(xs.min()), float(ys.min())
        w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        bbox = [x_min, y_min, w, h]
        area = w * h

        ann = {
            "id":           ann_id,
            "image_id":     img_id,
            "category_id":  cat_id,
            "bbox":         bbox,
            "area":         area,
            "segmentation": [],
            "iscrowd":      0,
            "keypoints":    flat_kp
        }
        annotations.append(ann)
        ann_id += 1

    # 5) Assemble final result
    out = {
        "info":       info,
        "licenses":   licenses,
        "categories": categories,
        "images":     images,
        "annotations": annotations
    }

    # 6) Save to file
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f" wrote {len(annotations)} annotations to `{OUTPUT_JSON_PATH}`")

if __name__ == "__main__":
    main()
