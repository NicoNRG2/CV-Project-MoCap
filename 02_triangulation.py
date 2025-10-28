
"""
Triangulates 2D keypoints from multi-view YOLO annotations into 3D
using camera projection matrices loaded from per-camera calibration files. Outputs a JSON with per-frame 3D joints.

USAGE:
> python 02_triangulation.py --input path/to/annotations.json --output path/to/skeleton_3d.json

"""

import json
import os
import cv2
import numpy as np
import argparse

CALIB_DIR = 'camera_data'

def main(input_json, output_json):
    # 1) Load annotations
    with open(input_json, 'r') as f:
        data = json.load(f)

    # 2) Load projection matrices
    proj_matrices = {}
    for cam_folder in os.listdir(CALIB_DIR):
        calib_file = os.path.join(CALIB_DIR, cam_folder, 'calib', 'camera_calib.json')
        if not os.path.exists(calib_file):
            continue
        calib = json.load(open(calib_file))
        K = np.array(calib['mtx'], dtype=np.float64)
        rvec = np.array(calib['rvecs'], dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        T = np.array(calib['tvecs'], dtype=np.float64).reshape(3, 1)
        P = K.dot(np.hstack((R, T)))
        idx = cam_folder.split('_')[-1]
        proj_matrices[idx] = P

    # 3) Group by frame
    annotations_by_frame = {}
    for ann in data['annotations']:
        img = next(i for i in data['images'] if i['id'] == ann['image_id'])
        parts = img['file_name'].split('_')
        cam = parts[0].replace('out', '')
        frame = parts[2]
        key = f"frame_{frame}"
        annotations_by_frame.setdefault(key, {})[cam] = ann['keypoints']

    # 4) Triangulate excluding occluded points (v<1)
    joints_3d = {}
    for frame_key, cams in annotations_by_frame.items():
        if len(cams) < 2:
            continue
        kp_count = len(next(iter(cams.values()))) // 3
        pts_3d = []
        for j in range(kp_count):
            A = []
            for cam_idx, kpts in cams.items():
                x = kpts[3 * j]
                y = kpts[3 * j + 1]
                v = kpts[3 * j + 2]
                if v < 1:  # 0=missing, 1=occluded, 2=visible
                    continue
                P = proj_matrices.get(cam_idx)
                if P is None:
                    continue
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])
            A = np.array(A)
            if A.shape[0] < 4:
                pts_3d.append([None, None, None])
            else:
                _, _, VT = np.linalg.svd(A)
                X = VT[-1] / VT[-1, 3]
                pts_3d.append(X[:3].tolist())
        joints_3d[frame_key] = pts_3d

    # 5) Save result
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({'skeleton_3d': joints_3d}, f, indent=2)

    print(f"Triangulated 3D skeleton saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triangulate 2D keypoints into 3D from YOLO annotations.")
    parser.add_argument("--input", "-i", required=True, help="Path to the annotations JSON file (input).")
    parser.add_argument("--output", "-o", required=True, help="Path to the 3D output JSON file.")
    args = parser.parse_args()

    main(args.input, args.output)
