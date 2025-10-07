import json
import os
import re
import cv2
import numpy as np

# Calibration files for each camera index
CALIB_FILES = {
    '2': 'camera_data/cam_2/calib/camera_calib.json',
    '5': 'camera_data/cam_5/calib/camera_calib.json',
    '8': 'camera_data/cam_8/calib/camera_calib.json',
    '13': 'camera_data/cam_13/calib/camera_calib.json',
}


def load_calibration(calib_path):
    """
    Load camera matrix and distortion coefficients from a JSON file.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib['mtx'], dtype=np.float32)
    dist = np.array(calib['dist'], dtype=np.float32)
    return mtx, dist


def rectify_annotations(coco_json_path, output_json_path):
    """
    Read COCO-format annotations, undistort keypoints and bboxes using the same maps
    that are used for video rectification, and save rectified JSON.
    """
    # Load annotations
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Prepare undistort maps per image
    maps = {}  # image_id -> (map_x, map_y)

    for img in data['images']:
        fname = img['file_name']
        match = re.match(r'.*out(\d+)_frame_.*', fname)
        if not match:
            raise ValueError(f"Cannot extract camera index from {fname}")
        cam_idx = match.group(1)
        if cam_idx not in CALIB_FILES:
            raise ValueError(f"No calibration for camera {cam_idx}")

        # Load calibration once per camera index
        mtx, dist = load_calibration(CALIB_FILES[cam_idx])
        w, h = img['width'], img['height']

        # Build undistort rectify maps (same as video)
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, (w, h), cv2.CV_32FC1
        )
        maps[img['id']] = (map_x, map_y)

    # Rectify annotations
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in maps:
            continue
        map_x, map_y = maps[img_id]
        h, w = map_x.shape

        # Rectify keypoints
        kpts = ann.get('keypoints', [])
        new_kpts = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i:i+3]
            xi = int(round(x))
            yi = int(round(y))
            # clamp to image bounds
            xi = min(max(xi, 0), w - 1)
            yi = min(max(yi, 0), h - 1)
            ux = float(map_x[yi, xi])
            uy = float(map_y[yi, xi])
            new_kpts += [ux, uy, v]
        ann['keypoints'] = new_kpts

        # Rectify bbox via its four corners
        x, y, bw, bh = ann['bbox']
        corners = [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]
        ux_list, uy_list = [], []
        for cx, cy in corners:
            xi = int(round(cx))
            yi = int(round(cy))
            xi = min(max(xi, 0), w - 1)
            yi = min(max(yi, 0), h - 1)
            ux_list.append(float(map_x[yi, xi]))
            uy_list.append(float(map_y[yi, xi]))
        x_min, x_max = min(ux_list), max(ux_list)
        y_min, y_max = min(uy_list), max(uy_list)
        ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Save rectified annotations
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    # adjust paths if needed
    input_json = '_annotations.coco.json'
    output_json = './_annotations.coco.rectified.json'
    print(f"Loading annotations from {input_json}...")
    rectify_annotations(input_json, output_json)
    print(f"Rectified annotations saved to {output_json}")


if __name__ == '__main__':
    main()
