import json
import os
import cv2
import numpy as np

# Paths (adjust as needed)
CALIB_DIR = 'camera_data'
RECT_ANN_PATH = './_annotations.coco.rectified.json'
OUTPUT_3D_JSON = './triangulated_3d_skeleton.json'

# Load rectified annotations
with open(RECT_ANN_PATH, 'r') as f:
    data = json.load(f)

# Load camera projection matrices
# Assumes for each camera: camera_data/cam_{idx}/calib/camera_calib_real.json contains mtx, dist, rvecs, tvecs
proj_matrices = {}
for cam_folder in os.listdir(CALIB_DIR):
    cam_path = os.path.join(CALIB_DIR, cam_folder)
    calib_file = os.path.join(cam_path, 'calib', 'camera_calib.json')
    if not os.path.exists(calib_file):
        continue
    calib = json.load(open(calib_file))
    mtx = np.array(calib['mtx'], dtype=np.float64)
    # Convert rotation vector to rotation matrix
    rvec = np.array(calib['rvecs'], dtype=np.float64).reshape(3, 1)
    R_mat, _ = cv2.Rodrigues(rvec)
    # Translation vector
    T = np.array(calib['tvecs'], dtype=np.float64).reshape(3, 1)
    # Build projection matrix P = K [R | T]
    RT = np.hstack((R_mat, T))
    P = mtx.dot(RT)
    idx = cam_folder.split('_')[-1]
    proj_matrices[idx] = P

# Group annotations by frame (frame id is in image file_name)
annotations_by_frame = {}
for ann in data['annotations']:
    img = next(i for i in data['images'] if i['id'] == ann['image_id'])
    fname = img['file_name']
    parts = fname.split('_')
    cam = parts[0].replace('out', '')
    frame = parts[2]
    key = f"frame_{frame}"
    annotations_by_frame.setdefault(key, {})[cam] = ann['keypoints']

# Triangulate per frame, per joint
joints_3d = {}
for frame_key, cams in annotations_by_frame.items():
    if len(cams) < 2:
        continue
    kp_count = len(next(iter(cams.values()))) // 3
    pts_3d = []
    for j in range(kp_count):
        A = []
        for cam_idx, kpts in cams.items():
            u = kpts[3*j]
            v = kpts[3*j + 1]
            P = proj_matrices.get(cam_idx)
            if P is None:
                continue
            # build rows: u*P[2,:] - P[0,:], v*P[2,:] - P[1,:]
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        A = np.array(A)
        if A.shape[0] < 4:
            # Not enough equations to solve
            pts_3d.append([None, None, None])
            continue
        # Solve using SVD
        _, _, VT = np.linalg.svd(A)
        X = VT[-1]
        X = X / X[3]
        pts_3d.append(X[:3].tolist())
    joints_3d[frame_key] = pts_3d

# Save 3D skeletons
os.makedirs(os.path.dirname(OUTPUT_3D_JSON), exist_ok=True)
with open(OUTPUT_3D_JSON, 'w') as f:
    json.dump({'skeleton_3d': joints_3d}, f, indent=2)

print(f"Triangulated 3D skeleton saved to {OUTPUT_3D_JSON}")
