#!/usr/bin/env python3
"""
Script per riproiettare il tuo scheletro 3D triangolato nelle viste delle 4 telecamere
e calcolare MSE e MPJPE rispetto alle annotazioni 2D rettificate in formato COCO.
"""

import os
import json
import numpy as np
import cv2
from collections import defaultdict

def load_camera_calib(calib_path):
    """Carica K, dist, rvec, tvec da camera_calib.json"""
    data = json.load(open(calib_path))
    K     = np.array(data['mtx'],  dtype=float)
    dist  = np.array(data['dist'], dtype=float)
    rvec  = np.array(data['rvecs'],dtype=float)  # shape (3,1)
    tvec  = np.array(data['tvecs'],dtype=float)  # shape (3,1)
    return K, dist, rvec, tvec

def build_image_map(coco_images):
    """
    Costruisce mappa (cam_id, frame_idx) -> image_id
    risolvendo da extra['name'] es. 'out2_frame_0001.png'
    """
    img_map = {}
    for img in coco_images:
        name = img.get('extra',{}).get('name', img['file_name'])
        if not name.startswith('out'): 
            continue
        parts = name.split('_')
        try:
            cam_id     = int(parts[0].replace('out',''))
            frame_idx  = int(parts[2].split('.')[0])
            img_map[(cam_id, frame_idx)] = img['id']
        except:
            continue
    return img_map

def load_gt2d(coco_ann):
    """Mappa image_id -> array(N_joints,2) di punti 2D """
    gt = {}
    for ann in coco_ann:
        img_id = ann['image_id']
        if img_id in gt:
            continue  # assumiamo una sola annotazione per immagine
        pts = np.array(ann['keypoints'], dtype=float).reshape(-1,3)
        gt[img_id] = pts[:,:2]
    return gt

def main():
    # --- CONFIGURAZIONE ---
    camera_ids      = [2,5,8,13]
    calib_base_dir  = "camera_data"
    annotations_file= "_annotations.coco.rectified.json"
    skeleton_file   = "triangulated_3d_skeleton.json"

    # 1) Carica annotazioni COCO rettificate
    coco = json.load(open(annotations_file))
    coco_images      = coco['images']
    coco_annotations = coco['annotations']
    image_map = build_image_map(coco_images)
    gt2d_map  = load_gt2d(coco_annotations)

    # 2) Carica scheletro 3D
    skel3d     = json.load(open(skeleton_file))['skeleton_3d']

    # 3) Carica calibrazioni
    cams = {}
    for cam_id in camera_ids:
        path = os.path.join(calib_base_dir, f"cam_{cam_id}", "calib", "camera_calib.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Calibration file mancante: {path}")
        cams[cam_id] = load_camera_calib(path)

    # 4) Riproiezione e raccolta errori
    all_errors = []
    per_joint  = defaultdict(list)

    for frame_name, pts3d in skel3d.items():
        # estrai indice numerico del frame: "frame_0001" → 1
        frame_idx = int(frame_name.split('_')[1])
        pts3d_arr = np.array(pts3d, dtype=float)  # shape (N_joints,3)

        for cam_id, (K, dist, rvec, tvec) in cams.items():
            key = (cam_id, frame_idx)
            if key not in image_map:
                continue
            img_id = image_map[key]
            if img_id not in gt2d_map:
                continue

            gt_pts2d = gt2d_map[img_id]            # (N_joints,2)
            # proietta tutti i punti in un colpo
            imgpts, _ = cv2.projectPoints(pts3d_arr, rvec, tvec, K, dist)
            proj2d    = imgpts.reshape(-1,2)       # (N_joints,2)

            # calcola errori per giunto
            errs = np.linalg.norm(proj2d - gt_pts2d, axis=1)
            for j,e in enumerate(errs):
                per_joint[j].append(e)
            all_errors.extend(errs.tolist())

    # 5) Metriche globali
    all_errors = np.array(all_errors)
    mse    = np.mean(all_errors**2)
    mpjpe  = np.mean(all_errors)

    print("=== Risultati Riproiezione 3D→2D ===")
    print(f"Frame totali: {len(skel3d)}  ×  Camere: {len(camera_ids)}")
    print(f"#errori calcolati = {all_errors.size}")
    print(f"MSE   (pixel²):       {mse:.3f}")
    print(f"MPJPE (pixel):        {mpjpe:.3f}\n")

    # 6) MPJPE per giunto (opzionale)
    print("MPJPE per giunto:")
    for j,errs in sorted(per_joint.items()):
        print(f"  giunto {j:02d}: {np.mean(errs):.2f} px")

if __name__ == "__main__":
    main()
