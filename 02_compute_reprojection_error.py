#!/usr/bin/env python3

"""
Compares original 2D COCO annotations with reprojected ones. 
Computes global and per-joint MSE/MPJPE using only keypoints with visibility v=2.

USAGE:
> python 02_compute_reprojection_error.py

"""

import json
import numpy as np
from collections import defaultdict

# === CONFIGURATION ===
gt_file   = "temp/02_temp/02_annotations.coco.rectified.json"   # ground truth
pred_file = "temp/02_temp/02_reprojected_annotations.json"      # reprojections

# === FUNCTIONS ===
def load_keypoints_map(coco_json):
    #Return a map image_id -> (coords, mask).
    data = json.load(open(coco_json))
    anns = data["annotations"]
    out = {}
    for ann in anns:
        pts = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)
        coords = pts[:, :2]
        mask   = pts[:, 2] == 2   # use only visible joints
        out[ann["image_id"]] = (coords, mask)
    return out

# === MAIN ===
gt_map   = load_keypoints_map(gt_file)
pred_map = load_keypoints_map(pred_file)

all_errs = []
per_joint = defaultdict(list)

# loop over images that have annotations in both files
common_ids = set(gt_map.keys()) & set(pred_map.keys())
print(f"Common images: {len(common_ids)}")

for img_id in common_ids:
    gt_coords, gt_mask     = gt_map[img_id]
    pred_coords, pred_mask = pred_map[img_id]

    # valid joints must be visible in both GT and predictions
    mask = gt_mask & pred_mask
    if not np.any(mask):
        continue

    errs = np.linalg.norm(pred_coords[mask] - gt_coords[mask], axis=1)
    all_errs.extend(errs.tolist())
    for j_idx, e in zip(np.where(mask)[0], errs):
        per_joint[j_idx].append(e)

# global metrics
all_errs = np.array(all_errs)
mse   = np.mean(all_errs**2) if all_errs.size > 0 else float("nan")
mpjpe = np.mean(all_errs)    if all_errs.size > 0 else float("nan")

print("=== Metrics: GT vs Reprojections Comparison ===")
print(f"# errors computed = {all_errs.size}")
print(f"MSE   (pixel^2): {mse:.3f}")
print(f"MPJPE (pixel):  {mpjpe:.3f}\n")

# per-joint metrics
print("Per-joint MPJPE:")
for j, errs in sorted(per_joint.items()):
    print(f"  joint {j:02d}: {np.mean(errs):.2f} px")


