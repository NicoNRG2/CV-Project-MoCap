#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv11-Pose batch inferencer with JSON export and simple tracking to keep the same person ID across frames.

Usage:
  python 04_yolo_pose.py --images images_rectified/cam_13 --output 04_labels13.json --weights yolo11l-pose.pt --imgsz 3840 --conf 0.20

Options:
  --no-track      Disable tracking for persistent IDs
  --device cuda:0 Use GPU if available (otherwise "cpu")
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ------------------------- Utils -------------------------
def iou_xyxy(a, b) -> float:
    """
    IoU between two bounding boxes in [x1, y1, x2, y2] format.
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    return inter / (areaA + areaB - inter + 1e-9)


def greedy_match(prev_boxes: List[List[float]],
                 cur_boxes: List[List[float]],
                 iou_thr: float = 0.4) -> List[Tuple[int, int]]:
    """
    Greedily matches boxes from frame t-1 to frame t using IoU.
    Returns a list of tuples (idx_prev, idx_cur).
    """
    matches = []
    if not prev_boxes or not cur_boxes:
        return matches

    used_prev = set()
    used_cur = set()

    # Compute all IoU pairs
    pairs = []
    for i, pb in enumerate(prev_boxes):
        for j, cb in enumerate(cur_boxes):
            iou = iou_xyxy(pb, cb)
            if iou >= iou_thr:
                pairs.append((iou, i, j))

    # Sort by IoU descending and match greedily
    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in used_prev or j in used_cur:
            continue
        used_prev.add(i)
        used_cur.add(j)
        matches.append((i, j))

    return matches


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder containing the images")
    ap.add_argument("--output", required=True, help="Path to the output JSON file")
    ap.add_argument("--weights", default="yolo11l-pose.pt", help="YOLO pose weights (e.g., yolo11l-pose.pt)")
    ap.add_argument("--imgsz", type=int, default=3840, help="Input size (longest side)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default=None, help='E.g., "cuda:0" or "cpu" (default: auto)')
    ap.add_argument("--no-track", action="store_true", help="Disable tracking (non-persistent IDs)")
    args = ap.parse_args()

    img_dir = Path(args.images)
    assert img_dir.is_dir(), f"Folder not found: {img_dir}"

    # Collect images (alphabetical order -> useful for sequences)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(str(img_dir / ext)))
    img_paths = sorted(img_paths)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Load model
    model = YOLO(args.weights)

    # Tracking structures
    tracking_enabled = not args.no_track
    next_track_id = 1
    prev_boxes = []
    prev_ids = []

    # Output JSON structure
    output = {
        "meta": {
            "model": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "tracking": tracking_enabled,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "images": []
    }

    # Inference loop
    for idx, img_path in enumerate(tqdm(img_paths, desc="Processing")):
        # For manual debug print, uncomment:
        # print(f"[DEBUG] Processing image {idx+1}/{len(img_paths)}: {img_path}")

        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )

        if len(results) == 0:
            # Should not happen, but handle anyway
            output["images"].append({
                "file": os.path.basename(img_path),
                "width": None,
                "height": None,
                "persons": []
            })
            prev_boxes, prev_ids = [], []
            continue

        res = results[0]
        H, W = res.orig_shape
        persons = []

        # Extract bbox, confidence, and keypoints
        # boxes.xyxy: (N,4) — boxes.conf: (N,) — keypoints.data: (N,K,3) with (x,y,vis)
        bboxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        kpts = res.keypoints.data.cpu().numpy().tolist() if res.keypoints is not None else []

        # Align lengths
        n = min(len(bboxes), len(scores), len(kpts))

        cur_boxes = bboxes[:n]
        cur_scores = scores[:n]
        cur_kpts = kpts[:n]

        # Simple IoU-based tracking
        cur_ids = [None] * n
        if tracking_enabled and prev_boxes:
            matches = greedy_match(prev_boxes, cur_boxes, iou_thr=0.4)
            for i_prev, j_cur in matches:
                cur_ids[j_cur] = prev_ids[i_prev]

        # Assign new IDs to unmatched detections
        if tracking_enabled:
            for j in range(n):
                if cur_ids[j] is None:
                    cur_ids[j] = next_track_id
                    next_track_id += 1
        else:
            # If tracking disabled, no persistent IDs
            cur_ids = [None] * n

        # Build "persons" list
        for j in range(n):
            persons.append({
                "id": cur_ids[j],
                "score": float(cur_scores[j]),
                "bbox": [float(x) for x in cur_boxes[j]],  # [x1,y1,x2,y2]
                "keypoints": [[float(x), float(y), float(v)] for (x, y, v) in cur_kpts[j]]
            })

        # Append per image
        output["images"].append({
            "file": os.path.basename(img_path),
            "width": int(W),
            "height": int(H),
            "persons": persons
        })

        # Update tracking state
        prev_boxes = cur_boxes
        prev_ids = cur_ids

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. JSON written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
