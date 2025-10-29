"""
Visualize YOLO Pose labels (bbox, ID, keypoints, skeleton) on top of images.

USAGE:
> python 04_test_labels.py --images images_rectified/cam_2 --json temp/04_temp/labels2.json --outdir temp/04_temp/cam2

Notes:
- The JSON file is the one produced by the inference script (field "images" with "persons":[{id,score,bbox,keypoints}]).
- Keypoints are in [x, y, v] format, with v = visibility (0/1 or [0..2]); only points with v > threshold are drawn.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Skeleton connections for the COCO-17 model (standard COCO order).
# Indices (0..16): 0-Nose,1-L.Eye,2-R.Eye,3-L.Ear,4-R.Ear,5-L.Shoulder,6-R.Shoulder,
# 7-L.Elbow,8-R.Elbow,9-L.Wrist,10-R.Wrist,11-L.Hip,12-R.Hip,13-L.Knee,14-R.Knee,15-L.Ankle,16-R.Ankle
COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8,10),  # arms + shoulders
    (11,12), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16),  # pelvis + legs
    (0,1), (0,2), (1,3), (2,4)  # head/eyes/ears
]


def draw_pose(
    img,
    bbox=None,
    keypoints=None,
    pid=None,
    score=None,
    kp_vis_thr=0.5,
    thickness=2,
    radius=3
):
    h, w = img.shape[:2]

    # Bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), thickness)

        # Label (ID + confidence)
        label = []
        if pid is not None: label.append(f"ID {pid}")
        if score is not None: label.append(f"{score:.2f}")
        if label:
            text = " | ".join(label)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 6, y1), (0, 255, 255), -1)
            cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Keypoints + skeleton
    if keypoints is not None and len(keypoints) >= 17:
        pts = np.array(keypoints, dtype=float)  # (17, 3) -> x,y,v
        # Lines
        for i, j in COCO_EDGES:
            vi = pts[i, 2]; vj = pts[j, 2]
            if vi > kp_vis_thr and vj > kp_vis_thr:
                pi = (int(round(pts[i, 0])), int(round(pts[i, 1])))
                pj = (int(round(pts[j, 0])), int(round(pts[j, 1])))
                cv2.line(img, pi, pj, (0, 180, 255), thickness)

        # Points
        for k in range(pts.shape[0]):
            x, y, v = pts[k]
            if v > kp_vis_thr:
                cv2.circle(img, (int(round(x)), int(round(y))), radius, (0, 255, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder containing the original images")
    ap.add_argument("--json", required=True, help="JSON file with labels (output from YOLO script)")
    ap.add_argument("--outdir", required=True, help="Output folder for annotated images")
    ap.add_argument("--kp-vis-thr", type=float, default=0.5, help="Keypoint visibility threshold")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness for bbox/skeleton")
    ap.add_argument("--radius", type=int, default=3, help="Radius for keypoints")
    ap.add_argument("--show", action="store_true", help="Display images during processing")
    args = ap.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        raise ValueError("JSON does not contain 'images' or it is empty.")

    for item in tqdm(images, desc="Annotating"):
        fname = item.get("file")
        if not fname:
            continue
        img_path = img_dir / fname
        if not img_path.is_file():
            print(f"[WARN] Missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        persons = item.get("persons", [])
        # Draw each person
        for p in persons:
            bbox = p.get("bbox")
            kpts = p.get("keypoints")
            pid = p.get("id")
            score = p.get("score")
            draw_pose(
                img,
                bbox=bbox,
                keypoints=kpts,
                pid=pid,
                score=score,
                kp_vis_thr=args.kp_vis_thr,
                thickness=args.thickness,
                radius=args.radius
            )

        # Save
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

        # Optional display
        if args.show:
            cv2.imshow("Pose Annotations", img)
            key = cv2.waitKey(1)
            # press 'q' to quit
            if key & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"Done! Annotated images saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
