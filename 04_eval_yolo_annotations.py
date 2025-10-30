#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict

# -------- config --------
GT_PATH = Path("temp/04_temp/04_original_annotations_filtered.json")
YOLO_PATH = Path("temp/04_temp/annotations_yolo.json")

# the expected keypoint ordering (13 joints)
JOINT_NAMES = [
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand",
]

NUM_JOINTS = len(JOINT_NAMES)

def load_coco_like(path: Path):
    """Return:
    - images_by_id: {image_id: file_name}
    - image_id_by_file: {file_name: image_id}
    - ann_by_image: {image_id: [annotation_dict, ...]}
    """
    with open(path, "r") as f:
        data = json.load(f)

    images_by_id = {}
    image_id_by_file = {}
    for img in data["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        images_by_id[img_id] = file_name
        image_id_by_file[file_name] = img_id

    ann_by_image = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    return images_by_id, image_id_by_file, ann_by_image

def kp_list_to_xy_array(kp_list):
    """
    kp_list = [x0,y0,v0, x1,y1,v1, ...]
    returns list of (x,y)
    """
    xy = []
    for j in range(0, len(kp_list), 3):
        x = kp_list[j + 0]
        y = kp_list[j + 1]
        # v = kp_list[j + 2]  # visibility, unused here
        xy.append((x, y))
    return xy

def per_joint_errors(gt_xy, pr_xy):
    """
    gt_xy/pr_xy: list of (x,y) for each joint index
    returns:
      - list of euclidean errors per joint
      - list of squared errors per joint (x,y MSE style)
    """
    errs = []
    sqerrs = []
    for (gx, gy), (px, py) in zip(gt_xy, pr_xy):
        dx = px - gx
        dy = py - gy
        dist = math.sqrt(dx*dx + dy*dy)
        errs.append(dist)

        # squared error on (x,y) for this joint
        sq = dx*dx + dy*dy
        sqerrs.append(sq)
    return errs, sqerrs

def main():
    # load GT and YOLO predictions
    gt_img_by_id, gt_id_by_file, gt_ann_by_image = load_coco_like(GT_PATH)
    yolo_img_by_id, yolo_id_by_file, yolo_ann_by_image = load_coco_like(YOLO_PATH)

    # We'll aggregate stats
    all_joint_errs = [[] for _ in range(NUM_JOINTS)]  # per-joint euclidean distance
    all_joint_sqerrs = [[] for _ in range(NUM_JOINTS)]  # per-joint squared error

    mpjpe_list = []  # mean euclidean over joints for each frame
    mse_list = []    # mean squared (x,y) error over joints for each frame

    # Match by file_name intersection
    common_files = set(gt_id_by_file.keys()) & set(yolo_id_by_file.keys())

    per_image_results = []  # store printable info per frame

    for file_name in sorted(common_files):
        gt_img_id = gt_id_by_file[file_name]
        yolo_img_id = yolo_id_by_file[file_name]

        gt_anns = gt_ann_by_image.get(gt_img_id, [])
        yolo_anns = yolo_ann_by_image.get(yolo_img_id, [])

        # We only handle the simple case: exactly one person in GT and in YOLO
        if len(gt_anns) != 1 or len(yolo_anns) != 1:
            continue

        gt_kp_xy = kp_list_to_xy_array(gt_anns[0]["keypoints"])
        pr_kp_xy = kp_list_to_xy_array(yolo_anns[0]["keypoints"])

        # Safety: need same number of joints
        if len(gt_kp_xy) != NUM_JOINTS or len(pr_kp_xy) != NUM_JOINTS:
            continue

        errs, sqerrs = per_joint_errors(gt_kp_xy, pr_kp_xy)

        # MPJPE for this image = mean euclidean error over all joints
        mpjpe = sum(errs) / NUM_JOINTS

        # MSE for this image = mean squared (x,y) error over all joints
        mse_img = sum(sqerrs) / NUM_JOINTS

        mpjpe_list.append(mpjpe)
        mse_list.append(mse_img)

        # save per-joint to global
        for j_idx, (e, se) in enumerate(zip(errs, sqerrs)):
            all_joint_errs[j_idx].append(e)
            all_joint_sqerrs[j_idx].append(se)

        per_image_results.append({
            "file_name": file_name,
            "mpjpe": mpjpe,
            "mse": mse_img,
            "per_joint_error": {
                JOINT_NAMES[j]: errs[j] for j in range(NUM_JOINTS)
            }
        })

    # --- summary ---
    def safe_mean(arr):
        return sum(arr) / len(arr) if arr else float("nan")

    overall_mpjpe = safe_mean(mpjpe_list)
    overall_mse = safe_mean(mse_list)

    per_joint_mean_err = {JOINT_NAMES[j]: safe_mean(all_joint_errs[j]) for j in range(NUM_JOINTS)}
    per_joint_mean_sqerr = {JOINT_NAMES[j]: safe_mean(all_joint_sqerrs[j]) for j in range(NUM_JOINTS)}

    # print nice report
    print("==== YOLO vs GT 2D Keypoint Evaluation ====")
    print(f"Images evaluated: {len(per_image_results)}")
    print(f"Overall MPJPE (px): {overall_mpjpe:.3f}")
    print(f"Overall MSE   (px^2): {overall_mse:.3f}")
    print()

    print("Per-joint mean Euclidean error (px):")
    for j in JOINT_NAMES:
        print(f"  {j:12s} : {per_joint_mean_err[j]:.3f}")

    print()
    print("Per-joint mean squared error (px^2):")
    for j in JOINT_NAMES:
        print(f"  {j:12s} : {per_joint_mean_sqerr[j]:.3f}")

    print()
    print("Per-image breakdown (first 10):")
    for r in per_image_results[:10]:
        print(f"- {r['file_name']}")
        print(f"    MPJPE: {r['mpjpe']:.3f} px")
        print(f"    MSE  : {r['mse']:.3f} px^2")

if __name__ == "__main__":
    main()
