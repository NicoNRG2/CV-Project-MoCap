"""
Description:
Script to adapt original COCO annotations by filtering and reordering keypoints to YOLO format.

USAGE:
> python 04_adapt_annotations.py --input temp/02_temp/02_annotations.coco.rectified.json --output temp/04_temp/04_original_annotations_filtered.json
"""

#!/usr/bin/env python3
import json
import argparse
from copy import deepcopy

# joints present in the input file, in the same order as in "categories[...]['keypoints']"
INPUT_JOINTS = [
    "Hips",
    "RHip",
    "RKnee",
    "RAnkle",
    "RFoot",
    "LHip",
    "LKnee",
    "LAnkle",
    "LFoot",
    "Spine",
    "Neck",
    "Head",
    "RShoulder",
    "RElbow",
    "RHand",
    "LShoulder",
    "LElbow",
    "LHand",
]

# joints to keep and their desired final order
OUTPUT_JOINTS = [
    "Head",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LHand", "RHand",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
]

# joints to completely discard
DROP_JOINTS = {"Hips", "Spine", "Neck", "LFoot", "RFoot"}


def slice_keypoints(flat_kps, idx):
    """
    flat_kps = [x0, y0, v0, x1, y1, v1, ...]
    idx = index of the joint (0-based) in INPUT_JOINTS order
    returns (x, y, v) for that joint
    """
    base = idx * 3
    return flat_kps[base:base+3]


def build_reordered_keypoints(flat_kps):
    """
    Takes the original keypoints (flattened) and returns a new flattened list
    containing only the desired joints in OUTPUT_JOINTS order.
    """
    # map joint_name -> index in the input
    name_to_input_idx = {name: i for i, name in enumerate(INPUT_JOINTS)}

    new_flat = []
    for joint_name in OUTPUT_JOINTS:
        if joint_name not in name_to_input_idx:
            # if the joint is missing in the input, append (0,0,0)
            new_flat.extend([0.0, 0.0, 0])
            continue

        i_in = name_to_input_idx[joint_name]
        x, y, v = slice_keypoints(flat_kps, i_in)

        # if, for any reason, the joint is in DROP_JOINTS, nullify it
        if joint_name in DROP_JOINTS:
            x, y, v = 0.0, 0.0, 0

        new_flat.extend([x, y, v])

    return new_flat


def main():
    parser = argparse.ArgumentParser(
        description="Filter and reorder COCO-style annotation keypoints."
    )
    parser.add_argument("--input", required=True, help="input JSON file")
    parser.add_argument("--output", required=True, help="output JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # --- update categories ---
    new_data = deepcopy(data)

    # find the 'person' category (or the first with 'keypoints')
    for cat in new_data.get("categories", []):
        if "keypoints" in cat and len(cat["keypoints"]) > 0:
            # replace the keypoints list with the new order
            cat["keypoints"] = OUTPUT_JOINTS[:]

            # optional: recreate a minimal skeleton consistent with the new order
            # each pair is (idx1, idx2) — COCO uses 1-based indexing
            SKELETON_DEF = [
                ("Head", "LShoulder"),
                ("Head", "RShoulder"),
                ("LShoulder", "LElbow"),
                ("LElbow", "LHand"),
                ("RShoulder", "RElbow"),
                ("RElbow", "RHand"),
                ("LHip", "LKnee"),
                ("LKnee", "LAnkle"),
                ("RHip", "RKnee"),
                ("RKnee", "RAnkle"),
                ("LHip", "RHip"),
            ]

            # map joint_name -> 1-based index in the new order
            out_index = {name: i+1 for i, name in enumerate(OUTPUT_JOINTS)}

            new_skel = []
            for a, b in SKELETON_DEF:
                if a in out_index and b in out_index:
                    new_skel.append([out_index[a], out_index[b]])

            cat["skeleton"] = new_skel
            break

    # --- update annotations ---
    for ann in new_data.get("annotations", []):
        if "keypoints" not in ann:
            continue

        old_kps = ann["keypoints"]
        new_kps = build_reordered_keypoints(old_kps)
        ann["keypoints"] = new_kps

        # optional: update num_keypoints if present
        if "num_keypoints" in ann:
            # count how many joints have visibility > 0
            vis_count = 0
            for i in range(0, len(new_kps), 3):
                v = new_kps[i+2]
                if v is not None and v != 0:
                    vis_count += 1
            ann["num_keypoints"] = vis_count

    # write output
    with open(args.output, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
