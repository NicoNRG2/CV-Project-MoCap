"""
Reorders joints in each frame of a MoCap JSON from an original schema to a new target schema.
Reads an input JSON and writes a reordered output JSON.

USAGE:
> python 03_reorder_triangulation_joints.py 

"""

import json
import argparse
from pathlib import Path

# Original joint order
ORIGINAL_ORDER = [
    "Hips",
    "RHip","RKnee","RAnkle","RFoot",
    "LHip","LKnee","LAnkle","LFoot",
    "Spine","Neck","Head",
    "RShoulder","RElbow","RHand",
    "LShoulder","LElbow","LHand"
]

# Desired new order
NEW_ORDER = [
    "Hips","Spine","Neck","Head",
    "LShoulder","LElbow","LHand",
    "RShoulder","RElbow","RHand",
    "LHip","LKnee","LAnkle","LFoot",
    "RHip","RKnee","RAnkle","RFoot"
]

# Map: new index -> old index
index_map = [ORIGINAL_ORDER.index(j) for j in NEW_ORDER]

def reorder_file(input_path: Path, output_path: Path):
    # Load JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Ignore the "skeleton_3d" field if present
    if "skeleton_3d" in data:
        data = data["skeleton_3d"]

    reordered = {}
    for frame, coords in data.items():
        if not isinstance(coords, list):
            raise ValueError(f"Frame {frame} does not contain a list of joints.")
        if len(coords) != len(ORIGINAL_ORDER):
            raise ValueError(f"Frame {frame}: expected {len(ORIGINAL_ORDER)} joints, found {len(coords)}.")
        reordered[frame] = [coords[i] for i in index_map]

    # Write result
    with open(output_path, "w") as f:
        json.dump(reordered, f, indent=2)

    print(f"File saved to {output_path} ({len(reordered)} frames processed)")

def main():
    parser = argparse.ArgumentParser(description="Reorder joints in the frames of a JSON file.")
    parser.add_argument("--input_json", default="temp/02_temp/02_triangulated_3d_skeleton.json", type=Path, help="Input JSON file")
    parser.add_argument("--output_json", default="temp/03_temp/03_final_triangulation.json", type=Path, help="Output JSON file")
    args = parser.parse_args()
    reorder_file(args.input_json, args.output_json)

if __name__ == "__main__":
    main()
