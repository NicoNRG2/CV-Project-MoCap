"""
Script to adapt keypoints in a JSON file by filtering out unwanted joints.

USAGE:
> python 04_adapt_keypoint.py [keypoint_number]
    n.b. in our case the keypoint to remove is 2,5,8,13: pass it as argument one by one.

"""

import json
from pathlib import Path
import sys

if len(sys.argv) != 2:
    print("Usage: python filter_keypoints.py <number>")
    sys.exit(1)

num = sys.argv[1]

input_path = Path(f"temp/04_temp/labels{num}_filtered.json")
output_path = Path(f"temp/04_temp/labels{num}_filtered_adapted.json")

# All 17 joints in the original YOLO Pose order
ALL_JOINTS = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]

# Joints to keep (by name)
KEEP_JOINTS = [
    "Nose",
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]
KEEP_INDICES = [ALL_JOINTS.index(j) for j in KEEP_JOINTS]

# === READ INPUT FILE ===
with open(input_path, "r") as f:
    data = json.load(f)

# === FILTER KEYPOINTS ===
for image in data.get("images", []):
    for person in image.get("persons", []):
        kpts = person.get("keypoints", [])
        # Keep only selected joints
        filtered_kpts = [kpts[i] for i in KEEP_INDICES if i < len(kpts)]
        person["keypoints"] = filtered_kpts

# === WRITE OUTPUT FILE ===
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ File saved to: {output_path}")
