"""
remove incompatible keypoints and reorder keypoints in the mocap json file

USAGE:
> python 04_adapt_mocap.py 

"""

import json
from pathlib import Path

# === INPUT / OUTPUT FILES ===
input_path = Path("temp/03_temp/03_final_mocap.json")
output_path = Path("temp/04_temp/04_adapted_final_mocap.json")

# === Original joint list (order in the file) ===
JOINTS_ORIG = [
    'Hips', 'Spine', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LHand',
    'RShoulder', 'RElbow', 'RHand',
    'LHip', 'LKnee', 'LAnkle', 'LFoot',
    'RHip', 'RKnee', 'RAnkle', 'RFoot'
]

# === Joints to remove ===
REMOVE = {'Hips', 'Spine', 'Neck', 'LFoot', 'RFoot'}

# === New desired joint order ===
JOINTS_NEW_ORDER = [
    'Head',
    'LShoulder', 'RShoulder',
    'LElbow', 'RElbow',
    'LHand', 'RHand',
    'LHip', 'RHip',
    'LKnee', 'RKnee',
    'LAnkle', 'RAnkle'
]

# === Create a mapping (joint -> original index) ===
idx_map = {name: i for i, name in enumerate(JOINTS_ORIG)}

# === Load JSON file ===
with open(input_path, 'r') as f:
    data = json.load(f)

# === Create a new filtered dictionary ===
filtered_data = {}

for frame, coords in data.items():
    new_frame = []
    for joint in JOINTS_NEW_ORDER:
        if joint in idx_map and joint not in REMOVE:
            new_frame.append(coords[idx_map[joint]])
    filtered_data[frame] = new_frame

# === Save output ===
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"File saved to: {output_path}")
