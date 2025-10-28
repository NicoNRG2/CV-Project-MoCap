#!/usr/bin/env python3

"""
Cuts a range of frames from a motion capture JSON file,
saves only the selected frames to a new JSON, and reports missing frames if any.

USAGE:
> python3 03_cut_frames.py

"""

import json
from pathlib import Path

def cut_frames(input_path, output_path, start_frame, end_frame):
    # Cut frames in the range [start_frame, end_frame] (inclusive).
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    missing = 0
    for i in range(start_frame, end_frame + 1):
        key = f"frame_{i}"
        if key in data:
            new_data[key] = data[key]
        else:
            missing += 1

    print(f"Selected {len(new_data)} frames out of requested {end_frame - start_frame + 1} (missing: {missing})")

    # Create the destination folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved cut frames to {output_path.resolve()}")

if __name__ == "__main__":
    input_path = "03_position_data_mocap.json"
    output_path = "temp/03_temp/03_selected_keypoints.json"
    start = 980
    end = 1372
    cut_frames(input_path, output_path, start, end)
