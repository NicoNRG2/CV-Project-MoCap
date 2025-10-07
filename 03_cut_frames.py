#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def cut_frames(input_path, output_path, start_frame, end_frame):
    """Cut frames in the range [start_frame, end_frame] (inclusive)."""
    with open(input_path, "r") as f:
        data = json.load(f)

    # Build new dict with only selected frames
    new_data = {}
    for i in range(start_frame, end_frame + 1):
        key = f"frame_{i}"
        if key in data:
            new_data[key] = data[key]

    print(f"Selected {len(new_data)} frames out of {len(data)}")

    # Save result
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Saved cut frames to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut frames from a JSON file by index range.")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--start", type=int, required=True, help="Start frame number (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End frame number (inclusive)")
    args = parser.parse_args()

    cut_frames(args.input, args.output, args.start, args.end)

# python 03_cut_frames.py --input keypoints_mocap.json --output selected_keypoints.json --start 980 --end 1372