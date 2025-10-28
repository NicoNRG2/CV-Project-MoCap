"""
Removes selected joint coordinates (by index) from each frame in a JSON file and saves the filtered result to a new JSON file.

USAGE:
> python3 03_adapt_skeleton.py

"""

import json
from pathlib import Path


def filter_coords(coords, remove_idx):
    # Removes coordinates at the specified indices.
    return [c for i, c in enumerate(coords) if i not in remove_idx]


def main():
    # Fixed paths
    input_file = "temp/03_temp/03_selected_keypoints.json"
    output_file = "temp/03_temp/03_selected_keypoints_adapted_joints.json"

    # Indices to remove (0-based → removes 2,4,7,10,12,15)
    to_remove = [1, 3, 6, 9, 11, 14]

    # Reads the input JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Applies the filter to each frame
    filtered_data = {}
    for frame, coords in data.items():
        filtered_data[frame] = filter_coords(coords, to_remove)

    # Creates the output directory if it does not exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Saves the result to the JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered {len(to_remove)} points. Result saved to {output_file}")


if __name__ == "__main__":
    main()
