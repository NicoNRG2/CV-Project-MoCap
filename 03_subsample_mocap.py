"""
Subsamples a motion-capture JSON starting from frame_980, then selects roughly one frame every 8.3 frames
using a Decimal accumulator to avoid drift.
Writes the reduced set to a new JSON and reports basic stats.

USAGE:
> python 03_subsample_mocap.py

"""

import json
from decimal import Decimal, getcontext
from pathlib import Path


def load_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract numeric indices: "frame_XXXX" -> XXXX
    idx = sorted(int(k.split("_")[1]) for k in data.keys())
    return data, idx


def next_existing_index(sorted_indices, target):
    # Returns the first index in sorted_indices >= target, or None if it doesn't exist.
    lo, hi = 0, len(sorted_indices)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_indices[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return sorted_indices[lo] if lo < len(sorted_indices) else None


def subsample_indices(sorted_indices, start_index, step_decimal):
    """
    Uses a decimal accumulator:
      - starts from start_index (inclusive)
      - at each iteration adds 'step_decimal'
      - rounds to the nearest integer
      - if rounding yields an index <= last selected,
        force target to (last + 1) to avoid duplicates
      - picks the first EXISTING frame >= target
    """
    selected = []
    if not sorted_indices:
        return selected

    start_real = next_existing_index(sorted_indices, start_index)
    if start_real is None:
        return selected

    selected.append(start_real)
    last_int = start_real

    acc = Decimal(start_real)
    step = Decimal(step_decimal)

    while True:
        acc += step
        cand = int(acc.to_integral_value(rounding=getcontext().rounding))  # round to nearest
        if cand <= last_int:
            cand = last_int + 1

        nxt = next_existing_index(sorted_indices, cand)
        if nxt is None:
            break

        selected.append(nxt)
        last_int = nxt

    return selected


def build_output(original_data, selected_indices):
    out = {}
    for i in selected_indices:
        key = f"frame_{i}"
        if key in original_data:
            out[key] = original_data[key]
    return out


def main():
    # Hardcoded parameters
    input_path = "temp/03_temp/03_selected_keypoints_adapted_joints.json"
    output_path = "temp/03_temp/03_selected_keypoints_adapted_joints_48frames.json"
    start = 980
    step = "8.3"

    getcontext().prec = 28  # high precision

    data, indices = load_frames(input_path)
    selected_indices = subsample_indices(indices, start, step)
    out = build_output(data, selected_indices)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Total frames: {len(indices)} | Selected: {len(selected_indices)}")
    print(f"First frame: frame_{selected_indices[0]}")
    print(f"Last frame: frame_{selected_indices[-1]}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
