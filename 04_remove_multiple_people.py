"""
Script to filter people with a specific ID from an annotation JSON file.

USAGE:
> python 04_remove_multiple_people.py --input temp/04_temp/labels2.json --output temp/04_temp/labels2_filtered.json --keep_id [id_number]
    n.b. Before this comands, go to temp/04_temp and, for each camera (cam2, cam5, cam8, cam13), check the player's id (black tracksuit): put it as [id_number].
"""

import json
from pathlib import Path
import argparse

# === ARGUMENT PARSER ===
parser = argparse.ArgumentParser(description="Filter people with a specific ID from an annotation JSON file.")
parser.add_argument("--input", type=Path, help="Path to the input JSON file")
parser.add_argument("--output", type=Path, help="Path to the output JSON file")
parser.add_argument("--keep_id", type=int, help="Person ID to keep (e.g., 1 or 2)")

args = parser.parse_args()

# === LOAD JSON ===
with open(args.input, "r") as f:
    data = json.load(f)

# === FILTER PEOPLE WITH THE SPECIFIED ID ===
filtered_images = []
for img in data["images"]:
    persons_filtered = [p for p in img["persons"] if p.get("id") == args.keep_id]
    if persons_filtered:  # add only images that contain at least one person with the selected ID
        filtered_images.append({
            "file": img["file"],
            "width": img["width"],
            "height": img["height"],
            "persons": persons_filtered
        })

# === CREATE THE NEW JSON ===
filtered_data = {
    "meta": data.get("meta", {}),
    "images": filtered_images
}

# === SAVE TO FILE ===
with open(args.output, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"File saved to: {args.output}")
print(f"Total images in the original file: {len(data['images'])}")
print(f"Images with id={args.keep_id}: {len(filtered_images)}")
