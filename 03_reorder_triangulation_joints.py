#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

# Ordine originale dei joint
ORIGINAL_ORDER = [
    "Hips",
    "RHip","RKnee","RAnkle","RFoot",
    "LHip","LKnee","LAnkle","LFoot",
    "Spine","Neck","Head",
    "RShoulder","RElbow","RHand",
    "LShoulder","LElbow","LHand"
]

# Nuovo ordine desiderato
NEW_ORDER = [
    "Hips","Spine","Neck","Head",
    "LShoulder","LElbow","LHand",
    "RShoulder","RElbow","RHand",
    "LHip","LKnee","LAnkle","LFoot",
    "RHip","RKnee","RAnkle","RFoot"
]

# Mappa: nuovo indice -> vecchio indice
index_map = [ORIGINAL_ORDER.index(j) for j in NEW_ORDER]

def reorder_file(input_path: Path, output_path: Path):
    # Carica il JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Ignora il campo "skeleton_3d" se presente
    if "skeleton_3d" in data:
        data = data["skeleton_3d"]

    reordered = {}
    for frame, coords in data.items():
        if not isinstance(coords, list):
            raise ValueError(f"Frame {frame} non contiene una lista di joint.")
        if len(coords) != len(ORIGINAL_ORDER):
            raise ValueError(f"Frame {frame}: attesi {len(ORIGINAL_ORDER)} joint, trovati {len(coords)}.")
        reordered[frame] = [coords[i] for i in index_map]

    # Scrivi il risultato
    with open(output_path, "w") as f:
        json.dump(reordered, f, indent=2)

    print(f"✅ File salvato in {output_path} ({len(reordered)} frame elaborati)")

def main():
    parser = argparse.ArgumentParser(description="Riordina i joint nei frame di un file JSON.")
    parser.add_argument("input_json", type=Path, help="File JSON di input")
    parser.add_argument("output_json", type=Path, help="File JSON di output")
    args = parser.parse_args()
    reorder_file(args.input_json, args.output_json)

if __name__ == "__main__":
    main()

# python reorder_triangulation_joints.py triangulated_3d_skeleton.json output.json