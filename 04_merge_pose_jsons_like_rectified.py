#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def load_yolo_pose_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def pick_person(persons, target_id=None):
    """
    Sceglie la persona:
      - se target_id è dato, usa quella con id==target_id (se presente)
      - altrimenti prende quella con score più alto
    """
    if not persons:
        return None
    if target_id is not None:
        for p in persons:
            if p.get("id") == target_id:
                return p
    # fallback: max score
    return max(persons, key=lambda p: p.get("score", 0.0))

def flatten_keypoints(kpts_xyz, conf_threshold=0.5):
    """
    Converte da [[x,y,conf], ...] a [x,y,v, x,y,v, ...] con v in {0,2}
    v=2 se conf >= threshold, altrimenti v=1
    """
    flat = []
    for trip in kpts_xyz:
        # gestisci keypoints incompleti con robustezza
        if trip is None or len(trip) < 2:
            flat.extend([0.0, 0.0, 0])   # “missing”
            continue
        x = float(trip[0])
        y = float(trip[1])
        c = float(trip[2]) if len(trip) >= 3 else 0.0
        v = 2 if c >= conf_threshold else 1
        flat.extend([x, y, v])
    return flat

def merge_inputs(inputs, out_path, person_id=2, conf_threshold=0.5, dedup=True):
    images = []
    annotations = []
    seen_files = set()
    img_id = 1
    ann_id = 1

    for in_path in inputs:
        data = load_yolo_pose_json(Path(in_path))
        for img in data.get("images", []):
            file_name = img.get("file")
            if not file_name:
                continue
            if dedup and file_name in seen_files:
                # già aggiunto da un altro JSON
                continue

            persons = img.get("persons", [])
            person = pick_person(persons, target_id=person_id)
            if not person:
                continue

            kpts_xyz = person.get("keypoints", [])
            flat = flatten_keypoints(kpts_xyz, conf_threshold=conf_threshold)

            # Costruisci record minimal compatibile con 02_triangulation.py
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": img.get("width"),
                "height": img.get("height"),
            })
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "keypoints": flat,
                # opzionali, NON necessari per la triangolazione:
                # "num_keypoints": len(kpts_xyz),
                # "bbox": person.get("bbox"),
            })

            seen_files.add(file_name)
            img_id += 1
            ann_id += 1

    merged = {
        "images": images,
        "annotations": annotations
        # "categories": [...]  # non necessario per lo script di triangolazione
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged COCO-like annotations to: {out_path}")
    print(f"  images: {len(images)}  annotations: {len(annotations)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge 4 YOLO-Pose jsons into a single COCO-like json for triangulation.")
    ap.add_argument("inputs", nargs="+", help="Input json files (e.g., cam_2.json cam_5.json cam_8.json cam_13.json)")
    ap.add_argument("-o", "--out", default="_annotations.coco.rectified.json",
                    help="Output json path (default: _annotations.coco.rectified.json)")
    ap.add_argument("--person-id", type=int, default=2, help="Keep only this person id (default: 2)")
    ap.add_argument("--conf-th", type=float, default=0.5, help="Confidence threshold to set visibility v=2 (default: 0.5)")
    ap.add_argument("--no-dedup", action="store_true", help="Do not deduplicate by file_name")
    args = ap.parse_args()

    merge_inputs(
        inputs=args.inputs,
        out_path=args.out,
        person_id=args.person_id,
        conf_threshold=args.conf_th,
        dedup=(not args.no_dedup),
    )

# python 04_merge_pose_jsons_like_rectified.py 04_temp/labels2_filtered_adapted.json 04_temp/labels5_filtered_adapted.json 04_temp/labels8_filtered_adapted.json 04_temp/labels13_filtered_adapted.json --out 04_annotations_yolo.json