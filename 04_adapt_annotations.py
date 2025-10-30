"""
This script filters a COCO-format JSON file to retain only specified keypoints.

Usage:
python 04_adapt_annotations.py --input temp/02_temp/02_annotations.coco.rectified.json --output temp/04_temp/04_original_annotations_filtered.json
"""

#!/usr/bin/env python3
import json
import argparse
from copy import deepcopy

# joints presenti nel file di input, nell'ordine in cui compaiono in "categories[...]['keypoints']"
INPUT_JOINTS = [
    "Hips",
    "RHip",
    "RKnee",
    "RAnkle",
    "RFoot",
    "LHip",
    "LKnee",
    "LAnkle",
    "LFoot",
    "Spine",
    "Neck",
    "Head",
    "RShoulder",
    "RElbow",
    "RHand",
    "LShoulder",
    "LElbow",
    "LHand",
]

# joint da tenere e loro ordine finale richiesto
OUTPUT_JOINTS = [
    "Head",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LHand", "RHand",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
]

# joint da scartare completamente
DROP_JOINTS = {"Hips", "Spine", "Neck", "LFoot", "RFoot"}

def slice_keypoints(flat_kps, idx):
    """
    flat_kps = [x0,y0,v0,x1,y1,v1,...]
    idx = indice del joint (0-based) nell'ordine INPUT_JOINTS
    ritorna (x,y,v) di quel joint
    """
    base = idx * 3
    return flat_kps[base:base+3]

def build_reordered_keypoints(flat_kps):
    """
    Prende i keypoints originali (tutti) e restituisce
    una nuova lista piatta coi soli joint richiesti
    in OUTPUT_JOINTS, nell'ordine richiesto.
    """
    # mappa nome_joint -> indice nell'input
    name_to_input_idx = {name: i for i, name in enumerate(INPUT_JOINTS)}

    new_flat = []
    for joint_name in OUTPUT_JOINTS:
        if joint_name not in name_to_input_idx:
            # se manca completamente nell'input metto (0,0,0)
            new_flat.extend([0.0, 0.0, 0])
            continue

        i_in = name_to_input_idx[joint_name]
        x, y, v = slice_keypoints(flat_kps, i_in)

        # se per qualche motivo il joint è nella DROP_JOINTS lo annullo
        if joint_name in DROP_JOINTS:
            x, y, v = 0.0, 0.0, 0

        new_flat.extend([x, y, v])

    return new_flat

def main():
    parser = argparse.ArgumentParser(
        description="Filtra e riordina i keypoints delle annotazioni COCO-style."
    )
    parser.add_argument("--input", required=True, help="file json di input")
    parser.add_argument("--output", required=True, help="file json di output")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # --- aggiorna categories ---
    new_data = deepcopy(data)

    # troviamo la category 'person' (o la prima con 'keypoints')
    for cat in new_data.get("categories", []):
        if "keypoints" in cat and len(cat["keypoints"]) > 0:
            # sostituiamo la lista dei keypoints con l'ordine nuovo
            cat["keypoints"] = OUTPUT_JOINTS[:]

            # opzionale: ricrea uno skeleton minimale coerente col nuovo ordine
            # qui facciamo qualcosa di semplice tipo braccia/gambe
            # ogni coppia è (idx1, idx2) 1-based come COCO
            SKELETON_DEF = [
                ("Head", "LShoulder"),
                ("Head", "RShoulder"),
                ("LShoulder", "LElbow"),
                ("LElbow", "LHand"),
                ("RShoulder", "RElbow"),
                ("RElbow", "RHand"),
                ("LHip", "LKnee"),
                ("LKnee", "LAnkle"),
                ("RHip", "RKnee"),
                ("RKnee", "RAnkle"),
                ("LHip", "RHip"),
            ]

            # crea mappa nome->1-based index nel nuovo ordine
            out_index = {name: i+1 for i, name in enumerate(OUTPUT_JOINTS)}

            new_skel = []
            for a, b in SKELETON_DEF:
                if a in out_index and b in out_index:
                    new_skel.append([out_index[a], out_index[b]])

            cat["skeleton"] = new_skel
            break

    # --- aggiorna annotations ---
    for ann in new_data.get("annotations", []):
        if "keypoints" not in ann:
            continue

        old_kps = ann["keypoints"]
        new_kps = build_reordered_keypoints(old_kps)
        ann["keypoints"] = new_kps

        # opzionale: aggiorna num_keypoints se presente
        if "num_keypoints" in ann:
            # conteggia quanti joint hanno visibility>0
            vis_count = 0
            for i in range(0, len(new_kps), 3):
                v = new_kps[i+2]
                if v is not None and v != 0:
                    vis_count += 1
            ann["num_keypoints"] = vis_count

    # scrivi output
    with open(args.output, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Fatto. Salvato in {args.output}")

if __name__ == "__main__":
    main()
