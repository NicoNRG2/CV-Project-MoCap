"""
DISEGNA keypoints rettificati vs riproiettati NELLO STESSO PLOT"""

#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_coco_annotations(path):
    """
    Carica il JSON COCO e ritorna:
      - annots: lista di dizionari 'annotation'
      - images: dict image_id → dizionario 'image'
    """
    data = json.load(open(path, 'r'))
    annots = data['annotations']
    images = {img['id']: img for img in data['images']}
    return annots, images

def get_keypoints_for_image(annots, image_id):
    """
    Cerca nelle annots l'unica annotation con image_id, estrae keypoints Nx3
    Ritorna array (N_joints, 3), oppure None se non trovato.
    """
    for ann in annots:
        if ann['image_id'] == image_id:
            kp = np.array(ann['keypoints'], dtype=float).reshape(-1, 3)
            return kp
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Confronta keypoints rettificati vs riproiettati per una stessa image_id")
    parser.add_argument("image_id", type=int, help="ID dell'immagine da plottare")
    parser.add_argument("--rectified", default="_annotations.coco.rectified.json",
                        help="Path al file COCO rettificato")
    parser.add_argument("--reproj", default="reprojected_annotations.json",
                        help="Path al file COCO con keypoints riproiettati")
    args = parser.parse_args()

    # 1) Carica le annotazioni
    rect_annots, rect_images = load_coco_annotations(args.rectified)
    reproj_annots, reproj_images = load_coco_annotations(args.reproj)

    # 2) Estrai keypoints
    kp_rect = get_keypoints_for_image(rect_annots, args.image_id)
    kp_reproj = get_keypoints_for_image(reproj_annots, args.image_id)

    if kp_rect is None:
        print(f"[ERRORE] Nessuna annotation rettificata per image_id {args.image_id}")
        return
    if kp_reproj is None:
        print(f"[ERRORE] Nessuna annotation riproiettata per image_id {args.image_id}")
        return

    # 3) Ricava informazioni immagine per titolo e dimensioni (opzionale)
    img_info = rect_images.get(args.image_id, {})
    fname = img_info.get('file_name', None)

    # 4) Prepara i dati XY
    xy_rect   = kp_rect[:,:2]
    xy_reproj = kp_reproj[:,:2]

    # 5) Plot
    plt.figure(figsize=(6,6))
    plt.scatter(xy_rect[:,0],   xy_rect[:,1],   c='g', marker='o', label='GT rettificati')
    plt.scatter(xy_reproj[:,0], xy_reproj[:,1], c='r', marker='x', label='Riproiettati')
    plt.legend(loc='upper right')
    plt.title(f"Keypoints confronto per image_id {args.image_id}" + (f"\n{fname}" if fname else ""))
    plt.gca().invert_yaxis()  # coord. immagine (0,0) in alto a sinistra
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
