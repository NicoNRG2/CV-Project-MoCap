#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv11-Pose batch inferencer con export JSON e tracking semplice per mantenere lo stesso person ID tra frame.

Uso:
  python yolo11_pose_to_json.py \
    --images /path/alla/cartella_immagini \
    --output /path/output_labels.json \
    --weights yolo11l-pose.pt \
    --imgsz 1280 \
    --conf 0.25

Opzioni:
  --no-track      Disattiva il tracking per gli ID persistenti
  --device cuda:0 Usa la GPU se disponibile (altrimenti "cpu")
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ------------------------- Util -------------------------
def iou_xyxy(a, b) -> float:
    """
    IoU tra due bbox in formato [x1, y1, x2, y2].
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    return inter / (areaA + areaB - inter + 1e-9)


def greedy_match(prev_boxes: List[List[float]],
                 cur_boxes: List[List[float]],
                 iou_thr: float = 0.4) -> List[Tuple[int, int]]:
    """
    Abbina in modo greedy i box di frame t-1 con quelli di frame t usando IoU.
    Ritorna una lista di tuple (idx_prev, idx_cur).
    """
    matches = []
    if not prev_boxes or not cur_boxes:
        return matches

    used_prev = set()
    used_cur = set()

    # Calcola tutte le coppie con IoU
    pairs = []
    for i, pb in enumerate(prev_boxes):
        for j, cb in enumerate(cur_boxes):
            iou = iou_xyxy(pb, cb)
            if iou >= iou_thr:
                pairs.append((iou, i, j))

    # Ordina per IoU desc e prendi greedy
    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in used_prev or j in used_cur:
            continue
        used_prev.add(i)
        used_cur.add(j)
        matches.append((i, j))

    return matches


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Cartella contenente le immagini")
    ap.add_argument("--output", required=True, help="Percorso del file JSON di output")
    ap.add_argument("--weights", default="yolo11l-pose.pt", help="Peso YOLO pose (es. yolo11l-pose.pt)")
    ap.add_argument("--imgsz", type=int, default=1280, help="Dimensione di input (lato lungo)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default=None, help='Es. "cuda:0" o "cpu" (default: auto)')
    ap.add_argument("--no-track", action="store_true", help="Disattiva tracking (ID non persistenti)")
    args = ap.parse_args()

    img_dir = Path(args.images)
    assert img_dir.is_dir(), f"Cartella non trovata: {img_dir}"

    # Raccogli immagini (ordina alfabeticamente -> utile sulle sequenze)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(str(img_dir / ext)))
    img_paths = sorted(img_paths)
    if not img_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata in {img_dir}")

    # Carica modello
    model = YOLO(args.weights)

    # Strutture per tracking
    tracking_enabled = not args.no_track
    next_track_id = 1
    prev_boxes = []
    prev_ids = []

    # JSON di output
    output = {
        "meta": {
            "model": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "tracking": tracking_enabled,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "images": []
    }

    # Inference loop
    for idx, img_path in enumerate(tqdm(img_paths, desc="Processing")):
        # Se vuoi un print manuale dei progressi, mettilo qui:
        # print(f"[DEBUG] Processando immagine {idx+1}/{len(img_paths)}: {img_path}")

        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )

        if len(results) == 0:
            # Non dovrebbe succedere, ma gestiamo ugualmente
            output["images"].append({
                "file": os.path.basename(img_path),
                "width": None,
                "height": None,
                "persons": []
            })
            prev_boxes, prev_ids = [], []
            continue

        res = results[0]
        H, W = res.orig_shape
        persons = []

        # Estrai bbox, conf e keypoints
        # boxes.xyxy: (N,4) — boxes.conf: (N,) — keypoints.data: (N,K,3) con (x,y,vis)
        bboxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        kpts = res.keypoints.data.cpu().numpy().tolist() if res.keypoints is not None else []

        # Allinea lunghezze
        n = min(len(bboxes), len(scores), len(kpts))

        cur_boxes = bboxes[:n]
        cur_scores = scores[:n]
        cur_kpts = kpts[:n]

        # Tracking semplice via IoU
        cur_ids = [None] * n
        if tracking_enabled and prev_boxes:
            matches = greedy_match(prev_boxes, cur_boxes, iou_thr=0.4)
            for i_prev, j_cur in matches:
                cur_ids[j_cur] = prev_ids[i_prev]

        # Assegna nuovi ID ai non abbinati
        if tracking_enabled:
            for j in range(n):
                if cur_ids[j] is None:
                    cur_ids[j] = next_track_id
                    next_track_id += 1
        else:
            # Se tracking off, niente ID persistenti
            cur_ids = [None] * n

        # Costruisci la lista "persons"
        for j in range(n):
            persons.append({
                "id": cur_ids[j],
                "score": float(cur_scores[j]),
                "bbox": [float(x) for x in cur_boxes[j]],  # [x1,y1,x2,y2]
                "keypoints": [[float(x), float(y), float(v)] for (x, y, v) in cur_kpts[j]]
            })

        # Append per immagine
        output["images"].append({
            "file": os.path.basename(img_path),
            "width": int(W),
            "height": int(H),
            "persons": persons
        })

        # Aggiorna stato tracking
        prev_boxes = cur_boxes
        prev_ids = cur_ids

    # Salva JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. JSON scritto in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
