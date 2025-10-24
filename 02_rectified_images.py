#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import os
import glob
import re
from pathlib import Path

# =========================
# Utilità calibrazione
# =========================
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

# Cache per mappe di rettifica (per camera e risoluzione)
_rectify_map_cache = {}  # key: (cam_idx, width, height) -> (map_x, map_y)

def get_rectify_maps(cam_idx, width, height, calib_path):
    """
    Crea (o recupera dalla cache) le mappe di remap per una data camera
    e una data risoluzione. Mantiene la stessa logica del tuo script video.
    """
    key = (cam_idx, width, height)
    if key in _rectify_map_cache:
        return _rectify_map_cache[key]

    mtx, dist = load_calibration(calib_path)

    # Griglia dei pixel (come nello script originale)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)

    # Stessa operazione di rettifica: undistortPoints con P=mtx
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]

    _rectify_map_cache[key] = (map_x, map_y)
    return map_x, map_y

# =========================
# Rettifica immagine singola
# =========================
def rectify_image(image_path, output_path, cam_idx, calib_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Immagine non leggibile: {image_path}")
        return False

    height, width = img.shape[:2]

    try:
        map_x, map_y = get_rectify_maps(cam_idx, width, height, calib_path)
    except FileNotFoundError:
        print(f"[ERRORE] File calibrazione non trovato: {calib_path}")
        return False
    except KeyError as e:
        print(f"[ERRORE] Chiavi mancanti nella calibrazione {calib_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERRORE] Problema nel calcolo mappe per {image_path}: {e}")
        return False

    # Stessa logica di remap dello script video
    rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Salva con lo stesso nome del file originale
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(output_path, rectified)
    if not ok:
        print(f"[ERRORE] Salvataggio fallito: {output_path}")
        return False

    return True

# =========================
# Main: batch su cartella images
# =========================
def main():
    input_dir = "train"
    output_dir = "images_rectified"
    os.makedirs(output_dir, exist_ok=True)

    # Pattern filename:
    # es. out8_frame_0004_png.rf.345f188c7aba71764ede395be914924c
    #    ^^^^ cam      ^^^^^^^ frame ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ hash
    # Regex robusta: inizio stringa, cattura numero camera dopo 'out',
    # poi 'frame_####', poi 'png.rf.' e hash
    name_re = re.compile(r'^out(?P<cam>\d+)_frame_(?P<frame>\d+)_png\.rf\..+$')

    # Prendi tutti i file nella cartella images (jpg/png ecc.)
    image_paths = sorted(
        [p for p in glob.glob(os.path.join(input_dir, '*')) if os.path.isfile(p)]
    )

    if not image_paths:
        print(f"[INFO] Nessuna immagine trovata in: {input_dir}")
        return

    processed = 0
    skipped = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        m = name_re.match(fname)
        if not m:
            # Non rispetta il pattern atteso: salta
            skipped += 1
            # Facoltativo: stampa avviso una volta ogni tanto
            if skipped <= 10:
                print(f"[WARN] Nome non conforme, salto: {fname}")
            continue

        cam_idx = m.group('cam')  # '2','5','8','13', ecc.
        calib_path = os.path.join("camera_data", f"cam_{cam_idx}", "calib", "camera_calib.json")

        out_path = os.path.join(output_dir, fname)

        ok = rectify_image(img_path, out_path, cam_idx, calib_path)
        if ok:
            processed += 1
            if processed % 50 == 0:
                print(f"[INFO] Rettificate {processed} immagini...")
        else:
            skipped += 1

    print(f"[DONE] Completato. Rettificate: {processed}, saltate: {skipped}")

if __name__ == "__main__":
    main()
