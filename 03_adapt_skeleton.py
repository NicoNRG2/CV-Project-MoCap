#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def filter_coords(coords, remove_idx):
    """Rimuove le coordinate agli indici specificati."""
    return [c for i, c in enumerate(coords) if i not in remove_idx]

def main():
    # Percorsi fissi
    input_file = "temp/03_temp/03_selected_keypoints.json"
    output_file = "temp/03_temp/03_selected_keypoints_adapted_joints.json"

    # Indici da rimuovere (0-based → rimuove 2,4,7,10,12,15)
    to_remove = [1, 3, 6, 9, 11, 14]

    # Legge il file JSON di input
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Applica il filtro a ogni frame
    filtered_data = {}
    for frame, coords in data.items():
        filtered_data[frame] = filter_coords(coords, to_remove)

    # Crea la directory di output se non esiste
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Salva il risultato su file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Filtrati {len(to_remove)} punti. Risultato salvato in {output_file}")
    

if __name__ == "__main__":
    main()


# rimove 2,4,7,10,12,15 le coordinate dello scheletro in più del mocap
# python 03_adapt_skeleton.py 