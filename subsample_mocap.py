#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subcampiona un file JSON di motion capture partendo da frame_980
e poi prendendo un frame ogni 8.3 frame usando un accumulatore decimale.

Uso:
  python subsample_mocap.py -i dati_tuta_filtrati.json -o dati_tuta_8p3.json --start 980 --step 8.3
"""

import argparse
import json
from decimal import Decimal, getcontext

def parse_args():
    p = argparse.ArgumentParser(description="Subcampionamento con accumulatore decimale.")
    p.add_argument("-i", "--input",  required=True, help="Percorso del file JSON in input.")
    p.add_argument("-o", "--output", required=True, help="Percorso del file JSON in output.")
    p.add_argument("--start", type=int, default=980, help="Frame di partenza (default: 980).")
    p.add_argument("--step",  type=str, default="8.3", help="Passo in frame, decimale (default: '8.3').")
    return p.parse_args()

def load_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Estrae indici numerici: "frame_XXXX" -> XXXX
    idx = sorted(int(k.split("_")[1]) for k in data.keys())
    return data, idx

def next_existing_index(sorted_indices, target):
    """Ritorna il primo indice in sorted_indices >= target, oppure None se non esiste."""
    # Ricerca binaria semplice
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
    Usa un accumulatore decimale:
      - parte da start_index (incluso)
      - ad ogni iterazione aggiunge 'step_decimal'
      - arrotonda all'intero più vicino
      - se l'arrotondamento produce un indice <= ultimo selezionato,
        forza il target al (ultimo + 1) per evitare duplicati
      - sceglie il primo frame ESISTENTE >= target
    """
    selected = []
    if not sorted_indices:
        return selected

    # Se il frame di start non esiste, usa il primo disponibile >= start
    start_real = next_existing_index(sorted_indices, start_index)
    if start_real is None:
        return selected

    selected.append(start_real)
    last_int = start_real

    acc = Decimal(start_real)
    step = Decimal(step_decimal)

    while True:
        acc += step
        cand = int(acc.to_integral_value(rounding=getcontext().rounding))  # arrotonda al più vicino
        if cand <= last_int:
            cand = last_int + 1

        nxt = next_existing_index(sorted_indices, cand)
        if nxt is None:
            break

        selected.append(nxt)
        last_int = nxt

    return selected

def build_output(original_data, selected_indices):
    # Mantiene l'ordine crescente dei frame
    out = {}
    for i in selected_indices:
        key = f"frame_{i}"
        if key in original_data:
            out[key] = original_data[key]
    return out

def main():
    args = parse_args()

    # precisione sufficiente per accumulare molti step senza drift
    getcontext().prec = 28

    data, indices = load_frames(args.input)
    selected_indices = subsample_indices(indices, args.start, args.step)
    out = build_output(data, selected_indices)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Frames totali: {len(indices)} | Selezionati: {len(selected_indices)}")
    print(f"Primo: frame_{selected_indices[0]} | Ultimo: frame_{selected_indices[-1]}")

if __name__ == "__main__":
    main()


# python subsample_mocap.py -i dati_tuta_filtrati.json -o dati_tuta_8p3.json --start 980 --step 8.3