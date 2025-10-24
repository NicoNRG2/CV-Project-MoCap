#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subcampiona un file JSON di motion capture partendo da frame_980
e poi prendendo un frame ogni 8.3 frame usando un accumulatore decimale.

Uso:
  python 03_subsample_mocap.py
"""

import json
from decimal import Decimal, getcontext
from pathlib import Path


def load_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Estrae indici numerici: "frame_XXXX" -> XXXX
    idx = sorted(int(k.split("_")[1]) for k in data.keys())
    return data, idx


def next_existing_index(sorted_indices, target):
    """Ritorna il primo indice in sorted_indices >= target, oppure None se non esiste."""
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
    out = {}
    for i in selected_indices:
        key = f"frame_{i}"
        if key in original_data:
            out[key] = original_data[key]
    return out


def main():
    # 🔧 Parametri hardcoded
    input_path = "temp/03_temp/03_selected_keypoints_adapted_joints.json"
    output_path = "temp/03_temp/03_selected_keypoints_adapted_joints_48frames.json"
    start = 980
    step = "8.3"

    getcontext().prec = 28  # alta precisione

    data, indices = load_frames(input_path)
    selected_indices = subsample_indices(indices, start, step)
    out = build_output(data, selected_indices)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Frames totali: {len(indices)} | Selezionati: {len(selected_indices)}")
    print(f"   Primo frame: frame_{selected_indices[0]}")
    print(f"   Ultimo frame: frame_{selected_indices[-1]}")
    print(f"💾 Salvato in: {output_path}")


if __name__ == "__main__":
    main()


# python 03_subsample_mocap.py