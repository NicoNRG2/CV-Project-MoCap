#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confronto frame-by-frame tra:
- MoCap JSON (frames al top-level): dati_tuta_8p3_renamed.json
- Triangolazione JSON (frames al top-level OPPURE sotto una chiave, es. 'skeleton_3d')

Output:
- summary.csv con una riga per frame (MPJPE, median, max, MAE per asse, MSE, RMSE)
- (opzionale) CSV per-frame con errori per joint

Uso tipico (nuovo JSON: frames al top-level):
python compare_mocap_vs_triang.py \
  --mocap /mnt/data/dati_tuta_8p3_renamed.json \
  --triang /mnt/data/triangulated_3d_skeleton.json \
  --out out_compare \
  --per-frame-csv \
  --align rigid        # none | rigid | similarity

Uso retro-compatibile (vecchio JSON con chiave):
  ... --triang-key skeleton_3d
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def load_frames_mocap(path: str) -> Dict[str, List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Atteso: { "frame_0001": [[x,y,z], ...], ... }
    return data

def _looks_like_frames_mapping(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    # euristica: almeno una chiave che inizia con "frame_" e valore lista di [x,y,z]
    for k, v in obj.items():
        if isinstance(k, str) and k.startswith("frame_") and isinstance(v, list):
            return True
    return False

def load_frames_triang(path: str, key: Optional[str] = None) -> Dict[str, List[List[float]]]:
    """
    Carica i frame di triangolazione:
    - Se key è fornita e presente -> usa data[key]
    - Altrimenti prova ad usare i frame al top-level
    - Se esiste una singola chiave che contiene i frame, usa quella
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Se l'utente ha indicato una chiave, prova ad usarla
    if key:
        if key in data and _looks_like_frames_mapping(data[key]):
            return data[key]
        else:
            print(f"[INFO] Chiave '{key}' non trovata o non valida, provo i frame al top-level...")

    # 2) Top-level?
    if _looks_like_frames_mapping(data):
        return data

    # 3) Un solo livello contenitore?
    if isinstance(data, dict) and len(data) == 1:
        only_key, only_val = next(iter(data.items()))
        if _looks_like_frames_mapping(only_val):
            print(f"[INFO] Uso la chiave auto-rilevata '{only_key}'.")
            return only_val

    raise KeyError("Formato JSON della triangolazione non riconosciuto: non trovo un mapping 'frame_xxxx'.")

def kabsch(P: np.ndarray, Q: np.ndarray, allow_scale: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    assert P.shape == Q.shape and P.shape[1] == 3
    Pc = P.mean(axis=0, keepdims=True)
    Qc = Q.mean(axis=0, keepdims=True)
    P0 = P - Pc
    Q0 = Q - Qc

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if allow_scale:
        varP = (P0**2).sum()
        s = (S.sum()) / varP if varP > 0 else 1.0
    else:
        s = 1.0

    t = (Qc.T - s * R @ Pc.T).reshape(3)
    return R, t, s

def compute_errors(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    diffs = A - B
    dists = np.linalg.norm(diffs, axis=1)
    mae_xyz = np.mean(np.abs(diffs), axis=0)
    mse = float(np.mean(dists**2))
    rmse = float(np.sqrt(mse))
    return {
        "mpjpe": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "max": float(np.max(dists)),
        "mae_x": float(mae_xyz[0]),
        "mae_y": float(mae_xyz[1]),
        "mae_z": float(mae_xyz[2]),
        "mse": mse,
        "rmse": rmse,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mocap", required=True, help="Path al JSON MoCap (frames al top-level).")
    ap.add_argument("--triang", required=True, help="Path al JSON triangolazione.")
    ap.add_argument("--triang-key", default=None, help="(Opzionale) Chiave che contiene i frame nel JSON triangolazione.")
    ap.add_argument("--out", default="out_compare", help="Cartella output.")
    ap.add_argument("--per-frame-csv", action="store_true", help="Scrive CSV per ogni frame con errori per joint.")
    ap.add_argument("--align", choices=["none", "rigid", "similarity"], default="none",
                   help="Allineamento: none, rigid (Kabsch), similarity (rot+trasl+scala).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = out_dir / "per_frame_errors"
    if args.per_frame_csv:
        per_frame_dir.mkdir(parents=True, exist_ok=True)

    mocap = load_frames_mocap(args.mocap)
    triang = load_frames_triang(args.triang, args.triang_key)

    mocap_frames = set(mocap.keys())
    triang_frames = set(triang.keys())
    common = sorted(mocap_frames & triang_frames)

    missing_in_triang = sorted(mocap_frames - triang_frames)
    missing_in_mocap = sorted(triang_frames - mocap_frames)

    print(f"[INFO] Frame in comune: {len(common)}")
    if missing_in_triang:
        print(f"[WARN] Frame presenti in MoCap ma non in Triangolazione: {len(missing_in_triang)} (es. {missing_in_triang[:5]})")
    if missing_in_mocap:
        print(f"[WARN] Frame presenti in Triangolazione ma non in MoCap: {len(missing_in_mocap)} (es. {missing_in_mocap[:5]})")

    rows = []
    for fr in common:
        A = np.asarray(mocap[fr], dtype=np.float64)
        B = np.asarray(triang[fr], dtype=np.float64)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3:
            print(f"[SKIP] {fr}: formato inatteso (shape A={A.shape}, B={B.shape})")
            continue
        if A.shape[0] != B.shape[0]:
            print(f"[WARN] {fr}: joint count diverso (A={A.shape[0]}, B={B.shape[0]}). Confronto sugli indici comuni.")
            J = min(A.shape[0], B.shape[0])
            A = A[:J]
            B = B[:J]

        A_aligned = A
        B_ref = B
        if args.align != "none":
            allow_scale = (args.align == "similarity")
            R, t, s = kabsch(A, B, allow_scale=allow_scale)
            A_aligned = (s * (A @ R.T)) + t

        diffs = A_aligned - B_ref
        dists = np.linalg.norm(diffs, axis=1)

        metrics = compute_errors(A_aligned, B_ref)
        metrics.update({
            "frame": fr,
            "num_joints": int(A_aligned.shape[0]),
            "align": args.align
        })
        rows.append(metrics)

        if args.per_frame_csv:
            df = pd.DataFrame({
                "joint_idx": np.arange(A_aligned.shape[0]),
                "mocap_x": A_aligned[:,0], "mocap_y": A_aligned[:,1], "mocap_z": A_aligned[:,2],
                "tri_x":  B_ref[:,0],      "tri_y":  B_ref[:,1],      "tri_z":  B_ref[:,2],
                "dx": diffs[:,0], "dy": diffs[:,1], "dz": diffs[:,2],
                "err_norm": dists
            })
            df.to_csv(per_frame_dir / f"{fr}.csv", index=False)

    if rows:
        summary = pd.DataFrame(rows).sort_values("frame")
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"[DONE] Frame analizzati: {len(summary)}")
        print(f"[STATS] MPJPE medio su tutti i frame: {summary['mpjpe'].mean():.3f}")
        print(f"[STATS] MPJPE mediano su tutti i frame: {summary['mpjpe'].median():.3f}")
        print(f"[STATS] MSE medio su tutti i frame: {summary['mse'].mean():.3f}")
        print(f"[STATS] RMSE medio su tutti i frame: {summary['rmse'].mean():.3f}")
        print(f"[OUT]   {out_dir/'summary.csv'}")
        if args.per_frame_csv:
            print(f"[OUT]   CSV per-frame in: {out_dir/'per_frame_errors'}")
    else:
        print("[DONE] Nessun frame in comune analizzabile.")

if __name__ == "__main__":
    main()

# python 03_step3compare.py --mocap final_mocap.json --triang final_triangulation.json --align similarity