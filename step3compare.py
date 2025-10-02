#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confronto frame-by-frame tra:
- MoCap JSON (frames al top-level): dati_tuta_8p3_renamed.json
- Triangolazione JSON (frames sotto "skeleton_3d"): triangulated_3d_skeleton.json

Output:
- summary.csv con una riga per frame (MPJPE, median, max, MAE per asse, ecc.)
- (opzionale) cartella per-frame con CSV degli errori per joint

Uso:
python compare_mocap_vs_triang.py \
  --mocap /mnt/data/dati_tuta_8p3_renamed.json \
  --triang /mnt/data/triangulated_3d_skeleton.json \
  --triang-key skeleton_3d \
  --out out_compare \
  --per-frame-csv \
  --align rigid        # oppure: none | rigid | similarity
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def load_frames_mocap(path: str) -> Dict[str, List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data: { "frame_0001": [[x,y,z], ...], ... }
    return data

def load_frames_triang(path: str, key: str) -> Dict[str, List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data: { key: { "frame_0001": [[x,y,z], ...], ... }, ... }
    if key not in data:
        raise KeyError(f"Chiave '{key}' non trovata nel file triangolazione.")
    return data[key]

def kabsch(P: np.ndarray, Q: np.ndarray, allow_scale: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Trova (R, t, s) che minimizzano || s*R*P + t - Q ||_F.
    - P, Q: (N,3) punti corrispondenti.
    - allow_scale=True -> stima anche scala s (similarity); altrimenti s=1 (rigid).
    Ritorna: (R, t, s)
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    # Centra
    Pc = P.mean(axis=0, keepdims=True)
    Qc = Q.mean(axis=0, keepdims=True)
    P0 = P - Pc
    Q0 = Q - Qc

    # Scala (opzionale)
    if allow_scale:
        varP = (P0**2).sum()
        # SVD su Q0^T P0
        H = P0.T @ Q0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        s = (S.sum()) / varP if varP > 0 else 1.0
    else:
        H = P0.T @ Q0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        s = 1.0

    t = (Qc.T - s * R @ Pc.T).reshape(3)
    return R, t, s

def compute_errors(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """
    A, B: (J,3) set di punti (stessa joint order).
    Ritorna metriche aggregate + per-asse.
    """
    diffs = A - B
    dists = np.linalg.norm(diffs, axis=1)  # per-joint Euclidee

    mae_xyz = np.mean(np.abs(diffs), axis=0)  # |Δx|, |Δy|, |Δz|
    mse = float(np.mean(dists**2))            # Mean Squared Error
    rmse = float(np.sqrt(mse))                # Root Mean Squared Error

    return {
        "mpjpe": float(np.mean(dists)),       # Mean Per Joint Position Error
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
    ap.add_argument("--triang-key", default="skeleton_3d", help="Chiave che contiene i frame nel JSON triangolazione.")
    ap.add_argument("--out", default="out_compare", help="Cartella output.")
    ap.add_argument("--per-frame-csv", action="store_true", help="Scrive CSV per ogni frame con errori per joint.")
    ap.add_argument("--align", choices=["none", "rigid", "similarity"], default="none",
                   help="Allineamento prima del confronto: none, rigid (Kabsch), similarity (rot+trasl+scala).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = out_dir / "per_frame_errors"
    if args.per_frame_csv:
        per_frame_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carica
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
        A = np.asarray(mocap[fr], dtype=float)        # (J,3)
        B = np.asarray(triang[fr], dtype=float)       # (J,3)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3:
            print(f"[SKIP] {fr}: formato inatteso (shape A={A.shape}, B={B.shape})")
            continue
        if A.shape[0] != B.shape[0]:
            print(f"[WARN] {fr}: joint count diverso (A={A.shape[0]}, B={B.shape[0]}). Confronto solo sugli indici comuni.")
            J = min(A.shape[0], B.shape[0])
            A = A[:J]
            B = B[:J]

        # opzionale: allineamento
        A_aligned = A.copy()
        B_ref = B.copy()
        if args.align != "none":
            allow_scale = (args.align == "similarity")
            # Allineo MoCap -> Triangolazione (per confronto)
            R, t, s = kabsch(A, B, allow_scale=allow_scale)
            A_aligned = (s * (A @ R.T)) + t  # attenzione: (J,3) * (3,3)^T

        # errori per joint
        diffs = A_aligned - B_ref
        dists = np.linalg.norm(diffs, axis=1)

        # metriche aggregate
        metrics = compute_errors(A_aligned, B_ref)
        metrics.update({
            "frame": fr,
            "num_joints": int(A_aligned.shape[0]),
            "align": args.align
        })
        rows.append(metrics)

        # CSV per-frame (opzionale)
        if args.per_frame_csv:
            df = pd.DataFrame({
                "joint_idx": np.arange(A_aligned.shape[0]),
                "mocap_x": A_aligned[:,0], "mocap_y": A_aligned[:,1], "mocap_z": A_aligned[:,2],
                "tri_x":  B_ref[:,0],      "tri_y":  B_ref[:,1],      "tri_z":  B_ref[:,2],
                "dx": diffs[:,0], "dy": diffs[:,1], "dz": diffs[:,2],
                "err_norm": dists
            })
            df.to_csv(per_frame_dir / f"{fr}.csv", index=False)

    # Summary
    if rows:
        summary = pd.DataFrame(rows).sort_values("frame")
        summary.to_csv(out_dir / "summary.csv", index=False)
        # stampa breve
        mpjpe_mean = summary["mpjpe"].mean()
        mpjpe_med = summary["mpjpe"].median()
        # print mse and rmse
        mse_mean = summary["mse"].mean()
        rmse_mean = summary["rmse"].mean()
        print(f"[DONE] Frame analizzati: {len(summary)}")
        print(f"[STATS] MPJPE medio su tutti i frame: {mpjpe_mean:.3f}")
        print(f"[STATS] MPJPE mediano su tutti i frame: {mpjpe_med:.3f}")
        print(f"[STATS] MSE medio su tutti i frame: {mse_mean:.3f}")
        print(f"[STATS] RMSE medio su tutti i frame: {rmse_mean:.3f}")
        print(f"[OUT]   {out_dir/'summary.csv'}")
        if args.per_frame_csv:
            print(f"[OUT]   CSV per-frame in: {per_frame_dir}")
    else:
        print("[DONE] Nessun frame in comune analizzabile.")

if __name__ == "__main__":
    main()

# python step3compare.py --mocap dati_tuta_8p3_renamed.json --triang triangulated_3d_skeleton.json --triang-key skeleton_3d --out out_compare --align rigid