#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compares MoCap joint positions with triangulated results frame-by-frame.
Optionally aligns via rigid/similarity Kabsch and prints MPJPE/RMSE/MSE statistics.

USAGE:
> python 03_step3compare.py

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
    # Expected: { "frame_0001": [[x,y,z], ...], ... }
    return data

def _looks_like_frames_mapping(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    # Heuristic: at least one key starting with "frame_" and value is a list of [x,y,z]
    for k, v in obj.items():
        if isinstance(k, str) and k.startswith("frame_") and isinstance(v, list):
            return True
    return False

def load_frames_triang(path: str, key: Optional[str] = None) -> Dict[str, List[List[float]]]:
    """
    Load triangulation frames:
    - If key is provided and present -> use data[key]
    - Otherwise, try to use frames at the top level
    - If there is a single container key that holds the frames, use that
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) If the user specified a key, try to use it
    if key:
        if key in data and _looks_like_frames_mapping(data[key]):
            return data[key]
        else:
            print(f"[INFO] Key '{key}' not found or invalid, trying top-level frames...")

    # 2) Top-level?
    if _looks_like_frames_mapping(data):
        return data

    # 3) Single-level container?
    if isinstance(data, dict) and len(data) == 1:
        only_key, only_val = next(iter(data.items()))
        if _looks_like_frames_mapping(only_val):
            print(f"[INFO] Using auto-detected key '{only_key}'.")
            return only_val

    raise KeyError("Unrecognized triangulation JSON format: cannot find a 'frame_xxxx' mapping.")

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
    ap.add_argument("--mocap", default="temp/03_temp/03_final_mocap.json" , help="Path to MoCap JSON (frames at top-level).")
    ap.add_argument("--triang", default="temp/03_temp/03_final_triangulation.json", help="Path to triangulation JSON.")
    ap.add_argument("--triang-key", default=None, help="(Optional) Key that contains the frames inside the triangulation JSON.")
    ap.add_argument("--out", default="out_compare", help="(Ignored) Output folder: CSVs are not saved.")
    ap.add_argument("--per-frame-csv", action="store_true", help="(Ignored) No per-frame CSV will be written.")
    ap.add_argument("--align", default="similarity", choices=["none", "rigid", "similarity"],
                   help="Alignment: none, rigid (Kabsch), similarity (rot+trans+scale).")
    args = ap.parse_args()

    mocap = load_frames_mocap(args.mocap)
    triang = load_frames_triang(args.triang, args.triang_key)

    mocap_frames = set(mocap.keys())
    triang_frames = set(triang.keys())
    common = sorted(mocap_frames & triang_frames)

    missing_in_triang = sorted(mocap_frames - triang_frames)
    missing_in_mocap = sorted(triang_frames - mocap_frames)

    print(f"[INFO] Common frames: {len(common)}")
    if missing_in_triang:
        print(f"[WARN] Frames present in MoCap but not in Triangulation: {len(missing_in_triang)} (e.g., {missing_in_triang[:5]})")
    if missing_in_mocap:
        print(f"[WARN] Frames present in Triangulation but not in MoCap: {len(missing_in_mocap)} (e.g., {missing_in_mocap[:5]})")

    rows = []
    for fr in common:
        A = np.asarray(mocap[fr], dtype=np.float64)
        B = np.asarray(triang[fr], dtype=np.float64)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3:
            print(f"[SKIP] {fr}: unexpected format (shape A={A.shape}, B={B.shape})")
            continue
        if A.shape[0] != B.shape[0]:
            print(f"[WARN] {fr}: different joint count (A={A.shape[0]}, B={B.shape[0]}). Comparing on common indices.")
            J = min(A.shape[0], B.shape[0])
            A = A[:J]
            B = B[:J]

        A_aligned = A
        B_ref = B
        if args.align != "none":
            allow_scale = (args.align == "similarity")
            R, t, s = kabsch(A, B, allow_scale=allow_scale)
            A_aligned = (s * (A @ R.T)) + t

        metrics = compute_errors(A_aligned, B_ref)
        metrics.update({
            "frame": fr,
            "num_joints": int(A_aligned.shape[0]),
            "align": args.align
        })
        rows.append(metrics)

    if rows:
        summary = pd.DataFrame(rows).sort_values("frame")
        print(f"[DONE] Frames analyzed: {len(summary)}")
        print(f"[STATS] Mean MPJPE over all frames: {summary['mpjpe'].mean():.3f} mm")
        print(f"[STATS] Median MPJPE over all frames: {summary['mpjpe'].median():.3f} mm")
        print(f"[STATS] Mean MSE over all frames: {summary['mse'].mean():.3f} mm²")
        print(f"[STATS] Mean RMSE over all frames: {summary['rmse'].mean():.3f} mm")
    else:
        print("[DONE] No analyzable common frames.")

if __name__ == "__main__":
    main()
