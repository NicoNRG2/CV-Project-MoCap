#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Joint names nell'ordine fornito
JOINTS = [
    "Hips",
    "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand",
]

# Coppie di indici 1-based come nel tuo JSON "skeleton"
_SKELETON_IDX_1BASED = [
    (1,2), (2,3), (3,4), (4,5),          # catena destra-anca → piede
    (1,6), (6,7), (7,8), (8,9),          # catena sinistra-anca → piede
    (1,10), (10,11), (11,12),            # tronco: Hips→Spine→Neck→Head
    (11,13), (13,14), (14,15),           # braccio destro
    (11,16), (16,17), (17,18),           # braccio sinistro
]

# Converte da indici 1-based a nomi joint (usa 0-based in Python)
BONES = [(JOINTS[i-1], JOINTS[j-1]) for (i, j) in _SKELETON_IDX_1BASED]

def load_frames(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Se contiene il campo "skeleton_3d", entra nel sotto-dizionario
    if "skeleton_3d" in data:
        data = data["skeleton_3d"]

    def frame_key(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    keys = sorted(list(data.keys()), key=frame_key)
    frames = [np.array(data[k], dtype=float) for k in keys]

    nJ = len(JOINTS)
    frames = [f for f in frames if f.shape == (nJ, 3)]
    if not frames:
        raise ValueError("Nessun frame valido nel JSON (attesi 18 joint per frame).")

    return frames

def compute_bounds(frames, pad_ratio=0.05):
    pts = np.concatenate(frames, axis=0)
    mins = pts.min(0); maxs = pts.max(0)
    pad = (maxs - mins).max() * pad_ratio
    return (mins - pad, maxs + pad)

def make_axes(fig, view, bounds):
    if view == "3d":
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=20, azim=-60)
        (mn, mx) = bounds
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        (mn, mx) = bounds
        if view == "xy":
            ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1])
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        elif view == "xz":
            ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[2], mx[2])
            ax.set_xlabel("X"); ax.set_ylabel("Z")
        elif view == "yz":
            ax.set_xlim(mn[1], mx[1]); ax.set_ylim(mn[2], mx[2])
            ax.set_xlabel("Y"); ax.set_ylabel("Z")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Projection: {view.upper()}")
    return ax

def project_points(p, view):
    if view == "3d":
        return p
    if view == "xy":
        return p[:, [0,1]]
    if view == "xz":
        return p[:, [0,2]]
    if view == "yz":
        return p[:, [1,2]]
    raise ValueError("view non valida")

def animate(frames, fps, out, view, downsample, max_frames, point_size):
    frames = frames[::max(1, downsample)]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    nF = len(frames)

    # Bounds stabili
    mins, maxs = compute_bounds(frames)
    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs

    # Indici ossa
    j2i = {n:i for i,n in enumerate(JOINTS)}
    edges = [(j2i[a], j2i[b]) for a,b in BONES]

    fig = plt.figure(figsize=(7,7))
    ax = make_axes(fig, view, (mins, maxs))
    ax.set_title("MoCap Skeleton")

    # Inizializzazione
    p0 = frames[0]
    if view == "3d":
        scat = ax.scatter(p0[:,0], p0[:,1], p0[:,2], s=point_size)
        lines = []
        for (i,j) in edges:
            (ln,) = ax.plot([p0[i,0],p0[j,0]],
                            [p0[i,1],p0[j,1]],
                            [p0[i,2],p0[j,2]], linewidth=2)
            lines.append(ln)
        # In 3D usa text2D per testo in coordinate normalizzate dell'axes
        time_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                              ha="left", va="top", fontsize=9)
    else:
        # Proiezione 2D
        def project_points(p, view_):
            if view_ == "xy": return p[:, [0,1]]
            if view_ == "xz": return p[:, [0,2]]
            if view_ == "yz": return p[:, [1,2]]
            raise ValueError("view non valida per 2D")

        p0p = project_points(p0, view)
        scat = ax.scatter(p0p[:,0], p0p[:,1], s=point_size)
        lines = []
        for (i,j) in edges:
            (ln,) = ax.plot([p0p[i,0],p0p[j,0]],
                            [p0p[i,1],p0p[j,1]], linewidth=2)
            lines.append(ln)
        # In 2D va bene ax.text con transAxes
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                            ha="left", va="top", fontsize=9)

    def update(fi):
        p = frames[fi]
        if view == "3d":
            scat._offsets3d = (p[:,0], p[:,1], p[:,2])
            for ln,(i,j) in zip(lines, edges):
                ln.set_data([p[i,0],p[j,0]], [p[i,1],p[j,1]])
                ln.set_3d_properties([p[i,2],p[j,2]])
        else:
            pp = project_points(p, view)
            scat.set_offsets(pp[:, :2])
            for ln,(i,j) in zip(lines, edges):
                ln.set_data([pp[i,0],pp[j,0]], [pp[i,1],pp[j,1]])

        t = fi / (fps / max(1, downsample))
        time_text.set_text(f"frame {fi+1}/{nF} — {t:.2f}s")
        return [scat, *lines, time_text]

    interval_ms = 1000.0 / (fps / max(1, downsample))
    anim = FuncAnimation(fig, update, frames=nF, interval=interval_ms, blit=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        try:
            writer = FFMpegWriter(fps=int(fps/max(1,downsample)), bitrate=3000)
            anim.save(str(out), writer=writer, dpi=120)
        except Exception as e:
            print(f"[INFO] MP4 fallito ({e}). Salvo GIF…")
            out = out.with_suffix(".gif")
            anim.save(str(out), writer=PillowWriter(fps=int(fps/max(1,downsample))), dpi=120)
    elif out.suffix.lower() == ".gif":
        anim.save(str(out), writer=PillowWriter(fps=int(fps/max(1,downsample))), dpi=120)
    else:
        # default GIF se estensione non riconosciuta
        out = out.with_suffix(".gif")
        anim.save(str(out), writer=PillowWriter(fps=int(fps/max(1,downsample))), dpi=120)

    print(f"[OK] Salvato: {out}")

def main():
    ap = argparse.ArgumentParser(description="Animazione 3D/2D MoCap da JSON")
    ap.add_argument("--input", "-i", type=str, default="triangulated_3d_skeleton.json",
                    help="Percorso al file JSON")
    ap.add_argument("--out", "-o", type=str, default="skeleton_animation.mp4",
                    help="Output (mp4/gif/png). mp4 richiede ffmpeg.")
    ap.add_argument("--fps", type=int, default=100, help="Frame rate di acquisizione")
    ap.add_argument("--downsample", type=int, default=1, help="Usa 2,4,10 per alleggerire")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutti, >0 = limita")
    ap.add_argument("--view", choices=["3d","xy","xz","yz"], default="3d",
                    help="Vista 3D o proiezione 2D")
    ap.add_argument("--point-size", type=float, default=20.0, help="Dimensione marker giunti")
    args = ap.parse_args()

    frames = load_frames(args.input)
    animate(frames=frames,
            fps=args.fps,
            out=args.out,
            view=args.view,
            downsample=args.downsample,
            max_frames=args.max_frames,
            point_size=args.point_size)

if __name__ == "__main__":
    main()

# python .\animate_triangulation.py --input .\triangulated_3d_skeleton.json --out triangulated.mp4 --fps 12