#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# NEW: imports per apertura file
import os
import sys
import platform
import subprocess

# --- JOINTS (13) nell'ordine richiesto ---
JOINTS = [
    "Nose",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
]

# Nessuna connessione: vogliamo solo i punti
BONES = []


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
        raise ValueError(f"Nessun frame valido nel JSON (attesi {nJ} joint per frame).")

    return frames


def rotate_frames_z(frames, deg=-90.0):
    """Ruota ogni frame di 'deg' gradi attorno a Z (default: -90°)."""
    th = radians(deg)
    Rz = np.array([
        [cos(th), -sin(th), 0.0],
        [sin(th),  cos(th), 0.0],
        [0.0,      0.0,     1.0],
    ], dtype=float)
    return [f @ Rz.T for f in frames]


def compute_bounds(frames, pad_ratio=0.05):
    pts = np.concatenate(frames, axis=0)
    mins = pts.min(0)
    maxs = pts.max(0)
    pad = (maxs - mins).max() * pad_ratio
    return (mins - pad, maxs + pad)


def make_axes(fig, view, bounds):
    if view == "3d":
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=20, azim=-60)
        (mn, mx) = bounds
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        ax.set_zlim(mn[2], mx[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        (mn, mx) = bounds
        if view == "xy":
            ax.set_xlim(mn[0], mx[0])
            ax.set_ylim(mn[1], mx[1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        elif view == "xz":
            ax.set_xlim(mn[0], mx[0])
            ax.set_ylim(mn[2], mx[2])
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        elif view == "yz":
            ax.set_xlim(mn[1], mx[1])
            ax.set_ylim(mn[2], mx[2])
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Projection: {view.upper()}")
    return ax


def project_points(p, view):
    if view == "3d":
        return p
    if view == "xy":
        return p[:, [0, 1]]
    if view == "xz":
        return p[:, [0, 2]]
    if view == "yz":
        return p[:, [1, 2]]
    raise ValueError("view non valida")


# NEW: apertura con app di default (Windows/macOS/Linux/WSL)
def open_with_default_app(path: Path):
    try:
        p = str(path)
        sysname = platform.system().lower()
        release = platform.uname().release.lower()

        # WSL detection
        is_wsl = ("microsoft" in release) or ("wsl" in release)

        if is_wsl:
            # wslview apre con l'app di Windows se disponibile
            ret = subprocess.run(["wslview", p], check=False)
            if ret.returncode == 0:
                return
            # fallback a xdg-open
            subprocess.run(["xdg-open", p], check=False)
            return

        if sysname == "windows":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sysname == "darwin":  # macOS
            subprocess.run(["open", p], check=False)
        else:  # Linux/Unix
            subprocess.run(["xdg-open", p], check=False)
    except Exception as e:
        print(f"[INFO] Impossibile aprire automaticamente il file: {e}")


def animate(frames, fps, out, view, downsample, max_frames, point_size):
    frames = frames[::max(1, downsample)]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    nF = len(frames)

    # Bounds stabili
    mins, maxs = compute_bounds(frames)

    fig = plt.figure(figsize=(7, 7))
    ax = make_axes(fig, view, (mins, maxs))
    ax.set_title("Triangulated Keypoints (YOLO)")

    # Inizializzazione solo punti
    p0 = frames[0]
    if view == "3d":
        scat = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=point_size)
        time_text = ax.text2D(
            0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=9
        )
    else:
        p0p = project_points(p0, view)
        scat = ax.scatter(p0p[:, 0], p0p[:, 1], s=point_size)
        time_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=9
        )

    def update(fi):
        p = frames[fi]
        if view == "3d":
            scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        else:
            pp = project_points(p, view)
            scat.set_offsets(pp[:, :2])

        t = fi / (fps / max(1, downsample))
        time_text.set_text(f"frame {fi+1}/{nF} — {t:.2f}s")
        return [scat, time_text]

    interval_ms = 1000.0 / (fps / max(1, downsample))
    anim = FuncAnimation(fig, update, frames=nF, interval=interval_ms, blit=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out.suffix.lower() == ".mp4":
            try:
                writer = FFMpegWriter(fps=int(fps / max(1, downsample)), bitrate=3000)
                anim.save(str(out), writer=writer, dpi=120)
            except Exception as e:
                print(f"[INFO] MP4 fallito ({e}). Salvo GIF…")
                out = out.with_suffix(".gif")
                anim.save(
                    str(out),
                    writer=PillowWriter(fps=int(fps / max(1, downsample))),
                    dpi=120,
                )
        elif out.suffix.lower() == ".gif":
            anim.save(
                str(out),
                writer=PillowWriter(fps=int(fps / max(1, downsample))),
                dpi=120,
            )
        else:
            out = out.with_suffix(".gif")
            anim.save(
                str(out),
                writer=PillowWriter(fps=int(fps / max(1, downsample))),
                dpi=120,
            )
    finally:
        # NEW: chiudi la figura per liberare il file prima di aprirlo
        plt.close(fig)

    print(f"[OK] Salvato: {out}")
    # NEW: prova ad aprire il file creato
    open_with_default_app(out)


def main():
    ap = argparse.ArgumentParser(description="Animazione 3D/2D keypoints (solo pallini) da JSON")
    ap.add_argument(
        "--input", "-i", type=str, default="triangulated_3d_keypoints.json",
        help="Percorso al file JSON"
    )
    ap.add_argument(
        "--out", "-o", type=str, default="keypoints_animation.mp4",
        help="Output (mp4/gif/png). mp4 richiede ffmpeg."
    )
    ap.add_argument("--fps", type=int, default=100, help="Frame rate di acquisizione")
    ap.add_argument("--downsample", type=int, default=1, help="Usa 2,4,10 per alleggerire")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutti, >0 = limita")
    ap.add_argument(
        "--view", choices=["3d", "xy", "xz", "yz"], default="3d",
        help="Vista 3D o proiezione 2D"
    )
    ap.add_argument("--point-size", type=float, default=20.0, help="Dimensione marker giunti")
    ap.add_argument(
        "--rotate-z", type=float, default=-90.0,
        help="Rotazione gradi attorno a Z (default -90). Usa 0 per nessuna rotazione."
    )
    args = ap.parse_args()

    frames = load_frames(args.input)
    if abs(args.rotate_z) > 1e-9:
        frames = rotate_frames_z(frames, deg=args.rotate_z)  # rotazione opzionale (default -90°)
    animate(
        frames=frames,
        fps=args.fps,
        out=args.out,
        view=args.view,
        downsample=args.downsample,
        max_frames=args.max_frames,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
