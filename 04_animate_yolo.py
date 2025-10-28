#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Animate 3D or 2D keypoints from a JSON file containing triangulated keypoints.
The animation can be saved as an MP4 video (requires ffmpeg) or as a GIF.

Usage:
python 04_animate_yolo.py --input temp/04_temp/04_triangulated_yolo.json --out temp/04_temp/04_yolo.gif --fps 12
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# NEW: imports for file opening
import os
import sys
import platform
import subprocess

# --- JOINTS (13) in the required order ---
JOINTS = [
    "Nose",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
]

# No connections: we only want to display points
BONES = []


def load_frames(path):
    with open(path, "r") as f:
        data = json.load(f)

    # If it contains the field "skeleton_3d", enter the sub-dictionary
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
        raise ValueError(f"No valid frames found in JSON (expected {nJ} joints per frame).")

    return frames


def rotate_frames_z(frames, deg=-90.0):
    """Rotate each frame by 'deg' degrees around the Z axis (default: -90°)."""
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
    raise ValueError("invalid view")


# NEW: open with default app (Windows/macOS/Linux/WSL)
def open_with_default_app(path: Path):
    try:
        p = str(path)
        sysname = platform.system().lower()
        release = platform.uname().release.lower()

        # Detect WSL
        is_wsl = ("microsoft" in release) or ("wsl" in release)

        if is_wsl:
            # wslview opens with the default Windows app if available
            ret = subprocess.run(["wslview", p], check=False)
            if ret.returncode == 0:
                return
            # fallback to xdg-open
            subprocess.run(["xdg-open", p], check=False)
            return

        if sysname == "windows":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sysname == "darwin":  # macOS
            subprocess.run(["open", p], check=False)
        else:  # Linux/Unix
            subprocess.run(["xdg-open", p], check=False)
    except Exception as e:
        print(f"[INFO] Unable to open the file automatically: {e}")


def animate(frames, fps, out, view, downsample, max_frames, point_size):
    frames = frames[::max(1, downsample)]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    nF = len(frames)

    # Stable bounds
    mins, maxs = compute_bounds(frames)

    fig = plt.figure(figsize=(7, 7))
    ax = make_axes(fig, view, (mins, maxs))
    ax.set_title("Triangulated Keypoints (YOLO)")

    # Initialize points only
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
                print(f"[INFO] MP4 export failed ({e}). Saving as GIF instead…")
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
        # NEW: close the figure to free the file before opening
        plt.close(fig)

    print(f"[OK] Saved: {out}")
    # NEW: try to open the created file
    open_with_default_app(out)


def main():
    ap = argparse.ArgumentParser(description="3D/2D keypoints animation (points only) from JSON")
    ap.add_argument(
        "--input", "-i", type=str, default="triangulated_3d_keypoints.json",
        help="Path to the JSON file"
    )
    ap.add_argument(
        "--out", "-o", type=str, default="keypoints_animation.mp4",
        help="Output file (mp4/gif/png). mp4 requires ffmpeg."
    )
    ap.add_argument("--fps", type=int, default=100, help="Acquisition frame rate")
    ap.add_argument("--downsample", type=int, default=1, help="Use 2,4,10 to reduce frames")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames, >0 = limit number")
    ap.add_argument(
        "--view", choices=["3d", "xy", "xz", "yz"], default="3d",
        help="3D view or 2D projection"
    )
    ap.add_argument("--point-size", type=float, default=20.0, help="Joint marker size")
    ap.add_argument(
        "--rotate-z", type=float, default=-90.0,
        help="Rotation in degrees around Z axis (default -90). Use 0 for no rotation."
    )
    args = ap.parse_args()

    frames = load_frames(args.input)
    if abs(args.rotate_z) > 1e-9:
        frames = rotate_frames_z(frames, deg=args.rotate_z)  # optional rotation (default -90°)
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
