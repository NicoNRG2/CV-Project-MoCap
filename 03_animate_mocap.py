"""
Animates a 3D/2D MoCap skeleton from a JSON with 18 joints, supporting optional Z-axis rotation, downsampling, and frame limits.
Exports to MP4/GIF and auto-opens the result with the system viewer.

USAGE:
> python 03_animate_mocap.py --input temp/03_temp/03_selected_keypoints_adapted_joints.json --out temp/03_temp/03_animate_mocap_100fps.mp4 --fps 100 --view 3d

"""

import json
import argparse
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl
from math import radians, cos, sin
import imageio_ffmpeg
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


# Joint names in the provided order
JOINTS = [
    "Hips","Spine","Neck","Head",
    "LShoulder","LElbow","LHand",
    "RShoulder","RElbow","RHand",
    "LHip","LKnee","LAnkle","LFoot",
    "RHip","RKnee","RAnkle","RFoot"
]

# Simple bone connections
BONES = [
    ("Hips","Spine"), ("Spine","Neck"), ("Neck","Head"),
    ("Neck","LShoulder"), ("LShoulder","LElbow"), ("LElbow","LHand"),
    ("Neck","RShoulder"), ("RShoulder","RElbow"), ("RElbow","RHand"),
    ("Hips","LHip"), ("LHip","LKnee"), ("LKnee","LAnkle"), ("LAnkle","LFoot"),
    ("Hips","RHip"), ("RHip","RKnee"), ("RKnee","RAnkle"), ("RAnkle","RFoot"),
]

def load_frames(path):
    with open(path, "r") as f:
        data = json.load(f)

    def frame_key(k):
        try:
            return int(k.split("_")[-1])
        except:
            return 0

    keys = sorted(list(data.keys()), key=frame_key)
    frames = [np.array(data[k], dtype=float) for k in keys]
    nJ = len(JOINTS)
    frames = [f for f in frames if f.shape == (nJ, 3)]
    if not frames:
        raise ValueError("No valid frames in JSON (expected 18 joints per frame).")
    return frames

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

def open_file_with_default_app(path: Path):
    # Open a file with the operating system's default viewer.
    p = Path(path).resolve()
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception as e:
        print(f"[WARN] Unable to open the file automatically: {e}")

def animate(frames, fps, out, view, downsample, max_frames, point_size, name):
    frames = frames[::max(1, downsample)]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    nF = len(frames)

    # Stable bounds
    mins, maxs = compute_bounds(frames)

    # Bone indices
    j2i = {n: i for i, n in enumerate(JOINTS)}
    edges = [(j2i[a], j2i[b]) for a, b in BONES]

    fig = plt.figure(figsize=(7, 7))
    ax = make_axes(fig, view, (mins, maxs))
    ax.set_title(name)

    # Initialization
    p0 = frames[0]
    if view == "3d":
        scat = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=point_size)
        lines = []
        for (i, j) in edges:
            (ln,) = ax.plot([p0[i, 0], p0[j, 0]],
                            [p0[i, 1], p0[j, 1]],
                            [p0[i, 2], p0[j, 2]], linewidth=2)
            lines.append(ln)
        time_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                              ha="left", va="top", fontsize=9)
    else:
        p0p = project_points(p0, view)
        scat = ax.scatter(p0p[:, 0], p0p[:, 1], s=point_size)
        lines = []
        for (i, j) in edges:
            (ln,) = ax.plot([p0p[i, 0], p0p[j, 0]],
                            [p0p[i, 1], p0p[j, 1]], linewidth=2)
            lines.append(ln)
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                            ha="left", va="top", fontsize=9)

    def update(fi):
        p = frames[fi]
        if view == "3d":
            scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            for ln, (i, j) in zip(lines, edges):
                ln.set_data([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]])
                ln.set_3d_properties([p[i, 2], p[j, 2]])
        else:
            pp = project_points(p, view)
            scat.set_offsets(pp[:, :2])
            for ln, (i, j) in zip(lines, edges):
                ln.set_data([pp[i, 0], pp[j, 0]], [pp[i, 1], pp[j, 1]])

        t = fi / (fps / max(1, downsample))
        time_text.set_text(f"frame {fi+1}/{nF} — {t:.2f}s")
        return [scat, *lines, time_text]

    interval_ms = 1000.0 / (fps / max(1, downsample))
    anim = FuncAnimation(fig, update, frames=nF, interval=interval_ms, blit=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_path = out

    if out.suffix.lower() == ".mp4":
        try:
            writer = FFMpegWriter(fps=int(fps / max(1, downsample)), bitrate=3000)
            anim.save(str(out), writer=writer, dpi=120)
        except Exception as e:
            print(f"[INFO] MP4 failed ({e}). Saving GIF…")
            out = out.with_suffix(".gif")
            anim.save(str(out), writer=PillowWriter(fps=int(fps / max(1, downsample))), dpi=120)
            final_path = out
    elif out.suffix.lower() == ".gif":
        anim.save(str(out), writer=PillowWriter(fps=int(fps / max(1, downsample))), dpi=120)
    else:
        out = out.with_suffix(".gif")
        anim.save(str(out), writer=PillowWriter(fps=int(fps / max(1, downsample))), dpi=120)
        final_path = out

    print(f"[OK] Saved: {final_path}")
    open_file_with_default_app(final_path)  # <--- automatic opening

def rotate_frames_z(frames, degrees):
    #Rotate all frames around the Z axis by a given angle (in degrees).
    if abs(degrees) < 1e-9:
        return frames  # no rotation
    th = radians(degrees)
    Rz = np.array([
        [cos(th), -sin(th), 0],
        [sin(th),  cos(th), 0],
        [0,        0,       1]
    ])
    return [f @ Rz.T for f in frames]

def main():
    ap = argparse.ArgumentParser(description="3D/2D MoCap animation from JSON")
    ap.add_argument("--input", "-i", type=str, default="temp/03_temp/03_selected_keypoints_adapted_joints.json", help="Path to the JSON file")
    ap.add_argument("--out", "-o", type=str, default="temp/03_temp/03_animate_mocap_100fps.mp4", help="Output (mp4/gif/png). mp4 requires ffmpeg.")
    ap.add_argument("--fps", type=int, default=100, help="Acquisition frame rate")
    ap.add_argument("--downsample", type=int, default=1, help="Use 2,4,10 to downsample")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all, >0 = limit")
    ap.add_argument("--view", choices=["3d","xy","xz","yz"], default="3d", help="3D view or 2D projection")
    ap.add_argument("--point-size", type=float, default=20.0, help="Joint marker size")
    ap.add_argument("--rotate", type=float, default=0, help="Rotate all frames around the Z axis by N degrees (default=0 = no rotation)")
    ap.add_argument("--name", type=str, default="MoCap Skeleton", help="plot title")
    args = ap.parse_args()

    frames = load_frames(args.input)
    frames = rotate_frames_z(frames, args.rotate)

    animate(frames=frames, fps=args.fps, out=args.out, view=args.view, downsample=args.downsample,
            max_frames=args.max_frames, point_size=args.point_size, name=args.name)

if __name__ == "__main__":
    main()
