"""
Plots a single 3D skeleton frame from a triangulated JSON, highlighting the head joint in red and drawing bone connections.
Saves a PNG preview and shows the figure.

USAGE:
> python 02_plot_3d_skeleton.py [frame_number]
    n.b. frame_number can be from 1 to 48

"""

import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines  # used to create the red dot in the legend

# Static definitions
KEYPOINTS = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand"
]

SKELETON = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12),
    (11, 13), (13, 14), (14, 15),
    (11, 16), (16, 17), (17, 18)
]

def plot_frame(frame_number, json_path="temp/02_temp/02_triangulated_3d_skeleton.json"):
    # Plot 3D skeleton for a given frame.
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"JSON file '{json_path}' not found.")
        return

    frames = data.get('skeleton_3d', {})
    try:
        idx = int(frame_number)
    except ValueError:
        print(f"Invalid frame number: {frame_number}")
        return

    key = f"frame_{idx:04d}"
    if key not in frames:
        print(f"Frame '{key}' not present in the JSON.")
        return

    points = frames[key]
    xs, ys, zs = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter: head in red, others in blue
    head_idx = KEYPOINTS.index("Head")
    for i, (x, y, z) in enumerate(points):
        c = 'r' if i == head_idx else 'b'
        s = 60 if i == head_idx else 20
        ax.scatter(x, y, z, c=c, s=s)

    # Skeleton connections
    for a, b in SKELETON:
        i, j = a-1, b-1
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='k')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 2])  # Aspect ratio
    plt.title(f"3D Skeleton — {key}")

    # Legend with red dot
    red_dot = mlines.Line2D([], [], color='r', marker='o', linestyle='None',
                            markersize=8, label='Head (red)')
    plt.legend(handles=[red_dot])

    plt.show()
    plt.savefig("temp/02_temp/02_skeleton.png")


if __name__ == "__main__":
    frame_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    plot_frame(frame_arg)
