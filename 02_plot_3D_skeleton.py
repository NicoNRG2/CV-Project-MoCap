import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines  # 👈 serve per creare il pallino rosso nella legenda

# Definizioni statiche
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
    """
    Plot 3D skeleton for a given frame.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File JSON '{json_path}' non trovato.")
        return

    frames = data.get('skeleton_3d', {})
    try:
        idx = int(frame_number)
    except ValueError:
        print(f"Numero di frame non valido: {frame_number}")
        return

    key = f"frame_{idx:04d}"
    if key not in frames:
        print(f"Frame '{key}' non presente nel JSON.")
        return

    points = frames[key]
    xs, ys, zs = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter: testa in rosso, gli altri in blu
    head_idx = KEYPOINTS.index("Head")
    for i, (x, y, z) in enumerate(points):
        c = 'r' if i == head_idx else 'b'
        s = 60 if i == head_idx else 20
        ax.scatter(x, y, z, c=c, s=s)

    # Connessioni scheletro
    for a, b in SKELETON:
        i, j = a-1, b-1
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='k')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 2])  # Aspect ratio
    plt.title(f"Scheletro 3D — {key}")

    # ✅ Legenda con pallino rosso
    red_dot = mlines.Line2D([], [], color='r', marker='o', linestyle='None',
                            markersize=8, label='Head (rosso)')
    plt.legend(handles=[red_dot])

    plt.show()


if __name__ == "__main__":
    frame_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    plot_frame(frame_arg)
