import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

def load_frames(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data.get('skeleton_3d', {})
    # Ordina i frame per nome
    keys = sorted(frames.keys())
    return [frames[k] for k in keys]

def animate_skeleton(frames, save_path="skeleton_video.mp4", fps=24, duration=2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.title("Scheletro 3D in movimento")

    scatters = []
    lines = []
    head_idx = KEYPOINTS.index("Head")

    def init():
        ax.cla()
        ax.set_box_aspect([1, 1, 2])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.title("Scheletro 3D in movimento")
        return []

    def update(frame_idx):
        ax.cla()
        ax.set_box_aspect([1, 1, 2])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.title("Scheletro 3D in movimento")
        points = frames[frame_idx]
        xs, ys, zs = zip(*points)
        for i, (x, y, z) in enumerate(points):
            c = 'r' if i == head_idx else 'b'
            s = 60 if i == head_idx else 20
            ax.scatter(x, y, z, c=c, s=s)
        for a, b in SKELETON:
            i, j = a-1, b-1
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='k')
        return []

    n_frames = min(len(frames), int(fps * duration))
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
    anim.save(save_path, fps=fps, dpi=150)
    print(f"Video salvato in {save_path}")

if __name__ == "__main__":
    frames = load_frames("triangulated_3d_skeleton.json")
    animate_skeleton(frames, save_path="skeleton_video.mp4", fps=24, duration=2)