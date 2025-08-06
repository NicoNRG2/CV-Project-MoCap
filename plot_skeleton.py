import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

def load_points_from_json(json_path, frame_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["skeleton_3d"][frame_name]

keypoints = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot", "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
]

skeleton = [
    [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9],
    [1, 10], [10, 11], [11, 12], [11, 13], [13, 14], [14, 15], [11, 16], [16, 17], [17, 18]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D skeleton for a specified frame.")
    parser.add_argument("--frame", type=int, default=1, help="Frame number to plot (e.g., 48 for frame_0048)")
    parser.add_argument("--json", type=str, default="triangulated_3d_skeleton.json", help="Path to JSON file")
    args = parser.parse_args()

    frame_name = f"frame_{args.frame:04d}"
    json_path = args.json
    points = load_points_from_json(json_path, frame_name)

    points = list(zip(*points))  # Transpose to get x, y, z lists

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter all points in blue except Head (index 11)
    ax.scatter(points[0][:11] + points[0][12:], points[1][:11] + points[1][12:], points[2][:11] + points[2][12:], c='b', marker='o')
    # Scatter Head point in red
    ax.scatter(points[0][11], points[1][11], points[2][11], c='r', marker='o', s=60, label='Head')

    # Draw skeleton connections
    for joint in skeleton:
        i, j = joint[0] - 1, joint[1] - 1  # Convert to 0-based index
        ax.plot([points[0][i], points[0][j]],
                [points[1][i], points[1][j]],
                [points[2][i], points[2][j]], c='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1, 1, 2])  # Aspect ratio for better visualization

    plt.title(f'3D Skeleton Plot - {frame_name}')
    plt.legend()
    plt.show()