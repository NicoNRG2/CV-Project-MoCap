import subprocess
import sys

# List of commands to execute in order
commands = [
    "python 02_download_roboflow.py",
    "python 02_rectified_videos.py",
    "python 02_rectified_images.py",
    "python 02_rectified_annotations.py",
    "python 02_debug_draw_keypoint_over_frame_ckeck.py --image train/out2_frame_0019_png.rf.aa99af7677dc057dc1f577a91cafef39.jpg --annotations train/_annotations.coco.json --image_id 48 --output temp/02_temp/02_debug_draw_normale.png",
    "python 02_debug_draw_keypoint_over_frame_ckeck.py --image images_rectified/out2_frame_0019_png.rf.aa99af7677dc057dc1f577a91cafef39.jpg --annotations temp/02_temp/02_annotations.coco.rectified.json --image_id 48 --output temp/02_temp/02_debug_draw_ret.png",
    "python 02_triangulation.py --input temp/02_temp/02_annotations.coco.rectified.json --output temp/02_temp/02_triangulated_3d_skeleton.json",
    "python 02_plot_3d_skeleton.py 1",
    "python 02_generate_reprojected_annotations.py",
    "python 02_compute_reprojection_error.py",
    "python 02_debug_plot_2D_compare_keypoints.py 10",
    "python 02_animate_triangulation.py  --input temp/02_temp/02_triangulated_3d_skeleton.json --out temp/02_temp/02_triangulated_skeleton.gif --fps 12"
]

def run_command(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error while executing: {cmd}")
        sys.exit(result.returncode)
    else:
        print(f"Completed: {cmd}")
        input("Press ENTER...")

def main():
    print("=== STEP 2: 3D player's position ===")
    for cmd in commands:
        run_command(cmd)
    print("\nDONE: all scripts have been executed successfully!")

if __name__ == "__main__":
    main()
