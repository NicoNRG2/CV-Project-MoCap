"""
This script automates the execution of the step2 pipeline scripts

USAGE:
> python 99_STEP2_PIPELINE.py

"""


import subprocess
import sys
from colorama import init, Fore, Style

# Initialize colorama (for Windows compatibility)
init(autoreset=True)

# List of commands to execute in order
commands = [
    ("python 02_download_roboflow.py","Downloaded the annotated dataset from Roboflow into the working directory."),
    ("python 02_rectified_videos.py","Performed geometric rectification on all camera videos using per-camera calibration (mtx, dist)."),
    ("python 02_rectified_images.py","Performed geometric rectification on all dataset images using per-camera calibration (mtx, dist)."),
    ("python 02_rectified_annotations.py","Rectified the 2D keypoint coordinates in the COCO JSON dataset."),
    ("python 02_debug_draw_keypoint_over_frame_check.py --image train/out2_frame_0019_png.rf.aa99af7677dc057dc1f577a91cafef39.jpg --annotations train/_annotations.coco.json --image_id 48 --output temp/02_temp/02_debug_draw_normale.png","Overlayed 2D keypoints on input frames to visually check the annotation alignment (normal)."),
    ("python 02_debug_draw_keypoint_over_frame_check.py --image images_rectified/out2_frame_0019_png.rf.aa99af7677dc057dc1f577a91cafef39.jpg --annotations temp/02_temp/02_annotations.coco.rectified.json --image_id 48 --output temp/02_temp/02_debug_draw_ret.png","Overlayed 2D keypoints on input frames to visually check the annotation alignment (rectified)."),
    ("python 02_triangulation.py --input temp/02_temp/02_annotations.coco.rectified.json --output temp/02_temp/02_triangulated_3d_skeleton.json","Triangulated 3D joint positions from the 2D keypoints across all camera views."),
    ("python 02_plot_3d_skeleton.py 1","Displayed a static 3D skeleton plot for a selected frame, useful for visual debugging."),
    ("python 02_generate_reprojected_annotations.py","Reprojected the 3D skeleton back into each camera view to verify geometric consistency"),
    ("python 02_compute_reprojection_error.py","Computed the reprojection error between the original 2D annotations and the reprojected points."),
    ("python 02_debug_plot_2D_compare_keypoints.py 10","Visualizes and compares 2D keypoints from the original and reprojected annotations for a specific frame."),
    ("python 02_animate_triangulation.py  --input temp/02_temp/02_triangulated_3d_skeleton.json --out temp/02_temp/02_triangulated_skeleton.gif --fps 12","Creates an animated 3D visualization (GIF) of the full reconstructed motion sequence.")
]

def run_command(cmd, desc):
    print(f"\n{Style.BRIGHT} {Fore.YELLOW}Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error while executing: {cmd}")
        sys.exit(result.returncode)
    else:
        print(Fore.GREEN + Style.BRIGHT + f"Completed: {desc}")
        input(Fore.WHITE + Style.BRIGHT + "Press ENTER to continue...")

def main():
    print(Fore.BLUE + Style.BRIGHT + "\n=== STEP 2 PIPELINE ===")
    for cmd, desc in commands:
        run_command(cmd, desc)
    print(Fore.BLUE + Style.BRIGHT + "\nDONE: all scripts have been executed successfully!")

if __name__ == "__main__":
    main()
