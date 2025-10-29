"""
This script automates the execution of the step3 pipeline scripts

USAGE:
> python 99_STEP3_PIPELINE.py

"""


import subprocess
import sys
from colorama import init, Fore, Style

# Initialize colorama (for Windows compatibility)
init(autoreset=True)

# List of commands to execute in order
commands = [
    ("python 03_cut_frames.py","Cut the shot segment of interest from the MoCap sequence."),
    ("python 03_adapt_skeleton.py","Removed unnecessary or unused bones from the skeleton."),
    ("python 03_animate_mocap.py","Generated an MP4 animation of the MoCap data (100 fps, 393 frames)."),
    ("python 03_subsample_mocap.py","Downsampled the MoCap sequence from 100 fps to 24 fps."),
    ("python 03_rename_frame.py","Renamed frames sequentially (e.g., frame_980 → frame_1)."),
    ("python 03_reorder_triangulation_joints.py","Reordered the triangulated joints to match the MoCap joint order."),
    ("python 03_step3compare.py","Performed a direct comparison between the triangulated and MoCap skeletons and compute some error metrics."),
    ("python 03_animate_mocap.py --input temp/03_temp/03_final_triangulation.json --out temp/03_temp/03_final_triangulation.gif --fps 12 --rotate -90 --name \"Triangulated Skeleton\"","Creates an animated GIF of the final 3D Triangulated skeleton."),
    ("python 03_animate_mocap.py --input temp/03_temp/03_final_mocap.json --out temp/03_temp/03_final_mocap.gif --fps 12","Creates an animated GIF of the final 3D MoCap skeleton.")
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
    print(Fore.BLUE + Style.BRIGHT + "\n=== STEP 3 PIPELINE ===")
    for cmd, desc in commands:
        run_command(cmd, desc)
    print(Fore.BLUE + Style.BRIGHT + "\nDONE: all scripts have been executed successfully!")

if __name__ == "__main__":
    main()
