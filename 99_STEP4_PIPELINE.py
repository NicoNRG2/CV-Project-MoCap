"""
This script automates the execution of the step4 pipeline scripts

USAGE:
> python 99_STEP4_PIPELINE.py

"""


import os
import sys
import subprocess

PY = "python"  # use the same interpreter that launches this script

def run(cmd_list, echo_prefix=""):
    # Run one command; stop on error. No pause here.
    print(f"\n{echo_prefix}Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"Error while executing: {' '.join(cmd_list)}")
        sys.exit(result.returncode)
    print("Completed.")
    input("Press ENTER...")

def run_group(cmd_lists, pause_message="Press ENTER to continue..."):
    # Run multiple commands sequentially; pause once at the end.
    for i, cmd in enumerate(cmd_lists, 1):
        run(cmd, echo_prefix=f"[{i}/{len(cmd_lists)}]")
    input(f"\n{pause_message}")

def ask_keep_id(cam_label: str) -> int:
    # Ask an integer > 0 to use as keep_id.
    while True:
        s = input(f"Enter the id for {cam_label}: ").strip()
        if s.isdigit() and int(s) > 0:
            return int(s)
        print("Invalid value. Try again (e.g., 1, 2, 3...).")

def ensure_dirs():
    os.makedirs("temp/04_temp", exist_ok=True)

def main():
    ensure_dirs()
    print("=== STEP 04: Starting YOLO Pose + Triangulation Pipeline ===")

    # 1) Split images (single, no pause)
    run([PY, "04_divide_images.py"])

    # 2) YOLO pose on each camera (group + single pause at end)
    run_group([
        [PY, "04_yolo_pose.py", "--images", "images_rectified/cam_2",
         "--output", "temp/04_temp/labels2.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"],
        [PY, "04_yolo_pose.py", "--images", "images_rectified/cam_5",
         "--output", "temp/04_temp/labels5.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"],
        [PY, "04_yolo_pose.py", "--images", "images_rectified/cam_8",
         "--output", "temp/04_temp/labels8.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"],
        [PY, "04_yolo_pose.py", "--images", "images_rectified/cam_13",
         "--output", "temp/04_temp/labels13.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"],
    ], pause_message="YOLO pose done for all cams. Press ENTER to continue...")

    # 3) Visual check (group + single pause)
    run_group([
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_2",
         "--json", "temp/04_temp/labels2.json", "--outdir", "temp/04_temp/cam2"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_5",
         "--json", "temp/04_temp/labels5.json", "--outdir", "temp/04_temp/cam5"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_8",
         "--json", "temp/04_temp/labels8.json", "--outdir", "temp/04_temp/cam8"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_13",
         "--json", "temp/04_temp/labels13.json", "--outdir", "temp/04_temp/cam13"],
    ], pause_message="Visual check exported. Press ENTER to continue...")

    print("Go to temp/04_temp and, for each camera (cam2, cam5, cam8, cam13), enter the player's id (black tracksuit).")
    keep2 = ask_keep_id("cam_2")
    keep5 = ask_keep_id("cam_5")
    keep8 = ask_keep_id("cam_8")
    keep13 = ask_keep_id("cam_13")

    # 5) Filter by keep_id (group + single pause)
    run_group([
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels2.json",
         "--output", "temp/04_temp/labels2_filtered.json", "--keep_id", str(keep2)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels5.json",
         "--output", "temp/04_temp/labels5_filtered.json", "--keep_id", str(keep5)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels8.json",
         "--output", "temp/04_temp/labels8_filtered.json", "--keep_id", str(keep8)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels13.json",
         "--output", "temp/04_temp/labels13_filtered.json", "--keep_id", str(keep13)],
    ], pause_message="Filtering done. Press ENTER...")

    # 6) Adapt keypoints (group + single pause)
    run_group([
        [PY, "04_adapt_keypoint.py", "2"],
        [PY, "04_adapt_keypoint.py", "5"],
        [PY, "04_adapt_keypoint.py", "8"],
        [PY, "04_adapt_keypoint.py", "13"],
    ], pause_message="Keypoint adaptation done. Press ENTER...")

    # 7) Merge (single)
    run([PY, "04_merge_pose_jsons_like_rectified.py",
         "temp/04_temp/labels2_filtered_adapted.json",
         "temp/04_temp/labels5_filtered_adapted.json",
         "temp/04_temp/labels8_filtered_adapted.json",
         "temp/04_temp/labels13_filtered_adapted.json",
         "--out", "temp/04_temp/annotations_yolo.json"])

    # 8) Triangulation (single)
    run([PY, "02_triangulation.py",
         "--input", "temp/04_temp/annotations_yolo.json",
         "--output", "temp/04_temp/04_triangulated_yolo.json"])

    # 9) YOLO animation (single)
    run([PY, "04_animate_yolo.py",
         "--input", "temp/04_temp/04_triangulated_yolo.json",
         "--out", "temp/04_temp/04_yolo.gif", "--fps", "12"])

    # 10) Adapt MoCap (single)
    run([PY, "04_adapt_mocap.py"])

    # 11) Final comparison (single)
    run([PY, "03_step3compare.py",
         "--mocap", "temp/04_temp/04_adapted_final_mocap.json",
         "--triang", "temp/04_temp/04_triangulated_yolo.json",
         "--align", "similarity"])

    print("\nPipeline 04 completed successfully!")
    

if __name__ == "__main__":
    main()
