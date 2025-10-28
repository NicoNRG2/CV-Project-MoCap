import os
import sys
import subprocess

PY = sys.executable  # use the same interpreter that launches this script

def run(cmd_list):
    # Runs a command (list of args) and stops the pipeline on error.
    print("\nRunning:", " ".join(cmd_list))
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"Error while executing: {' '.join(cmd_list)}")
        sys.exit(result.returncode)
    print("Completed.")
    input("Press ENTER to continue to the next script...")

def ask_keep_id(cam_label: str) -> int:
    # Interactively asks for an integer > 0 to use as keep_id.
    
    while True:
       
        s = input(f"Enter the id for {cam_label}: ").strip()
        if s.isdigit() and int(s) > 0:
            return int(s)
        print("Invalid value. Try again (e.g., 1, 2, 3...).")

def ensure_dirs():
    os.makedirs("temp/04_temp", exist_ok=True)

def main():
    ensure_dirs()
    print("=== Starting YOLO Pose + Triangulation Pipeline (Step 04) ===")

    # 1) Split images into 4 folders
    run([PY, "04_divide_images.py"])

    # 2) YOLO pose on each camera
    run([PY, "04_yolo_pose.py", "--images", "images_rectified/cam_2",
         "--output", "temp/04_temp/labels2.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"])

    run([PY, "04_yolo_pose.py", "--images", "images_rectified/cam_5",
         "--output", "temp/04_temp/labels5.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"])

    run([PY, "04_yolo_pose.py", "--images", "images_rectified/cam_8",
         "--output", "temp/04_temp/labels8.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"])

    run([PY, "04_yolo_pose.py", "--images", "images_rectified/cam_13",
         "--output", "temp/04_temp/labels13.json", "--weights", "yolo11l-pose.pt",
         "--imgsz", "3840", "--conf", "0.20", "--device", "cuda:0"])

    # 3) Visual check of labels
    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_2",
         "--json", "temp/04_temp/labels2.json", "--outdir", "temp/04_temp/cam2"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_5",
         "--json", "temp/04_temp/labels5.json", "--outdir", "temp/04_temp/cam5"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_8",
         "--json", "temp/04_temp/labels8.json", "--outdir", "temp/04_temp/cam8"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_13",
         "--json", "temp/04_temp/labels13.json", "--outdir", "temp/04_temp/cam13"])

  
    print("Go to /temp/04_temp and, for each camera (cam2, cam5, cam8, cam13), enter the player's id (black tracksuit).")
    # 4) INTERACTIVE INPUT: keep_id for the 4 cameras
    keep2 = ask_keep_id("cam_2")
    keep5 = ask_keep_id("cam_5")
    keep8 = ask_keep_id("cam_8")
    keep13 = ask_keep_id("cam_13")

    # 5) Filter JSONs keeping only the selected keep_id
    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels2.json",
         "--output", "temp/04_temp/labels2_filtered.json", "--keep_id", str(keep2)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels5.json",
         "--output", "temp/04_temp/labels5_filtered.json", "--keep_id", str(keep5)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels8.json",
         "--output", "temp/04_temp/labels8_filtered.json", "--keep_id", str(keep8)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels13.json",
         "--output", "temp/04_temp/labels13_filtered.json", "--keep_id", str(keep13)])

    # 6) Remove/Adapt incompatible keypoints (one per camera)
    run([PY, "04_adapt_keypoint.py", "2"])
    run([PY, "04_adapt_keypoint.py", "5"])
    run([PY, "04_adapt_keypoint.py", "8"])
    run([PY, "04_adapt_keypoint.py", "13"])

    # 7) Merge poses (same order as rectified)
    run([PY, "04_merge_pose_jsons_like_rectified.py",
         "temp/04_temp/labels2_filtered_adapted.json",
         "temp/04_temp/labels5_filtered_adapted.json",
         "temp/04_temp/labels8_filtered_adapted.json",
         "temp/04_temp/labels13_filtered_adapted.json",
         "--out", "temp/04_temp/annotations_yolo.json"])

    # 8) Triangulation
    run([PY, "02_triangulation.py",
         "--input", "temp/04_temp/annotations_yolo.json",
         "--output", "temp/04_temp/04_triangulated_yolo.json"])

    # 9) YOLO animation
    run([PY, "04_animate_yolo.py",
         "--input", "temp/04_temp/04_triangulated_yolo.json",
         "--out", "temp/04_temp/04_yolo.gif", "--fps", "12"])

    # 10) Adapt MoCap (remove extra joints)
    run([PY, "04_adapt_mocap.py"])

    # 11) Final comparison (similarity alignment)
    run([PY, "03_step3compare.py",
         "--mocap", "temp/04_temp/04_adapted_final_mocap.json",
         "--triang", "temp/04_temp/04_triangulated_yolo.json",
         "--align", "similarity"])

    print("\nPipeline 04 completed successfully!")
    

if __name__ == "__main__":
    main()
