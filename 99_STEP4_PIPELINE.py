"""
STEP 04 pipeline runner

Esegue gli script della pipeline 04 in sequenza.
- I comandi "singoli" chiedono SEMPRE ENTER al termine (pause=True).
- I blocchi run_group(...) chiedono ENTER una sola volta alla fine del blocco.

USO:
> python 99_STEP4_PIPELINE.py
"""

import os
import sys
import subprocess
from colorama import init, Fore, Style

# Initialize colorama (for Windows compatibility)
init(autoreset=True)

PY = sys.executable  # use the same interpreter you use to launch this script

def run(cmd_list, desc="", echo_prefix="",  pause=False):
    """Esegue un comando; stop su errore. Pausa opzionale alla fine."""
    print(f"\n{Style.BRIGHT}{Fore.YELLOW}{echo_prefix}Running: {' '.join(cmd_list)}{Style.RESET_ALL}", flush=True)

    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"Error while executing: {' '.join(cmd_list)}")
        sys.exit(result.returncode)
    print(Fore.GREEN + Style.BRIGHT + f"Completed {desc}" , flush=True)
    if pause:
        input(Fore.WHITE + Style.BRIGHT + "Press ENTER to continue...")

def run_group(cmd_lists, pause_message=Fore.WHITE + Style.BRIGHT +"Press ENTER to continue..."):
    """Esegue più comandi in sequenza; UNA pausa alla fine."""
    for i, cmd in enumerate(cmd_lists, 1):
        run(cmd, echo_prefix=f"[{i}/{len(cmd_lists)}] ")
    input(Fore.WHITE + Style.BRIGHT + f"\n{Style.BRIGHT}{Fore.WHITE}{pause_message}")

def ask_keep_id(cam_label: str) -> int:
    """Chiede un intero > 0 da usare come keep_id."""
    while True:
        s = input(f"Enter the id for {cam_label}: ").strip()
        if s.isdigit() and int(s) > 0:
            return int(s)
        print("Invalid value. Try again (e.g., 1, 2, 3...).")

def ensure_dirs():
    os.makedirs("temp/04_temp", exist_ok=True)

def main():
    ensure_dirs()
    print(Fore.BLUE + Style.BRIGHT + "\n=== STEP 4 PIPELINE ===")

    # 1) Split images (SINGOLO, pausa)
    run([PY, "04_divide_images.py"],desc=Fore.GREEN + Style.BRIGHT +": splitted all rectified images into subfolders based on their camera ID (cam_2, cam_5, cam_8, cam_13).",  pause=True)

    # 2) YOLO pose su ciascuna camera (BLOCCO, UNA pausa alla fine)
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
    ], pause_message=Fore.GREEN + Style.BRIGHT +"Completed: run YOLO-Pose inference on each camera’s images to detect human keypoints and export them as JSON files."+Fore.WHITE+ Style.BRIGHT+"\nPress ENTER to continue...")

    # 3) Controllo visivo (BLOCCO, UNA pausa alla fine)
    run_group([
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_2",
         "--json", "temp/04_temp/labels2.json", "--outdir", "temp/04_temp/cam2"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_5",
         "--json", "temp/04_temp/labels5.json", "--outdir", "temp/04_temp/cam5"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_8",
         "--json", "temp/04_temp/labels8.json", "--outdir", "temp/04_temp/cam8"],
        [PY, "04_test_labels.py", "--images", "images_rectified/cam_13",
         "--json", "temp/04_temp/labels13.json", "--outdir", "temp/04_temp/cam13"],
    ], pause_message=Fore.GREEN + Style.BRIGHT +"Completed: visualized detected keypoints on sample images to verify YOLO-Pose results for each camera.\nPress ENTER to continue...")

    print("Go to temp/04_temp and, for each camera (cam2, cam5, cam8, cam13), enter the player's id (black tracksuit).")
    keep2 = ask_keep_id("cam_2")
    keep5 = ask_keep_id("cam_5")
    keep8 = ask_keep_id("cam_8")
    keep13 = ask_keep_id("cam_13")

    # 5) Filtra per keep_id (BLOCCO, UNA pausa alla fine)
    run_group([
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels2.json",
         "--output", "temp/04_temp/labels2_filtered.json", "--keep_id", str(keep2)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels5.json",
         "--output", "temp/04_temp/labels5_filtered.json", "--keep_id", str(keep5)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels8.json",
         "--output", "temp/04_temp/labels8_filtered.json", "--keep_id", str(keep8)],
        [PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels13.json",
         "--output", "temp/04_temp/labels13_filtered.json", "--keep_id", str(keep13)],
    ], pause_message=Fore.GREEN + Style.BRIGHT +"Completed: filtered frames containing multiple detections, keeping only the player of interest across all cameras.\nPress ENTER to continue...")

    # 6) Adattamento keypoint (BLOCCO, UNA pausa alla fine)
    run_group([
        [PY, "04_adapt_keypoint.py", "2"],
        [PY, "04_adapt_keypoint.py", "5"],
        [PY, "04_adapt_keypoint.py", "8"],
        [PY, "04_adapt_keypoint.py", "13"],
    ], pause_message=Fore.GREEN + Style.BRIGHT +"Completed: removes incompatible or irrelevant joints from the YOLO output to match the MoCap joint set.\nPress ENTER to continue...")

    # 7) Merge (SINGOLO, pausa)
    run([PY, "04_merge_pose_jsons_like_rectified.py",
         "temp/04_temp/labels2_filtered_adapted.json",
         "temp/04_temp/labels5_filtered_adapted.json",
         "temp/04_temp/labels8_filtered_adapted.json",
         "temp/04_temp/labels13_filtered_adapted.json",
         "--out", "temp/04_temp/annotations_yolo.json"],desc=Fore.GREEN + Style.BRIGHT +": merged all filtered YOLO JSONs (cam_2, cam_5, cam_8, cam_13) into a single COCO-style annotation file.", pause=True)

    # 8) Triangolazione (SINGOLO, pausa)
    run([PY, "02_triangulation.py",
         "--input", "temp/04_temp/annotations_yolo.json",
         "--output", "temp/04_temp/04_triangulated_yolo.json"], desc=Fore.GREEN + Style.BRIGHT +": triangulated 3D joint positions from the YOLO-Pose 2D detections (using the same script of step 2).", pause=True)

    # 9) Animazione YOLO (SINGOLO, pausa)
    run([PY, "04_animate_yolo.py",
         "--input", "temp/04_temp/04_triangulated_yolo.json",
         "--out", "temp/04_temp/04_yolo.gif", "--fps", "12"], desc=Fore.GREEN + Style.BRIGHT +": created a 3D animated GIF of the reconstructed skeleton from YOLO detections.", pause=True)

    # 10) Adattamento MoCap (SINGOLO, pausa)
    run([PY, "04_adapt_mocap.py"], desc=Fore.GREEN + Style.BRIGHT +": removed extra joints from the MoCap data to make it compatible with the YOLO skeleton.", pause=True)

    # 11) Confronto finale (SINGOLO, pausa)
    run([PY, "03_step3compare.py",
         "--mocap", "temp/04_temp/04_adapted_final_mocap.json",
         "--triang", "temp/04_temp/04_triangulated_yolo.json",
         "--align", "similarity"], desc=Fore.GREEN + Style.BRIGHT +": compared the triangulated YOLO skeleton with the adapted MoCap skeleton, aligning them via similarity transformation.", pause=True)

    print(Fore.BLUE + Style.BRIGHT + "\nDONE: all scripts have been executed successfully!")

if __name__ == "__main__":
    main()
