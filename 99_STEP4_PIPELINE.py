import os
import sys
import subprocess

PY = sys.executable  # usa lo stesso interprete con cui lanci questo script

def run(cmd_list):
    """Esegue un comando (lista di argomenti) e ferma la pipeline in caso di errore."""
    print("\n🚀 Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione: {' '.join(cmd_list)}")
        sys.exit(result.returncode)
    print("✅ Completato.")
    input("👉 Premi INVIO per continuare al prossimo script...")

def ask_keep_id(cam_label: str) -> int:
    """Chiede interattivamente un intero >0 per keep_id."""
    
    while True:
       
        s = input(f"👉 Inserisci l'id per {cam_label}: ").strip()
        if s.isdigit() and int(s) > 0:
            return int(s)
        print("⚠️  Valore non valido. Riprova (es. 1, 2, 3...).")

def ensure_dirs():
    os.makedirs("temp/04_temp", exist_ok=True)

def main():
    ensure_dirs()
    print("=== Starting YOLO Pose + Triangulation Pipeline (Step 04) ===")

    # 1) Divide immagini nelle 4 cartelle
    run([PY, "04_divide_images.py"])

    # 2) YOLO pose su ciascuna camera
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

    # 3) Visual check delle etichette
    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_2",
         "--json", "temp/04_temp/labels2.json", "--outdir", "temp/04_temp/cam2"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_5",
         "--json", "temp/04_temp/labels5.json", "--outdir", "temp/04_temp/cam5"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_8",
         "--json", "temp/04_temp/labels8.json", "--outdir", "temp/04_temp/cam8"])

    run([PY, "04_test_labels.py", "--images", "images_rectified/cam_13",
         "--json", "temp/04_temp/labels13.json", "--outdir", "temp/04_temp/cam13"])

  
    print("Entra in /temp/04_temp e per ogni camera (cam2, cam5, cam8, cam13), inserisci l'id del giocatore con la tuta")
    # 4) INPUT INTERATTIVO: keep_id per le 4 camere
    keep2 = ask_keep_id("cam_2")
    keep5 = ask_keep_id("cam_5")
    keep8 = ask_keep_id("cam_8")
    keep13 = ask_keep_id("cam_13")

    # 5) Filtra i JSON tenendo solo il keep_id scelto
    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels2.json",
         "--output", "temp/04_temp/labels2_filtered.json", "--keep_id", str(keep2)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels5.json",
         "--output", "temp/04_temp/labels5_filtered.json", "--keep_id", str(keep5)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels8.json",
         "--output", "temp/04_temp/labels8_filtered.json", "--keep_id", str(keep8)])

    run([PY, "04_remove_multiple_people.py", "--input", "temp/04_temp/labels13.json",
         "--output", "temp/04_temp/labels13_filtered.json", "--keep_id", str(keep13)])

    # 6) Rimuove/Adatta keypoint incompatibili (uno per camera)
    run([PY, "04_adapt_keypoint.py", "2"])
    run([PY, "04_adapt_keypoint.py", "5"])
    run([PY, "04_adapt_keypoint.py", "8"])
    run([PY, "04_adapt_keypoint.py", "13"])

    # 7) Merge delle pose (ordine come le rettificate)
    run([PY, "04_merge_pose_jsons_like_rectified.py",
         "temp/04_temp/labels2_filtered_adapted.json",
         "temp/04_temp/labels5_filtered_adapted.json",
         "temp/04_temp/labels8_filtered_adapted.json",
         "temp/04_temp/labels13_filtered_adapted.json",
         "--out", "temp/04_temp/annotations_yolo.json"])

    # 8) Triangolazione
    run([PY, "02_triangulation.py",
         "--input", "temp/04_temp/annotations_yolo.json",
         "--output", "temp/04_temp/04_triangulated_yolo.json"])

    # 9) Animazione YOLO
    run([PY, "04_animate_yolo.py",
         "--input", "temp/04_temp/04_triangulated_yolo.json",
         "--out", "temp/04_temp/04_yolo.gif", "--fps", "12"])

    # 10) Adatta MoCap (rimozione joint extra)
    run([PY, "04_adapt_mocap.py"])

    # 11) Confronto finale (allineamento di similarità)
    run([PY, "03_step3compare.py",
         "--mocap", "temp/04_temp/04_adapted_final_mocap.json",
         "--triang", "temp/04_temp/04_triangulated_yolo.json",
         "--align", "similarity"])

    print("\n🎉 Pipeline 04 completata con successo!")
    

if __name__ == "__main__":
    main()
