import subprocess
import sys

# Lista dei comandi da eseguire in ordine
commands = [
    "python 03_cut_frames.py",
    "python 03_adapt_skeleton.py",
    "python 03_animate_mocap.py",
    "python 03_subsample_mocap.py",
    "python 03_rename_frame.py",
    "python 03_reorder_triangulation_joints.py",
    "python 03_step3compare.py",
    "python 03_animate_mocap.py --input temp/03_temp/03_final_triangulation.json --out temp/03_temp/03_final_triangulation.gif --fps 12 --rotate -90 --name \"Triangulated Skeleton\"",
    "python 03_animate_mocap.py --input temp/03_temp/03_final_mocap.json --out temp/03_temp/03_final_mocap.gif --fps 12"
    
]
def run_command(cmd):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione di: {cmd}")
        sys.exit(result.returncode)
    else:
        print(f"✅ Completato: {cmd}")
        input("👉 Premi INVIO per continuare al prossimo script...")

def main():
    print("=== STEP 3: Align with MoCap Data ===")
    for cmd in commands:
        run_command(cmd)
    print("\n🎉 Tutti gli script sono stati eseguiti con successo!")

if __name__ == "__main__":
    main()
