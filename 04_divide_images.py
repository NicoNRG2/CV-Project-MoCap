#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import shutil

CAM_IDS = {"2", "5", "8", "13"}
PATTERN = re.compile(r"^out(2|5|8|13)_", re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser(
        description="Divide le immagini in sottocartelle cam_2, cam_5, cam_8, cam_13 in base al nome outN_*."
    )
    parser.add_argument("--images_dir", default="images_rectified", type=Path, help="Percorso alla cartella images_rectified") 
    parser.add_argument("--dry-run", action="store_true", help="Mostra cosa verrebbe fatto senza spostare i file")
    args = parser.parse_args()

    root = args.images_dir
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Errore: {root} non esiste o non è una cartella.")

    # Crea le sottocartelle se non esistono
    dest_dirs = {cid: root / f"cam_{cid}" for cid in CAM_IDS}
    for d in dest_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    # Scorri solo i file direttamente nella cartella (non nelle sottocartelle)
    for p in root.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.match(p.name)
        if not m:
            skipped += 1
            continue

        cam_id = m.group(1)
        dest_dir = dest_dirs[cam_id]
        dest = dest_dir / p.name

        # Gestione di eventuali conflitti di nome: rinomina con suffisso numerico
        if dest.exists():
            base = dest.stem
            ext = dest.suffix
            i = 1
            while True:
                candidate = dest_dir / f"{base}__{i}{ext}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1

        if args.dry_run:
            print(f"[DRY] {p.name}  ->  {dest}")
        else:
            shutil.move(str(p), str(dest))
        moved += 1

    print(f"Completato. Spostati: {moved} file. Ignorati: {skipped} file (nome non conforme a outN_*).")
    print(f"Cartelle di destinazione: {', '.join(str(d) for d in dest_dirs.values())}")

if __name__ == "__main__":
    main()
