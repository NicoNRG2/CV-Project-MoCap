"""
This Python script automatically sorts and moves images from a folder (default: images_rectified) into four subfolders based on their camera ID found in the filename.

USAGE:
> python 04_divide_images.py

"""
import argparse
import re
from pathlib import Path
import shutil

CAM_IDS = {"2", "5", "8", "13"}
PATTERN = re.compile(r"^out(2|5|8|13)_", re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser(
        description="Divide images into subfolders cam_2, cam_5, cam_8, cam_13 based on the filename outN_*."
    )
    parser.add_argument("--images_dir", default="images_rectified", type=Path, help="Path to the images_rectified folder") 
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually moving the files")
    args = parser.parse_args()

    root = args.images_dir
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Error: {root} does not exist or is not a directory.")

    # Create subfolders if they do not exist
    dest_dirs = {cid: root / f"cam_{cid}" for cid in CAM_IDS}
    for d in dest_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    # Iterate only over files directly in the folder (not in subfolders)
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

        # Handle name conflicts: rename with numeric suffix
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

    print(f"Moved: {moved} files. Skipped: {skipped} files (name not matching outN_* pattern).")
    print(f"Destination folders: {', '.join(str(d) for d in dest_dirs.values())}")

if __name__ == "__main__":
    main()
