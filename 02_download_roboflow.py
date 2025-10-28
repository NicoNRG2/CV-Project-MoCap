"""
Downloads a ZIP dataset (e.g., from Roboflow), extracts it safely, deletes the ZIP, and removes optional README files.
Works on Linux/macOS/Windows/PowerShell.

USAGE:
> python 02_download_roboflow.py

"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

DEFAULT_URL = "https://app.roboflow.com/ds/fFEVpEaLNe?key=gZjXq6fQYi"

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def download(url: str, dest: Path, timeout: int = 60) -> None:
    #Download a file with a minimal progress indicator.
    req = Request(url, headers={"User-Agent": "python-downloader/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            total = resp.length
            chunk = 64 * 1024
            downloaded = 0

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                last_print = time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    now = time.time()
                    if now - last_print >= 0.1:
                        if total:
                            pct = downloaded * 100 / total
                            sys.stdout.write(
                                f"\rDownloading: {human_size(downloaded)} / {human_size(total)} ({pct:5.1f}%)"
                            )
                        else:
                            sys.stdout.write(f"\rDownloading: {human_size(downloaded)}")
                        sys.stdout.flush()
                        last_print = now
            if total:
                print(f"Downloading: {human_size(downloaded)} / {human_size(total)} (100.0%)")
            else:
                print(f"Downloading: {human_size(downloaded)}")
    except HTTPError as e:
        raise SystemExit(f"HTTP error {e.code}: {e.reason}")
    except URLError as e:
        raise SystemExit(f"network error: {e.reason}")
    except Exception as e:
        raise SystemExit(f"Download failed: {e}")

def safe_unzip(zip_path: Path, out_dir: Path) -> None:
    #Safely extract the zip (prevents path traversal; Windows-compatible).
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            extracted_path = out_dir / member.filename
            # Prevent path traversal (convert to string before startswith)
            if not str(extracted_path.resolve()).startswith(str(out_dir.resolve())):
                raise SystemExit(f"Suspicious path in zip: {member.filename}")
        zf.extractall(out_dir)

def remove_readme_files(out_dir: Path) -> None:
    #Remove README.dataset.txt and README.roboflow.txt if present.
    targets = ["README.dataset.txt", "README.roboflow.txt"]
    removed_any = False
    for name in targets:
        path = out_dir / name
        if path.exists():
            try:
                path.unlink()
                print(f"Removed: {path}")
                removed_any = True
            except Exception as e:
                print(f"Not possible to remove {path}: {e}")
    if not removed_any:
        print("No README files to remove.")

def main():
    parser = argparse.ArgumentParser(description="Download, extract, and clean a Roboflow ZIP dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="ZIP file URL (follows redirects).")
    parser.add_argument("--outdir", default=".", help="Extraction folder (default: current dir).")
    parser.add_argument("--filename", default="roboflow.zip", help="Local zip filename (default: roboflow.zip).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite zip if it exists.")
    args = parser.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    zip_path = out_dir / args.filename

    # 1 - Download
    if zip_path.exists() and not args.overwrite:
        print(f"File already exists: {zip_path}. Use --overwrite to re-download.")
    else:
        print(f"URL: {args.url}")
        print(f"Downloading to: {zip_path}")
        download(args.url, zip_path)

    # 2 - Extract
    print(f"Extraction in: {out_dir}")
    try:
        safe_unzip(zip_path, out_dir)
    except zipfile.BadZipFile:
        raise SystemExit("Corrupted or invalid ZIP archive.")
    except Exception as e:
        raise SystemExit(f"Error during the extraction: {e}")

    # 3 - Remove the zip
    try:
        zip_path.unlink()
        print(f"Removed: {zip_path}")
    except Exception as e:
        print(f"Not possible to remove the zip ({e})")

    # 4 - Remove READMEs
    remove_readme_files(out_dir)

    print("All Done!")

if __name__ == "__main__":
    main()

