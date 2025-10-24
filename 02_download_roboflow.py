#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scarica un dataset ZIP da Roboflow (o qualunque URL), lo estrae,
rimuove lo zip e i file di README inutili.
Compatibile con Linux/macOS/Windows/PowerShell.
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
    """Scarica un file con una barra di avanzamento minimale."""
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
                                f"\rScaricando: {human_size(downloaded)} / {human_size(total)} ({pct:5.1f}%)"
                            )
                        else:
                            sys.stdout.write(f"\rScaricando: {human_size(downloaded)}")
                        sys.stdout.flush()
                        last_print = now
            if total:
                print(f"\rScaricando: {human_size(downloaded)} / {human_size(total)} (100.0%)")
            else:
                print(f"\rScaricando: {human_size(downloaded)}")
    except HTTPError as e:
        raise SystemExit(f"Errore HTTP {e.code}: {e.reason}")
    except URLError as e:
        raise SystemExit(f"Errore di rete: {e.reason}")
    except Exception as e:
        raise SystemExit(f"Download fallito: {e}")

def safe_unzip(zip_path: Path, out_dir: Path) -> None:
    """Estrae lo zip in modo sicuro (evita path traversal, compatibile con Windows)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            extracted_path = out_dir / member.filename
            # Prevenzione path traversal (converti in stringa prima di startswith)
            if not str(extracted_path.resolve()).startswith(str(out_dir.resolve())):
                raise SystemExit(f"Percorso sospetto nello zip: {member.filename}")
        zf.extractall(out_dir)

def remove_readme_files(out_dir: Path) -> None:
    """Rimuove i file README.dataset.txt e README.roboflow.txt se presenti."""
    targets = ["README.dataset.txt", "README.roboflow.txt"]
    removed_any = False
    for name in targets:
        path = out_dir / name
        if path.exists():
            try:
                path.unlink()
                print(f"Rimosso: {path}")
                removed_any = True
            except Exception as e:
                print(f"⚠️  Impossibile rimuovere {path}: {e}")
    if not removed_any:
        print("Nessun file README da rimuovere.")

def main():
    parser = argparse.ArgumentParser(description="Scarica, estrai e ripulisci un dataset ZIP.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL del file ZIP (segue redirect).")
    parser.add_argument("--outdir", default=".", help="Cartella di estrazione (default: current dir).")
    parser.add_argument("--filename", default="roboflow.zip", help="Nome file zip locale (default: roboflow.zip).")
    parser.add_argument("--overwrite", action="store_true", help="Sovrascrivi file zip se esiste.")
    args = parser.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    zip_path = out_dir / args.filename

    # 1️⃣ Scarico
    if zip_path.exists() and not args.overwrite:
        print(f"File già presente: {zip_path}. Usa --overwrite per riscaricare.")
    else:
        print(f"URL: {args.url}")
        print(f"Scarico in: {zip_path}")
        download(args.url, zip_path)

    # 2️⃣ Estraggo
    print(f"Estrazione in: {out_dir}")
    try:
        safe_unzip(zip_path, out_dir)
    except zipfile.BadZipFile:
        raise SystemExit("Archivio ZIP corrotto o non valido.")
    except Exception as e:
        raise SystemExit(f"Errore durante l'estrazione: {e}")

    # 3️⃣ Rimuovo lo zip
    try:
        zip_path.unlink()
        print(f"Rimosso: {zip_path}")
    except Exception as e:
        print(f"⚠️  Impossibile rimuovere lo zip ({e})")

    # 4️⃣ Rimuovo i README
    remove_readme_files(out_dir)

    print("✅ Tutto completato con successo!")

if __name__ == "__main__":
    main()
