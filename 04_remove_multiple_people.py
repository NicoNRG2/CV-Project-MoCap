import json
from pathlib import Path
import argparse

# === PARSER DEGLI ARGOMENTI ===
parser = argparse.ArgumentParser(description="Filtra persone con uno specifico ID da un file JSON di annotazioni.")
parser.add_argument("--input", type=Path, help="Percorso del file JSON di input")
parser.add_argument("--output", type=Path, help="Percorso del file JSON di output")
parser.add_argument("--keep_id", type=int, help="ID della persona da mantenere (es. 1 o 2)")

args = parser.parse_args()

# === CARICA JSON ===
with open(args.input, "r") as f:
    data = json.load(f)

# === FILTRA LE PERSONE CON ID SPECIFICATO ===
filtered_images = []
for img in data["images"]:
    persons_filtered = [p for p in img["persons"] if p.get("id") == args.keep_id]
    if persons_filtered:  # aggiungi solo immagini dove esiste almeno una persona con l'ID scelto
        filtered_images.append({
            "file": img["file"],
            "width": img["width"],
            "height": img["height"],
            "persons": persons_filtered
        })

# === CREA IL NUOVO JSON ===
filtered_data = {
    "meta": data.get("meta", {}),
    "images": filtered_images
}

# === SALVA SU FILE ===
with open(args.output, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"✅ File salvato in: {args.output}")
print(f"📸 Immagini totali nel file originale: {len(data['images'])}")
print(f"📸 Immagini con id={args.keep_id}: {len(filtered_images)}")