import json
from pathlib import Path

# === PARAMETRI ===
input_path = Path("04_labels2.json")   # percorso del file originale
output_path = Path("04_label2_filtered.json")     # percorso di salvataggio

# === CARICA JSON ===
with open(input_path, "r") as f:
    data = json.load(f)

# === FILTRA LE PERSONE CON ID == 2 ===
filtered_images = []
for img in data["images"]:
    persons_id2 = [p for p in img["persons"] if p.get("id") == 2]
    if persons_id2:  # aggiungi solo immagini dove esiste id==2
        filtered_images.append({
            "file": img["file"],
            "width": img["width"],
            "height": img["height"],
            "persons": persons_id2
        })

# === CREA IL NUOVO JSON ===
filtered_data = {
    "meta": data.get("meta", {}),
    "images": filtered_images
}

# === SALVA SU FILE ===
with open(output_path, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"✅ File salvato in: {output_path}")
print(f"📸 Immagini totali nel file originale: {len(data['images'])}")
print(f"📸 Immagini con id=2: {len(filtered_images)}")
