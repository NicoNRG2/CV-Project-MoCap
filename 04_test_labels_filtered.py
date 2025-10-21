import json
from pathlib import Path

# === PARAMETRI: cambiare con numero della camera desiderato e ricordarsi di scegliere id giusto per ogni camera!!! (mettere nel readme quale scegliere per ogni camera) ===
input_path = Path("04_labels13.json")   # percorso del file originale
output_path = Path("04_label13_filtered.json")     # percorso di salvataggio

# === CARICA JSON ===
with open(input_path, "r") as f:
    data = json.load(f)

# === FILTRA LE PERSONE CON ID == 2 ===
filtered_images = []
for img in data["images"]:
    persons_id1 = [p for p in img["persons"] if p.get("id") == 1] #SCEGLIERE ID DA TENERE
    if persons_id1:  # aggiungi solo immagini dove esiste id==1
        filtered_images.append({
            "file": img["file"],
            "width": img["width"],
            "height": img["height"],
            "persons": persons_id1
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
