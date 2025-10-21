import json
from pathlib import Path
import sys

if len(sys.argv) != 2:
    print("Uso: python filter_keypoints.py <numero>")
    sys.exit(1)

num = sys.argv[1]

input_path = Path(f"04_temp/labels{num}_filtered.json")
output_path = Path(f"04_temp/labels{num}_filtered_adapted.json")

# Tutti i 17 joint nell'ordine originale YOLO Pose
ALL_JOINTS = [
    "Naso",
    "Occhio sinistro",
    "Occhio destro",
    "Orecchio sinistro",
    "Orecchio destro",
    "Spalla sinistra",
    "Spalla destra",
    "Gomito sinistro",
    "Gomito destro",
    "Polso sinistro",
    "Polso destro",
    "Anca sinistra",
    "Anca destra",
    "Ginocchio sinistro",
    "Ginocchio destro",
    "Caviglia sinistra",
    "Caviglia destra"
]

# Indici (0-based) dei joint da tenere
KEEP_JOINTS = [
    "Naso",
    "Spalla sinistra", "Spalla destra",
    "Gomito sinistro", "Gomito destro",
    "Polso sinistro", "Polso destro",
    "Anca sinistra", "Anca destra",
    "Ginocchio sinistro", "Ginocchio destro",
    "Caviglia sinistra", "Caviglia destra"
]
KEEP_INDICES = [ALL_JOINTS.index(j) for j in KEEP_JOINTS]

# === LETTURA FILE ===
with open(input_path, "r") as f:
    data = json.load(f)

# === FILTRAGGIO ===
for image in data.get("images", []):
    for person in image.get("persons", []):
        kpts = person.get("keypoints", [])
        # Tieni solo i joint selezionati
        filtered_kpts = [kpts[i] for i in KEEP_INDICES if i < len(kpts)]
        person["keypoints"] = filtered_kpts

# === SCRITTURA OUTPUT ===
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ File salvato in: {output_path}")
