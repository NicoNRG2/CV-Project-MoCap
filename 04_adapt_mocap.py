import json
from pathlib import Path

# === INPUT / OUTPUT FILES ===
input_path = Path("temp/03_temp/03_final_mocap.json")
output_path = Path("temp/04_temp/04_adapted_final_mocap.json")

# === Joint list originale (ordine del file) ===
JOINTS_ORIG = [
    'Hips', 'Spine', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LHand',
    'RShoulder', 'RElbow', 'RHand',
    'LHip', 'LKnee', 'LAnkle', 'LFoot',
    'RHip', 'RKnee', 'RAnkle', 'RFoot'
]

# === Joint da rimuovere ===
REMOVE = {'Hips', 'Spine', 'Neck', 'LFoot', 'RFoot'}

# === Nuovo ordine richiesto ===
JOINTS_NEW_ORDER = [
    'Head',
    'LShoulder', 'RShoulder',
    'LElbow', 'RElbow',
    'LHand', 'RHand',
    'LHip', 'RHip',
    'LKnee', 'RKnee',
    'LAnkle', 'RAnkle'
]

# === Creiamo mappa (joint -> indice originale) ===
idx_map = {name: i for i, name in enumerate(JOINTS_ORIG)}

# === Carichiamo il JSON ===
with open(input_path, 'r') as f:
    data = json.load(f)

# === Creiamo nuovo dizionario filtrato ===
filtered_data = {}

for frame, coords in data.items():
    new_frame = []
    for joint in JOINTS_NEW_ORDER:
        if joint in idx_map and joint not in REMOVE:
            new_frame.append(coords[idx_map[joint]])
    filtered_data[frame] = new_frame

# === Salviamo ===
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"✅ File salvato in: {output_path}")
