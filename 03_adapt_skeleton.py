import json
import sys

def filter_coords(coords, remove_idx):
    """Rimuove le coordinate agli indici specificati."""
    return [c for i, c in enumerate(coords) if i not in remove_idx]

def main(input_file, output_file):
    # Indici da rimuovere (0-based, quindi 2→1, 4→3, 7→6, 10→9, 12→11, 15→14)
    to_remove = [1, 3, 6, 9, 11, 14]

    # Legge il file JSON di input
    with open(input_file, "r") as f:
        data = json.load(f)

    # Applica il filtro a ogni frame
    filtered_data = {}
    for frame, coords in data.items():
        filtered_data[frame] = filter_coords(coords, to_remove)

    # Salva il risultato su file JSON
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ Filtrati {len(to_remove)} punti. Risultato salvato in {output_file}")
    for frame in filtered_data:
        print(f"{frame}: {len(filtered_data[frame])} coordinate")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python adapt_skeleton.py input.json output.json")
    else:
        main(sys.argv[1], sys.argv[2])

# rimove 2,4,7,10,12,15 le coordinate dello scheletro in più del mocap
# python 03_adapt_skeleton.py selected_keypoints.json selected_keypoints_adapted_joints.json