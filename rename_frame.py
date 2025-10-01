import json

# Read the JSON file
input_file = "c:/Users/nicol/Desktop/Computer Vision/CVLaboratories2024-2025/CV-Project-MoCap/dati_tuta_8p3.json"
output_file = "c:/Users/nicol/Desktop/Computer Vision/CVLaboratories2024-2025/CV-Project-MoCap/dati_tuta_8p3_renamed.json"

# Load JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Create new dictionary with renamed frames
new_data = {}
for i, (old_key, value) in enumerate(data.items(), start=1):
    new_key = f"frame_{i:04d}"  # Create new key with padding zeros
    new_data[new_key] = value

# Save the renamed JSON
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=2)

print(f"File saved as {output_file}")