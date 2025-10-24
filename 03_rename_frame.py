import json

# Read the JSON file
input_file = "temp/03_temp/03_selected_keypoints_adapted_joints_48frames.json"
output_file = "temp/03_temp/03_final_mocap.json"

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