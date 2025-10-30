"""
Create animated spheres in Blender from mocap/tracking data stored in a JSON file

Important: this script can only be executed inside Blender's scripting environment.
"""

import bpy
import json
import math
from mathutils import Vector

# ============== PARAMETERS ==============
#JSON_PATH = r"C:\Users\nicol\Desktop\03_final_mocap.json"
#FBX_OUT   = r"C:\Users\nicol\Desktop\anim_mocap_spheres.fbx"
JSON_PATH = r"C:\Users\nicol\Desktop\03_final_triangulation.json"
FBX_OUT   = r"C:\Users\nicol\Desktop\anim_triangulation_spheres.fbx"

FPS       = 12             # frame rate
SCALE     = 0.001          # 0.001 = millimeters -> meters
SPH_RADIUS = 0.04          # sphere radius
COLLECTION_NAME = "MoCap_Spheres"
#MATERIAL_NAME   = "MocapRed"
MATERIAL_NAME   = "MocapBlue"

# Use linear interpolation between keyframes (recommended for mocap)
USE_LINEAR_INTERP = True
# =======================================

# Remove all mesh/empty objects from the scene (optional)
def clean_scene():
    for o in list(bpy.data.objects):
        if o.type in {'MESH', 'EMPTY'} and o.users_scene:
            bpy.data.objects.remove(o, do_unlink=True)
# Get or create a dedicated collection for mocap spheres
def ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll
# Get or create a simple red material
def ensure_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        if hasattr(mat, "diffuse_color"):
            #mat.diffuse_color = (1.0, 0.2, 0.2, 1.0)   # Red
            mat.diffuse_color = (0.1, 0.3, 1.0, 1.0)    # Blue
    return mat
# Load frames from JSON and return them sorted by frame index.
def load_frames(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    def frame_key(k):
        try:
            return int(k.split("_")[-1])
        except:
            return k

    ordered_keys = sorted(data.keys(), key=frame_key)
    frames = [data[k] for k in ordered_keys]
    return frames, ordered_keys

# Create one sphere per joint and return them as a list.
def create_joint_spheres(n_joints, collection, radius, material=None):
    spheres = []
    for j in range(n_joints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"joint_{j:03d}"

        # Move to target collection
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collection.objects.link(obj)

        if material:
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

        spheres.append(obj)
    return spheres

# Set all keyframes of an object to linear interpolation.
def set_linear_interpolation(obj):
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'


def main():
    # 1) Clean up the scene
    clean_scene()

    # 2) Load mocap data
    frames, frame_keys = load_frames(JSON_PATH)
    if not frames:
        raise RuntimeError("No frames found in the JSON!")

    # 3) Determine frame/joint counts
    n_frames = len(frames)
    n_joints = len(frames[0])
    for f in frames:
        n_joints = min(n_joints, len(f))
    if n_joints == 0:
        raise RuntimeError("No joints found in the first frame!")

    print(f"{n_frames} frames, {n_joints} joints per frame.")

    # 4) Scene setup
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = n_frames

    # 5) Create collection and material
    coll = ensure_collection(COLLECTION_NAME)
    mat = ensure_material(MATERIAL_NAME)

    # 6) Create spheres for joints
    spheres = create_joint_spheres(n_joints, coll, SPH_RADIUS, mat)

    # 7) Animate spheres (insert location keyframes for every frame)
    for i, joints in enumerate(frames):
        frame_num = i + 1
        for j in range(n_joints):
            x, y, z = joints[j]
            loc = Vector((x * SCALE, y * SCALE, z * SCALE))
            spheres[j].location = loc
            spheres[j].keyframe_insert(data_path="location", frame=frame_num)

    # 8) Apply linear interpolation if requested
    if USE_LINEAR_INTERP:
        for obj in spheres:
            set_linear_interpolation(obj)

    # 9) Export to FBX
    bpy.ops.export_scene.fbx(
        filepath=FBX_OUT,
        use_selection=False,
        apply_unit_scale=True,
        bake_anim=True,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        path_mode='AUTO',
        add_leaf_bones=False,
    )

    print(f"FBX export completed: {FBX_OUT}")


if __name__ == "__main__":
    main()
