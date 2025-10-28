import bpy
import json
import math
from mathutils import Vector

# ============== PARAMETRI ==============
JSON_PATH = r"C:\Users\nicol\Desktop\03_final_mocap.json"
FBX_OUT   = r"C:\Users\nicol\Desktop\anim_mocap_spheres.fbx"

FPS       = 12
SCALE     = 0.001          # 0.001 = millimetri -> metri
SPH_RADIUS = 0.04          # raggio delle sfere
COLLECTION_NAME = "MoCap_Spheres"
MATERIAL_NAME   = "MocapRed"

# Interpolazione lineare (più prevedibile per dati di mocap)
USE_LINEAR_INTERP = True
# =======================================

# -- pulizia scena di base (opzionale: lascia Camera/Lights)
def clean_scene():
    for o in list(bpy.data.objects):
        if o.type in {'MESH', 'EMPTY'} and o.users_scene:
            bpy.data.objects.remove(o, do_unlink=True)

# -- crea (o ottiene) una Collection dedicata
def ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

# -- materiale semplice per visibilità
def ensure_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        # colore base (rosso tenue)
        if hasattr(mat, "diffuse_color"):
            mat.diffuse_color = (1.0, 0.2, 0.2, 1.0)
    return mat

# -- lettura JSON e ordinamento frame
def load_frames(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Chiavi del tipo "frame_0001", "frame_0002", ...
    # Ordiniamo in base al numero (naturale)
    def frame_key(k):
        # estrai la parte numerica alla fine della chiave
        # es. "frame_0001" -> 1
        try:
            return int(k.split("_")[-1])
        except:
            return k

    ordered_keys = sorted(data.keys(), key=frame_key)
    frames = [data[k] for k in ordered_keys]
    return frames, ordered_keys

# -- crea N sfere (una per joint index) e le restituisce in una lista
def create_joint_spheres(n_joints, collection, radius, material=None):
    spheres = []
    for j in range(n_joints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"joint_{j:03d}"
        # link nella collection dedicata (e rimuovi dalla principale)
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

# -- keyframe lineari
def set_linear_interpolation(obj):
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'

# -- main
def main():
    # 1) pulizia (facoltativa)
    clean_scene()

    # 2) carica i dati
    frames, frame_keys = load_frames(JSON_PATH)
    if not frames:
        raise RuntimeError("Nessun frame nel JSON!")

    # 3) dimensioni: n_frame e n_joints
    n_frames = len(frames)
    n_joints = len(frames[0])
    # Verifica consistenza: se qualche frame ha meno joint, si usa il minimo comune
    for f in frames:
        n_joints = min(n_joints, len(f))
    if n_joints == 0:
        raise RuntimeError("Nessun joint trovato nel primo frame valido!")

    print(f"Caricati {n_frames} frame, {n_joints} joint per frame.")

    # 4) setup scena (FPS e range)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = n_frames

    # 5) collection + materiale
    coll = ensure_collection(COLLECTION_NAME)
    mat = ensure_material(MATERIAL_NAME)

    # 6) crea le sfere
    spheres = create_joint_spheres(n_joints, coll, SPH_RADIUS, mat)

    # 7) inserisci keyframe per TUTTI i frame
    #    Il JSON ha coordinate [x,y,z] per joint; applichiamo la scala.
    #    frame Blender parte da 1, mappiamo frame_i -> i+1
    for i, joints in enumerate(frames):
        frame_num = i + 1
        # sicurezza: tronca/usa solo i primi n_joints
        for j in range(n_joints):
            x, y, z = joints[j]
            loc = Vector((x * SCALE, y * SCALE, z * SCALE))
            spheres[j].location = loc
            spheres[j].keyframe_insert(data_path="location", frame=frame_num)

    # 8) opzionale: imposta interpolazione lineare
    if USE_LINEAR_INTERP:
        for obj in spheres:
            set_linear_interpolation(obj)

    # 9) esporta FBX con animazione
    bpy.ops.export_scene.fbx(
        filepath=FBX_OUT,
        use_selection=False,           # esporta tutto
        apply_unit_scale=True,
        bake_anim=True,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        path_mode='AUTO',
        add_leaf_bones=False,
    )

    print(f"Esportazione FBX completata: {FBX_OUT}")

# Esegui
if __name__ == "__main__":
    main()
