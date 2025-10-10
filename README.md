# Multiview Motion Capture Evaluation Project

## 🏀 Project Goal

Estimate the player's 3D poses using a **multiview camera setup** recorded at *Sanbàpolis*, and evaluate the accuracy of the triangulated 3D skeletons by comparing them with **motion capture (MoCap)** data.

---

## 📋 Project Steps

### 1. Annotate Player’s Poses
- Each action is captured by **4 synchronized camera views** (`cam_2`, `cam_5`, `cam_8`, `cam_13`).
- One action per group of 2 people.
- Around **100 frames per person**.
- Annotate **only the person wearing the black MoCap suit**.
- Annotations were created using **Roboflow** and exported in **COCO JSON format**.

> ✅ Step completed: all frames have been annotated and downloaded in COCO JSON format.

---

### 2. 3D Player Reconstruction via Triangulation

Using the 2D keypoints annotated from the multiview cameras, we reconstruct the player’s 3D pose via **triangulation**.

#### Pipeline
1. **Rectify input videos and images**  
   The rectification removes lens distortion and aligns all cameras to a common epipolar geometry.  
   *(Important: the same transformation must also be applied to the ground-truth annotations).*

2. **Triangulation**  
   3D points are computed from corresponding 2D detections across views using the **Direct Linear Transform (DLT)** method.

3. **Visualization**  
   Display the reconstructed 3D skeleton for a given frame.

4. **Reprojection & Evaluation**  
   Reproject the triangulated 3D skeleton back to each camera view and compare it with the original 2D annotations using standard metrics:
   - **Mean Per Joint Position Error (MPJPE)**
   - **Mean Squared Error (MSE)**

#### Executed scripts
```bash
python 01_rectified_videos.py
python 01_rectified_images.py
python 01_rectified_annotations.py
python 01_draw_keypoint_over_frame_check.py

python 02_triangulation.py
python 02_plot_3d_skeleton.py [frame_number]
python 02_generate_reprojected_annotations.py
python 02_compute_reprojection_error.py
python 02_animate_triangulation.py  --input triangulated_3d_skeleton.json --out 02_triangulated_skeleton.gif --fps 12
```
### 3. Alignment with Motion Capture Data

The MoCap system and the multiview RGB setup are **not synchronized**, so alignment must be performed manually or algorithmically.

#### Procedure
1. Identify reference poses (e.g., player raising arms before a shot) in both datasets.
2. Match corresponding poses to align the MoCap and triangulation timelines.
3. Subsample and rename frames to achieve consistent frame rates and naming schemes.
4. Align and compare the 3D skeletons using the Kabsch–Umeyama algorithm for rigid alignment.

Executed scripts
```bash
python 03_animate_mocap.py                  # generate a video of the mocap data
python 03_cut_frames.py --input keypoints_mocap.json --output selected_keypoints.json --start 980 --end 1372                     # cut the shot part of our interest
python 03_adapt_skeleton.py selected_keypoints.json selected_keypoints_adapted_joints.json                 # remove extra bones
python 03_subsample_mocap.py -i selected_keypoints_adapted_joints.json -o selected_keypoints_adapted_joints_48frames.json --start 980 --step 8.3                # from 100 fps to 24 fps
python 03_rename_frame.py                   # e.g., frame_980 → frame_1
python 03_reorder_triangulation_joints.py   # order the triangulation joint as the mocap
python 03_step3compare.py --mocap final_mocap.json --triang final_triangulation.json --align similarity
```

| MoCap Frame | Triangulation Frame | Notes              |
|-------------|---------------------|--------------------|
| 1322        | 42                  | baseline alignment |
| 1372        | 48                  | 8.3 fps x 6 frames |
| 980         | 1                   | 8.3 fps x 41 frames|

### Evaluation Metrics
- Mean Per Joint Position Error (MPJPE)
- Mean Squared Error (MSE)
- Median Joint Error

## 🧩 Skeleton Mapping

### Motion Capture Keypoints
```bash
'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftForeArmRoll', 'LeftHand',
'RightShoulder', 'RightArm', 'RightForeArm', 'RightForeArmRoll', 'RightHand',
'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
```

Removed 6 extra joints:  
`Spine`, `Spine2`, `LeftShoulder`, `LeftForeArmRoll`, `RightShoulder`, `RightForeArmRoll`.

---

### Triangulation Keypoints
```bash
"Hips", "RHip", "RKnee", "RAnkle", "RFoot",
"LHip", "LKnee", "LAnkle", "LFoot",
"Spine", "Neck", "Head",
"RShoulder", "RElbow", "RHand",
"LShoulder", "LElbow", "LHand"
```

---

### Unified Skeleton Order (used for comparison)
```bash
'Hips', 'Spine', 'Neck', 'Head',
'LShoulder', 'LElbow', 'LHand',
'RShoulder', 'RElbow', 'RHand',
'LHip', 'LKnee', 'LAnkle', 'LFoot',
'RHip', 'RKnee', 'RAnkle', 'RFoot'
```

## 4. (Skipped) Human Pose Estimation

We planned to test a human pose estimation model (e.g., YOLO Pose), reprojecting its detections and comparing them with the MoCap ground truth.
However, this step was skipped because the YOLO Pose skeleton structure is not compatible with the MoCap skeleton (only limbs are comparable).

## Results
<p align="center">
  <img src="02_triangulated_skeleton.gif" width="45%" />
  <img src="03_mocap_skeleton.gif" width="45%" />
</p>

### Triangulation vs MoCap (final accuracy):

MPJPE 69.7 mm (mean), 69.8 mm (median)<br>
MSE 5767.8 mm², RMSE 75.3 mm<br>
Coherent 3D reconstruction with ~7–8 cm average joint error.

## 👥 Authors

### Group members:
Nicola Cappellaro
Riccardo Zannoni