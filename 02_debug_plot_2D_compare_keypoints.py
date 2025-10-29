"""
Plots, on the same figure, rectified COCO keypoints vs reprojected keypoints for a given image_id. 
Useful to visually check reprojection consistency.

USAGE:
> python 02_debug_plot_2D_compare_keypoints.py [frame_number]
    n.b. frame_number can be from 1 to 48
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_coco_annotations(path):
    """
    Load the COCO JSON and return:
      - annots: list of 'annotation' dictionaries
      - images: dict image_id → 'image' dictionary
    """
    data = json.load(open(path, 'r'))
    annots = data['annotations']
    images = {img['id']: img for img in data['images']}
    return annots, images

def get_keypoints_for_image(annots, image_id):
    """
    Search in annots for the single annotation with image_id, extract keypoints Nx3.
    Returns array (N_joints, 3), or None if not found.
    """
    for ann in annots:
        if ann['image_id'] == image_id:
            kp = np.array(ann['keypoints'], dtype=float).reshape(-1, 3)
            return kp
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Compare rectified vs reprojected keypoints for the same image_id")
    parser.add_argument("image_id", type=int, help="ID of the image to plot")
    parser.add_argument("--rectified", default="temp/02_temp/02_annotations.coco.rectified.json",
                        help="Path to the rectified COCO file")
    parser.add_argument("--reproj", default="temp/02_temp/02_reprojected_annotations.json",
                        help="Path to the COCO file with reprojected keypoints")
    args = parser.parse_args()

    # 1) Load annotations
    rect_annots, rect_images = load_coco_annotations(args.rectified)
    reproj_annots, reproj_images = load_coco_annotations(args.reproj)

    # 2) Extract keypoints
    kp_rect = get_keypoints_for_image(rect_annots, args.image_id)
    kp_reproj = get_keypoints_for_image(reproj_annots, args.image_id)

    if kp_rect is None:
        print(f"[ERROR] No rectified annotation for image_id {args.image_id}")
        return
    if kp_reproj is None:
        print(f"[ERROR] No reprojected annotation for image_id {args.image_id}")
        return

    # 3) Get image info for title and size (optional)
    img_info = rect_images.get(args.image_id, {})
    fname = img_info.get('file_name', None)

    # 4) Prepare XY data
    xy_rect   = kp_rect[:,:2]
    xy_reproj = kp_reproj[:,:2]

    # 5) Plot
    plt.figure(figsize=(6,6))
    plt.scatter(xy_rect[:,0],   xy_rect[:,1],   c='g', marker='o', label='Rectified GT')
    plt.scatter(xy_reproj[:,0], xy_reproj[:,1], c='r', marker='x', label='Reprojected')
    plt.legend(loc='upper right')
    plt.title(f"Keypoints comparison on image_id {args.image_id}" + (f"\n{fname}" if fname else ""))
    plt.gca().invert_yaxis()  # image coordinates: (0,0) at top-left
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

