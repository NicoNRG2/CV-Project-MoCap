import cv2
import json
import argparse
import os

# Draw keypoints and skeleton on a frame given COCO annotations

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def draw_keypoints_on_image(img, ann, skeleton, kp_radius=4, color=(0,255,0)):
    """
    Draws keypoints and skeleton connections on the image.
    ann: single annotation dict with 'keypoints'
    skeleton: list of [i,j] index pairs (1-based indices)
    """
    kpts = ann['keypoints']  # flat list: [x1,y1,v1, x2,y2,v2, ...]
    num = len(kpts) // 3
    pts = []
    for i in range(num):
        x, y, v = kpts[3*i:3*i+3]
        if v > 0:
            pts.append((int(round(x)), int(round(y))))
            # draw circle
            cv2.circle(img, (int(round(x)), int(round(y))), kp_radius, color, -1)
        else:
            pts.append(None)
    # draw skeleton
    for link in skeleton:
        # link indices in COCO are 1-based
        idx1, idx2 = link[0]-1, link[1]-1
        p1, p2 = pts[idx1], pts[idx2]
        if p1 is not None and p2 is not None:
            cv2.line(img, p1, p2, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser(description='Draw keypoints on an image frame')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--annotations', required=True, help='Path to COCO-format JSON annotations')
    parser.add_argument('--image_id', type=int, required=True, help='Image ID to overlay')
    parser.add_argument('--output', default=None, help='Path to save the output image')
    args = parser.parse_args()

    data = load_annotations(args.annotations)

    # find image entry
    img_entry = next((im for im in data['images'] if im['id'] == args.image_id), None)
    if img_entry is None:
        print(f"Image ID {args.image_id} not found in annotations")
        return

    # load image
    img_path = args.image
    if not os.path.isfile(img_path):
        print(f"Image file not found: {img_path}")
        return
    img = cv2.imread(img_path)

    # get skeleton from categories
    cat = data['categories'][0]
    skeleton = cat.get('skeleton', [])

    # find all annotations for this image
    anns = [ann for ann in data['annotations'] if ann['image_id'] == args.image_id]
    if not anns:
        print(f"No annotations found for image ID {args.image_id}")
        return

    # draw each annotation
    for ann in anns:
        img = draw_keypoints_on_image(img, ann, skeleton)

    # show or save
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"Annotated image saved to {args.output}")
    else:
        cv2.imshow('Keypoints', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
