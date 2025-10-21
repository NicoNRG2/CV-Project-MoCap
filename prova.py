from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
#image size 1280




# Predict with the model
results = model("images_rectified/cam_5/out5_frame_0047_png.rf.54ae154e2678707107d7c43f4ef75b40.jpg", imgsz=3840)  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)

    #print reults
    print("xy:", xy)
    print("xyn:", xyn)
    print("kpts:", kpts)
