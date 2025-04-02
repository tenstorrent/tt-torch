# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import os

sys.path.append("tests/models/yolov10/src")

import cv2
import torch
import urllib.request
import supervision as sv
from src.ultralytics import YOLOv10

# Create directories if they don't exist
os.makedirs("tests/models/yolov10/weights", exist_ok=True)
os.makedirs("tests/models/yolov10/data", exist_ok=True)

# Define paths
weights_path = "tests/models/yolov10/weights/yolov10n.pt"
image_path = "tests/models/yolov10/data/dog.jpeg"

# Download weights file if it doesn't exist
if not os.path.exists(weights_path):
    print("Downloading YOLOv10 weights...")
    urllib.request.urlretrieve(
        "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
        weights_path,
    )

# Download image file if it doesn't exist
if not os.path.exists(image_path):
    print("Downloading sample image...")
    urllib.request.urlretrieve(
        "https://media.roboflow.com/notebooks/examples/dog.jpeg", image_path
    )
model = YOLOv10(weights_path)
image = cv2.imread(image_path)
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
cv2.imwrite("tests/models/yolov10/data/annotated_dog.jpeg", annotated_image)
