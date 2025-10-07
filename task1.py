# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Author: Jack Searle (21502396)


import os
import cv2
import glob
import sys
from ultralytics import YOLO

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

def save_image(output_path, image):
    """Save image to output_path, creating directories if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[INFO] Cropped image saved at: {output_path}")

def get_image_list(image_path):
    """Return list of image file paths given a file or folder."""
    if os.path.isfile(image_path):
        _, ext = os.path.splitext(image_path)
        if ext in VALID_EXTS:
            return [image_path]
        else:
            print(f"[ERROR] {image_path} is not a valid image file.")
            sys.exit(1)

    elif os.path.isdir(image_path):
        filelist = glob.glob(os.path.join(image_path, '*'))
        return [f for f in filelist if os.path.splitext(f)[1] in VALID_EXTS]

    else:
        print(f"[ERROR] Input path {image_path} does not exist.")
        sys.exit(1)

def run_inference(model, frame, threshold=0.5):
    """Run YOLO inference and return the best bounding box as (x1,y1,x2,y2)."""
    results = model.predict(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > float(threshold):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

    if not detections:
        return None

    # Pick the highest confidence detection
    best = max(detections, key=lambda det: det['confidence'])
    return best['bbox']

def run_task1(image_path, config):
    """
    Task 1: Object Detection and Cropping
    Crops detected feature from image(s) and saves to outputs/task1/
    """
    print(f"[INFO] Running Task 1 on: {image_path}")
    model_path = config.get('model_path', 'data/task2YOLO.pt')
    model = YOLO(model_path)

    images = get_image_list(image_path)

    for img in images:
        # note img is file path string

        frame = cv2.imread(img)
        if frame is None:
            print(f"[WARNING] Could not read {img}, skipping.")
            continue

        bbox = run_inference(model, frame, threshold=config.get('threshold', 0.5))
        if bbox is None:
            print(f"[WARNING] No detections above threshold for {img}")
            continue

        # crop image by taking slice of array
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"[WARNING] Cropped image has zero size for {img}, skipping save.")
            continue

        output_dir = 'output/task1'
        img_base = os.path.splitext(os.path.basename(img))[0]
        output_name = f"bn{img_base[3]}.png"
        output_path = os.path.join(output_dir, output_name)
        save_image(output_path, cropped)