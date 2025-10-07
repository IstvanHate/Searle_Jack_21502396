

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Jack Searle

import os
import cv2
import numpy as np
import glob
import sys

from ultralytics import YOLO


def save_output(output_path, content, output_type):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def run_inference(model, frame, threshold=0.5):
    # Run inference
    results = model(frame)

    # extract bounding boxes and confidence scores
    bounding_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            if confidence > threshold:
                bounding_boxes.append({x1, y1, x2, y2})

    #check for multiple detections, take highest
    if len(bounding_boxes) > 1:
        bounding_boxes = sort_confidence(bounding_boxes)
    bounding_box = bounding_boxes[0]

    return bounding_box

def sort_confidence(bounding_boxes):
    # Sort bounding boxes by confidence score in descending order
    return sorted(bounding_boxes, key=lambda x: x['confidence'], reverse=True)


def run_task1(image_path, config):
    ''' Task 1: Object Detection and Cropping '''

    print(f"Running Task 1 on image: {image_path} with config: {config}")

    # load model
    model = YOLO('data/task1YOLO.pt')

    # check image_path exists
    if (not os.path.exists(image_path)):
        print(f"Error: Image path {image_path} not found.")
        sys.exit(1)

    # determine if image_path is a directory or a file
    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']

    if os.path.isdir(image_path):
        source_type = 'directory'
    elif os.path.isfile(image_path):
        _, ext = os.path.splitext(image_path)
        if ext in img_ext_list:
            source_type = 'image'
        else:
            print(f'Input {image_path} is not a valid image file.')
            sys.exit(1)
    else:
        print(f'Input {image_path} is invalid.')
        sys.exit(1)

    #load up image(s)
    if source_type == 'image':
        #list is single item, just the image
        imgs_list = [image_path]
    elif source_type == 'folder':
        #use glob to find all valid files, append to list
        imgs_list = []
        filelist = glob.glob(image_path + '/*')
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)

    for img in imgs_list:

        #read image path
        frame = cv2.imread(img)

        #run image inference
        bounding_box = run_inference(model, frame)

        #crop image to bounding box
        x1, y1, x2, y2 = bounding_box
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            print(f"Warning: Cropped image has zero size for {img}. Skipping save.")
            continue

        # Save cropped image
        output_path = f"output/task1/cropped_{os.path.basename(img)}"
        save_output(output_path, cropped_img, output_type='image')
