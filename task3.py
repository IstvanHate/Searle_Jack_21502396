

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

# Author: 21502396

import os
import cv2
import glob
import sys
from ultralytics import YOLO

# torch extensions
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}


def save_output(output_path, content, output_type='txt'):
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

def run_inference(model, frames, threshold=0.5):
    """Run YOLO inference and return all detections above threshold."""

    # for running inference
    model.eval()
    outputs = []

    # run inference
    with torch.no_grad():
        for frame in frames:
            outputs = model(frame)
            _, predicted = torch.max(outputs, 1)

    if len(outputs) == 0:
        print("[WARNING] No detections found.")
        outputs = None

    return outputs
def run_task2(image_path, config):
    """
    Task 1: Object Detection and Cropping
    Crops detected feature from image(s) and saves to outputs/task1/
    """
    print(f"[INFO] Running Task 1 on: {image_path}")
    model_path = config.get('model_path_tsk2', 'data/task2YOLO.pt')
    model = YOLO(model_path)

    images = get_image_list(image_path)

    for img in images:
        # note img is file path string

        frame = cv2.imread(img)
        if frame is None:
            print(f"[WARNING] Could not read {img}, skipping.")
            continue

        detections = run_inference(model, frame, threshold=config.get('threshold', 0.5))
        if detections is None:
            print(f"[WARNING] No detections above threshold for {img}")
            continue

        # crop image by taking slice of array for each detection
        #remember detection['bbox'][x] is (x1, y1, x2, y2)
        cropped_list = [frame[det['bbox'][1]:det['bbox'][3], det['bbox'][0]:det['bbox'][2]] for det in detections]

        for i, cropped in enumerate(cropped_list):
            # create output directory based on bnX where X is the number in the filename of img
            X = os.path.basename(img)[2]  # gets the number after 'bn'
            output_dir = f'output/task2/bn{X}'
            # create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # save each cropped image as cx.png in the output directory
            img_base = f'c{i}.png'
            output_path = os.path.join(output_dir, img_base)
            save_image(output_path, cropped)

if __name__ == "__main__":
    print("umm hello go run assignment.py")