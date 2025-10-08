# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Jack Searle (21502396)
# Disclaimer: AI debugging tools used (GitHub Copilot, ChatGPT) e.g. /explain, /fix, /suggest
# Starting point of skeleton code provided by COMP3007 (almost completely rewritten)

# imports
import os
import sys
import cv2
from ultralytics import YOLO

# Import functions from other tasks
from task1 import run_inference as run_inference_task1, get_image_list as get_image_list_task1
from task2 import run_inference as run_inference_task2
from task3 import run_inference as run_inference_task3, CHAR_DICT, load_model

# write content string to txt file
def save_output(output_path, content):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"[INFO] Output saved at: {output_path}")

def run_task4(image_path, config):
    """
    Task 4: Complete Perception Pipeline imgx.png -> imgx.txt.
    - If image is negative (no building number detected by Task 1), produce no output.
    - If positive, output a text file with recognised building number.
    """
    print(f"[INFO] Running Task 4 on: {image_path}")

    # Load models for all tasks

    #YOLOV8s models for image detection tasks
    model_tsk1 = YOLO(config.get('model_path_tsk1', 'data/task1YOLO.pt'))
    model_tsk2 = YOLO(config.get('model_path_tsk2', 'data/task2YOLO.pt'))
    model_tsk3 = load_model(config.get('model_path_tsk3', 'data/digit_classifier.pth'))

    # small CNN model for digit classification task
    '''from my_model_defs import DigitCNN
    import torch
    model_tsk3 = DigitCNN()
    model_tsk3.load_state_dict(torch.load(config.get('model_path_tsk3', 'data/digit_classifier.pth')))
    model_tsk3.eval()'''

    # get list of image(s) file paths
    images = get_image_list_task1(image_path)

    if not images:
        print(f"[ERROR] No valid images found in {image_path}")
        return

    #run through whole pipeline for each image in image list
    for img_path in images:

        # Read image to BGR format (3D numpy array)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARNING] Could not read {img_path}, skipping.")
            continue

        # --- Task 1: Detect building number ---
        bbox = run_inference_task1(model_tsk1, frame, threshold=config.get('threshold', 0.5))
        if bbox is None:
            print(f"[INFO] Negative image detected ({os.path.basename(img_path)}), no output.")
            continue  # Skip if no detection

        # crop to building number
        x1, y1, x2, y2 = bbox
        cropped_bn = frame[y1:y2, x1:x2]

        # --- Task 2: Detect individual digits ---
        detections = run_inference_task2(model_tsk2, cropped_bn, threshold=config.get('threshold', 0.5))
        if detections is None or len(detections) == 0:
            print(f"[WARNING] No digit detections for {img_path}, skipping.")
            continue    # skip if no detections

        # Sort left-to-right to maintain digit order
        sorted_detections = sorted(detections, key=lambda d: d['bbox'][0])

        # --- Task 3: Classify digits ---
        recognised_chars = []
        for det in sorted_detections:

            #crop every digit in building number
            digit_crop = cropped_bn[det['bbox'][1]:det['bbox'][3], det['bbox'][0]:det['bbox'][2]]
            temp_filename = "_temp_digit.png"
            cv2.imwrite(temp_filename, digit_crop)

            cls_idx = run_inference_task3(model_tsk3, temp_filename, threshold=config.get('threshold', 0.5))
            if cls_idx is not None:

                #append to output char list
                recognised_chars.append(CHAR_DICT[cls_idx])
            else:
                recognised_chars.append("?")  # fallback

            os.remove(temp_filename)

        # list -> string
        bn_string = "".join(recognised_chars)

        # --- Output ---
        output_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        save_output(os.path.join("output/task4", output_name), bn_string)

if __name__ == "__main__":
    print("umm hello go run assignment.py")