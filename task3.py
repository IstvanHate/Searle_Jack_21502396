

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
# Disclaimer: AI used
# This code was developed with the assistance of AI tools, including GitHub Copilot and ChatGPT.
# GitHub Copilot commands used: /fix, /explain, /refactor

import os
from xml.parsers.expat import model
import cv2
import glob
import sys

# torch extensions
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# my model definitions
from my_model_defs import DigitCNN

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

def load_model(model_path, device_str='cuda'):
    """
    fml
    Assumes the model's class (DigitCNN) is already imported.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found.")
        return None

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Load full model object
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    model.to(device)

    print(f"[INFO] Model loaded successfully from {model_path}")
    return model


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

def run_inference(model, img, threshold):
    """
    Run CNN inference on a single grayscale frame.

    Params:
        model      - Trained PyTorch classification model (expects grayscale).
                     The model should output logits for classification, typically as a tensor of shape [batch_size, num_classes].
        img        - Image file path (string)
        threshold  - Minimum probability [0–1] to accept prediction

    Returns:
        predicted_class (int) if above threshold, otherwise None
    """
    # Preprocessing
    # read image as grayscale (to match training with single channel)
    frame = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print(f"[WARNING] Could not read {img}, skipping.")
        return None

    # Resize to training dimension
    frame = cv2.resize(frame, (28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Add batch dimension (unsqueeze)
    frame_tensor = transform(frame).unsqueeze(0)  # add batch dim → [1, 1, H, W]

    # Move to same device as the model
    device = next(model.parameters()).device
    frame_tensor = frame_tensor.to(device)

    # --- Inference ---
    with torch.no_grad():
        outputs = model(frame_tensor)  # shape [1, num_classes]
        probs = torch.softmax(outputs, dim=1)  # convert logits -> probabilities
        confidence, predicted_idx = torch.max(probs, 1)  # take only best class
        confidence = confidence.item()
        predicted_class = predicted_idx.item()

    # --- Confidence check ---

    #just in case they supply an invalid test iamge for task 3 to throw you off
    if confidence < float(threshold):
        print(f"[INFO] No detections above threshold {threshold:.2f} (best was {confidence:.2f})")
        return None

    print(f"[DEBUG] Prediction: class {predicted_class}, confidence {confidence:.3f}")
    return predicted_class

def run_task3(image_path, config):
    """
    Task 3: Digit classification
    Classifies digit cropped from task 2 using a simple CNN model
    """
    print(f"[INFO] Running Task 3 on: {image_path}")
    model_path = config.get('model_path_tsk3', 'data/digit_classifier.pth')

    model = load_model(model_path, device_str='cuda')
    if model is None:
        print(f"[ERROR] Could not load model from {model_path}")
        sys.exit(1)

    # Get list of images
    images = get_image_list(image_path)

    if not images: #to this day idk if this or images = None is the best way
        print(f"[ERROR] No valid images found in {image_path}")
        sys.exit(1)

    for img in images:
        # note img is file path string

        detection = run_inference(model, img, threshold=config.get('threshold', 0.5))
        if detection is None:
            print(f"[WARNING] No detections above threshold for {img}")
            continue
        else:
            content = str(detection[0].item())  # convert tensor to int and then to string
            print(f"[INFO] Detected digit: {content} in {img}")
            # create output directory based on bnX where X is the number in the filename of img
            X = os.path.basename(img)[2]  # gets the number after 'bn'
            output_dir = f'output/task3/bn{X}'
            # create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # save each classified image as cx.txt in the output directory
            img_base = f'c{X}.txt'
            output_path = os.path.join(output_dir, img_base)
            save_output(output_path, content, output_type='txt')

if __name__ == "__main__":
    print("umm hello go run assignment.py")