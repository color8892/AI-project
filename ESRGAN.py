"""
This script enhances image quality using the ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model.

Functionality:
- Downloads the ESRGAN model from TensorFlow Hub.
- Defines functions for preprocessing and postprocessing images.
- Enhances the image quality by passing images through the ESRGAN model.
- Replaces the original image files with the enhanced versions.

"""

import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

def preprocess_image(image):
    image = image.convert("RGB")
    return (np.asarray(image, dtype=np.float32) / 255.0)[np.newaxis, ...]

def postprocess_image(image):
    return np.clip(image[0] * 255, 0, 255).astype(np.uint8)

for subdir, dirs, files in os.walk(r"C:\Users\mnb35\Downloads\PYTHON\SIGN\CAR"):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".png") or filepath.endswith(".jpeg") or filepath.endswith(".jpg"):
            print(f"Processing file: {filepath}")
            img = Image.open(filepath)
            width, height = img.size
            if width > 100 or height > 50:
                print(f"Skipping file due to size: {filepath}")
                continue
            processed_img = preprocess_image(img)
            sr_img = model(processed_img)
            sr_img = postprocess_image(sr_img)
            result_img = Image.fromarray(sr_img)
            result_img.save(filepath)
