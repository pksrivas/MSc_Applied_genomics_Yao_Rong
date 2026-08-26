import os
import cv2
import torch
import numpy as np
from segment_anything import (sam_model_registry, SamPredictor)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sam_checkpoint = os.path.join(BASE_DIR, "training", "models", "sam_vit_h_4b8939.pth")
model_type = "vit_h"
device = ("cuda"
    if torch.cuda.is_available()
    else "cpu")
sam = sam_model_registry[model_type](
    checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
current_image = None
current_image_rgb = None

def set_image(image_bgr):
    global current_image
    global current_image_rgb
    current_image = image_bgr
    current_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(current_image_rgb)

def segment_by_click(x, y):
    global current_image
    global current_image_rgb
    if current_image is None:
        return None, None, None
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)
    mask = masks[np.argmax(scores)]
    overlay = current_image_rgb.copy()
    overlay[mask] = [0, 255, 0]
    ys, xs = np.where(mask)
    x1 = np.min(xs)
    x2 = np.max(xs)
    y1 = np.min(ys)
    y2 = np.max(ys)
    crop = current_image_rgb[y1:y2, x1:x2]
    return overlay, crop, mask