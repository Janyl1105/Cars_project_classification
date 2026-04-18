from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

from src.data.download import detect_dataset_layout
from src.data.image_io import safe_imread
from src.models.multihead_classifier import COLOR_CLASSES, TYPE_CLASSES

try:
    import cv2
except Exception:
    cv2 = None

TYPE2ID = {name: index for index, name in enumerate(TYPE_CLASSES)}
COLOR2ID = {name: index for index, name in enumerate(COLOR_CLASSES)}


def infer_type_from_name(name: str) -> str:
    value = name.lower()
    if "suv" in value or "sport utility" in value:
        return "suv"
    if "hatchback" in value:
        return "hatchback"
    if "wagon" in value:
        return "wagon"
    if "coupe" in value:
        return "coupe"
    if "convertible" in value or "cabriolet" in value:
        return "convertible"
    if "van" in value or "minivan" in value:
        return "van"
    if "pickup" in value or "pick-up" in value:
        return "pickup"
    if "truck" in value:
        return "truck"
    if any(token in value for token in ["911", "ferrari", "lamborghini", "mclaren", "gt-r", "gtr", "corvette"]):
        return "sports"
    if "sedan" in value:
        return "sedan"
    return "other"


def _central_crop(image_bgr: np.ndarray, frac: float = 0.65) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    crop_height = int(height * frac)
    crop_width = int(width * frac)
    y1 = (height - crop_height) // 2
    x1 = (width - crop_width) // 2
    return image_bgr[y1 : y1 + crop_height, x1 : x1 + crop_width]
