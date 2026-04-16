from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torchvision.transforms as T
except Exception:
    T = None


def safe_imread(path: str):
    if cv2 is None:
        return None
    try:
        return cv2.imread(path)
    except Exception:
        return None


def bgr_to_rgb(image):
    if image is None:
        return None
    if cv2 is None:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image_any(path: str):
    image = safe_imread(path)
    if image is not None:
        return bgr_to_rgb(image)

    try:
        with Image.open(path) as handle:
            return np.array(handle.convert("RGB"))
    except Exception:
        return None


def load_pil_image(path: str) -> Image.Image:
    array = read_image_any(path)
    if array is None:
        raise ValueError(f"Unable to read image: {path}")
    return Image.fromarray(array)


def build_eval_transforms(image_size: int = 224):
    if T is None:
        raise ImportError("torchvision is required for transforms")
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def list_image_files(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]