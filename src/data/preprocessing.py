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


def dominant_color_label_hsv_fast(image_bgr: np.ndarray | None) -> str:
    if image_bgr is None or image_bgr.size == 0 or cv2 is None:
        return "other"

    image = cv2.resize(image_bgr, (128, 128))
    crop = _central_crop(image)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    mask = ((value > 50) & (saturation > 35)) | (value > 220)
    hue = hue[mask]
    saturation = saturation[mask]
    value = value[mask]

    if len(hue) < 20:
        return "other"

    hue_median = int(np.median(hue))
    saturation_median = int(np.median(saturation))
    value_median = int(np.median(value))

    if value_median > 220 and saturation_median < 55:
        return "white"
    if value_median < 60:
        return "black"
    if saturation_median < 45:
        return "gray" if value_median < 150 else "silver"
    if hue_median < 10 or hue_median > 170:
        return "red"
    if 10 <= hue_median < 25:
        return "yellow"
    if 25 <= hue_median < 85:
        return "green"
    if 85 <= hue_median < 130:
        return "blue"
    return "other"


def dominant_color_label_bgr_kmeans(image_bgr: np.ndarray | None, k: int = 3) -> str:
    if image_bgr is None or image_bgr.size == 0 or cv2 is None:
        return "other"

    image = cv2.resize(image_bgr, (128, 128))
    crop = _central_crop(image)
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return dominant_color_label_hsv_fast(image_bgr)

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)].astype(np.uint8)
    return dominant_color_label_hsv_fast(dominant.reshape(1, 1, 3))


def crop_by_bbox(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: float = 0.0) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    if pad > 0:
        pad_x = int((x2 - x1) * pad / 2)
        pad_y = int((y2 - y1) * pad / 2)
        x1 = max(0, x1 - pad_x)
        x2 = min(width, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(height, y2 + pad_y)
    else:
        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


def _mat_to_class_names(meta_path: Path) -> list[str]:
    names = sio.loadmat(meta_path)["class_names"][0]
    values = []
    for item in names:
        if isinstance(item, np.ndarray):
            values.append(str(item[0]))
        else:
            values.append(str(item))
    return values


def _load_annotations(annos_path: Path):
    return sio.loadmat(annos_path)["annotations"][0]


def build_annotation_frame(dataset_root: str | Path) -> pd.DataFrame:
    layout = detect_dataset_layout(dataset_root)
    class_names = _mat_to_class_names(layout["devkit"] / "cars_meta.mat")

    rows: list[dict[str, object]] = []
    split_specs = [
        ("train", layout["train"], layout["devkit"] / "cars_train_annos.mat"),
        ("test", layout["test"], layout["devkit"] / "cars_test_annos.mat"),
    ]

    for source_split, image_dir, annos_path in split_specs:
        annotations = _load_annotations(annos_path)
        for annotation in annotations:
            file_name = str(annotation["fname"][0])
            bbox_x1 = int(annotation["bbox_x1"][0][0])
            bbox_y1 = int(annotation["bbox_y1"][0][0])
            bbox_x2 = int(annotation["bbox_x2"][0][0])
            bbox_y2 = int(annotation["bbox_y2"][0][0])
            class_id = int(annotation["class"][0][0]) - 1 if "class" in annotation.dtype.names else -1
            class_label = class_names[class_id] if class_id >= 0 else "unknown"
            image_path = image_dir / file_name
            rows.append(
                {
                    "source_split": source_split,
                    "image_path": str(image_path),
                    "class_id": class_id,
                    "class_label": class_label,
                    "bbox_x1": bbox_x1,
                    "bbox_y1": bbox_y1,
                    "bbox_x2": bbox_x2,
                    "bbox_y2": bbox_y2,
                }
            )

    return pd.DataFrame(rows)


def crop_annotation_frame(frame: pd.DataFrame, out_images_dir: str | Path, pad: float = 0.0) -> pd.DataFrame:
    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sample_index = 0
    for row in frame.itertuples(index=False):
        image = safe_imread(str(row.image_path))
        if image is None:
            continue

        cropped = crop_by_bbox(image, row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2, pad=pad)
        if cropped is None:
            continue

        out_name = f"crop_{sample_index:07d}.jpg"
        out_path = out_images_dir / out_name
        if cv2 is None:
            continue
        success = cv2.imwrite(str(out_path), cropped)
        if not success:
            continue

        rows.append(
            {
                "image_path": str(out_path),
                "source_image_path": row.image_path,
                "source_split": row.source_split,
                "class_id": row.class_id,
                "class_label": row.class_label,
                "bbox_x1": row.bbox_x1,
                "bbox_y1": row.bbox_y1,
                "bbox_x2": row.bbox_x2,
                "bbox_y2": row.bbox_y2,
            }
        )
        sample_index += 1

    return pd.DataFrame(rows)


def add_type_color_labels(frame: pd.DataFrame, use_kmeans: bool = False) -> pd.DataFrame:
    frame = frame.copy()
    type_names: list[str] = []
    type_ids: list[int] = []
    color_names: list[str] = []
    color_ids: list[int] = []

    for row in frame.itertuples(index=False):
        type_name = infer_type_from_name(str(row.class_label)) if pd.notna(row.class_label) else "other"
        type_id = TYPE2ID.get(type_name, TYPE2ID["other"])

        image = safe_imread(str(row.image_path))
        if image is None:
            color_name = "other"
        elif use_kmeans:
            color_name = dominant_color_label_bgr_kmeans(image)
        else:
            color_name = dominant_color_label_hsv_fast(image)
        color_id = COLOR2ID.get(color_name, COLOR2ID["other"])

        type_names.append(type_name)
        type_ids.append(type_id)
        color_names.append(color_name)
        color_ids.append(color_id)

    frame["type_name"] = type_names
    frame["type_id"] = type_ids
    frame["color_name"] = color_names
    frame["color_id"] = color_ids
    return frame


def add_splits(
    frame: pd.DataFrame,
    strategy: str = "official",
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    frame = frame.copy()

    if strategy == "official":
        train_frame = frame[frame["source_split"] == "train"].copy()
        test_frame = frame[frame["source_split"] == "test"].copy()
        train_frame, val_frame = train_test_split(
            train_frame,
            test_size=val_size,
            stratify=train_frame["type_id"],
            random_state=random_state,
        )
        train_frame["split"] = "train"
        val_frame["split"] = "val"
        test_frame["split"] = "test"
        return pd.concat([train_frame, val_frame, test_frame], ignore_index=True)

    if strategy == "random80_10_10":
        train_frame, temp_frame = train_test_split(
            frame,
            test_size=val_size + test_size,
            stratify=frame["type_id"],
            random_state=random_state,
        )
        ratio = test_size / (val_size + test_size)
        val_frame, test_frame = train_test_split(
            temp_frame,
            test_size=ratio,
            stratify=temp_frame["type_id"],
            random_state=random_state,
        )
        train_frame["split"] = "train"
        val_frame["split"] = "val"
        test_frame["split"] = "test"
        return pd.concat([train_frame, val_frame, test_frame], ignore_index=True)

    raise ValueError(f"Unknown split strategy: {strategy}")


def prepare_cropped_dataset(
    dataset_root: str | Path,
    cropped_dir: str | Path,
    csv_path: str | Path,
    use_kmeans: bool = False,
    pad: float = 0.0,
    split_strategy: str = "official",
    random_state: int = 42,
) -> pd.DataFrame:
    annotation_frame = build_annotation_frame(dataset_root)
    cropped_frame = crop_annotation_frame(annotation_frame, cropped_dir, pad=pad)
    labeled_frame = add_type_color_labels(cropped_frame, use_kmeans=use_kmeans)
    final_frame = add_splits(labeled_frame, strategy=split_strategy, random_state=random_state)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_csv(csv_path, index=False)
    return final_frame
