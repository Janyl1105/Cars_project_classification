"""Data utilities package for Stanford Cars dataset."""

from src.data.download import detect_dataset_layout, summarize_layout, unpack_local_archives
from src.data.preprocessing import add_type_color_labels, build_annotation_frame, prepare_cropped_dataset

__all__ = [
    "detect_dataset_layout",
    "summarize_layout",
    "unpack_local_archives",
    "add_type_color_labels",
    "build_annotation_frame",
    "prepare_cropped_dataset",
]