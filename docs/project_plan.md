# Project Plan

## Objective

Build a real-time system that:

1. detects cars in video frames,
2. classifies each detected car by type,
3. classifies each detected car by color,
4. writes an annotated output video.

## Why the two-stage approach

The notebook already contains a working attribute-classification pipeline. Reusing it reduces project risk.

Recommended design:

1. YOLO for object detection.
2. Multi-head classifier for car type and color.
3. Video pipeline that combines both models.

## Work packages

### WP1. Dataset consolidation

- Verify remote dataset completeness.
- Standardize paths for `devkit`, `cars_train`, and `cars_test`.
- Generate CSVs with image paths, class labels, type labels, and color labels.

### WP2. Detection dataset

- Convert bounding box annotations into YOLO format.
- Create train/val/test manifests for detection.
- Train `yolo11n` or `yolo11s` as the detector baseline.

### WP3. Attribute classifier

- Move notebook classifier code into Python modules.
- Keep multi-head output:
  - head 1: `type`
  - head 2: `color`
- Support checkpoint loading for inference.

### WP4. Real-time inference

- Read input video.
- Run detector per frame.
- Crop detected cars.
- Run attribute classifier on crops.
- Draw bounding boxes and labels.
- Save processed video.

### WP5. MLOps packaging

- Port code into a `lightning-hydra-template` compatible structure.
- Add Hydra configs.
- Add DVC tracking for data and models.
- Add Dockerfile.
- Export models to ONNX.

## Deliverables

1. GitHub repository with clean structure.
2. DVC-tracked dataset and model artifacts.
3. Trained detector weights.
4. Trained multi-head classifier weights.
5. ONNX export.
6. Final annotated video.
7. Technical report.