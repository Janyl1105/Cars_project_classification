# Real-Time Car Detection And Multi-Class Classification

This workspace now contains a starter project for the graduation topic: real-time car detection plus attribute classification by type and color.

## Current status

The original experimentation still lives in `Untitled_ssh.ipynb`.

The new project scaffold moves the main ideas into reusable Python modules:

- `src/models/multihead_classifier.py` - multi-head classifier for type and color
- `src/models/lightning_module.py` - Lightning training module for multi-task learning
- `src/models/losses.py` - weighted CE and focal loss helpers
- `src/data/image_io.py` - robust image loading and standard transforms
- `src/data/datamodule.py` - dataset and datamodule for training
- `src/train.py` - classifier training entrypoint
- `src/inference/video_inference.py` - two-stage real-time pipeline: YOLO detector + classifier
- `configs/` - Hydra-style configuration files
- `docs/project_plan.md` - implementation roadmap
- `docs/report_outline.md` - technical report structure

## Recommended final architecture

This project should use a two-stage pipeline:

1. YOLO detects cars on each frame.
2. Each detected car crop is sent to a multi-head classifier.
3. The classifier predicts `type` and `color`.
4. The system renders boxes and labels on video frames.

This is the most practical path because the current notebook already contains a strong attribute-classification baseline.

## What still needs to be done

1. Move the remaining training logic from the notebook into Python modules.
2. Prepare YOLO-format annotations for vehicle detection.
3. Train and validate the detector.
4. Export the final inference stack to ONNX where appropriate.
5. Add DVC, Docker, and experiment tracking around the project.

## Suggested repo layout

```text
.github/
configs/
docs/
notebooks/
scripts/
src/
tests/
Untitled_ssh.ipynb
README.md
requirements.txt
```

## Minimal run example

Dataset download:

```bash
python scripts/download_stanford_cars.py
```

Dataset preparation and bbox cropping:

```bash
python scripts/prepare_stanford_cars.py
```

Current dataset layout used by the project:

```text
stanford_dataset/
  cars_train/
  cars_test/
  cars_cropped/
  cars_bbox_cropped.csv
devkit/
```

After dependencies are installed and model weights are prepared, the real-time video pipeline is intended to run like this:

```bash
python -m src.inference.video_inference \
  --video input.mp4 \
  --output output/annotated.mp4 \
  --detector weights/yolo.pt \
  --classifier weights/classifier.ckpt
```

Classifier training entrypoint:

```bash
python -m src.train data.csv_path=stanford_dataset/cars_bbox_cropped.csv
```

Evaluation entrypoint:

```bash
python -m src.eval checkpoint_path=output/checkpoints/best.ckpt
```
