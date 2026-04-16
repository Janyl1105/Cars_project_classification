import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from src.data.image_io import build_eval_transforms
from src.models.multihead_classifier import ClassifierPredictor


def draw_prediction(frame, xyxy, label: str, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def crop_boxes(frame, boxes):
    crops = []
    coords = []
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(frame[y1:y2, x1:x2])
        coords.append((x1, y1, x2, y2))
    return crops, coords


def main():
    parser = argparse.ArgumentParser(description="Real-time car detection and attribute classification")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--detector", required=True, help="YOLO weights path")
    parser.add_argument("--classifier", required=True, help="Classifier checkpoint path")
    parser.add_argument("--backbone", default="resnet50", help="Classifier backbone name")
    parser.add_argument("--image-size", type=int, default=224, help="Classifier crop size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-id", type=int, default=2, help="Detector class id for car if the detector is COCO-trained")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.detector)
    classifier = ClassifierPredictor(args.classifier, device=args.device, backbone_name=args.backbone)
    transform = build_eval_transforms(args.image_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.predict(frame, conf=args.conf, verbose=False, device=args.device)[0]
        selected_boxes = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id != args.class_id:
                continue
            selected_boxes.append(box.xyxy[0].tolist())

        crops_bgr, coords = crop_boxes(frame, selected_boxes)
        if crops_bgr:
            batch = []
            for crop in crops_bgr:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = transform(crop_rgb)
                batch.append(tensor)
            batch_tensor = torch.stack(batch, dim=0)
            predictions = classifier.predict(batch_tensor)

            for xyxy, type_name, color_name in zip(coords, predictions["type_name"], predictions["color_name"]):
                draw_prediction(frame, xyxy, f"car | {type_name} | {color_name}")

        writer.write(frame)

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()