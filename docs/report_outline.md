# Technical Report Outline

## Title

Real-Time Car Detection And Multi-Class Attribute Classification Using A Two-Stage Deep Learning Pipeline

## 1. Introduction

- Problem statement
- Motivation for intelligent transport systems and video analytics
- Project goals
- Main contributions

## 2. Related Work

- Real-time object detection
- YOLO family for transportation tasks
- Attribute classification for vehicles
- Multi-task learning for type and color prediction

## 3. Dataset And Preprocessing

- Dataset sources
- Bounding box annotations
- Type and color label generation
- Data splitting strategy
- Data augmentation

## 4. Methodology

### 4.1 Detection stage

- Detector choice: YOLOv11n or YOLOv11s
- Fine-tuning setup
- Detection metrics

### 4.2 Classification stage

- Multi-head classifier architecture
- Backbone choice
- Type and color heads
- Losses and imbalance handling

### 4.3 Inference pipeline

- Frame-by-frame detection
- Crop extraction
- Attribute prediction
- Visualization and output video generation

## 5. Experimental Setup

- Hardware and software
- Training configuration
- Hyperparameters
- Logging and reproducibility tools

## 6. Results

- Detection performance: mAP
- Classification performance: accuracy, precision, recall, F1
- Runtime performance: FPS and latency
- Qualitative examples

## 7. Discussion

- Strengths of the two-stage approach
- Failure cases
- Accuracy versus latency tradeoff
- Limits of the current dataset

## 8. Conclusion

- Summary of results
- Practical value of the system
- Future work

## Appendix

- Config tables
- Extra visualizations
- Example commands