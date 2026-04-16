# Scripts

Place dataset preparation, export, and helper automation scripts in this directory.

Current entrypoint:

- `python /workspace/scripts/download_stanford_cars.py` downloads and extracts the Stanford Cars Kaggle dataset into `/workspace/stanford_dataset_raw`.
- `python /workspace/scripts/prepare_stanford_cars.py` reads images from `/workspace/stanford_dataset`, crops them by bbox, and writes `/workspace/stanford_dataset/cars_bbox_cropped.csv`.
- `python /workspace/scripts/unpack_stanford_archives.py` unpacks `/workspace/archives/cars_train.zip` and `/workspace/archives/cars_test.zip` into `/workspace/stanford_dataset` and optionally `/workspace/archives/devkit.zip` into `/workspace/devkit`.

