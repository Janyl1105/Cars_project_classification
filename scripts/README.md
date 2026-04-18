# Scripts

Place dataset preparation, export, and helper automation scripts in this directory.

Current entrypoint:

* `python /workspace/scripts/download_stanford_cars.py` downloads and extracts the Stanford Cars Kaggle dataset into `/workspace/stanford_dataset_raw`.
* `python /workspace/scripts/prepare_stanford_cars.py` reads images from `/workspace/stanford_dataset`, crops them by bbox, and writes `/workspace/stanford_dataset/cars_bbox_cropped.csv`.
* `python /workspace/scripts/unpack_stanford_archives.py` unpacks `/workspace/archives/cars_train.zip` and `/workspace/archives/cars_test.zip` into `/workspace/stanford_dataset` and optionally `/workspace/archives/devkit.zip` into `/workspace/devkit`.

Recommended order for SSH/local archive setup:

1. Transfer `cars_train.zip` and `cars_test.zip` to the remote host.
2. Run:

```bash
python /workspace/scripts/unpack_stanford_archives.py
```

3. Then run:

```bash
python /workspace/scripts/prepare_stanford_cars.py
```

4. After that, train or evaluate with the normal commands from the main README.

If you already have `/workspace/devkit` extracted on the remote host, `devkit.zip` is not required.

This avoids downloading or extracting large files directly over SSH and keeps the remote workflow lightweight.

