from pathlib import Path

from src.data.preprocessing import prepare_cropped_dataset

# Запуск: python -m src.data.prepare_stanford_cars
# Обрезает изображения по bbox-аннотациям и генерирует cars_bbox_cropped.csv
# ✅ Запускать после распаковки архивов, перед обучением


def main() -> None:
    dataset_root = Path("/workspace/stanford_dataset")
    # старый путь (относительный): project_root / "data" / "stanford_dataset"
    cropped_dir = dataset_root / "cars_cropped"
    csv_path = dataset_root / "cars_bbox_cropped.csv"

    frame = prepare_cropped_dataset(
        dataset_root=dataset_root,
        cropped_dir=cropped_dir,
        csv_path=csv_path,
        use_kmeans=False,
        pad=0.0,
        split_strategy="official",
        random_state=42,
    )

    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Saved cropped images to: {cropped_dir}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Total rows: {len(frame)}")
    print(frame["split"].value_counts().to_string())


if __name__ == "__main__":
    main()