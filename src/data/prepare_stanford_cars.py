from pathlib import Path

from src.data.preprocessing import prepare_cropped_dataset
#он импортирует prepare_cropped_dataset из src.data.preprocessing
# берёт dataset_root, cropped_dir, csv_path
#он запускает весь процесс один раз
#То есть:
# для подготовки данных Обрезает изображения по аннотациям
#Организует в папки по классам
# ✅ После распаковки, перед обучением
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "stanford_dataset"
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