from pathlib import Path

# OPTIONAL: Требует Kaggle API.
# Альтернатива без API: python -m src.data.unpack_stanford_archives
#
# Настройка:
#   1. pip install kaggle
#   2. Получи ключ на https://www.kaggle.com/settings/account
#   3. Сохрани kaggle.json в ~/.kaggle/kaggle.json
#
# Затем раскомментируй код ниже и запусти:
#   python -m src.data.download_stanford_cars

"""
from src.data.download import detect_dataset_layout, download_kaggle_dataset, summarize_layout


def main() -> None:
    dataset = "ashc004/stanford-car-dataset"
    raw_dir = Path("/workspace/stanford_dataset_raw")

    print("Downloading Stanford Cars dataset from Kaggle...")
    extracted_root = download_kaggle_dataset(dataset, raw_dir, force=False)
    print(f"Downloaded to: {extracted_root}")

    try:
        layout = detect_dataset_layout(extracted_root)
    except FileNotFoundError:
        print("Download complete, but dataset layout not recognized.")
        return

    print("Dataset ready:")
    for name, path in layout.items():
        print(f"  {name}: {path}")

    stats = summarize_layout(layout)
    print("Counts:")
    for name, value in stats.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
"""

if __name__ == "__main__":
    print("Kaggle download is OPTIONAL. Use 'python -m src.data.unpack_stanford_archives' for local zip archives.")