from pathlib import Path

# OPTIONAL: This script requires Kaggle API setup.
# If you have a local cars_dataset.zip, use scripts/unpack_stanford_archives.py instead.
# 
# To use this script:
# 1. Install kaggle: pip install kaggle
# 2. Get API credentials from https://www.kaggle.com/settings/account
# 3. Save kaggle.json to ~/.kaggle/kaggle.json
#
# Then uncomment the code below and run: python scripts/download_stanford_cars.py

"""
from src.data.download import download_kaggle_dataset, detect_dataset_layout, summarize_layout

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = "ashc004/stanford-car-dataset"
    raw_dir = project_root / "data" / "stanford_dataset_raw"

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
    print("Kaggle download is OPTIONAL. Use scripts/unpack_stanford_archives.py for local datasets.")