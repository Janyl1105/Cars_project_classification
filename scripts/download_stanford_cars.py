from pathlib import Path

from src.data.download import detect_dataset_layout, download_kaggle_dataset, summarize_layout


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = "ashc004/stanford-car-dataset"
    raw_dir = project_root / "stanford_dataset_raw"

    extracted_root = download_kaggle_dataset(dataset, raw_dir, force=False)
    print(f"Downloaded to: {extracted_root}")
    print(f"Project root: {project_root}")

    try:
        layout = detect_dataset_layout(extracted_root)
    except FileNotFoundError:
        print("Archive downloaded, but the ready-to-train layout was not detected inside the extracted folder.")
        return

    print("Detected layout:")
    for name, path in layout.items():
        print(f"  {name}: {path}")

    stats = summarize_layout(layout)
    print("Counts:")
    for name, value in stats.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
