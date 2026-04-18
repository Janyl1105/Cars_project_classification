from pathlib import Path

from src.data.download import summarize_layout, unpack_local_archives

#Берёт уже скачанные zip-файлы
#Распаковывает их в нужную папку
#✅ Когда вы скачали zip локально и передали на сервер

def main() -> None:
    dataset_dir = Path("/workspace/data/stanford_dataset")
    devkit_dir = Path("/workspace/data/devkit")

    # Check for cars_dataset.zip in /workspace/tests and unzip it first
    cars_dataset_zip = Path("/workspace/tests/cars_dataset.zip")
    if cars_dataset_zip.exists():
        print(f"Found {cars_dataset_zip}, unpacking...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        from zipfile import ZipFile
        with ZipFile(cars_dataset_zip, 'r') as z:
            z.extractall(dataset_dir)
        print(f"Unpacked to {dataset_dir}")
        layout = {
            "train": dataset_dir / "train",
            "test": dataset_dir / "test",
            "devkit": devkit_dir,
        }
        from src.data.download import summarize_layout
        counts = summarize_layout(layout)
        print("Dataset ready. Counts:")
        for name, value in counts.items():
            print(f"  {name}: {value}")
        return

    candidate_roots = [Path("/workspace/archives"), dataset_dir, Path("/workspace")]

    def find_first(name: str) -> Path | None:
        for root in candidate_roots:
            candidate = root / name
            if candidate.exists():
                return candidate
        return None

    combined_zip = find_first("cars_train_test.zip")
    train_zip = find_first("cars_train.zip")
    test_zip = find_first("cars_test.zip")

    if combined_zip is None and (train_zip is None or test_zip is None):
        raise FileNotFoundError(
            "Provide either cars_train_test.zip or both cars_train.zip and cars_test.zip in /workspace, /workspace/archives, or /workspace/data/stanford_dataset."
        )

    layout = unpack_local_archives(
        train_zip=train_zip or "",
        test_zip=test_zip or "",
        dataset_dir=dataset_dir,
        combined_zip=combined_zip,
        devkit_dir=devkit_dir,
    )

    print("Archives unpacked successfully.")