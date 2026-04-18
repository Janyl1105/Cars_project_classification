from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

#поиск структуры датасета
def detect_dataset_layout(dataset_root: str | Path) -> dict[str, Path]:
    dataset_root = Path(dataset_root)

    candidates = [
        {
            "train": dataset_root / "data" / "stanford_dataset" / "train",
            "test": dataset_root / "data" / "stanford_dataset" / "test",
            "devkit": dataset_root / "data" / "devkit",
        },
        {
            "train": dataset_root / "data" / "stanford_dataset" / "cars_train",
            "test": dataset_root / "data" / "stanford_dataset" / "cars_test",
            "devkit": dataset_root / "data" / "devkit",
        },
        {
            "train": dataset_root / "stanford_dataset" / "train",
            "test": dataset_root / "stanford_dataset" / "test",
            "devkit": dataset_root / "devkit",
        },
        {
            "train": dataset_root / "cars_train",
            "test": dataset_root / "cars_test",
            "devkit": dataset_root / "devkit",
        },
        {
            "train": dataset_root / "train",
            "test": dataset_root / "test",
            "devkit": dataset_root / "devkit",
        },
        {
            "train": dataset_root / "cars_train",
            "test": dataset_root / "cars_test",
            "devkit": dataset_root.parent / "devkit",
        },
        {
            "train": dataset_root / "train",
            "test": dataset_root / "test",
            "devkit": dataset_root.parent / "devkit",
        },
        {
            "train": dataset_root / "data" / "cars_train",
            "test": dataset_root / "data" / "cars_test",
            "devkit": dataset_root / "data" / "devkit",
        },
        {
            "train": Path("/workspace/data/stanford_dataset/train"),
            "test": Path("/workspace/data/stanford_dataset/test"),
            "devkit": Path("/workspace/data/devkit"),
        },
        {
            "train": Path("/workspace/data/stanford_dataset/cars_train"),
            "test": Path("/workspace/data/stanford_dataset/cars_test"),
            "devkit": Path("/workspace/data/devkit"),
        },
        {
            "train": Path("/workspace/stanford_dataset/cars_train"),
            "test": Path("/workspace/stanford_dataset/cars_test"),
            "devkit": Path("/workspace/devkit"),
        },
    ]

    for candidate in candidates:
        if all(path.exists() for path in candidate.values()):
            return candidate

    raise FileNotFoundError(
        "Could not detect Stanford Cars layout. Expected train/test (or cars_train/cars_test) and devkit directories."
    )


def summarize_layout(layout: dict[str, Path]) -> dict[str, int]:
    return {
        "train_images": len(list(layout["train"].glob("*.jpg"))),
        "test_images": len(list(layout["test"].glob("*.jpg"))),
        "devkit_files": len(list(layout["devkit"].glob("*"))),
    }


def unpack_local_archives(
    train_zip: str | Path,
    test_zip: str | Path,
    dataset_dir: str | Path,
    devkit_zip: str | Path | None = None,
    devkit_dir: str | Path | None = None,
    combined_zip: str | Path | None = None,
) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_dir / "cars_train"
    test_dir = dataset_dir / "cars_test"

    if combined_zip is not None:
        combined_archive = Path(combined_zip)
        if not combined_archive.exists():
            raise FileNotFoundError(f"Archive not found: {combined_archive}")
        for target_dir in [train_dir, test_dir]:
            if target_dir.exists():
                shutil.rmtree(target_dir)
        with ZipFile(combined_archive) as archive:
            archive.extractall(dataset_dir)
    else:
        for archive_path, target_dir in [(Path(train_zip), train_dir), (Path(test_zip), test_dir)]:
            if not archive_path.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path) as archive:
                archive.extractall(target_dir)

    if devkit_zip is not None:
        devkit_archive = Path(devkit_zip)
        if not devkit_archive.exists():
            raise FileNotFoundError(f"Archive not found: {devkit_archive}")
        resolved_devkit_dir = Path(devkit_dir) if devkit_dir is not None else dataset_dir.parent / "devkit"
        if resolved_devkit_dir.exists():
            shutil.rmtree(resolved_devkit_dir)
        resolved_devkit_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(devkit_archive) as archive:
            archive.extractall(resolved_devkit_dir)

    return detect_dataset_layout(dataset_dir.parent if dataset_dir.name == "stanford_dataset" else dataset_dir)

