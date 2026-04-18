from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}")

def ensure_kaggle_credentials() -> Path:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Put kaggle.json into ~/.kaggle/kaggle.json on the remote machine."
        )
    return kaggle_json

def download_kaggle_dataset(dataset: str, out_dir: str | Path, force: bool = False) -> Path:
    ensure_kaggle_credentials()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = dataset.split("/")[-1]
    extract_dir = out_dir / dataset_slug
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        return extract_dir

    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--force"]
    _run_command(command)

    zip_candidates = sorted(out_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_candidates:
        raise FileNotFoundError(f"No zip archive found in {out_dir} after downloading {dataset}.")

    zip_path = zip_candidates[0]
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(extract_dir))
    return extract_dir

def detect_dataset_layout(dataset_root: str | Path) -> dict[str, Path]:
    dataset_root = Path(dataset_root)

    candidates = [
        {
            "train": dataset_root / "stanford_dataset" / "cars_train",
            "test": dataset_root / "stanford_dataset" / "cars_test",
            "devkit": dataset_root / "devkit",
        },
        {
            "train": dataset_root / "cars_train",
            "test": dataset_root / "cars_test",
            "devkit": dataset_root.parent / "devkit",
        },
        {
            "train": dataset_root / "data" / "cars_train",
            "test": dataset_root / "data" / "cars_test",
            "devkit": dataset_root / "devkit",
        },
        {
            "train": Path("/workspace/stanford_dataset/cars_train"),
            "test": Path("/workspace/stanford_dataset/cars_test"),
            "devkit": Path("/workspace/devkit"),
        },
        {
            "train": Path("/workspace/data/cars_train"),
            "test": Path("/workspace/data/cars_test"),
            "devkit": Path("/workspace/devkit"),
        },
    ]

    for candidate in candidates:
        if all(path.exists() for path in candidate.values()):
            return candidate

    raise FileNotFoundError(
        "Could not detect Stanford Cars layout. Expected cars_train, cars_test, and devkit directories."
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
from __future__ import annotations

import shutil
<<<<<<< HEAD
from pathlib import Path
from zipfile import ZipFile

#поиск структуры датасета
=======
import subprocess
from pathlib import Path
from zipfile import ZipFile


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}")


def ensure_kaggle_credentials() -> Path:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Put kaggle.json into ~/.kaggle/kaggle.json on the remote machine."
        )
    return kaggle_json


def download_kaggle_dataset(dataset: str, out_dir: str | Path, force: bool = False) -> Path:
    ensure_kaggle_credentials()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = dataset.split("/")[-1]
    extract_dir = out_dir / dataset_slug
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        return extract_dir

    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--force"]
    _run_command(command)

    zip_candidates = sorted(out_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_candidates:
        raise FileNotFoundError(f"No zip archive found in {out_dir} after downloading {dataset}.")

    zip_path = zip_candidates[0]
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(extract_dir))
    return extract_dir


>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
def detect_dataset_layout(dataset_root: str | Path) -> dict[str, Path]:
    dataset_root = Path(dataset_root)

    candidates = [
        {
<<<<<<< HEAD
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
=======
            "train": dataset_root / "stanford_dataset" / "cars_train",
            "test": dataset_root / "stanford_dataset" / "cars_test",
>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
            "devkit": dataset_root / "devkit",
        },
        {
            "train": dataset_root / "cars_train",
            "test": dataset_root / "cars_test",
            "devkit": dataset_root / "devkit",
        },
        {
<<<<<<< HEAD
            "train": dataset_root / "train",
            "test": dataset_root / "test",
            "devkit": dataset_root / "devkit",
        },
        {
=======
>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
            "train": dataset_root / "cars_train",
            "test": dataset_root / "cars_test",
            "devkit": dataset_root.parent / "devkit",
        },
        {
<<<<<<< HEAD
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
=======
            "train": dataset_root / "data" / "cars_train",
            "test": dataset_root / "data" / "cars_test",
            "devkit": dataset_root / "devkit",
>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
        },
        {
            "train": Path("/workspace/stanford_dataset/cars_train"),
            "test": Path("/workspace/stanford_dataset/cars_test"),
            "devkit": Path("/workspace/devkit"),
        },
<<<<<<< HEAD
=======
        {
            "train": Path("/workspace/data/cars_train"),
            "test": Path("/workspace/data/cars_test"),
            "devkit": Path("/workspace/devkit"),
        },
>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
    ]

    for candidate in candidates:
        if all(path.exists() for path in candidate.values()):
            return candidate

    raise FileNotFoundError(
<<<<<<< HEAD
        "Could not detect Stanford Cars layout. Expected train/test (or cars_train/cars_test) and devkit directories."
=======
        "Could not detect Stanford Cars layout. Expected cars_train, cars_test, and devkit directories."
>>>>>>> a5a5f798cee2227cc55414b5bebd66c603cd7071
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

