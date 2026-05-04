from pathlib import Path

from src.data.download import summarize_layout, unpack_local_archives

#Берёт уже скачанные zip-файлы
#Распаковывает их в нужную папку
#✅ Когда вы скачали zip локально и передали на сервер

def main() -> None:
    dataset_dir = Path("/workspace/stanford_dataset")
    devkit_dir = Path("/workspace/devkit")
    # старые пути (закомментированы):
    # dataset_dir = Path("/workspace/data/stanford_dataset")
    # devkit_dir = Path("/workspace/data/devkit")

    # старая логика для /workspace/tests/cars_dataset.zip — закомментирована:
    # cars_dataset_zip = Path("/workspace/tests/cars_dataset.zip")
    # if cars_dataset_zip.exists(): ...

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
    devkit_zip = find_first("devkit.zip")

    if combined_zip is None and (train_zip is None or test_zip is None):
        raise FileNotFoundError(
            "Provide either cars_train_test.zip or both cars_train.zip and cars_test.zip in /workspace, /workspace/archives, or /workspace/stanford_dataset."
        )

    layout = unpack_local_archives(
        train_zip=train_zip or "",
        test_zip=test_zip or "",
        dataset_dir=dataset_dir,
        combined_zip=combined_zip,
        devkit_zip=devkit_zip if devkit_zip is not None else None,
        devkit_dir=devkit_dir,
    )

    print("Archives unpacked successfully.")