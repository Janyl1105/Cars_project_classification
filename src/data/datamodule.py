from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.image_io import build_eval_transforms, load_pil_image

try:
    import torchvision.transforms as T
except Exception:
    T = None


def build_train_transforms(image_size: int = 224):
    if T is None:
        raise ImportError("torchvision is required for transforms")
    return T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(image_size, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class StanfordMultiHeadDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        start_index = index
        while True:
            row = self.frame.iloc[index]
            try:
                image = load_pil_image(str(Path(row["image_path"])))
                if not isinstance(image, Image.Image):
                    raise ValueError("Image loader did not return a PIL image")

                x = self.transform(image)
                return {
                    "x": x,
                    "y_type": torch.tensor(int(row["type_id"]), dtype=torch.long),
                    "y_color": torch.tensor(int(row["color_id"]), dtype=torch.long),
                    "image_path": row["image_path"],
                    "class_label": row.get("class_label", "unknown"),
                }
            except Exception as exc:
                print(f"Skipped sample: {row['image_path']} ({type(exc).__name__}: {exc})", flush=True)

            index = (index + 1) % len(self.frame)
            if index == start_index:
                raise RuntimeError("All samples failed to load")


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, csv_path: str, batch_size: int = 64, num_workers: int = 2, image_size: int = 224):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.type_counts = None
        self.color_counts = None

    def setup(self, stage=None):
        frame = pd.read_csv(self.csv_path)

        train_frame = frame[frame["split"] == "train"].copy()
        val_frame = frame[frame["split"] == "val"].copy()
        test_frame = frame[frame["split"] == "test"].copy()

        self.ds_train = StanfordMultiHeadDataset(train_frame, build_train_transforms(self.image_size))
        self.ds_val = StanfordMultiHeadDataset(val_frame, build_eval_transforms(self.image_size))
        self.ds_test = StanfordMultiHeadDataset(test_frame, build_eval_transforms(self.image_size))

        self.type_counts = train_frame["type_id"].value_counts().sort_index()
        self.color_counts = train_frame["color_id"].value_counts().sort_index()

    def _loader(self, dataset, shuffle: bool, drop_last: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self._loader(self.ds_train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.ds_test, shuffle=False)