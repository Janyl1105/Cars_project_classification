from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import StanfordCarsDataModule
from src.models.lightning_module import CarsLitModule


def _abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    csv_path = _abs_path(cfg.data.csv_path)
    checkpoint_path = _abs_path(cfg.checkpoint_path)
    logs_dir = _abs_path(cfg.paths.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    datamodule = StanfordCarsDataModule(
        csv_path=str(csv_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.model.image_size,
    )
    datamodule.setup("test")

    model = CarsLitModule.load_from_checkpoint(str(checkpoint_path))
    logger = TensorBoardLogger(save_dir=str(logs_dir), name="cars_multihead_eval")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=logger,
    )
    results = trainer.test(model, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    main()
