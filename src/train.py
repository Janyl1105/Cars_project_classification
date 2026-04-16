from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import StanfordCarsDataModule
from src.models.lightning_module import CarsLitModule


def _abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, workers=True)

    csv_path = _abs_path(cfg.data.csv_path)
    outputs_dir = _abs_path(cfg.paths.output_dir) / "checkpoints"
    logs_dir = _abs_path(cfg.paths.output_dir) / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    datamodule = StanfordCarsDataModule(
        csv_path=str(csv_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.model.image_size,
    )
    datamodule.setup()

    model = CarsLitModule(
        backbone_name=cfg.model.backbone,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        lambda_color=cfg.training.lambda_color,
        loss_name=cfg.training.loss_name,
        focal_gamma=cfg.training.focal_gamma,
        freeze_backbone=cfg.training.freeze_backbone,
        head_hidden=cfg.model.head_hidden,
        head_dropout=cfg.model.head_dropout,
        t_max_epochs=cfg.training.epochs,
        pretrained=cfg.training.pretrained,
    )
    model.set_backbone_trainable(not cfg.training.freeze_backbone)

    logger = TensorBoardLogger(save_dir=str(logs_dir), name="cars_multihead")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(outputs_dir),
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        auto_insert_metric_name=False,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    trainer.fit(model, datamodule=datamodule)

    best_path = checkpoint_cb.best_model_path
    best = Path(best_path) if best_path else None
    print(f"TensorBoard logdir: {logger.log_dir}")
    print(f"Best checkpoint: {best}")

    metrics = trainer.callback_metrics
    for metric_name in ["train/acc_type", "train/acc_color", "val/acc_type", "val/acc_color"]:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name].item():.4f}")


if __name__ == "__main__":
    main()