from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from src.models.losses import FocalLoss, make_class_weights
from src.models.multihead_classifier import COLOR_CLASSES, TYPE_CLASSES, MultiHeadCarModel


class CarsLitModule(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_type_classes: int = len(TYPE_CLASSES),
        num_color_classes: int = len(COLOR_CLASSES),
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        lambda_color: float = 1.5,
        loss_name: str = "weighted_ce",
        focal_gamma: float = 2.0,
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_dropout: float = 0.3,
        t_max_epochs: int = 20,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = MultiHeadCarModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
            num_type_classes=num_type_classes,
            num_color_classes=num_color_classes,
        )

        self.type_weights: Optional[torch.Tensor] = None
        self.color_weights: Optional[torch.Tensor] = None
        self.crit_type_focal: Optional[nn.Module] = None
        self.crit_color_focal: Optional[nn.Module] = None

        self.train_acc_t = MulticlassAccuracy(num_classes=num_type_classes)
        self.train_prec_t = MulticlassPrecision(num_classes=num_type_classes, average="macro")
        self.train_rec_t = MulticlassRecall(num_classes=num_type_classes, average="macro")
        self.train_f1_t_macro = MulticlassF1Score(num_classes=num_type_classes, average="macro")
        self.train_f1_t_weighted = MulticlassF1Score(num_classes=num_type_classes, average="weighted")

        self.train_acc_c = MulticlassAccuracy(num_classes=num_color_classes)
        self.train_prec_c = MulticlassPrecision(num_classes=num_color_classes, average="macro")
        self.train_rec_c = MulticlassRecall(num_classes=num_color_classes, average="macro")
        self.train_f1_c_macro = MulticlassF1Score(num_classes=num_color_classes, average="macro")
        self.train_f1_c_weighted = MulticlassF1Score(num_classes=num_color_classes, average="weighted")

        self.val_acc_t = MulticlassAccuracy(num_classes=num_type_classes)
        self.val_prec_t = MulticlassPrecision(num_classes=num_type_classes, average="macro")
        self.val_rec_t = MulticlassRecall(num_classes=num_type_classes, average="macro")
        self.val_f1_t_macro = MulticlassF1Score(num_classes=num_type_classes, average="macro")
        self.val_f1_t_weighted = MulticlassF1Score(num_classes=num_type_classes, average="weighted")

        self.val_acc_c = MulticlassAccuracy(num_classes=num_color_classes)
        self.val_prec_c = MulticlassPrecision(num_classes=num_color_classes, average="macro")
        self.val_rec_c = MulticlassRecall(num_classes=num_color_classes, average="macro")
        self.val_f1_c_macro = MulticlassF1Score(num_classes=num_color_classes, average="macro")
        self.val_f1_c_weighted = MulticlassF1Score(num_classes=num_color_classes, average="weighted")

        self.test_acc_t = MulticlassAccuracy(num_classes=num_type_classes)
        self.test_prec_t = MulticlassPrecision(num_classes=num_type_classes, average="macro")
        self.test_rec_t = MulticlassRecall(num_classes=num_type_classes, average="macro")
        self.test_f1_t_macro = MulticlassF1Score(num_classes=num_type_classes, average="macro")
        self.test_f1_t_weighted = MulticlassF1Score(num_classes=num_type_classes, average="weighted")

        self.test_acc_c = MulticlassAccuracy(num_classes=num_color_classes)
        self.test_prec_c = MulticlassPrecision(num_classes=num_color_classes, average="macro")
        self.test_rec_c = MulticlassRecall(num_classes=num_color_classes, average="macro")
        self.test_f1_c_macro = MulticlassF1Score(num_classes=num_color_classes, average="macro")
        self.test_f1_c_weighted = MulticlassF1Score(num_classes=num_color_classes, average="weighted")

        self._apply_backbone_freeze()

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self.net(x)

    def _apply_backbone_freeze(self):
        for parameter in self.net.backbone.parameters():
            parameter.requires_grad = not self.hparams.freeze_backbone

    def set_backbone_trainable(self, trainable: bool = True):
        for parameter in self.net.backbone.parameters():
            parameter.requires_grad = trainable

    def on_fit_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return

        if hasattr(datamodule, "type_counts") and datamodule.type_counts is not None:
            self.type_weights = make_class_weights(datamodule.type_counts, self.hparams.num_type_classes, self.device)
        if hasattr(datamodule, "color_counts") and datamodule.color_counts is not None:
            self.color_weights = make_class_weights(datamodule.color_counts, self.hparams.num_color_classes, self.device)

        if self.hparams.loss_name == "focal":
            self.crit_type_focal = FocalLoss(alpha=self.type_weights, gamma=self.hparams.focal_gamma, reduction="mean")
            self.crit_color_focal = FocalLoss(alpha=self.color_weights, gamma=self.hparams.focal_gamma, reduction="mean")

    def _ce_or_weighted_ce(self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        targets = targets.long()
        if self.hparams.loss_name == "ce":
            return F.cross_entropy(logits, targets)
        if self.hparams.loss_name == "weighted_ce":
            if weights is not None:
                return F.cross_entropy(logits, targets, weight=weights.to(logits.device))
            return F.cross_entropy(logits, targets)
        raise ValueError(f"Wrong loss_name for CE path: {self.hparams.loss_name}")

    def _compute_task_loss(self, logits: torch.Tensor, targets: torch.Tensor, task_name: str) -> torch.Tensor:
        targets = targets.long()
        if self.hparams.loss_name in ["ce", "weighted_ce"]:
            weights = self.type_weights if task_name == "type" else self.color_weights
            return self._ce_or_weighted_ce(logits, targets, weights)

        if self.hparams.loss_name == "focal":
            if task_name == "type":
                if self.crit_type_focal is None:
                    self.crit_type_focal = FocalLoss(alpha=self.type_weights, gamma=self.hparams.focal_gamma, reduction="mean")
                return self.crit_type_focal(logits, targets)
            if self.crit_color_focal is None:
                self.crit_color_focal = FocalLoss(alpha=self.color_weights, gamma=self.hparams.focal_gamma, reduction="mean")
            return self.crit_color_focal(logits, targets)

        raise ValueError(f"Unknown loss_name: {self.hparams.loss_name}")

    def _select_metrics(self, stage: str):
        if stage == "train":
            return (
                self.train_acc_t,
                self.train_prec_t,
                self.train_rec_t,
                self.train_f1_t_macro,
                self.train_f1_t_weighted,
                self.train_acc_c,
                self.train_prec_c,
                self.train_rec_c,
                self.train_f1_c_macro,
                self.train_f1_c_weighted,
            )
        if stage == "val":
            return (
                self.val_acc_t,
                self.val_prec_t,
                self.val_rec_t,
                self.val_f1_t_macro,
                self.val_f1_t_weighted,
                self.val_acc_c,
                self.val_prec_c,
                self.val_rec_c,
                self.val_f1_c_macro,
                self.val_f1_c_weighted,
            )
        if stage == "test":
            return (
                self.test_acc_t,
                self.test_prec_t,
                self.test_rec_t,
                self.test_f1_t_macro,
                self.test_f1_t_weighted,
                self.test_acc_c,
                self.test_prec_c,
                self.test_rec_c,
                self.test_f1_c_macro,
                self.test_f1_c_weighted,
            )
        raise ValueError(f"Unknown stage: {stage}")

    def _log_metrics(self, stage: str, logits_type: torch.Tensor, y_type: torch.Tensor, logits_color: torch.Tensor, y_color: torch.Tensor, batch_size: int):
        (
            acc_t,
            prec_t,
            rec_t,
            f1_t_macro,
            f1_t_weighted,
            acc_c,
            prec_c,
            rec_c,
            f1_c_macro,
            f1_c_weighted,
        ) = self._select_metrics(stage)

        self.log(f"{stage}/acc_type", acc_t(logits_type, y_type), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/acc_color", acc_c(logits_color, y_color), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/prec_type_macro", prec_t(logits_type, y_type), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/rec_type_macro", rec_t(logits_type, y_type), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/f1_type_macro", f1_t_macro(logits_type, y_type), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/f1_type_weighted", f1_t_weighted(logits_type, y_type), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/prec_color_macro", prec_c(logits_color, y_color), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/rec_color_macro", rec_c(logits_color, y_color), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/f1_color_macro", f1_c_macro(logits_color, y_color), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/f1_color_weighted", f1_c_weighted(logits_color, y_color), on_step=False, on_epoch=True, batch_size=batch_size)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["x"]
        y_type = batch["y_type"].long()
        y_color = batch["y_color"].long()

        out = self(x)
        logits_type = out["type"]
        logits_color = out["color"]

        loss_type = self._compute_task_loss(logits_type, y_type, task_name="type")
        loss_color = self._compute_task_loss(logits_color, y_color, task_name="color")
        loss = loss_type + float(self.hparams.lambda_color) * loss_color

        batch_size = x.size(0)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/loss_type", loss_type, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_color", loss_color, on_step=False, on_epoch=True, batch_size=batch_size)

        self._log_metrics(stage, logits_type, y_type, logits_color, y_color, batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.hparams.t_max_epochs - 3))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }