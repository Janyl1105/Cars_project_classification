from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_class_weights(counts: pd.Series, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if counts is None or len(counts) == 0:
        return None

    counts = counts.reindex(range(num_classes), fill_value=0)
    freq = torch.tensor(counts.values, dtype=torch.float32, device=device)

    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    valid = freq > 0
    weights[valid] = 1.0 / freq[valid]
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            if self.alpha.ndim != 1:
                raise ValueError(f"alpha must be 1D, got shape={self.alpha.shape}")
            if self.alpha.numel() <= targets.max().item():
                raise ValueError(f"Alpha shape mismatch: alpha.shape={self.alpha.shape}, max target={targets.max().item()}")
            loss = self.alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")