from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch


def seed_everything(seed: int = 42, workers: bool = True) -> None:
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=workers)


def get_device() -> torch.device:
    """Return the currently available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    device = torch.device("cpu")
    print("✅ Using CPU")
    return device
