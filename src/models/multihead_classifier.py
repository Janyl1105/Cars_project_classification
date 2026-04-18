from typing import Dict, Sequence

import torch
import torch.nn as nn
from torchvision import models


TYPE_CLASSES: Sequence[str] = (
    "sedan",
    "suv",
    "hatchback",
    "wagon",
    "coupe",
    "convertible",
    "van",
    "pickup",
    "truck",
    "sports",
    "other",
)

COLOR_CLASSES: Sequence[str] = (
    "white",
    "black",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "other",
)


def build_backbone(backbone_name: str, pretrained: bool = True):
    name = backbone_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        feat_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim

    raise ValueError(f"Unknown backbone: {backbone_name}")


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MultiHeadCarModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_dropout: float = 0.3,
        num_type_classes: int = len(TYPE_CLASSES),
        num_color_classes: int = len(COLOR_CLASSES),
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        if freeze_backbone:
            self.freeze_backbone()

        self.head_type = MLPHead(feat_dim, num_type_classes, hidden_dim=head_hidden, dropout=head_dropout)
        self.head_color = MLPHead(feat_dim, num_color_classes, hidden_dim=head_hidden, dropout=head_dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim == 4:
            features = self.pool(features)

        features = torch.flatten(features, 1)
        features = self.dropout(features)

        return {
            "type": self.head_type(features),
            "color": self.head_color(features),
        }

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


class ClassifierPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu", backbone_name: str = "resnet50"):
        self.device = torch.device(device)
        self.model = MultiHeadCarModel(backbone_name=backbone_name, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cleaned = {}
            for key, value in state_dict.items():
                key = key.replace("net.", "")
                cleaned[key] = value
            self.model.load_state_dict(cleaned, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device))
        type_idx = outputs["type"].argmax(dim=1)
        color_idx = outputs["color"].argmax(dim=1)
        return {
            "type_idx": type_idx.cpu(),
            "color_idx": color_idx.cpu(),
            "type_name": [TYPE_CLASSES[index] for index in type_idx.tolist()],
            "color_name": [COLOR_CLASSES[index] for index in color_idx.tolist()],
        }