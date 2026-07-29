from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
)


def build_rgb_classifier(name: str, num_classes: int = 2, pretrained: bool = True):
    name = name.lower().replace("-", "_")

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported RGB model: {name}")
