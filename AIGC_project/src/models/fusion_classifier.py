from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
)

from src.models.fft_classifier import FFTFeatureExtractor


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.feature_dim = model.fc.in_features
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(images), 1)


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.feature_dim = model.classifier[1].in_features
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        self.feature_dim = model.classifier[2].in_features
        self.features = model.features
        self.avgpool = model.avgpool
        self.norm = model.classifier[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)


def build_rgb_feature_extractor(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name == "resnet50":
        return ResNetFeatureExtractor(pretrained=pretrained)
    if name == "efficientnet_b0":
        return EfficientNetFeatureExtractor(pretrained=pretrained)
    if name == "convnext_tiny":
        return ConvNeXtFeatureExtractor(pretrained=pretrained)
    raise ValueError(f"Unsupported RGB backbone for fusion: {name}")


class RGBFFTFusionClassifier(nn.Module):
    def __init__(
        self,
        rgb_backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 2,
        fft_feature_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rgb_branch = build_rgb_feature_extractor(rgb_backbone, pretrained=pretrained)
        self.fft_branch = FFTFeatureExtractor(feature_dim=fft_feature_dim)
        fusion_dim = self.rgb_branch.feature_dim + fft_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rgb_features = self.rgb_branch(images)
        fft_features = self.fft_branch(images)
        return self.classifier(torch.cat([rgb_features, fft_features], dim=1))
