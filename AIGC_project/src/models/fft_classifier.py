from __future__ import annotations

import torch
import torch.nn as nn


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalized_rgb_to_fft(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    spectrum = torch.fft.fft2(gray, norm="ortho")
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(spectrum))

    flat = magnitude.flatten(1)
    min_values = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (magnitude - min_values) / (max_values - min_values + 1e-6)


class FFTFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(normalized_rgb_to_fft(images))


class FFTOnlyClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, feature_dim: int = 128):
        super().__init__()
        self.features = FFTFeatureExtractor(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
