from __future__ import annotations

from src.models.fft_classifier import FFTOnlyClassifier
from src.models.fusion_classifier import RGBFFTFusionClassifier
from src.models.rgb_classifier import build_rgb_classifier


def normalize_model_name(name: str) -> str:
    return name.lower().replace("-", "_")


def build_model(model_cfg: dict):
    name = normalize_model_name(model_cfg["name"])
    num_classes = int(model_cfg.get("num_classes", 2))
    pretrained = bool(model_cfg.get("pretrained", True))

    if name == "fft_only":
        return FFTOnlyClassifier(
            num_classes=num_classes,
            feature_dim=int(model_cfg.get("fft_feature_dim", 128)),
        )

    if name in {"rgb_fft_fusion", "rgb_fft"}:
        return RGBFFTFusionClassifier(
            rgb_backbone=model_cfg.get("rgb_backbone", "convnext_tiny"),
            pretrained=pretrained,
            num_classes=num_classes,
            fft_feature_dim=int(model_cfg.get("fft_feature_dim", 128)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )

    return build_rgb_classifier(name, num_classes=num_classes, pretrained=pretrained)
