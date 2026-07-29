from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.transforms import build_transforms
from src.utils import load_config, load_model_state, select_device


def load_inference_model(config_path: str | Path, ckpt_path: str | Path, device_name: str = "auto"):
    config = load_config(config_path)
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    device = select_device(device_name)
    model = build_model(model_cfg).to(device)
    checkpoint = load_model_state(ckpt_path, model, map_location=device)
    model.eval()
    _, transform = build_transforms(int(config["data"]["input_size"]))
    return model, transform, device, config, checkpoint


@torch.no_grad()
def predict_image(model, transform, device, image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    label = "AI-generated" if prob_fake >= 0.5 else "Real"
    return {"label": label, "fake_probability": prob_fake}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, transform, device, _, _ = load_inference_model(args.config, args.ckpt, args.device)
    image = Image.open(args.image)
    result = predict_image(model, transform, device, image)
    print(f"label={result['label']} fake_probability={result['fake_probability']:.4f}")


if __name__ == "__main__":
    main()
