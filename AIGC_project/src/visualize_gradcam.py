from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import BinaryImageDataset, limit_samples_balanced, samples_from_config
from src.models import build_model
from src.transforms import build_transforms
from src.utils import ensure_dir, load_config, load_model_state, select_device


def find_target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "rgb_branch") and hasattr(model.rgb_branch, "features"):
        return model.rgb_branch.features[-1]
    raise ValueError("Could not infer a target layer for Grad-CAM.")


class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob_fake = torch.softmax(logits.detach(), dim=1)[0, 1].item()
        pred = int(torch.argmax(logits.detach(), dim=1).item())
        return cam.cpu(), pred, prob_fake


def overlay_cam(image: Image.Image, cam: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    heat = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(base.size)
    heat = ImageOps.colorize(heat, black="#001f3f", white="#ff4136")
    return Image.blend(base, heat, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--out-dir", default="results/figures/gradcam")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.test_csv:
        dataset = BinaryImageDataset(csv_file=args.test_csv)
    elif args.data:
        dataset = BinaryImageDataset(args.data)
    else:
        dataset = BinaryImageDataset(samples=samples_from_config(config["data"], "test"))

    samples = limit_samples_balanced(dataset.samples, args.num_samples, args.seed)
    random.Random(args.seed).shuffle(samples)

    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    device = select_device(config.get("eval", {}).get("device", config.get("train", {}).get("device", "auto")))
    model = build_model(model_cfg).to(device)
    load_model_state(args.ckpt, model, map_location=device)
    model.eval()
    _, transform = build_transforms(int(config["data"]["input_size"]))
    cam = SimpleGradCAM(model, find_target_layer(model))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    try:
        for idx, (path, label, source) in enumerate(samples[: args.num_samples], start=1):
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            heatmap, pred, prob_fake = cam(tensor)
            overlay = overlay_cam(image, heatmap)
            class_name = "fake" if label == 1 else "real"
            pred_name = "fake" if pred == 1 else "real"
            target = out_dir / f"{idx:02d}_{class_name}_{source}_pred-{pred_name}_{prob_fake:.3f}.png"
            overlay.save(target)
            print(f"Saved {target}")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
