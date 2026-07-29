from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import BinaryImageDataset, count_by_label, limit_samples_balanced, samples_from_config
from src.metrics import compute_binary_metrics
from src.models import build_model
from src.transforms import build_transforms
from src.utils import ensure_dir, format_result_row, load_config, load_model_state, select_device


DEGRADATIONS = [
    "clean",
    "jpeg_q95",
    "jpeg_q75",
    "jpeg_q50",
    "jpeg_q30",
    "blur_k3",
    "blur_k5",
    "resize_0.5",
    "center_crop",
    "gaussian_noise",
]


def apply_degradation(image: Image.Image, name: str) -> Image.Image:
    image = image.convert("RGB")
    if name == "clean":
        return image
    if name.startswith("jpeg_q"):
        quality = int(name.replace("jpeg_q", ""))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name.startswith("blur_k"):
        radius = max((int(name.replace("blur_k", "")) - 1) / 2.0, 0.1)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if name == "resize_0.5":
        w, h = image.size
        small = image.resize((max(w // 2, 1), max(h // 2, 1)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if name == "center_crop":
        w, h = image.size
        side_w, side_h = int(w * 0.85), int(h * 0.85)
        left = (w - side_w) // 2
        top = (h - side_h) // 2
        return image.crop((left, top, left + side_w, top + side_h)).resize((w, h), Image.BICUBIC)
    if name == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32)
        noise = np.random.default_rng(42).normal(0, 8, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"Unknown degradation: {name}")


class RobustnessDataset(Dataset):
    def __init__(self, samples, transform, degradation: str):
        self.samples = list(samples)
        self.transform = transform
        self.degradation = degradation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, source = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = apply_degradation(image, self.degradation)
        return self.transform(image), label, str(path), source


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true = []
    y_prob = []
    for images, labels, _, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def write_row(path: Path, row: dict):
    ensure_dir(path.parent)
    row = format_result_row(row)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--degradations", nargs="*", default=DEGRADATIONS)
    args = parser.parse_args()

    config = load_config(args.config)
    _, eval_transform = build_transforms(int(config["data"]["input_size"]))
    if args.test_csv:
        base_dataset = BinaryImageDataset(transform=eval_transform, csv_file=args.test_csv)
        data_label = args.test_csv
    elif args.data:
        base_dataset = BinaryImageDataset(args.data, transform=eval_transform)
        data_label = args.data
    else:
        base_dataset = BinaryImageDataset(transform=eval_transform, samples=samples_from_config(config["data"], "test"))
        data_label = config["data"].get("test_csv") or config["data"].get("test_dir")

    max_test = config["data"].get("max_test_samples")
    samples = base_dataset.samples
    if max_test:
        samples = limit_samples_balanced(samples, int(max_test), int(config.get("train", {}).get("seed", 42)))
    print(f"Robustness samples: {len(samples)} {count_by_label(samples)}")

    device = select_device(config.get("eval", {}).get("device", config.get("train", {}).get("device", "auto")))
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    model = build_model(model_cfg).to(device)
    checkpoint = load_model_state(args.ckpt, model, map_location=device)
    criterion = nn.CrossEntropyLoss()
    out_path = Path(args.out or Path(config["output"]["table_dir"]) / "robustness_results.csv")

    for degradation in args.degradations:
        dataset = RobustnessDataset(samples, eval_transform, degradation)
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("eval", {}).get("batch_size", 64)),
            shuffle=False,
            num_workers=int(config.get("eval", {}).get("num_workers", 2)),
            pin_memory=True,
        )
        metrics = evaluate(model, loader, device, criterion)
        row = {
            "model": model_cfg["name"],
            "checkpoint": str(args.ckpt),
            "data": str(data_label),
            "epoch": checkpoint.get("epoch", ""),
            "degradation": degradation,
            **metrics,
        }
        write_row(out_path, row)
        print(format_result_row(row))

    print(f"Saved robustness results to {out_path}")


if __name__ == "__main__":
    main()
