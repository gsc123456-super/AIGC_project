from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import BinaryImageDataset, limit_samples_balanced, samples_from_config
from src.utils import ensure_dir, load_config


def fft_spectrum_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image.convert("RGB"))
    arr = np.asarray(gray).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return Image.fromarray((magnitude * 255).astype(np.uint8))


def make_side_by_side(image: Image.Image, spectrum: Image.Image, size: int) -> Image.Image:
    left = image.convert("RGB").resize((size, size))
    right = spectrum.convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size * 2, size), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (size, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--out-dir", default="results/figures/fft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if args.test_csv:
        dataset = BinaryImageDataset(csv_file=args.test_csv)
    elif args.data:
        dataset = BinaryImageDataset(args.data)
    elif args.config:
        config = load_config(args.config)
        dataset = BinaryImageDataset(samples=samples_from_config(config["data"], "test"))
    else:
        raise SystemExit("Please pass --data, --test-csv, or --config.")

    samples = limit_samples_balanced(dataset.samples, args.num_samples, args.seed)
    random.Random(args.seed).shuffle(samples)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    for idx, (path, label, source) in enumerate(samples[: args.num_samples], start=1):
        image = Image.open(path).convert("RGB")
        spectrum = fft_spectrum_image(image)
        canvas = make_side_by_side(image, spectrum, args.size)
        class_name = "fake" if label == 1 else "real"
        target = out_dir / f"{idx:02d}_{class_name}_{source}.png"
        canvas.save(target)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
