from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {
    "real": 0,
    "nature": 0,
    "fake": 1,
    "ai": 1,
}


class BinaryImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        transform=None,
        samples=None,
        csv_file: str | Path | None = None,
    ):
        self.root = Path(root) if root is not None else None
        self.transform = transform
        if samples is not None:
            self.samples = list(samples)
        elif csv_file is not None:
            self.samples = load_csv_samples(csv_file)
        else:
            self.samples = collect_binary_samples(self.root)

        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, source = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path), source


def collect_binary_samples(root: str | Path) -> list[tuple[Path, int, str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: list[tuple[Path, int, str]] = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        label = CLASS_TO_LABEL.get(class_dir.name.lower())
        if label is None:
            continue

        for image_path in iter_image_files(class_dir):
            source = image_path.parent.name if image_path.parent != class_dir else class_dir.name
            samples.append((image_path, label, source))

    return samples


def load_csv_samples(csv_file: str | Path) -> list[tuple[Path, int, str]]:
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV manifest does not exist: {csv_file}")

    samples: list[tuple[Path, int, str]] = []
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_file} is missing required columns: {sorted(missing)}")

        for row in reader:
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = (csv_file.parent / image_path).resolve()
            label = int(row["label"])
            source = row.get("source") or image_path.parent.name
            samples.append((image_path, label, source))

    return samples


def samples_from_config(data_cfg: dict, split: str) -> list[tuple[Path, int, str]]:
    csv_key = f"{split}_csv"
    dir_key = f"{split}_dir"
    if data_cfg.get(csv_key):
        return load_csv_samples(data_cfg[csv_key])
    if data_cfg.get(dir_key):
        return collect_binary_samples(data_cfg[dir_key])
    raise KeyError(f"Config data section must define {csv_key} or {dir_key}.")


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def split_samples(samples, val_ratio: float, seed: int):
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    return train_samples, val_samples


def limit_samples_balanced(samples, max_samples: int | None, seed: int):
    if not max_samples or max_samples >= len(samples):
        return list(samples)

    rng = random.Random(seed)
    groups: dict[int, list] = {}
    for sample in samples:
        groups.setdefault(int(sample[1]), []).append(sample)

    if len(groups) < 2:
        limited = list(samples)
        rng.shuffle(limited)
        return limited[:max_samples]

    per_class = max_samples // len(groups)
    selected = []
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        selected.extend(group[:per_class])

    remaining = max_samples - len(selected)
    if remaining > 0:
        leftovers = [sample for group in groups.values() for sample in group if sample not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)
    return selected


def count_by_label(samples):
    counts = {0: 0, 1: 0}
    for _, label, _ in samples:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts
