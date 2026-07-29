from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import IMAGE_EXTENSIONS, CLASS_TO_LABEL
from src.utils import ensure_dir


CLASS_NAMES = {"real", "nature", "fake", "ai"}


def manifest_relative_path(image_path: Path, manifest_path: Path) -> str:
    try:
        return Path(os.path.relpath(image_path.resolve(), manifest_path.parent.resolve())).as_posix()
    except ValueError:
        return image_path.as_posix()


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_labeled_images(raw_root: Path):
    samples = []
    for image_path in iter_images(raw_root):
        parts = [part.lower() for part in image_path.parts]
        label_name = next((part for part in reversed(parts) if part in CLASS_NAMES), None)
        if label_name is None:
            continue
        label = CLASS_TO_LABEL[label_name]
        source = infer_source(raw_root, image_path, label_name)
        samples.append((image_path, label, source))
    return samples


def infer_source(raw_root: Path, image_path: Path, label_name: str) -> str:
    rel_parts = image_path.relative_to(raw_root).parts
    for part in rel_parts:
        if part.lower() in {"train", "val", "test", label_name}:
            continue
        return part
    return "genimage"


def split_samples(samples, seed: int):
    by_key = {}
    for sample in samples:
        _, label, source = sample
        by_key.setdefault((source, label), []).append(sample)

    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for group in by_key.values():
        rng.shuffle(group)
        n = len(group)
        train_n = int(n * 0.7)
        val_n = int(n * 0.15)
        result["train"].extend(group[:train_n])
        result["val"].extend(group[train_n : train_n + val_n])
        result["test"].extend(group[train_n + val_n :])
    for split in result:
        rng.shuffle(result[split])
    return result


def maybe_limit(split_map, max_per_source_class: int | None, seed: int):
    if not max_per_source_class:
        return split_map
    rng = random.Random(seed)
    limited = {}
    for split, samples in split_map.items():
        groups = {}
        for sample in samples:
            _, label, source = sample
            groups.setdefault((source, label), []).append(sample)
        limited[split] = []
        for group in groups.values():
            rng.shuffle(group)
            limited[split].extend(group[:max_per_source_class])
        rng.shuffle(limited[split])
    return limited


def write_manifest(path: Path, samples, split: str):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        for image_path, label, source in samples:
            writer.writerow(
                {"path": manifest_relative_path(image_path, path), "label": label, "source": source, "split": split}
            )


def export_imagefolder(out_root: Path, split: str, samples, mode: str):
    for image_path, label, source in samples:
        class_name = "fake" if label == 1 else "real"
        target_dir = out_root / split / class_name / source
        ensure_dir(target_dir)
        target = target_dir / image_path.name
        if target.exists():
            continue
        if mode == "copy":
            shutil.copy2(image_path, target)
        elif mode == "hardlink":
            try:
                target.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw/genimage")
    parser.add_argument("--out-root", default="data/genimage_subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source-class", type=int, default=None)
    parser.add_argument("--export", choices=["manifest", "copy", "hardlink"], default="manifest")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"GenImage raw root not found: {raw_root}")

    samples = find_labeled_images(raw_root)
    if not samples:
        archives = [p for p in raw_root.rglob("*") if p.suffix.lower() in {".zip", ".z01", ".z02"}]
        hint = " Found archive files; please extract BigGAN/ADM first." if archives else ""
        raise RuntimeError(f"No labeled GenImage images found under {raw_root}.{hint}")

    split_map = maybe_limit(split_samples(samples, args.seed), args.max_per_source_class, args.seed)
    for split, split_samples_ in split_map.items():
        write_manifest(out_root / "meta" / f"{split}.csv", split_samples_, split)
        if args.export in {"copy", "hardlink"}:
            export_imagefolder(out_root, split, split_samples_, args.export)
        print(f"{split}: {len(split_samples_)} images -> {out_root / 'meta' / f'{split}.csv'}")


if __name__ == "__main__":
    main()
