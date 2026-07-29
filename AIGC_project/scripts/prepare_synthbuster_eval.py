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

from src.datasets import IMAGE_EXTENSIONS, collect_binary_samples, load_csv_samples
from src.utils import ensure_dir


def manifest_relative_path(image_path: Path, manifest_path: Path) -> str:
    try:
        return Path(os.path.relpath(image_path.resolve(), manifest_path.parent.resolve())).as_posix()
    except ValueError:
        return image_path.as_posix()


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_fake_samples(fake_root: Path, max_per_source: int | None, seed: int):
    groups = {}
    for path in iter_images(fake_root):
        source = path.parent.name
        groups.setdefault(source, []).append((path, 1, source))

    rng = random.Random(seed)
    samples = []
    for source, group in groups.items():
        rng.shuffle(group)
        samples.extend(group[:max_per_source] if max_per_source else group)
    rng.shuffle(samples)
    return samples


def collect_real_samples(real_root: str | None, real_csv: str | None, target_count: int, seed: int):
    if real_csv:
        samples = load_csv_samples(real_csv)
    elif real_root:
        samples = collect_binary_samples(real_root)
        samples = [sample for sample in samples if sample[1] == 0]
    else:
        raise ValueError("Please provide --real-root or --real-csv for Synthbuster evaluation.")

    rng = random.Random(seed)
    rng.shuffle(samples)
    return [(path, 0, source) for path, _, source in samples[:target_count]]


def write_manifest(path: Path, samples):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        for image_path, label, source in samples:
            writer.writerow(
                {"path": manifest_relative_path(image_path, path), "label": label, "source": source, "split": "test"}
            )


def export_imagefolder(out_root: Path, samples, mode: str):
    for image_path, label, source in samples:
        class_name = "fake" if label == 1 else "real"
        target_dir = out_root / class_name / source
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
    parser.add_argument("--fake-root", default="data/raw/synthbuster")
    parser.add_argument("--real-root", default=None)
    parser.add_argument("--real-csv", default=None)
    parser.add_argument("--out-root", default="data/synthbuster_eval")
    parser.add_argument("--max-per-source", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", choices=["manifest", "copy", "hardlink"], default="manifest")
    args = parser.parse_args()

    fake_root = Path(args.fake_root)
    out_root = Path(args.out_root)
    if not fake_root.exists():
        raise FileNotFoundError(f"Synthbuster root not found: {fake_root}")

    fake_samples = collect_fake_samples(fake_root, args.max_per_source, args.seed)
    if not fake_samples:
        archives = list(fake_root.glob("*.zip"))
        hint = " Found synthbuster.zip; please expand it before preparing eval data." if archives else ""
        raise RuntimeError(f"No Synthbuster images found under {fake_root}.{hint}")

    real_samples = collect_real_samples(args.real_root, args.real_csv, len(fake_samples), args.seed)
    samples = real_samples + fake_samples
    random.Random(args.seed).shuffle(samples)

    write_manifest(out_root / "meta" / "synthbuster_eval.csv", samples)
    if args.export in {"copy", "hardlink"}:
        export_imagefolder(out_root, samples, args.export)
    print(f"real: {len(real_samples)} fake: {len(fake_samples)} -> {out_root / 'meta' / 'synthbuster_eval.csv'}")


if __name__ == "__main__":
    main()
