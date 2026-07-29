from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import collect_binary_samples
from src.utils import ensure_dir


def manifest_relative_path(image_path: Path, manifest_path: Path) -> str:
    try:
        return Path(os.path.relpath(image_path.resolve(), manifest_path.parent.resolve())).as_posix()
    except ValueError:
        return image_path.as_posix()


def write_manifest(path: Path, samples, split: str):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        for image_path, label, source in samples:
            writer.writerow(
                {"path": manifest_relative_path(image_path, path), "label": label, "source": source, "split": split}
            )


def copy_tree_if_requested(src: Path, dst: Path, copy_files: bool):
    if not copy_files:
        return
    ensure_dir(dst)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif child.is_file():
            shutil.copy2(child, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/cifake")
    parser.add_argument("--out-root", default="data/cifake")
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"CIFAKE root not found: {raw_root}")

    for split in ["train", "test"]:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        samples = collect_binary_samples(split_dir)
        write_manifest(out_root / "meta" / f"{split}.csv", samples, split)
        copy_tree_if_requested(split_dir, out_root / split, args.copy_files)
        print(f"{split}: {len(samples)} images -> {out_root / 'meta' / f'{split}.csv'}")


if __name__ == "__main__":
    main()
