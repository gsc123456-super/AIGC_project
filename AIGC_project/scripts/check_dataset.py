from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import collect_binary_samples, count_by_label, load_csv_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.csv:
        samples = load_csv_samples(args.csv)
        print(f"{args.csv}: {len(samples)} images {count_by_label(samples)}")
        by_source = {}
        for _, label, source in samples:
            by_source.setdefault(source, {0: 0, 1: 0})
            by_source[source][int(label)] += 1
        for source, counts in sorted(by_source.items()):
            print(f"  {source}: {counts}")
        return

    if not args.root:
        raise SystemExit("Please pass --root or --csv.")

    root = Path(args.root)
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists():
            samples = collect_binary_samples(split_dir)
            print(f"{split}: {len(samples)} images {count_by_label(samples)}")

    if (root / "FAKE").exists() or (root / "REAL").exists():
        samples = collect_binary_samples(root)
        print(f"{root}: {len(samples)} images {count_by_label(samples)}")


if __name__ == "__main__":
    main()
