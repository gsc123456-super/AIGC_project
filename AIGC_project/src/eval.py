from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import BinaryImageDataset, count_by_label, limit_samples_balanced, samples_from_config
from src.metrics import compute_binary_metrics
from src.models import build_model
from src.transforms import build_transforms
from src.utils import ensure_dir, format_result_row, load_config, load_model_state, select_device


RESULT_FIELDS = [
    "model",
    "checkpoint",
    "data",
    "epoch",
    "source",
    "acc",
    "precision",
    "recall",
    "f1",
    "auc",
    "loss",
    "seconds",
    "time_per_image_ms",
]


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true = []
    y_prob = []
    sources = []
    start = time.time()

    for images, labels, _, batch_sources in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        sources.extend(list(batch_sources))

    elapsed = time.time() - start
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["seconds"] = elapsed
    metrics["time_per_image_ms"] = elapsed * 1000.0 / max(total_count, 1)
    return metrics, y_true, y_prob, sources


def write_results(path, row, fieldnames=None):
    ensure_dir(Path(path).parent)
    row = format_result_row(row)
    fieldnames = fieldnames or [key for key in RESULT_FIELDS if key in row] + [
        key for key in row.keys() if key not in RESULT_FIELDS
    ]
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=None, help="Override test_dir in config.")
    parser.add_argument("--test-csv", default=None, help="Override test_csv in config.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--group-by-source", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    _, eval_transform = build_transforms(int(config["data"]["input_size"]))
    if args.test_csv:
        test_label = args.test_csv
        dataset = BinaryImageDataset(transform=eval_transform, csv_file=args.test_csv)
    elif args.data:
        test_label = args.data
        dataset = BinaryImageDataset(args.data, transform=eval_transform)
    elif config["data"].get("test_csv") or config["data"].get("test_dir"):
        test_label = config["data"].get("test_csv") or config["data"].get("test_dir")
        dataset = BinaryImageDataset(transform=eval_transform, samples=samples_from_config(config["data"], "test"))
    else:
        raise KeyError("Config data section must define test_dir/test_csv, or pass --data/--test-csv.")
    max_test = config["data"].get("max_test_samples")
    if max_test:
        seed = int(config.get("train", {}).get("seed", 42))
        dataset.samples = limit_samples_balanced(dataset.samples, int(max_test), seed)
    print(f"Test samples: {len(dataset)} {count_by_label(dataset.samples)}")

    loader = DataLoader(
        dataset,
        batch_size=int(config.get("eval", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("eval", {}).get("num_workers", 2)),
        pin_memory=True,
    )

    device = select_device(config.get("eval", {}).get("device", config.get("train", {}).get("device", "auto")))
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    model = build_model(model_cfg).to(device)
    checkpoint = load_model_state(args.ckpt, model, map_location=device)
    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_prob, sources = run_eval(model, loader, device, criterion)

    row = {
        "model": model_cfg["name"],
        "checkpoint": str(args.ckpt),
        "data": str(test_label),
        "epoch": checkpoint.get("epoch", ""),
        **metrics,
    }
    print(format_result_row(row))

    out_path = args.out or Path(config["output"]["table_dir"]) / "cifake_eval_results.csv"
    if args.group_by_source:
        write_results(out_path, {**row, "source": "all"}, fieldnames=RESULT_FIELDS)
        for source in sorted(set(sources)):
            idx = [i for i, item in enumerate(sources) if item == source]
            source_metrics = compute_binary_metrics([y_true[i] for i in idx], [y_prob[i] for i in idx])
            write_results(
                out_path,
                {
                    "model": model_cfg["name"],
                    "checkpoint": str(args.ckpt),
                    "data": str(test_label),
                    "source": source,
                    "epoch": checkpoint.get("epoch", ""),
                    **source_metrics,
                },
                fieldnames=RESULT_FIELDS,
            )
    else:
        write_results(out_path, row)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
