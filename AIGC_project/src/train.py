from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import (
    BinaryImageDataset,
    collect_binary_samples,
    count_by_label,
    limit_samples_balanced,
    samples_from_config,
    split_samples,
)
from src.metrics import compute_binary_metrics
from src.models import build_model
from src.transforms import build_transforms
from src.utils import ensure_dir, format_result_row, load_config, save_checkpoint, seed_worker, select_device, set_seed


def make_loaders(config):
    data_cfg = config["data"]
    train_cfg = config["train"]
    train_transform, eval_transform = build_transforms(data_cfg["input_size"])

    all_train_samples = samples_from_config(data_cfg, "train")
    if data_cfg.get("val_csv") or data_cfg.get("val_dir"):
        train_samples = all_train_samples
        val_samples = samples_from_config(data_cfg, "val")
    else:
        train_samples, val_samples = split_samples(
            all_train_samples,
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            seed=int(train_cfg.get("seed", 42)),
        )
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    seed = int(train_cfg.get("seed", 42))
    if max_train:
        train_samples = limit_samples_balanced(train_samples, int(max_train), seed)
    if max_val:
        val_samples = limit_samples_balanced(val_samples, int(max_val), seed + 1)

    train_dataset = BinaryImageDataset(transform=train_transform, samples=train_samples)
    val_dataset = BinaryImageDataset(transform=eval_transform, samples=val_samples)

    generator = torch.Generator()
    generator.manual_seed(int(train_cfg.get("seed", 42)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("eval", {}).get("batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(config.get("eval", {}).get("num_workers", train_cfg.get("num_workers", 2))),
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)} {count_by_label(train_samples)}")
    print(f"Val samples:   {len(val_dataset)} {count_by_label(val_samples)}")
    return train_loader, val_loader


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

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, labels, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def append_history(path, row):
    ensure_dir(Path(path).parent)
    row = format_result_row(row)
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["train"].get("seed", 42)))
    torch.backends.cudnn.benchmark = True

    device = select_device(config["train"].get("device", "auto"))
    print(f"Using device: {device}")

    model = build_model(config["model"]).to(device)

    train_loader, val_loader = make_loaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(config["output"]["ckpt_dir"])
    log_dir = Path(config["output"]["log_dir"])
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    best_f1 = -1.0
    history_path = log_dir / "history.csv"
    epochs = int(config["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "seconds": elapsed,
        }
        append_history(history_path, row)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} time={elapsed:.1f}s"
        )

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, epoch, val_metrics, config)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(ckpt_dir / "best_f1.pth", model, optimizer, epoch, val_metrics, config)
            print(f"Saved best checkpoint with F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
