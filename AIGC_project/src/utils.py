from __future__ import annotations

import os
import random
import math
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path, root):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str = "auto"):
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_model_state(path, model, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def format_result_row(row):
    digits = {
        "acc": 4,
        "precision": 4,
        "recall": 4,
        "f1": 4,
        "auc": 4,
        "val_acc": 4,
        "val_precision": 4,
        "val_recall": 4,
        "val_f1": 4,
        "val_auc": 4,
        "train_loss": 6,
        "val_loss": 6,
        "loss": 6,
        "seconds": 3,
        "time_per_image_ms": 3,
    }
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isnan(value):
                formatted[key] = "nan"
            else:
                formatted[key] = f"{value:.{digits.get(key, 4)}f}"
        else:
            formatted[key] = value
    return formatted
