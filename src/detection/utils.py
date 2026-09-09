"""Shared utilities for the Landslide Detection module.

    - set_seed / seeded_worker_init: reproducible RNG state for
      Python / NumPy / PyTorch / DataLoader workers.
    - ExperimentTracker: append-only CSV row logger for controlled
      experiments; each row is one training run with its config and
      final validation metrics.
    - load_config / save_config: YAML I/O for detection configs.
    - device_summary: one-line "cuda: <name>, mem X GiB" or "cpu".
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def seeded_worker_init(worker_id: int) -> None:
    """Passable to `DataLoader(worker_init_fn=...)`."""
    base = torch.initial_seed() % (2 ** 32)
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


# ---------------------------------------------------------------------------
# device summary
# ---------------------------------------------------------------------------

def device_summary(device: torch.device | str | None = None) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device.index or 0)
        return f"cuda: {p.name} ({p.total_memory / 1024 ** 3:.1f} GiB)"
    return "cpu"


# ---------------------------------------------------------------------------
# YAML config I/O
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def save_config(cfg: dict, path: str | Path) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


# ---------------------------------------------------------------------------
# experiment tracking
# ---------------------------------------------------------------------------

EXPERIMENT_FIELDS = [
    "experiment_id",
    "model",
    "in_channels", "out_channels", "base_features",
    "loss", "loss_params",
    "optimizer", "learning_rate", "weight_decay",
    "scheduler",
    "batch_size", "epochs_planned", "epochs_actually_ran", "best_epoch",
    "seed",
    "val_loss", "val_dice", "val_iou", "val_precision",
    "val_recall", "val_f1", "val_specificity", "val_accuracy", "val_pr_auc",
    "threshold",
    "checkpoint",
    "seconds_total",
    "notes",
]


@dataclass
class ExperimentRow:
    experiment_id: str
    model: str = "unet"
    in_channels: int = 14
    out_channels: int = 1
    base_features: int = 32
    loss: str = ""
    loss_params: str = ""
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    batch_size: int = 16
    epochs_planned: int = 0
    epochs_actually_ran: int = 0
    best_epoch: int = 0
    seed: int = 42
    val_loss: float = 0.0
    val_dice: float = 0.0
    val_iou: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_specificity: float = 0.0
    val_accuracy: float = 0.0
    val_pr_auc: float = 0.0
    threshold: float = 0.5
    checkpoint: str = ""
    seconds_total: float = 0.0
    notes: str = ""


class ExperimentTracker:
    """Append-only CSV row logger for training experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS).writeheader()

    def append(self, row: ExperimentRow | dict) -> None:
        if isinstance(row, ExperimentRow):
            row = asdict(row)
        # coerce any list/dict fields to json strings so CSV round-trips
        for k, v in list(row.items()):
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v)
        # ensure all fields present
        for f in EXPERIMENT_FIELDS:
            row.setdefault(f, "")
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS).writerow(
                {k: row[k] for k in EXPERIMENT_FIELDS})

    def rows(self) -> list[dict[str, str]]:
        with self.path.open("r") as f:
            return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# early stopping
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """Monitor a validation metric; call `step(value)` per epoch.

    Args:
        patience: epochs without improvement before `should_stop` flips true.
        mode: "max" (bigger is better) or "min".
        min_delta: minimum change to count as an improvement.
    """

    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best: float = field(init=False)
    counter: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        assert self.mode in {"max", "min"}
        self.best = -float("inf") if self.mode == "max" else float("inf")

    def step(self, value: float) -> bool:
        if self.mode == "max":
            improved = value > (self.best + self.min_delta)
        else:
            improved = value < (self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


__all__ = [
    "set_seed", "seeded_worker_init", "device_summary",
    "load_config", "save_config",
    "ExperimentRow", "ExperimentTracker", "EXPERIMENT_FIELDS",
    "EarlyStopping",
]
