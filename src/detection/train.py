"""Training and validation loops for the Landslide Detection U-Net.

Kept intentionally small so the notebook stays thin. The functions do not
depend on the notebook; they can be called from any Python entry point.

Design notes:
    - Trainer holds model / optimizer / scheduler / loss / device / AMP scaler.
    - `train_one_epoch` steps LR every epoch (cosine schedules expect that).
    - `validate` returns a metrics dict computed exactly (accumulated
      confusion counts, not a batch-average of ratios).
    - Best checkpoint selection is by validation IoU by default.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import BinaryMetricAccumulator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EpochStats:
    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    val_iou: float
    val_f1: float
    val_precision: float
    val_recall: float
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("epoch", "lr", "train_loss", "val_loss",
                 "val_iou", "val_f1", "val_precision", "val_recall",
                 "seconds")}


class Trainer:
    """Minimal training controller for a single-GPU (or CPU) run."""

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 use_amp: bool = True,
                 grad_clip: Optional[float] = None,
                 threshold: float = 0.5) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = grad_clip
        self.threshold = threshold

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        return running / max(n, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        acc = BinaryMetricAccumulator(threshold=self.threshold)
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            acc.update(logits.float(), yb)
        return running / max(n, 1), acc.compute()


@dataclass
class RunHistory:
    epochs: list[EpochStats] = field(default_factory=list)

    def append(self, s: EpochStats) -> None:
        self.epochs.append(s)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([e.as_dict() for e in self.epochs],
                                         indent=2))


def fit(trainer: Trainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: str | Path,
        history_path: str | Path | None = None,
        log_fn=print,
        select_by: str = "val_iou") -> RunHistory:
    """Train `epochs` epochs, keep best-val checkpoint, return history.

    The best checkpoint is a plain torch.save({'model': state_dict, ...}).
    """
    assert select_by in {"val_iou", "val_f1", "val_loss"}
    hist = RunHistory()
    best = -float("inf") if select_by != "val_loss" else float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_metrics = trainer.validate(val_loader)
        lr = trainer.optimizer.param_groups[0]["lr"]
        if trainer.scheduler is not None:
            trainer.scheduler.step()

        stats = EpochStats(
            epoch=ep, lr=lr,
            train_loss=train_loss, val_loss=val_loss,
            val_iou=val_metrics["iou"], val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            seconds=time.time() - t0)
        hist.append(stats)

        selected = -val_loss if select_by == "val_loss" else val_metrics[
            {"val_iou": "iou", "val_f1": "f1"}[select_by]]
        improved = selected > best
        if improved:
            best = selected
            torch.save(
                {
                    "model": trainer.model.state_dict(),
                    "epoch": ep,
                    "val_metrics": val_metrics,
                    "select_by": select_by,
                },
                str(checkpoint_path),
            )

        log_fn(
            f"epoch {ep:03d}/{epochs:03d}  "
            f"lr={lr:.2e}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_iou={val_metrics['iou']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}  "
            f"({stats.seconds:.1f}s){'  <- best' if improved else ''}"
        )
        if history_path is not None:
            hist.save_json(history_path)

    return hist


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> dict[str, float]:
    """One-pass exact metrics over a DataLoader (used for the test set)."""
    model.eval()
    acc = BinaryMetricAccumulator(threshold=threshold)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        acc.update(logits.float(), yb)
    return acc.compute()


__all__ = [
    "set_seed",
    "EpochStats", "RunHistory",
    "Trainer", "fit", "evaluate",
]
