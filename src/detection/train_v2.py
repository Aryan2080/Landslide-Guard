"""Detection V2 training driver: fair-comparison protocol.

The whole point of `train_v2` versus `train.py` is that every experiment (loss
arm, architecture arm) is guaranteed to receive the same training protocol:

    - same seed
    - same optimizer + LR + weight decay
    - same LR scheduler (cosine to `epochs`)
    - same max epochs and early-stopping patience
    - same batch size
    - same AMP setting on CUDA
    - same gradient clipping

Only the model (arch) and loss (spec) differ per arm. Everything is logged for
Phase-9 reporting.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import BinaryMetricAccumulator, PRAUCAccumulator
from .utils import EarlyStopping, set_seed


@dataclass
class TrainProtocol:
    """Everything held constant across V2 experiments."""

    seed: int = 42
    epochs: int = 100
    early_stop_patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    select_by: str = "val_dice"    # {val_dice, val_iou, val_f1, val_pr_auc, val_loss}
    threshold_for_val_metrics: float = 0.5


@dataclass
class EpochStatsV2:
    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_specificity: float
    val_accuracy: float
    val_pr_auc: float
    seconds: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("epoch", "lr", "train_loss", "val_loss",
                 "val_dice", "val_iou", "val_precision", "val_recall",
                 "val_f1", "val_specificity", "val_accuracy",
                 "val_pr_auc", "seconds")}


class TrainerV2:
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
        running = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite train loss {loss.item():.4f}")
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
    def validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        acc = BinaryMetricAccumulator(threshold=self.threshold)
        pr = PRAUCAccumulator()
        running = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            acc.update(logits.float(), yb)
            pr.update(logits.float(), yb)
        metrics = acc.compute()
        metrics["pr_auc"] = pr.compute()
        return running / max(n, 1), metrics


@dataclass
class RunHistoryV2:
    epochs: list[EpochStatsV2] = field(default_factory=list)

    def append(self, s: EpochStatsV2) -> None:
        self.epochs.append(s)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([e.as_dict() for e in self.epochs],
                                         indent=2))


def fit_v2(trainer: TrainerV2,
           train_loader: DataLoader,
           val_loader: DataLoader,
           protocol: TrainProtocol,
           checkpoint_path: str | Path,
           history_path: str | Path | None = None,
           log_fn: Callable[[str], None] = print,
           extra_state: dict | None = None,
           ) -> RunHistoryV2:
    """Train under the fixed protocol; keep the best-val checkpoint."""
    metric_key = {"val_loss": None,
                  "val_dice": "dice",
                  "val_iou":  "iou",
                  "val_f1":   "f1",
                  "val_pr_auc": "pr_auc"}[protocol.select_by]

    hist = RunHistoryV2()
    best = -float("inf") if protocol.select_by != "val_loss" else float("inf")
    stopper = EarlyStopping(patience=protocol.early_stop_patience, mode="max")

    for ep in range(1, protocol.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, vm = trainer.validate(val_loader)
        lr = trainer.optimizer.param_groups[0]["lr"]
        if trainer.scheduler is not None:
            trainer.scheduler.step()

        stats = EpochStatsV2(
            epoch=ep, lr=lr,
            train_loss=train_loss, val_loss=val_loss,
            val_dice=vm["dice"], val_iou=vm["iou"],
            val_precision=vm["precision"], val_recall=vm["recall"],
            val_f1=vm["f1"], val_specificity=vm["specificity"],
            val_accuracy=vm["accuracy"],
            val_pr_auc=float(vm.get("pr_auc", 0.0)),
            seconds=time.time() - t0)
        hist.append(stats)

        selected = (-val_loss if protocol.select_by == "val_loss"
                    else vm[metric_key])
        improved = selected > best
        if improved:
            best = selected
            payload = {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": (trainer.scheduler.state_dict()
                              if trainer.scheduler is not None else None),
                "epoch": ep, "val_metrics": vm,
                "val_loss": val_loss,
                "select_by": protocol.select_by,
                "protocol": protocol.__dict__,
            }
            if extra_state:
                payload.update(extra_state)
            torch.save(payload, str(checkpoint_path))

        log_fn(
            f"epoch {ep:03d}/{protocol.epochs:03d}  lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice={vm['dice']:.4f}  iou={vm['iou']:.4f}  "
            f"P={vm['precision']:.4f} R={vm['recall']:.4f}  "
            f"pr_auc={vm.get('pr_auc', 0.0):.4f}  "
            f"({stats.seconds:.1f}s){'  <- best' if improved else ''}"
        )
        if history_path is not None:
            hist.save_json(history_path)

        stopper.step(selected if protocol.select_by != "val_loss" else -selected)
        if stopper.should_stop:
            log_fn(f"early stopping at epoch {ep} "
                   f"(no improvement in {stopper.patience} epochs)")
            break

    return hist


def build_optimizer(model: nn.Module, protocol: TrainProtocol) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(),
                             lr=protocol.lr,
                             weight_decay=protocol.weight_decay)


def build_scheduler(opt: torch.optim.Optimizer, protocol: TrainProtocol
                    ) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=protocol.epochs)


__all__ = [
    "TrainProtocol", "EpochStatsV2", "RunHistoryV2",
    "TrainerV2", "fit_v2",
    "build_optimizer", "build_scheduler",
]
