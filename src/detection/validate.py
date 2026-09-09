"""Validation-only helpers for the Landslide Detection module.

Separated from train.py so downstream code (threshold sweep, error
analysis, final test evaluation) can import a stable "compute exact
metrics on this DataLoader" surface.
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import (
    BinaryMetricAccumulator, PRAUCAccumulator, sweep_thresholds,
)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader,
                   device: torch.device
                   ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (logits, target) tensors for each batch. Model must be in eval."""
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yield model(xb), yb


@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader,
                    device: torch.device, threshold: float = 0.5,
                    loss_fn: nn.Module | None = None,
                    with_pr_auc: bool = True) -> dict[str, float]:
    """Full validation pass: loss (if loss_fn given) + threshold metrics + PR-AUC."""
    model.eval()
    acc = BinaryMetricAccumulator(threshold=threshold)
    pr = PRAUCAccumulator() if with_pr_auc else None
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        acc.update(logits.float(), yb)
        if pr is not None:
            pr.update(logits.float(), yb)
        if loss_fn is not None:
            total_loss += float(loss_fn(logits, yb).item()) * xb.size(0)
            n += xb.size(0)
    metrics = acc.compute()
    if loss_fn is not None:
        metrics["loss"] = total_loss / max(n, 1)
    if pr is not None:
        metrics["pr_auc"] = pr.compute()
    return metrics


def sweep_threshold(model: nn.Module, loader: DataLoader,
                    device: torch.device,
                    thresholds: np.ndarray | None = None,
                    ) -> dict[str, np.ndarray]:
    """Run the model once and evaluate all thresholds. Returns per-threshold
    Dice/IoU/P/R/F1 arrays aligned with `thresholds`."""
    model.eval()
    def _iter():
        # We cache logits per batch to avoid a second forward pass at
        # each threshold. `sweep_thresholds` iterates over batches once.
        for logits, target in collect_logits(model, loader, device):
            yield logits.float(), target
    return sweep_thresholds(_iter(), thresholds=thresholds)


__all__ = ["collect_logits", "compute_metrics", "sweep_threshold"]
