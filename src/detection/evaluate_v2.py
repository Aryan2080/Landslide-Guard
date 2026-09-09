"""Detection V2 one-shot test evaluator.

Fails loudly unless the locked-model marker file exists at the checkpoint's
parent directory. That marker (`LOCKED`) is written only after the model and
threshold have been locked (Phase 12). The idea is to make it hard to
accidentally invoke the test evaluator during hyperparameter tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import BinaryMetricAccumulator, PRAUCAccumulator


LOCKED_MARKER = "LOCKED"  # empty file next to the checkpoint


def write_lock(marker_dir: str | Path, reason: str = "phase 12 lock") -> Path:
    """Write the LOCKED marker file after Phase 12."""
    p = Path(marker_dir) / LOCKED_MARKER
    p.write_text(reason)
    return p


def is_locked(marker_dir: str | Path) -> bool:
    return (Path(marker_dir) / LOCKED_MARKER).is_file()


@torch.no_grad()
def evaluate_test(model: nn.Module,
                  loader: DataLoader,
                  device: torch.device,
                  threshold: float,
                  require_lock_dir: str | Path | None = None,
                  ) -> dict[str, Any]:
    """One-shot exact evaluation on a locked-DataLoader (typically the test set).

    Args:
        model: trained V2 model (weights already loaded).
        loader: test DataLoader (no augmentation).
        device: cuda / cpu.
        threshold: locked threshold from Phase 12.
        require_lock_dir: if set, the LOCKED marker must exist here.

    Returns a dict with dice / iou / precision / recall / f1 / specificity /
    accuracy / pr_auc / tp / fp / fn / tn / threshold.
    """
    if require_lock_dir is not None and not is_locked(require_lock_dir):
        raise RuntimeError(
            f"Test evaluation blocked: {LOCKED_MARKER} missing in {require_lock_dir}. "
            "Lock the model and threshold in Phase 12 first.")
    model.eval()
    acc = BinaryMetricAccumulator(threshold=float(threshold))
    pr = PRAUCAccumulator()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        acc.update(logits.float(), yb)
        pr.update(logits.float(), yb)
    out = acc.compute()
    out["pr_auc"] = pr.compute()
    return out


__all__ = ["evaluate_test", "write_lock", "is_locked", "LOCKED_MARKER"]
