"""Binary segmentation metrics for the Landslide Detection module.

All metrics are computed against `target ∈ {0, 1}` and predicted probabilities
thresholded at `threshold` (default 0.5). Confusion-matrix cells (TP/FP/FN/TN)
are accumulated as int64 in `BinaryMetricAccumulator` so the epoch-level
metric is exact rather than a batch-average of ratios.

Public API:
    - confusion_counts(prob, target, threshold) -> (tp, fp, fn, tn) as ints
    - iou_from_counts(tp, fp, fn)                -> float
    - f1_from_counts(tp, fp, fn)                 -> float
    - precision_from_counts(tp, fp)              -> float
    - recall_from_counts(tp, fn)                 -> float
    - BinaryMetricAccumulator: .update(logits, target) / .compute() -> dict
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _flatten(prob: torch.Tensor, target: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
    if prob.dim() == 4 and prob.shape[1] == 1:
        prob = prob.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    return prob.reshape(-1), target.reshape(-1)


def confusion_counts(prob: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for one batch."""
    p, t = _flatten(prob, target)
    pred_pos = p >= threshold
    targ_pos = t >= 0.5  # target is {0.0, 1.0} already
    tp = int((pred_pos & targ_pos).sum().item())
    fp = int((pred_pos & ~targ_pos).sum().item())
    fn = int((~pred_pos & targ_pos).sum().item())
    tn = int((~pred_pos & ~targ_pos).sum().item())
    return tp, fp, fn, tn


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def precision_from_counts(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall_from_counts(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = precision_from_counts(tp, fp)
    r = recall_from_counts(tp, fn)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


@dataclass
class BinaryMetricAccumulator:
    """Accumulates exact confusion counts across many batches."""

    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = torch.sigmoid(logits)
        tp, fp, fn, tn = confusion_counts(prob, target, self.threshold)
        self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn

    def compute(self) -> dict[str, float]:
        return {
            "iou":       iou_from_counts(self.tp, self.fp, self.fn),
            "f1":        f1_from_counts(self.tp, self.fp, self.fn),
            "precision": precision_from_counts(self.tp, self.fp),
            "recall":    recall_from_counts(self.tp, self.fn),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "threshold": self.threshold,
        }


__all__ = [
    "confusion_counts",
    "iou_from_counts", "f1_from_counts",
    "precision_from_counts", "recall_from_counts",
    "BinaryMetricAccumulator",
]
