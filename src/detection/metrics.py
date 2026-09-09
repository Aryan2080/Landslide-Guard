"""Binary segmentation metrics for the Landslide Detection module.

Two families of metrics live here:

1. **Threshold metrics** - accumulated from confusion-matrix cells at a
   configurable threshold. Exact rather than a batch-average of ratios.
   Metrics: Dice, IoU/Jaccard, precision, recall, F1, specificity,
   pixel accuracy.

2. **Threshold-free metric** - PR-AUC (average precision) computed from
   sorted per-pixel probabilities. Accumulated with `PRAUCAccumulator`.

Both accumulators are streaming, so an entire epoch's metric can be
computed exactly without materializing every pixel probability at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _flatten(prob: torch.Tensor, target: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
    if prob.dim() == 4 and prob.shape[1] == 1:
        prob = prob.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    return prob.reshape(-1), target.reshape(-1)


# ---------------------------------------------------------------------------
# threshold metrics: functions on confusion counts
# ---------------------------------------------------------------------------

def confusion_counts(prob: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for one batch as ints."""
    p, t = _flatten(prob, target)
    pred_pos = p >= threshold
    targ_pos = t >= 0.5
    tp = int((pred_pos & targ_pos).sum().item())
    fp = int((pred_pos & ~targ_pos).sum().item())
    fn = int((~pred_pos & targ_pos).sum().item())
    tn = int((~pred_pos & ~targ_pos).sum().item())
    return tp, fp, fn, tn


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    """IoU / Jaccard on the positive class. Undefined when there are
    no positive predictions AND no positive targets - defined here to
    return 1.0 in that case (perfect on an empty task)."""
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """Sorensen-Dice on the positive class. Returns 1.0 if there is
    nothing to detect and nothing was predicted."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return 2 * tp / denom


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


def specificity_from_counts(tn: int, fp: int) -> float:
    """TN / (TN + FP)."""
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def accuracy_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fp + fn + tn
    return (tp + tn) / denom if denom > 0 else 0.0


@dataclass
class BinaryMetricAccumulator:
    """Accumulate exact confusion counts across many batches."""

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
            "dice":        dice_from_counts(self.tp, self.fp, self.fn),
            "iou":         iou_from_counts(self.tp, self.fp, self.fn),
            "precision":   precision_from_counts(self.tp, self.fp),
            "recall":      recall_from_counts(self.tp, self.fn),
            "f1":          f1_from_counts(self.tp, self.fp, self.fn),
            "specificity": specificity_from_counts(self.tn, self.fp),
            "accuracy":    accuracy_from_counts(self.tp, self.fp, self.fn, self.tn),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# threshold-free metric: PR-AUC (average precision)
# ---------------------------------------------------------------------------

@dataclass
class PRAUCAccumulator:
    """Streams positive-class probabilities and labels; computes AP at end.

    We store per-pixel (prob, label) pairs in float32/int8 numpy arrays for
    memory efficiency. On Landslide4Sense at 128x128, each 245-item
    validation split contributes 245*16384 = 4,014,080 pixels - trivial to
    keep in RAM. If ever needed for larger data, swap in binning here.
    """

    _probs: list[np.ndarray] = field(default_factory=list)
    _labels: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self._probs.clear(); self._labels.clear()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).ravel()
        lab = (target.detach().cpu().numpy() >= 0.5).astype(np.uint8).ravel()
        self._probs.append(prob)
        self._labels.append(lab)

    def compute(self) -> float:
        if not self._probs:
            return 0.0
        probs = np.concatenate(self._probs)
        labels = np.concatenate(self._labels)
        return average_precision(labels, probs)


def average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    """AP (area under the precision-recall curve) for binary labels.

    Uses the trapezoidal reduction; equivalent to sklearn's average_precision
    computation up to tie handling. Ties are broken by placing all positives
    first (a slight optimistic bias, symmetric across models being compared).
    """
    if labels.size == 0:
        return 0.0
    if labels.sum() == 0:
        return 0.0  # AP is undefined; return 0 (worst) for stability
    # sort by score descending; break ties by label desc (positives first)
    order = np.lexsort((-labels, -probs))
    lab_sorted = labels[order]
    cum_tp = np.cumsum(lab_sorted)
    cum_fp = np.cumsum(1 - lab_sorted)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / labels.sum()
    # AP = sum over positive samples of (delta recall) * precision
    # delta recall is 1/n_pos at each positive, 0 elsewhere
    return float((precision * (lab_sorted == 1)).sum() / labels.sum())


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    logits_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute Dice / IoU / P / R / F1 for a grid of thresholds in one pass.

    Args:
        logits_iter: iterable of (logits, target) tensors (e.g. the
                     validation DataLoader wrapped in a generator that
                     runs the model per batch).
        thresholds:  1-D numpy array of thresholds to evaluate. Defaults
                     to np.arange(0.1, 0.91, 0.05).

    Returns:
        dict with 'thresholds' + per-metric arrays of the same length.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.905, 0.05)
    T = len(thresholds)
    tps = np.zeros(T, dtype=np.int64)
    fps = np.zeros(T, dtype=np.int64)
    fns = np.zeros(T, dtype=np.int64)
    tns = np.zeros(T, dtype=np.int64)
    for logits, target in logits_iter:
        prob = torch.sigmoid(logits.detach()).cpu().numpy().ravel()
        lab  = (target.detach().cpu().numpy() >= 0.5).astype(np.uint8).ravel()
        for i, thr in enumerate(thresholds):
            pred = prob >= thr
            tp = int((pred & lab.astype(bool)).sum())
            fp = int((pred & ~lab.astype(bool)).sum())
            fn = int((~pred & lab.astype(bool)).sum())
            tn = int((~pred & ~lab.astype(bool)).sum())
            tps[i] += tp; fps[i] += fp; fns[i] += fn; tns[i] += tn
    out = {
        "thresholds": thresholds,
        "dice":      np.array([dice_from_counts(t, fp, fn) for t, fp, fn in zip(tps, fps, fns)]),
        "iou":       np.array([iou_from_counts (t, fp, fn) for t, fp, fn in zip(tps, fps, fns)]),
        "precision": np.array([precision_from_counts(t, fp) for t, fp in zip(tps, fps)]),
        "recall":    np.array([recall_from_counts   (t, fn) for t, fn in zip(tps, fns)]),
        "f1":        np.array([f1_from_counts(t, fp, fn) for t, fp, fn in zip(tps, fps, fns)]),
        "tp": tps, "fp": fps, "fn": fns, "tn": tns,
    }
    return out


__all__ = [
    "confusion_counts",
    "dice_from_counts", "iou_from_counts", "f1_from_counts",
    "precision_from_counts", "recall_from_counts",
    "specificity_from_counts", "accuracy_from_counts",
    "BinaryMetricAccumulator", "PRAUCAccumulator", "average_precision",
    "sweep_thresholds",
]
