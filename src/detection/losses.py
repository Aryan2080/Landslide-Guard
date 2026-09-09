"""Segmentation losses for binary landslide segmentation.

Public API:
    - DiceLoss      : soft Dice on sigmoid(logits) vs. float targets in {0, 1}.
    - BCEDiceLoss   : convex combination BCE(logits) + Dice(logits, target).
                      BCE weight and Dice weight are configurable (default 0.5 / 0.5).

Inputs to every loss below:
    logits : (B, 1, H, W) or (B, H, W) raw logits from the model.
    target : (B, 1, H, W) or (B, H, W) float32, values in {0.0, 1.0}.

Both losses reduce to a scalar tensor.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _align(logits: torch.Tensor, target: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Squeeze/broadcast so logits and target both look like (B, 1, H, W)."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.dtype != logits.dtype:
        target = target.to(logits.dtype)
    return logits, target


class DiceLoss(nn.Module):
    """Soft Dice loss on the sigmoid of the logits.

        dice = (2 * sum(p * y) + eps) / (sum(p) + sum(y) + eps)
        loss = 1 - dice   (per batch, averaged)

    A per-sample smoothing epsilon prevents div-by-zero on all-background
    patches. Sum is taken over spatial dims (H, W).
    """

    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)  # H, W
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)  # (B, 1)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCEWithLogits + soft Dice.

    Motivation: Landslide4Sense is heavily imbalanced (~2% positive pixels
    on training). Dice provides shape/region supervision; BCE stabilises
    the pixel-wise signal early in training.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 dice_eps: float = 1.0,
                 pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        assert bce_weight >= 0 and dice_weight >= 0
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(eps=dice_eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        logits_a, target_a = _align(logits, target)
        return (self.bce_weight * self.bce(logits_a, target_a)
                + self.dice_weight * self.dice(logits_a, target_a))


__all__ = ["DiceLoss", "BCEDiceLoss"]
