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


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

        pt = p * y + (1 - p) * (1 - y)
        loss = -alpha_t * (1 - pt) ** gamma * log(pt)

    - `alpha` shifts weight toward the positive class in [0, 1].
    - `gamma >= 0` down-weights well-classified examples (0 = standard BCE).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        assert reduction in {"mean", "sum", "none"}
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalDiceLoss(nn.Module):
    """Weighted Focal + soft Dice. Default weights 0.5 / 0.5.

    Alternative to BCEDiceLoss for datasets with heavy imbalance where BCE
    alone over-weights easy background. We use it as one of the
    class-imbalance experiment arms in Stage 2 (Section 16).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 focal_weight: float = 0.5, dice_weight: float = 0.5,
                 dice_eps: float = 1.0) -> None:
        super().__init__()
        assert focal_weight >= 0 and dice_weight >= 0
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss(eps=dice_eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, target)
                + self.dice_weight * self.dice(logits, target))


class BCEWithLogitsLossAligned(nn.Module):
    """`nn.BCEWithLogitsLoss` that first shape-aligns logits and target.

    Our model emits (B, 1, H, W) and masks are (B, H, W). Rather than force
    every caller to unsqueeze, this wrapper broadcasts to (B, 1, H, W).
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.inner = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        return self.inner(logits, target)


def build_loss(spec: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Factory: dict spec -> loss module.

    Supported names:
        bce            {name: bce, pos_weight: optional}
        dice           {name: dice, dice_eps: 1.0}
        bce_dice       {name: bce_dice, bce_weight: 0.5, dice_weight: 0.5, dice_eps: 1.0, pos_weight: optional}
        focal          {name: focal, gamma: 2.0, alpha: 0.25}
        focal_dice     {name: focal_dice, gamma: 2.0, alpha: 0.25, focal_weight: 0.5, dice_weight: 0.5, dice_eps: 1.0}
    """
    name = spec.get("name", "bce_dice").lower()
    if name == "bce":
        return BCEWithLogitsLossAligned(pos_weight=pos_weight)
    if name == "dice":
        return DiceLoss(eps=float(spec.get("dice_eps", 1.0)))
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(spec.get("bce_weight", 0.5)),
            dice_weight=float(spec.get("dice_weight", 0.5)),
            dice_eps=float(spec.get("dice_eps", 1.0)),
            pos_weight=pos_weight,
        )
    if name == "focal":
        return FocalLoss(gamma=float(spec.get("gamma", 2.0)),
                         alpha=float(spec.get("alpha", 0.25)))
    if name == "focal_dice":
        return FocalDiceLoss(
            gamma=float(spec.get("gamma", 2.0)),
            alpha=float(spec.get("alpha", 0.25)),
            focal_weight=float(spec.get("focal_weight", 0.5)),
            dice_weight=float(spec.get("dice_weight", 0.5)),
            dice_eps=float(spec.get("dice_eps", 1.0)),
        )
    raise ValueError(f"unknown loss name: {name!r}")


__all__ = ["DiceLoss", "BCEDiceLoss", "FocalLoss", "FocalDiceLoss",
           "BCEWithLogitsLossAligned", "build_loss"]
