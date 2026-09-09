"""Detection V2 losses.

All losses share a single call signature:
    loss(logits, target) -> torch.Tensor  (scalar)

logits: (B, 1, H, W) or (B, H, W) raw logits from the V2 U-Net.
target: (B, 1, H, W) or (B, H, W) float32, values in {0, 1}.

`build_loss_v2(spec, pos_weight=None)` is the single entry point. Every loss
also implements `.describe()` returning a small dict for experiment logging.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# alignment helper
# ---------------------------------------------------------------------------

def _align(logits: torch.Tensor, target: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.dtype != logits.dtype:
        target = target.to(logits.dtype)
    return logits, target


# ---------------------------------------------------------------------------
# individual losses
# ---------------------------------------------------------------------------

class BCEWithLogitsLossAligned(nn.Module):
    """Shape-aligning wrapper over `nn.BCEWithLogitsLoss`."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.inner = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def describe(self) -> dict:
        return {
            "name": "bce",
            "pos_weight": (float(self.pos_weight.item())
                           if self.pos_weight is not None else None),
        }

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        return self.inner(logits, target)


class DiceLossV2(nn.Module):
    """Soft-Dice loss on sigmoid(logits).

    dice = (2 * sum(p * y) + eps) / (sum(p) + sum(y) + eps)
    loss = 1 - mean(dice_per_sample)
    """

    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = float(eps)

    def describe(self) -> dict:
        return {"name": "dice", "eps": self.eps}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class FocalLossV2(nn.Module):
    """Binary focal loss.

    pt = p * y + (1 - p) * (1 - y)
    loss = mean(-alpha_t * (1 - pt)^gamma * log(pt))

    Numerically stable via `binary_cross_entropy_with_logits` for the log-pt term.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def describe(self) -> dict:
        return {"name": "focal", "gamma": self.gamma, "alpha": self.alpha}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class TverskyLossV2(nn.Module):
    """Tversky loss.

        T = TP / (TP + alpha*FN + beta*FP + eps)
        loss = 1 - T

    Choosing alpha > beta emphasises recall (penalises false negatives more
    than false positives). Landslide detection is recall-sensitive.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 eps: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def describe(self) -> dict:
        return {"name": "tversky", "alpha": self.alpha, "beta": self.beta,
                "eps": self.eps}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _align(logits, target)
        p = torch.sigmoid(logits)
        dims = (2, 3)
        tp = (p * target).sum(dim=dims)
        fn = ((1 - p) * target).sum(dim=dims)
        fp = (p * (1 - target)).sum(dim=dims)
        t = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - t.mean()


class FocalTverskyLossV2(nn.Module):
    """(1 - Tversky) ** gamma."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 4.0 / 3.0, eps: float = 1.0) -> None:
        super().__init__()
        self.tversky = TverskyLossV2(alpha=alpha, beta=beta, eps=eps)
        self.gamma = float(gamma)

    def describe(self) -> dict:
        d = self.tversky.describe()
        d["name"] = "focal_tversky"
        d["gamma"] = self.gamma
        return d

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, target).pow(self.gamma)


# ---------------------------------------------------------------------------
# composite losses (fair-comparison workhorses)
# ---------------------------------------------------------------------------

class WeightedComposite(nn.Module):
    """Weighted sum of two losses. Prints the exact recipe via describe()."""

    def __init__(self, a: nn.Module, b: nn.Module,
                 a_weight: float, b_weight: float) -> None:
        super().__init__()
        assert a_weight >= 0 and b_weight >= 0
        self.a = a; self.b = b
        self.a_weight = float(a_weight); self.b_weight = float(b_weight)

    def describe(self) -> dict:
        return {
            "name": f"{self.a.describe()['name']}+{self.b.describe()['name']}",
            "a_weight": self.a_weight, "b_weight": self.b_weight,
            "a": self.a.describe(), "b": self.b.describe(),
        }

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.a_weight * self.a(logits, target) + self.b_weight * self.b(logits, target)


def bce_dice(bce_weight: float = 0.5, dice_weight: float = 0.5,
             dice_eps: float = 1.0,
             pos_weight: Optional[torch.Tensor] = None) -> WeightedComposite:
    return WeightedComposite(
        BCEWithLogitsLossAligned(pos_weight=pos_weight),
        DiceLossV2(eps=dice_eps),
        bce_weight, dice_weight)


def focal_dice(gamma: float = 2.0, alpha: float = 0.25,
               focal_weight: float = 0.5, dice_weight: float = 0.5,
               dice_eps: float = 1.0) -> WeightedComposite:
    return WeightedComposite(
        FocalLossV2(gamma=gamma, alpha=alpha),
        DiceLossV2(eps=dice_eps),
        focal_weight, dice_weight)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_loss_v2(spec: dict,
                  pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Build a loss from a spec dict.

    Recognised names:
        "bce"                 - BCEWithLogitsLoss (accepts pos_weight)
        "dice"                - soft Dice
        "bce_dice"            - w1 * BCE + w2 * Dice        (bce_weight, dice_weight)
        "focal"               - focal loss                  (gamma, alpha)
        "focal_dice"          - w1 * Focal + w2 * Dice      (focal_weight, dice_weight, gamma, alpha)
        "weighted_bce_dice"   - w1 * BCE_pw + w2 * Dice, pos_weight forced on
        "tversky"             - Tversky loss                (alpha, beta)
        "focal_tversky"       - (1 - Tversky)^gamma         (alpha, beta, gamma)

    Any other keys in `spec` are ignored (recorded elsewhere).
    """
    name = spec.get("name", "bce_dice").lower()
    if name == "bce":
        return BCEWithLogitsLossAligned(pos_weight=pos_weight)
    if name == "dice":
        return DiceLossV2(eps=float(spec.get("dice_eps", 1.0)))
    if name == "bce_dice":
        return bce_dice(
            bce_weight=float(spec.get("bce_weight", 0.5)),
            dice_weight=float(spec.get("dice_weight", 0.5)),
            dice_eps=float(spec.get("dice_eps", 1.0)),
            pos_weight=None,
        )
    if name == "weighted_bce_dice":
        assert pos_weight is not None, "weighted_bce_dice requires pos_weight"
        return bce_dice(
            bce_weight=float(spec.get("bce_weight", 0.5)),
            dice_weight=float(spec.get("dice_weight", 0.5)),
            dice_eps=float(spec.get("dice_eps", 1.0)),
            pos_weight=pos_weight,
        )
    if name == "focal":
        return FocalLossV2(gamma=float(spec.get("gamma", 2.0)),
                           alpha=float(spec.get("alpha", 0.25)))
    if name == "focal_dice":
        return focal_dice(
            gamma=float(spec.get("gamma", 2.0)),
            alpha=float(spec.get("alpha", 0.25)),
            focal_weight=float(spec.get("focal_weight", 0.5)),
            dice_weight=float(spec.get("dice_weight", 0.5)),
            dice_eps=float(spec.get("dice_eps", 1.0)),
        )
    if name == "tversky":
        return TverskyLossV2(alpha=float(spec.get("alpha", 0.7)),
                             beta=float(spec.get("beta", 0.3)),
                             eps=float(spec.get("eps", 1.0)))
    if name == "focal_tversky":
        return FocalTverskyLossV2(alpha=float(spec.get("alpha", 0.7)),
                                  beta=float(spec.get("beta", 0.3)),
                                  gamma=float(spec.get("gamma", 4.0 / 3.0)),
                                  eps=float(spec.get("eps", 1.0)))
    raise ValueError(f"unknown loss name: {name!r}")


__all__ = [
    "BCEWithLogitsLossAligned",
    "DiceLossV2", "FocalLossV2", "TverskyLossV2", "FocalTverskyLossV2",
    "WeightedComposite", "bce_dice", "focal_dice",
    "build_loss_v2",
]
