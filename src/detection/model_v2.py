"""Detection V2 U-Net.

Improvements over V1 (`model.py`):
    - Configurable normalization (BatchNorm / GroupNorm / None).
    - Optional residual double-conv blocks (default False; enable per experiment).
    - Optional dropout in the bottleneck (default 0.0).
    - Kaiming initialization for conv layers.
    - Same tensor contract as V1: input (B, 14, 128, 128), output (B, 1, 128, 128)
      raw logits (no built-in sigmoid), so `BCEWithLogitsLoss` and friends stay
      numerically stable.

Configurable `base_features` in {16, 24, 32, 48, 64}. V2 baseline is 32.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import nn


NormKind = Literal["batchnorm", "groupnorm", "none"]


def _make_norm(kind: NormKind, num_features: int, gn_groups: int = 8) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm2d(num_features)
    if kind == "groupnorm":
        g = min(gn_groups, num_features)
        # Ensure divisibility
        while num_features % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(num_groups=max(g, 1), num_channels=num_features)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r}")


class DoubleConvV2(nn.Module):
    """(Conv3x3 -> Norm -> ReLU) x 2, with optional residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, norm: NormKind = "batchnorm",
                 residual: bool = False) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                               bias=(norm == "none"))
        self.n1 = _make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                               bias=(norm == "none"))
        self.n2 = _make_norm(norm, out_ch)
        self.relu = nn.ReLU(inplace=True)
        # Residual shortcut projection when channel counts differ
        self.shortcut = (nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                         if residual and in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        if self.residual:
            y = y + self.shortcut(x)
        return self.relu(y)


class DownV2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: NormKind = "batchnorm",
                 residual: bool = False) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvV2(in_ch, out_ch, norm=norm, residual=residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpV2(nn.Module):
    """Transpose-conv upsample, concat skip, DoubleConvV2."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 norm: NormKind = "batchnorm",
                 residual: bool = False) -> None:
        super().__init__()
        # Halve channels via transpose-conv, then concat with skip.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvV2(in_ch // 2 + skip_ch, out_ch,
                                 norm=norm, residual=residual)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


@dataclass
class UNetV2Config:
    in_channels: int = 14
    out_channels: int = 1
    base_features: int = 32
    norm: NormKind = "batchnorm"
    residual: bool = False
    bottleneck_dropout: float = 0.0

    def as_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_features": self.base_features,
            "norm": self.norm,
            "residual": self.residual,
            "bottleneck_dropout": self.bottleneck_dropout,
        }


class UNetV2(nn.Module):
    """Detection V2 U-Net (14 -> 1 logits, 128x128 spatial)."""

    def __init__(self,
                 in_channels: int = 14,
                 out_channels: int = 1,
                 base_features: int = 32,
                 norm: NormKind = "batchnorm",
                 residual: bool = False,
                 bottleneck_dropout: float = 0.0) -> None:
        super().__init__()
        self.cfg = UNetV2Config(in_channels, out_channels, base_features,
                                norm, residual, bottleneck_dropout)
        f = base_features

        # Encoder
        self.inc = DoubleConvV2(in_channels, f,        norm=norm, residual=residual)  # 128
        self.d1  = DownV2(f,        f * 2,             norm=norm, residual=residual)  #  64
        self.d2  = DownV2(f * 2,    f * 4,             norm=norm, residual=residual)  #  32
        self.d3  = DownV2(f * 4,    f * 8,             norm=norm, residual=residual)  #  16
        # Bottleneck
        self.d4  = DownV2(f * 8,    f * 16,            norm=norm, residual=residual)  #   8
        self.drop = (nn.Dropout2d(bottleneck_dropout) if bottleneck_dropout > 0
                     else nn.Identity())
        # Decoder
        self.u1  = UpV2(f * 16, f * 8, f * 8,          norm=norm, residual=residual)  #  16
        self.u2  = UpV2(f * 8,  f * 4, f * 4,          norm=norm, residual=residual)  #  32
        self.u3  = UpV2(f * 4,  f * 2, f * 2,          norm=norm, residual=residual)  #  64
        self.u4  = UpV2(f * 2,  f,     f,              norm=norm, residual=residual)  # 128
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.drop(self.d4(x4))
        y = self.u1(x5, x4)
        y = self.u2(y, x3)
        y = self.u3(y, x2)
        y = self.u4(y, x1)
        return self.outc(y)


def count_parameters_v2(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_unet_v2(cfg: dict | UNetV2Config | None = None, **overrides) -> UNetV2:
    """Factory: dict/UNetV2Config + kwargs overrides -> UNetV2."""
    if cfg is None:
        cfg = UNetV2Config()
    elif isinstance(cfg, dict):
        cfg = UNetV2Config(**cfg)
    payload = cfg.as_dict()
    payload.update(overrides)
    return UNetV2(**payload)


__all__ = ["UNetV2", "UNetV2Config", "build_unet_v2", "count_parameters_v2",
           "DoubleConvV2", "DownV2", "UpV2"]
