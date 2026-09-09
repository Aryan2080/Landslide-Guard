"""U-Net for landslide segmentation (14-channel input, 1-channel logits output).

From-scratch implementation - deliberately dependency-free (only torch.nn).
No pretrained weights: the 14-channel Sentinel-2 + terrain input is not
compatible with standard ImageNet 3-channel pretrained encoders.

Architecture:
    encoder: 4 down blocks (base -> 2x -> 4x -> 8x features)
    bottleneck: 1 block (16x features)
    decoder: 4 up blocks with skip connections
    output: 1x1 conv -> single-channel logits (no sigmoid)

The forward pass emits raw logits so callers can plug them into
BCEWithLogitsLoss (numerically stable). Take torch.sigmoid(logits) at
inference to get probabilities.
"""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2, no spatial downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transpose-conv upsample, concat skip, DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Sizes should match on 128x128 inputs; guard anyway.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net for LandslideGuard Detection.

    Args:
        in_channels: number of input channels (default 14).
        out_channels: number of output logit channels (default 1 for binary).
        base_features: channels after the first DoubleConv (default 32).
    """

    def __init__(self,
                 in_channels: int = 14,
                 out_channels: int = 1,
                 base_features: int = 32) -> None:
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)          # 128x128 -> f
        self.d1 = Down(f, f * 2)                       #  64      -> 2f
        self.d2 = Down(f * 2, f * 4)                   #  32      -> 4f
        self.d3 = Down(f * 4, f * 8)                   #  16      -> 8f
        self.d4 = Down(f * 8, f * 16)                  #   8      -> 16f (bottleneck)
        self.u1 = Up(f * 16, f * 8, f * 8)             #  16      -> 8f
        self.u2 = Up(f * 8,  f * 4, f * 4)             #  32      -> 4f
        self.u3 = Up(f * 4,  f * 2, f * 2)             #  64      -> 2f
        self.u4 = Up(f * 2,  f,     f)                 # 128      -> f
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        y = self.u1(x5, x4)
        y = self.u2(y, x3)
        y = self.u3(y, x2)
        y = self.u4(y, x1)
        return self.outc(y)  # (B, out_channels, H, W) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["UNet", "count_parameters"]
