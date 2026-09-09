"""Preprocessing for the Landslide Detection module (Stage 1).

Design decisions (all derived from the verification pass on the actual
Landslide4Sense files - see notebooks/01_detection_development.ipynb):

    - Image HDF5 key: "img", shape (128, 128, 14), dtype float64.
    - Mask  HDF5 key: "mask", shape (128, 128), dtype uint8, values in {0, 1}.
    - All 14 channels are strictly non-negative on the training set with no
      NaN and no Inf observed. To stay robust for later data, NaN/Inf are
      still replaced with 0.0 defensively before standardization.
    - Normalization: per-channel z-score (x - mean) / max(std, EPS) using
      statistics computed on the TRAINING split only.
    - Tensor convention: (C, H, W) float32 for the image and (H, W) float32
      for the mask (0.0 / 1.0). The mask is emitted as float32 so it can be
      fed directly to BCEWithLogitsLoss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch

IMG_KEY = "img"
MASK_KEY = "mask"
N_CHANNELS = 14
PATCH_H = 128
PATCH_W = 128
STD_FLOOR = 1e-6
NAN_INF_REPLACEMENT = 0.0


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel normalization statistics (train-only)."""

    mean: np.ndarray  # shape (C,), float64
    std: np.ndarray   # shape (C,), float64
    std_floor: float = STD_FLOOR

    def __post_init__(self) -> None:
        assert self.mean.shape == (N_CHANNELS,), self.mean.shape
        assert self.std.shape == (N_CHANNELS,), self.std.shape

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text())
        mean = np.asarray(payload["mean"], dtype=np.float64)
        std = np.asarray(payload["std"], dtype=np.float64)
        floor = float(payload.get("std_floor", STD_FLOOR))
        return cls(mean=mean, std=std, std_floor=floor)


def read_image(path: str | Path) -> np.ndarray:
    """Read a Landslide4Sense image patch as float32 (H, W, C)."""
    with h5py.File(path, "r") as f:
        arr = f[IMG_KEY][()]
    if arr.shape != (PATCH_H, PATCH_W, N_CHANNELS):
        raise ValueError(f"Unexpected image shape {arr.shape} in {path}")
    return arr.astype(np.float32, copy=False)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a Landslide4Sense mask as uint8 (H, W)."""
    with h5py.File(path, "r") as f:
        arr = f[MASK_KEY][()]
    if arr.shape != (PATCH_H, PATCH_W):
        raise ValueError(f"Unexpected mask shape {arr.shape} in {path}")
    return arr.astype(np.uint8, copy=False)


def sanitize(image: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf/-Inf with a defensive fill value (0.0).

    Verification showed zero NaN / Inf on the training split, but this guard
    keeps downstream tensors finite if a corrupt patch ever appears.
    """
    return np.nan_to_num(image, nan=NAN_INF_REPLACEMENT,
                         posinf=NAN_INF_REPLACEMENT,
                         neginf=NAN_INF_REPLACEMENT)


def normalize(image: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Per-channel z-score using training statistics.

    image: (H, W, C) float32
    returns: (H, W, C) float32
    """
    denom = np.maximum(stats.std, stats.std_floor).astype(np.float32)
    mean = stats.mean.astype(np.float32)
    return (image - mean) / denom


def to_chw_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    """(H, W, C) numpy -> (C, H, W) float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(
        image_hwc.transpose(2, 0, 1).astype(np.float32, copy=False)))


def mask_to_tensor(mask_hw: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 -> (H, W) float32 torch tensor with values in {0.0, 1.0}."""
    return torch.from_numpy(mask_hw.astype(np.float32, copy=False))


def preprocess_pair(image_path: str | Path,
                    mask_path: str | Path,
                    stats: NormalizationStats,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full non-augmentation pipeline for a single (image, mask) pair.

    Returns:
        (image_tensor CHW float32, mask_tensor HW float32)
    """
    img = sanitize(read_image(image_path))
    img = normalize(img, stats)
    msk = read_mask(mask_path)
    return to_chw_tensor(img), mask_to_tensor(msk)
