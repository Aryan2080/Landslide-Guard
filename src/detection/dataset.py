"""PyTorch Dataset and DataLoader utilities for Landslide4Sense.

Design decisions (Stage 1):

    - Filenames pair as image_N.h5  <->  mask_N.h5 by matching the numeric
      suffix. Any file without a match is dropped and reported via
      __init__'s report attribute.
    - Training augmentation: geometric only - horizontal flip, vertical flip,
      and 90-degree rotations (all four multiples). Any spatial transform is
      applied identically to the image and to the mask.
    - No augmentation is applied to the validation or test splits.
    - No color/photometric augmentation is applied, because 12 of the 14
      channels are physically-anchored Sentinel-2 reflectances and the last
      two are DEM/slope; color jitter on such channels would be meaningless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    N_CHANNELS,
    NormalizationStats,
    mask_to_tensor,
    normalize,
    read_image,
    read_mask,
    sanitize,
    to_chw_tensor,
)

_NUM_RE = re.compile(r"(\d+)")

VALID_SPLITS = ("train", "valid", "test")


def _numeric_id(filename: str) -> str:
    m = _NUM_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot extract numeric id from filename: {filename}")
    return m.group(1)


def _pair_files(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Pair image_N.h5 with mask_N.h5 by numeric id."""
    imgs = {_numeric_id(p.name): p for p in img_dir.iterdir()
            if p.suffix == ".h5"}
    masks = {_numeric_id(p.name): p for p in mask_dir.iterdir()
             if p.suffix == ".h5"}
    common = sorted(set(imgs) & set(masks), key=int)
    return [(imgs[k], masks[k]) for k in common]


class _GeometricAug:
    """Segmentation-safe geometric augmentation.

    Applies the same random transform to the image (C, H, W) and mask (H, W).
    Chosen operations preserve the raster grid: flips and 90-degree rotations.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img_chw: np.ndarray, mask_hw: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        # horizontal flip
        if self.rng.random() < 0.5:
            img_chw = np.ascontiguousarray(img_chw[:, :, ::-1])
            mask_hw = np.ascontiguousarray(mask_hw[:, ::-1])
        # vertical flip
        if self.rng.random() < 0.5:
            img_chw = np.ascontiguousarray(img_chw[:, ::-1, :])
            mask_hw = np.ascontiguousarray(mask_hw[::-1, :])
        # 90-degree rotation (0/1/2/3 quarter turns)
        k = int(self.rng.integers(0, 4))
        if k:
            img_chw = np.ascontiguousarray(np.rot90(img_chw, k=k, axes=(1, 2)))
            mask_hw = np.ascontiguousarray(np.rot90(mask_hw, k=k, axes=(0, 1)))
        return img_chw, mask_hw


@dataclass
class SplitReport:
    split: str
    n_images: int
    n_masks: int
    n_paired: int
    unpaired_image_ids: list[str] = field(default_factory=list)
    unpaired_mask_ids: list[str] = field(default_factory=list)


class Landslide4SenseDataset(Dataset):
    """Landslide4Sense HDF5 dataset with train-only geometric augmentation.

    Args:
        img_dir: directory of image_*.h5 files.
        mask_dir: directory of mask_*.h5 files.
        stats: normalization statistics (must be train-only).
        split: one of "train" / "valid" / "test". "train" enables augmentation.
        augment_seed: seed for the augmentation RNG (optional).

    __getitem__ returns:
        image: torch.float32, shape (14, 128, 128)
        mask:  torch.float32, shape (128, 128), values in {0.0, 1.0}
    """

    def __init__(self,
                 img_dir: str | Path,
                 mask_dir: str | Path,
                 stats: NormalizationStats,
                 split: str,
                 augment_seed: int | None = None):
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.stats = stats
        self.split = split
        self._pairs = _pair_files(self.img_dir, self.mask_dir)
        self._aug = _GeometricAug(seed=augment_seed) if split == "train" else None

        # Report unpaired items for debug/leakage inspection.
        img_ids = {_numeric_id(p.name) for p in self.img_dir.iterdir()
                   if p.suffix == ".h5"}
        mask_ids = {_numeric_id(p.name) for p in self.mask_dir.iterdir()
                    if p.suffix == ".h5"}
        self.report = SplitReport(
            split=split,
            n_images=len(img_ids),
            n_masks=len(mask_ids),
            n_paired=len(self._pairs),
            unpaired_image_ids=sorted(img_ids - mask_ids, key=int),
            unpaired_mask_ids=sorted(mask_ids - img_ids, key=int),
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p = self._pairs[idx]
        img_hwc = sanitize(read_image(img_p))            # (H, W, C) float32
        img_hwc = normalize(img_hwc, self.stats)         # z-score
        img_chw = np.ascontiguousarray(img_hwc.transpose(2, 0, 1))  # (C, H, W)
        mask_hw = read_mask(mask_p)                      # (H, W) uint8
        if self._aug is not None:
            img_chw, mask_hw = self._aug(img_chw, mask_hw)
        image_t = torch.from_numpy(img_chw.astype(np.float32, copy=False))
        mask_t = mask_to_tensor(mask_hw)
        return image_t, mask_t


def build_dataloader(dataset: Landslide4SenseDataset,
                     batch_size: int,
                     num_workers: int = 0,
                     shuffle: bool | None = None,
                     drop_last: bool = False,
                     pin_memory: bool = False,
                     ) -> DataLoader:
    """DataLoader factory. Defaults shuffle=True only for the training split."""
    if shuffle is None:
        shuffle = dataset.split == "train"
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last,
                      pin_memory=pin_memory)


def build_all_splits(data_root: str | Path,
                     stats: NormalizationStats,
                     augment_seed: int | None = None,
                     ) -> dict[str, Landslide4SenseDataset]:
    """Construct all three Landslide4Sense splits from the canonical layout."""
    root = Path(data_root)
    layout = {
        "train": (root / "TrainData" / "TrainData" / "img",
                  root / "TrainData" / "TrainData" / "mask"),
        "valid": (root / "ValidData" / "ValidData" / "img",
                  root / "ValidData" / "ValidData" / "mask"),
        "test":  (root / "TestData"  / "TestData"  / "img",
                  root / "TestData"  / "TestData"  / "mask"),
    }
    return {name: Landslide4SenseDataset(img, mask, stats, split=name,
                                         augment_seed=augment_seed)
            for name, (img, mask) in layout.items()}


__all__ = [
    "N_CHANNELS",
    "Landslide4SenseDataset",
    "SplitReport",
    "build_all_splits",
    "build_dataloader",
]
