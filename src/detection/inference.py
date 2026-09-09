"""Inference pipeline for the trained Landslide Detection U-Net.

Flow:

    raw HDF5 image -> preprocessing.preprocess_pair -> model forward
        -> sigmoid -> LOCKED threshold -> postprocessing -> binary mask

Nothing here reads any part of a validation or test file to configure
itself; the normalization statistics and threshold both come from the
locked artifacts written during Stage 2 (validation-derived).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import UNet
from .postprocessing import (
    PostprocessingConfig, apply as apply_postprocessing,
)
from .preprocessing import (
    NormalizationStats, normalize, read_image, sanitize,
)


@dataclass
class DetectionInference:
    """Loadable inference bundle: model + normalization + threshold."""

    model: torch.nn.Module
    stats: NormalizationStats
    threshold: float
    device: torch.device
    postproc: PostprocessingConfig | None = None

    @classmethod
    def from_files(cls,
                   checkpoint_path: str | Path,
                   normalization_path: str | Path,
                   threshold: float,
                   device: torch.device | str = "cpu",
                   in_channels: int = 14,
                   out_channels: int = 1,
                   base_features: int = 32,
                   postproc: PostprocessingConfig | None = None
                   ) -> "DetectionInference":
        device = torch.device(device)
        model = UNet(in_channels=in_channels, out_channels=out_channels,
                     base_features=base_features)
        payload = torch.load(str(checkpoint_path), map_location=device,
                             weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
        model.to(device).eval()
        stats = NormalizationStats.from_json(normalization_path)
        return cls(model=model, stats=stats, threshold=float(threshold),
                   device=device,
                   postproc=postproc or PostprocessingConfig(threshold=float(threshold)))

    @torch.no_grad()
    def infer_array(self, image_hwc: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Run on one raw (H, W, 14) image; return (prob, mask)."""
        assert image_hwc.ndim == 3 and image_hwc.shape[-1] == 14, image_hwc.shape
        img = normalize(sanitize(image_hwc.astype(np.float32)), self.stats)
        t = torch.from_numpy(np.ascontiguousarray(
            img.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        logits = self.model(t)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pp = self.postproc or PostprocessingConfig(threshold=self.threshold)
        mask = apply_postprocessing(prob, pp)
        return prob, mask

    @torch.no_grad()
    def infer_file(self, image_h5_path: str | Path
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: read one HDF5 image and run inference."""
        img = read_image(image_h5_path)  # (H, W, 14) float32
        return self.infer_array(img)


__all__ = ["DetectionInference"]
