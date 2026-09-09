"""Inference pipeline for the trained Landslide Detection model.

Supports both V1 (`model.UNet`) and V2 (`model_v2.UNetV2`) checkpoints. The
V2 checkpoint carries an `arch` dict specifying `base_features`, `norm`,
`residual`, etc. When present the V2 architecture is instantiated;
otherwise the V1 architecture is used.

Flow:
    raw HDF5 image -> preprocessing.preprocess_pair -> model forward
        -> sigmoid -> LOCKED threshold -> postprocessing -> binary mask

Nothing here reads validation/test data to configure itself. Normalization
statistics come from the Stage-1 file (train-only), and the threshold from
the Phase-12 locked config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import UNet
from .model_v2 import UNetV2, UNetV2Config
from .postprocess import PostprocessingConfig, apply as apply_postprocessing
from .preprocessing import (
    NormalizationStats, normalize, read_image, sanitize,
)


def _build_model(arch: dict) -> torch.nn.Module:
    """Instantiate V1 or V2 U-Net from an `arch` dict."""
    kind = arch.get("kind", "v1")
    if kind == "v2":
        cfg = UNetV2Config(
            in_channels=int(arch.get("in_channels", 14)),
            out_channels=int(arch.get("out_channels", 1)),
            base_features=int(arch.get("base_features", 32)),
            norm=arch.get("norm", "batchnorm"),
            residual=bool(arch.get("residual", False)),
            bottleneck_dropout=float(arch.get("bottleneck_dropout", 0.0)),
        )
        return UNetV2(**cfg.as_dict())
    if kind == "v1":
        return UNet(
            in_channels=int(arch.get("in_channels", 14)),
            out_channels=int(arch.get("out_channels", 1)),
            base_features=int(arch.get("base_features", 32)),
        )
    raise ValueError(f"unknown architecture kind: {kind!r}")


@dataclass
class DetectionInference:
    """Loadable inference bundle: model + normalization + threshold + postproc."""

    model: torch.nn.Module
    stats: NormalizationStats
    threshold: float
    device: torch.device
    postproc: PostprocessingConfig | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_files(cls,
                   checkpoint_path: str | Path,
                   normalization_path: str | Path,
                   threshold: float,
                   device: torch.device | str = "cpu",
                   arch: dict | None = None,
                   postproc: PostprocessingConfig | None = None,
                   ) -> "DetectionInference":
        """Load a bundle for inference.

        `arch` is a dict describing the architecture. If omitted, we look for
        `arch` inside the checkpoint payload; if still missing, we fall back
        to V1 U-Net (base_features=32).
        """
        device = torch.device(device)
        payload = torch.load(str(checkpoint_path), map_location=device,
                             weights_only=False)
        if arch is None:
            arch = payload.get("arch") if isinstance(payload, dict) else None
        if arch is None:
            arch = {"kind": "v1", "in_channels": 14, "out_channels": 1,
                    "base_features": 32}
        model = _build_model(arch)
        state = (payload["model"] if isinstance(payload, dict) and "model" in payload
                 else payload)
        model.load_state_dict(state)
        model.to(device).eval()
        stats = NormalizationStats.from_json(normalization_path)
        return cls(model=model, stats=stats, threshold=float(threshold),
                   device=device,
                   postproc=(postproc
                             or PostprocessingConfig(threshold=float(threshold))),
                   metadata={"arch": arch,
                             "checkpoint": str(checkpoint_path),
                             "normalization": str(normalization_path)})

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
        img = read_image(image_h5_path)
        return self.infer_array(img)


__all__ = ["DetectionInference"]
