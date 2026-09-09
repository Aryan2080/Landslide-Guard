"""Post-processing for landslide probability maps.

    - probability_to_mask: threshold a (H, W) probability map at a locked
      threshold, returning uint8 {0, 1}.
    - remove_small_components: drop connected components below a
      configurable pixel-area threshold. Uses scipy.ndimage.label.
    - fill_small_holes: fill background holes inside a landslide region.
    - extract_boundaries: pixel-space boundary (uint8) of a binary mask
      via morphological gradient.

Geolocation is NOT fabricated: Landslide4Sense HDF5 files do not carry a
CRS or affine transform, so we operate strictly in pixel coordinates.
`extract_boundaries` returns a pixel mask, not geo-polygons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def probability_to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a probability map. Accepts (H, W) or (B, H, W) floats."""
    return (prob >= float(threshold)).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 4) -> np.ndarray:
    """Drop connected components smaller than `min_area` pixels.

    Args:
        mask: (H, W) uint8 in {0, 1}.
        min_area: minimum component size to keep (in pixels).

    Returns cleaned mask (same shape / dtype). Requires scipy; if scipy is
    unavailable and no components need removing this is a no-op.
    """
    if ndi is None:
        return mask
    if min_area <= 1:
        return mask
    labels, n = ndi.label(mask > 0)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    keep = sizes >= min_area
    if not keep.any():
        return np.zeros_like(mask)
    return (keep[labels]).astype(mask.dtype)


def fill_small_holes(mask: np.ndarray, max_hole: int = 8) -> np.ndarray:
    """Fill background holes with area < `max_hole` pixels inside foreground."""
    if ndi is None or max_hole <= 1:
        return mask
    inv = mask == 0
    labels, n = ndi.label(inv)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    small = sizes < max_hole
    fill = small[labels]
    out = mask.copy()
    out[fill] = 1
    return out


def extract_boundaries(mask: np.ndarray) -> np.ndarray:
    """Pixel-space boundary of a binary mask.

    Uses a 3x3 morphological gradient: boundary = mask - erode(mask).
    Returns uint8 {0, 1}.
    """
    if ndi is None:
        # Simple fallback: XOR with shifted versions.
        h_shift = np.zeros_like(mask); h_shift[:, 1:] = mask[:, :-1]
        v_shift = np.zeros_like(mask); v_shift[1:, :] = mask[:-1, :]
        b = (mask != h_shift) | (mask != v_shift)
        return (b & (mask > 0)).astype(np.uint8)
    struct = ndi.generate_binary_structure(2, 1)
    eroded = ndi.binary_erosion(mask > 0, structure=struct).astype(np.uint8)
    return ((mask > 0).astype(np.uint8) - eroded).astype(np.uint8)


@dataclass
class PostprocessingConfig:
    threshold: float = 0.5
    min_area: int = 0   # 0 = disabled by default; enable in the config
    max_hole: int = 0   # 0 = disabled by default


def apply(prob: np.ndarray, cfg: PostprocessingConfig) -> np.ndarray:
    """Convenience: probability -> mask -> optional cleanup."""
    mask = probability_to_mask(prob, cfg.threshold)
    if cfg.max_hole > 1:
        mask = fill_small_holes(mask, cfg.max_hole)
    if cfg.min_area > 1:
        mask = remove_small_components(mask, cfg.min_area)
    return mask


__all__ = [
    "probability_to_mask",
    "remove_small_components", "fill_small_holes", "extract_boundaries",
    "PostprocessingConfig", "apply",
]
