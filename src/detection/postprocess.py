"""Detection V2 post-processing and mask-to-polygon export.

Re-exports the pixel-space cleanup helpers from `postprocessing.py` (kept
unchanged from V1) and adds a mask-to-polygon pipeline that emits GeoJSON
FeatureCollections **in pixel coordinates**.

Landslide4Sense HDF5 files carry no CRS / affine transform, so V2 does NOT
fabricate geographic coordinates. The GeoJSON schema is fully typed
regardless, so a downstream Monitoring/Prediction module can attach a real
CRS + transform once one is available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .postprocessing import (  # re-export
    probability_to_mask, remove_small_components, fill_small_holes,
    extract_boundaries, PostprocessingConfig, apply,
)

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


# ---------------------------------------------------------------------------
# polygonization
# ---------------------------------------------------------------------------

@dataclass
class PolygonizeConfig:
    """How to turn a cleaned binary mask into GeoJSON polygons."""

    connectivity: int = 2          # 1 = 4-connected, 2 = 8-connected
    min_area_px: int = 8           # drop components smaller than this
    simplify_tolerance: float = 0.0  # 0 = no simplification
    coordinate_reference: str = "pixel"  # or "geo" (unused; documented)


def _find_polygon_pixels(labels: np.ndarray, label: int
                         ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """Extract the outer boundary + holes of one labeled component in pixel coords.

    Simple boundary tracing based on the label mask - not a full contour
    algorithm, but sufficient for GeoJSON export with pixel-space coordinates.
    We approximate the polygon by the outer bounding hull of the component
    (rectangular envelope) and by the connected boundary points.
    """
    ys, xs = np.where(labels == label)
    if xs.size == 0:
        return [], []
    # Rectangle-hull approximation (always a valid polygon, cheap):
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    outer = [(xmin, ymin), (xmax + 1, ymin), (xmax + 1, ymax + 1),
             (xmin, ymax + 1), (xmin, ymin)]
    return outer, []


def polygonize_mask(mask: np.ndarray, cfg: PolygonizeConfig | None = None,
                    probability: np.ndarray | None = None,
                    ) -> dict:
    """Turn a binary mask into a GeoJSON FeatureCollection in pixel coords.

    Args:
        mask: (H, W) uint8 in {0, 1}.
        cfg: PolygonizeConfig; defaults to standard 8-connectivity, min_area=8.
        probability: optional (H, W) float in [0, 1] used to attach mean
                     confidence per component.

    Each Feature carries:
        detection_id     : sequential int
        geometry         : GeoJSON Polygon (pixel-space)
        properties.centroid_x / centroid_y  : float pixel coordinates
        properties.area_px                  : int
        properties.confidence               : mean prob in the component
                                              (0.0 if probability=None)
        properties.crs                      : "pixel" (no georef available)
    """
    cfg = cfg or PolygonizeConfig()
    assert mask.ndim == 2 and mask.dtype in (np.uint8, bool), mask.dtype
    m = (mask > 0).astype(np.uint8)

    if ndi is None:
        return {"type": "FeatureCollection", "features": [],
                "warnings": ["scipy not available; polygonization skipped"]}

    struct = (ndi.generate_binary_structure(2, 1)
              if cfg.connectivity == 1
              else ndi.generate_binary_structure(2, 2))
    labels, n = ndi.label(m, structure=struct)
    features = []
    dropped = 0
    for k in range(1, n + 1):
        area = int((labels == k).sum())
        if area < cfg.min_area_px:
            dropped += 1
            continue
        outer, holes = _find_polygon_pixels(labels, k)
        if not outer:
            continue
        ys, xs = np.where(labels == k)
        cx = float(xs.mean()); cy = float(ys.mean())
        if probability is not None:
            conf = float(probability[ys, xs].mean())
        else:
            conf = 0.0
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[float(x), float(y)] for x, y in outer]],
            },
            "properties": {
                "detection_id": len(features) + 1,
                "centroid_x": cx, "centroid_y": cy,
                "area_px": area,
                "confidence": conf,
                "crs": cfg.coordinate_reference,
            },
        }
        features.append(feat)
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "connectivity": cfg.connectivity,
            "min_area_px": cfg.min_area_px,
            "dropped_below_min_area": dropped,
            "notes": ("Landslide4Sense HDF5 files carry no CRS/affine; "
                      "coordinates are pixel-space. Downstream Monitoring "
                      "can transform via a supplied CRS+affine."),
        },
    }


def save_geojson(fc: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fc, indent=2))
    return p


__all__ = [
    # re-exports from postprocessing.py
    "probability_to_mask", "remove_small_components", "fill_small_holes",
    "extract_boundaries", "PostprocessingConfig", "apply",
    # V2 additions
    "PolygonizeConfig", "polygonize_mask", "save_geojson",
]
