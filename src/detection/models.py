"""Segmentation model definitions for landslide detection.

Planned responsibilities (not yet implemented):
    - Baseline U-Net for 14-channel input to single-channel landslide mask.
    - Possible U-Net variants (residual, attention, etc.) if later required.

No architecture is defined yet; the input tensor shape and output channels
must be pinned down against verified dataset shapes first.
"""
