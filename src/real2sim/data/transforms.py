"""Aspect-preserving letterbox transform shared by real data preprocessing and sim rendering.

Decision D-004 (delegated): the 1280x720 iPhone frame is resized by a single scale factor so it
fits inside the model tensor, then padded symmetrically. Padding geometry is returned so camera
intrinsics can be adjusted and so the inference adapter applies an identical transform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

DEFAULT_PAD_VALUE = 114  # mid-grey, as used by common letterbox conventions


@dataclass(frozen=True)
class LetterboxGeometry:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    resized_w: int
    resized_h: int
    pad_left: int
    pad_top: int

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox_geometry(src_w: int, src_h: int, dst_w: int, dst_h: int) -> LetterboxGeometry:
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        raise ValueError("image sizes must be positive")
    scale = min(dst_w / src_w, dst_h / src_h)
    rw = min(dst_w, max(1, int(round(src_w * scale))))  # never 0 (cv2.resize would fail)
    rh = min(dst_h, max(1, int(round(src_h * scale))))
    return LetterboxGeometry(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale=scale,
        resized_w=rw,
        resized_h=rh,
        pad_left=(dst_w - rw) // 2,
        pad_top=(dst_h - rh) // 2,
    )


def letterbox_rgb(
    img: np.ndarray, dst_h: int, dst_w: int, pad_value: int = DEFAULT_PAD_VALUE
) -> tuple[np.ndarray, LetterboxGeometry]:
    """Letterbox an ``HxWx3`` uint8 RGB image into ``dst_h x dst_w``."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {img.shape}")
    h, w = img.shape[:2]
    g = letterbox_geometry(w, h, dst_w, dst_h)
    interp = cv2.INTER_AREA if g.scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (g.resized_w, g.resized_h), interpolation=interp)
    out = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    out[g.pad_top : g.pad_top + g.resized_h, g.pad_left : g.pad_left + g.resized_w] = resized
    return out, g


def to_chw_float(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """``HxWx3`` uint8 -> ``3xHxW`` float32 in [0, 1] (LeRobot's expected image range)."""
    return np.ascontiguousarray(img_hwc_uint8.transpose(2, 0, 1)).astype(np.float32) / 255.0


def adjust_intrinsics(K: np.ndarray, g: LetterboxGeometry) -> np.ndarray:
    """Map a 3x3 pinhole matrix from the source image to the letterboxed image.

    Uses the per-axis scale actually applied (``resized/src``, which differs from ``g.scale`` by
    rounding) and the pixel-centre convention of ``cv2.resize``: ``x' = (x + 0.5) s - 0.5``.
    """
    K2 = np.array(K, dtype=np.float64).copy()
    sx = g.resized_w / g.src_w
    sy = g.resized_h / g.src_h
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5 + g.pad_left
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5 + g.pad_top
    return K2
