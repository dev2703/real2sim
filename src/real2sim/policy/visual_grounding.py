"""Causal image-space ball features with one training/inference implementation.

This deliberately has no camera/FK backprojection: image coordinates do not become
metric object positions until independent camera registration is accepted. The
fixed colour/shape rule is an experimental detector, not annotated ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from real2sim.data.labels import PROPRIO_NAMES
from real2sim.perception.ball_hsv import BALL_HSV

BALL_FEATURE_NAMES = ["ball_image_u_fraction", "ball_image_v_fraction",
                      "ball_image_radius_fraction", "ball_image_visible"]
_GROUNDING = BALL_HSV["grounding_v1"]


@dataclass(frozen=True)
class BallImageFeatures:
    version: str = "hsv_ball_v1"
    enabled: bool = True
    camera_key: str = "observation.images.iphone"
    height: int = 192
    width: int = 256
    hue_min: int = _GROUNDING.hue_min
    hue_max: int = _GROUNDING.hue_max
    saturation_min: int = _GROUNDING.saturation_min
    value_min: int = _GROUNDING.value_min
    min_area_px: int = 18
    max_area_fraction: float = 0.08
    min_circularity: float = 0.35

    def __post_init__(self):
        if (self.version != "hsv_ball_v1" or type(self.enabled) is not bool or self.camera_key != "observation.images.iphone"
                or (self.height, self.width) != (192, 256)):
            raise ValueError("unsupported visual grounding version, camera or image shape")
        if (not 0 <= self.hue_min <= self.hue_max <= 179
                or not 0 <= self.saturation_min <= 255 or not 0 <= self.value_min <= 255
                or not isinstance(self.min_area_px, int) or self.min_area_px < 1
                or not 0 < self.max_area_fraction < 1
                or not 0 < self.min_circularity <= 1):
            raise ValueError("invalid ball detector thresholds")

    @classmethod
    def from_dict(cls, value: dict) -> BallImageFeatures:
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def grounded_state_names() -> list[str]:
    return [*PROPRIO_NAMES, *BALL_FEATURE_NAMES]


def ball_features(image: np.ndarray, config: BallImageFeatures) -> np.ndarray:
    """Uint8 letterboxed RGB -> u/w, v/h, enclosing radius/w, visible.

    A missing detection is four zeros, including an explicit zero visibility flag.
    There is no previous-frame latch, future lookahead or privileged fallback.
    Multiple candidates are ordered by contour area with deterministic tie breaks.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (config.height, config.width, 3):
        raise ValueError("ball features require contract-sized uint8 RGB")
    if not config.enabled:
        return np.zeros(4, np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (config.hue_min, config.saturation_min, config.value_min),
                       (config.hue_max, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (area < config.min_area_px or area > image.shape[0] * image.shape[1] * config.max_area_fraction
                or perimeter <= 0 or 4 * np.pi * area / perimeter ** 2 < config.min_circularity):
            continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        candidates.append((area, u, v, radius))
    if not candidates:
        return np.zeros(4, np.float32)
    _, u, v, radius = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[0]
    return np.asarray([u / config.width, v / config.height, radius / config.width, 1], np.float32)


def append_ball_features(canonical: np.ndarray, image: np.ndarray,
                         config: BallImageFeatures) -> np.ndarray:
    canonical = np.asarray(canonical, np.float32)
    if canonical.shape != (13,) or not np.isfinite(canonical).all():
        raise ValueError("visual grounding requires finite canonical proprioception[13]")
    return np.r_[canonical, ball_features(image, config)].astype(np.float32)


def rgb_from_env(image: np.ndarray, config: BallImageFeatures) -> np.ndarray:
    """Invert the env's exact uint8/255 conversion before the shared detector."""
    image = np.asarray(image)
    if (image.shape != (3, config.height, config.width) or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError("visual grounding requires finite CHW image in [0,1]")
    return np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
