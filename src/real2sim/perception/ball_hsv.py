"""Named tennis-ball HSV windows used across audit, grounding, and metric targets.

These presets deliberately differ: audit/camera-fit use a wide yellow gate for coverage
diagnostics, grounding uses a tighter letterbox detector for ACT state features, and metric
targets use a looser gate before geometric silhouette fitting. Call sites must pick a named
preset instead of inventing new anonymous thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallHsvWindow:
    name: str
    hue_min: int
    hue_max: int
    saturation_min: int
    value_min: int
    saturation_max: int = 255
    value_max: int = 255

    @property
    def lower(self) -> tuple[int, int, int]:
        return (self.hue_min, self.saturation_min, self.value_min)

    @property
    def upper(self) -> tuple[int, int, int]:
        return (self.hue_max, self.saturation_max, self.value_max)


BALL_HSV = {
    # data.quality.ball_area_px and sim.camera_fit centroid detector
    "audit": BallHsvWindow("audit", hue_min=22, hue_max=50, saturation_min=70, value_min=90),
    # policy.visual_grounding BallImageFeatures defaults
    "grounding_v1": BallHsvWindow(
        "grounding_v1", hue_min=24, hue_max=44, saturation_min=90, value_min=85),
    # policy.visual_targets.locate_objects yellow/green gate before silhouette fit
    "metric_targets": BallHsvWindow(
        "metric_targets", hue_min=20, hue_max=50, saturation_min=75, value_min=30),
}
