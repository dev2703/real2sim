"""Captioned MP4 capture of the policy view (plan M7.3: every clip says what controls the robot)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def policy_view_to_bgr(chw_float01: np.ndarray, upscale: int = 2) -> np.ndarray:
    hwc = (np.clip(np.asarray(chw_float01, np.float32), 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    if upscale != 1:
        hwc = cv2.resize(hwc, (hwc.shape[1] * upscale, hwc.shape[0] * upscale),
                         interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)


def burn_caption(bgr: np.ndarray, caption: str) -> np.ndarray:
    out = bgr.copy()
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 26), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, caption, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class CaptionedVideo:
    """``cv2.VideoWriter`` wrapper; captions are mandatory for every frame."""

    def __init__(self, path: str | Path, fps: float, frame_hw: tuple[int, int], codec: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame_hw
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer at {self.path}")
        self.frames = 0

    def write(self, bgr: np.ndarray, caption: str) -> None:
        if not caption.strip():
            raise ValueError("caption is mandatory")
        self.writer.write(burn_caption(bgr, caption))
        self.frames += 1

    def close(self) -> None:
        self.writer.release()
