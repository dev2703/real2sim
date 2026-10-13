"""Train-only colour jitter and bounded image translation; labels and padding are preserved."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class ImageAugmentation:
    def __init__(self, config: dict, letterbox: dict):
        allowed = {"brightness", "contrast", "saturation", "shift_pixels"}
        if config.keys() - allowed:
            raise ValueError(f"unknown augmentation options: {config.keys() - allowed}")
        self.config = {k: float(config.get(k, 0)) for k in allowed}
        for key in ("brightness", "contrast", "saturation"):
            if not np.isfinite(self.config[key]) or not 0 <= self.config[key] <= 0.5:
                raise ValueError(f"{key} must be finite and in [0, 0.5]")
        shift = self.config["shift_pixels"]
        if not np.isfinite(shift) or shift != int(shift) or not 0 <= shift <= 8:
            raise ValueError("shift_pixels must be an integer in [0, 8]")
        self.geometry = letterbox

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Each DataLoader worker receives a reproducible torch seed from the loader generator.
        out = image.copy()
        g = self.geometry
        y, x = g["pad_top"], g["pad_left"]
        h, w = g["resized_h"], g["resized_w"]
        content = image[y:y + h, x:x + w].astype(np.float32) / 255
        brightness, contrast, saturation = [
            1 + (2 * torch.rand(()).item() - 1) * self.config[key]
            for key in ("brightness", "contrast", "saturation")
        ]
        content *= brightness
        grey = content @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        content = (content - grey.mean()) * contrast + grey.mean()
        grey = content @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        content = grey[..., None] + (content - grey[..., None]) * saturation
        content = np.round(np.clip(content, 0, 1) * 255).astype(np.uint8)
        shift = int(self.config["shift_pixels"])
        if shift:
            dx, dy = torch.randint(-shift, shift + 1, (2,)).tolist()
            content = cv2.warpAffine(content, np.array([[1, 0, dx], [0, 1, dy]], dtype=float),
                                     (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        out[y:y + h, x:x + w] = content
        return out
