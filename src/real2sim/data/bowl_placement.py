"""Offline bowl-rim observations for pairing recorded images with static renders."""

import cv2
import numpy as np


def detect_bowl_rim(rgb):
    """Return an ellipse candidate in the recorded policy view, or None.

    This scene-specific detector is for reviewed offline setup only, not policy
    observations or a generic bowl detector. Callers must visually review results.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (192, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected uint8 192x256 RGB policy image")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 25:
            continue
        ellipse = cv2.fitEllipse(contour)
        (x, y), (a, b), _ = ellipse
        if not (120 < x < 225 and 90 < y < 165 and 22 < min(a, b) < 55 and 35 < max(a, b) < 75):
            continue
        ratio = cv2.contourArea(contour) / (np.pi * a * b / 4)
        if ratio < .6:
            continue
        score = abs(max(a, b) - 48) / 15 + abs(min(a, b) / max(a, b) - .72) + abs(ratio - 1) * 3
        candidates.append({"score": float(score), "pixel_policy": [float(x), float(y)],
                           "ellipse": ellipse})
    return min(candidates, key=lambda r: r["score"]) if candidates else None


def validate_placements(record, manifest):
    """Refuse stale, unreviewed, incomplete or mislabeled episode setup."""
    if record.get("review_status") != "visually_reviewed":
        raise ValueError("bowl placements must be visually reviewed")
    if record.get("derived_files_sha256") != manifest["files_sha256"]:
        raise ValueError("bowl placements refer to different derived data")
    rows = record["episodes"]
    expected = {ep: split for split in ("train", "val") for ep in manifest["split"][split]}
    if len(rows) != len(expected) or {r["episode"] for r in rows} != set(expected):
        raise ValueError("need exactly one bowl placement per retained episode")
    for row in rows:
        if row["split"] != expected[row["episode"]]:
            raise ValueError("bowl placement split mismatch")
        pixel = np.asarray(row["pixel_policy"], float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or not (0 <= pixel[0] < 256 and 24 <= pixel[1] < 168):
            raise ValueError("invalid bowl rim pixel")
    return {r["episode"]: r for r in rows}
