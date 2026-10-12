"""Minimal YAML config loading shared by the CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root unless it is already absolute (``~`` expanded)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


def validate_camera_keys(config: dict[str, Any], available_keys: list[str]) -> list[str]:
    """Return the ordered camera keys from config, refusing any key the dataset does not have."""
    keys = list(config.get("camera_keys") or [])
    if not keys:
        raise ValueError("configs/data.yaml must list at least one camera key (audit-selected)")
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"camera keys listed more than once: {dupes}")
    missing = [k for k in keys if k not in available_keys]
    if missing:
        raise ValueError(f"camera keys not present in dataset: {missing}; available: {available_keys}")
    excluded = {e["key"] for e in config.get("excluded_cameras", [])}
    clash = [k for k in keys if k in excluded]
    if clash:
        raise ValueError(f"camera keys both selected and excluded: {clash}")
    return keys
