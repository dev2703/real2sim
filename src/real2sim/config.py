"""Minimal YAML config loading and shared IO helpers used by the CLIs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root unless it is already absolute (``~`` expanded)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex digest of a file; used for contract and provenance checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """``json.dumps`` default for numpy scalars/arrays, tuples and ``Path`` values."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


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
    entries = config.get("excluded_cameras") or []
    if not all(isinstance(e, dict) and "key" in e for e in entries):
        raise ValueError("excluded_cameras entries must be mappings with a 'key' field")
    excluded = {e["key"] for e in entries}
    clash = [k for k in keys if k in excluded]
    if clash:
        raise ValueError(f"camera keys both selected and excluded: {clash}")
    return keys
