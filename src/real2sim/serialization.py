"""JSON conversion and report writing for numerical experiment artifacts."""

import json
from pathlib import Path

import numpy as np


def json_default(value):
    """Convert NumPy values and paths; reject unsupported objects explicitly."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def write_json(path: str | Path, value, *, atomic: bool = False) -> None:
    """Write an indented, newline-terminated report, refusing non-finite numbers."""
    path = Path(path)
    content = json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n"
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    target.write_text(content)
    if atomic:
        target.replace(path)
