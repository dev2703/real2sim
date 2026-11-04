"""Shared config/IO helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real2sim.config import json_default, sha256_file


def test_sha256_file_is_stable(tmp_path: Path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"real2sim")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path)
    assert digest == sha256_file(str(path))


def test_json_default_covers_numpy_and_path(tmp_path: Path):
    payload = {
        "i": np.int64(3),
        "f": np.float64(1.5),
        "a": np.arange(3),
        "t": (1, 2),
        "p": tmp_path / "x",
        "b": np.bool_(True),
    }
    encoded = json.loads(json.dumps(payload, default=json_default))
    assert encoded == {
        "i": 3,
        "f": 1.5,
        "a": [0, 1, 2],
        "t": [1, 2],
        "p": str(tmp_path / "x"),
        "b": True,
    }
