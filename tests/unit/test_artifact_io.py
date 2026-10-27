"""Integrity and report compatibility checks for the shared artifact helpers."""

import hashlib
import json

import numpy as np
import pytest

from real2sim.artifacts import sha256_file
from real2sim.serialization import json_default, write_json


@pytest.mark.parametrize("size", [0, 31, (1 << 20) + 13])
def test_file_fingerprint_matches_standard_digest(tmp_path, size):
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(path) == sha256_file(str(path), chunk=73) == expected
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")


@pytest.mark.parametrize("atomic", [False, True])
def test_numerical_reports_roundtrip(tmp_path, atomic):
    path = tmp_path / "report.json"
    sibling = tmp_path / "report.tmp"
    sibling.write_text("unrelated sibling")
    payload = {"matrix": np.arange(6).reshape(2, 3), "score": np.float32(.5),
               "count": np.int64(2), "success": np.bool_(True), "path": path}
    write_json(path, payload, atomic=atomic)
    assert json.loads(path.read_text()) == {"matrix": [[0, 1, 2], [3, 4, 5]],
        "score": .5, "count": 2, "success": True, "path": str(path)}
    assert path.read_text().endswith("\n")
    assert sibling.read_text() == "unrelated sibling"
    assert not path.with_suffix(path.suffix + ".tmp").exists()


@pytest.mark.parametrize("invalid", [float("nan"), np.float64("inf"), np.array([np.nan])])
@pytest.mark.parametrize("atomic", [False, True])
def test_invalid_report_preserves_previous_contents(tmp_path, invalid, atomic):
    path = tmp_path / "report.json"
    path.write_text('{"complete": true}\n')
    with pytest.raises(ValueError):
        write_json(path, {"score": invalid}, atomic=atomic)
    assert json.loads(path.read_text()) == {"complete": True}


def test_unknown_report_objects_are_rejected():
    with pytest.raises(TypeError):
        json.dumps({"unsupported": object()}, default=json_default)


def test_script_report_writer_preserves_payload_keyword(tmp_path):
    from scripts.diag_v2_full_policy_rollouts import write_json as write_report

    path = tmp_path / "report.json"
    write_report(path=path, payload={"score": np.int64(3)})
    assert json.loads(path.read_text()) == {"score": 3}
