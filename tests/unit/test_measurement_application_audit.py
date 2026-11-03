"""Available scalar measurements should never be reported as an absent file."""

from pathlib import Path

import pytest

from scripts.audit_v2_measurement_application import audit, compare


def test_current_measured_scenes_apply_all_scalars_and_preserve_lighting():
    report = audit(Path("physical_measurements.md"), [
        Path(f"configs/v2/measured_registration_20261028/sim_{robot}_positive_all_scalars.yaml")
        for robot in ("so101", "panda")
    ])
    assert report["measurement_file_present"]
    assert not report["missing_measurement_file"]
    assert not report["simulation_requires_new_capture"]
    for scene in report["scenes"]:
        assert scene["source_hash_matches_task"]
        assert scene["all_reported_scalars_within_bound"]
        assert scene["lighting_config_unchanged"]


def test_audit_detects_a_real_dimensional_mismatch():
    result = compare(.4, .3, .05, interpretation="total distance")
    assert not result["within_reported_bound"]
    assert result["difference_mm"] == pytest.approx([100])
