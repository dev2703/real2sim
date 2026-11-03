"""Embodiment retargeting preserves explicit task points and gripper meaning."""

import json

import numpy as np
import pytest

from scripts.diag_v2_learned_panda_retarget_20261027 import (
    PANDA_OFFSET,
    VirtualSource,
    panda_opening_from_percent,
    target_from_source,
    validate_learned_trace,
)

pytestmark = pytest.mark.integration


def test_task_point_mapping_compensates_distinct_tool_origins():
    source = VirtualSource("configs/v2/sim_so101.waypoint_20261026.yaml")
    try:
        q = source.source.initial_qpos
        target = target_from_source(source, q)
        np.testing.assert_allclose(target[:3, 3] + target[:3, :3] @ PANDA_OFFSET,
                                   source.ball_point(q), atol=1e-12)
        assert np.linalg.det(target[:3, :3]) == pytest.approx(1.)
        assert np.all(np.diff(source.gaps) > 0)
    finally:
        source.close()


def test_gripper_intent_maps_endpoints_instead_of_delayed_metric_clipping():
    assert panda_opening_from_percent(0., .08) == 0.
    assert panda_opening_from_percent(90., .08) == .08
    assert panda_opening_from_percent(45., .08) == .04
    assert panda_opening_from_percent(100., .08) == .08
    with pytest.raises(ValueError, match="finite"):
        panda_opening_from_percent(np.nan, .08)


def test_trajectory_requires_matching_checkpoint_and_no_teacher_actions(tmp_path):
    trace = tmp_path / "seed_300" / "ticks.jsonl"
    trace.parent.mkdir()
    trace.write_text(json.dumps({"control": {"applied_radians": [0.] * 6}}) + "\n")
    report = {"checkpoint_sha256": "weights", "teacher_or_ik_corrections": False,
        "trials": [{"trace": str(trace), "ticks": 1, "seed": 300,
            "teacher_goals_or_ik_corrections": False, "pick_place_success": True}]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    rows, provenance = validate_learned_trace(trace, "weights")
    assert len(rows) == 1 and provenance["source_seed"] == 300
    with pytest.raises(ValueError, match="same frozen ACT"):
        validate_learned_trace(trace, "different_weights")
    report["teacher_or_ik_corrections"] = True
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="without teacher"):
        validate_learned_trace(trace, "weights")
