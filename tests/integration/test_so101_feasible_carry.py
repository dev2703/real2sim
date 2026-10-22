"""Physical carry diagnostic succeeds without the legacy oracle's relaxed IK gates."""

import importlib.util
from pathlib import Path

import pytest

from real2sim.evaluation.oracle import OracleParams
from real2sim.sim.env import So101PickPlaceEnv

SPEC = importlib.util.spec_from_file_location(
    "so101_feasible_carry", Path("scripts/diag_so101_feasible_carry.py"))
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
pytestmark = [pytest.mark.integration]
CONFIG = Path("configs/sim_so101.yaml")


def test_feasible_carry_preserves_native_ik_acceptance():
    with So101PickPlaceEnv(CONFIG, render_images=False) as env:
        ctrl = diagnostic.FeasibleCarry(env)
        for solver in (ctrl.ik_grasp, ctrl.ik_carry):
            for key in ("position_tolerance_m", "orientation_tolerance_rad",
                        "orientation_weight_m_per_rad", "penetration_tolerance_m"):
                assert solver.config[key] == env.config["ik"][key]
        with pytest.raises(ValueError, match="unchanged native IK"):
            diagnostic.FeasibleCarry(env, OracleParams())


def test_joint_space_carry_completes_physical_pick_place_with_strict_gates():
    baseline = diagnostic.run(CONFIG, "matched_pose", [0])[0]
    feasible = diagnostic.run(CONFIG, "feasible_carry", [0])[0]
    assert baseline["held"] and baseline["lifted"]
    assert not baseline["pick_place_success"]
    assert baseline["phase_failed"] == "transport"
    assert feasible["pick_place_success"] and feasible["success"]
    assert feasible["max_table_penetration_m"] < .002
    assert not any(t["ik_stalled"] for t in feasible["phase_log"])
