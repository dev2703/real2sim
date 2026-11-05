"""A learned step can shrink for feasibility without inventing an object goal."""

from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.policy.common_recovery import CommonCartesianExecutor, soft_stop_seed


def executor(accept_distance):
    commands, targets = [], []

    def step(command):
        commands.append(command.copy())
        return {}, 0., False, False, {"control": {"limit_clipped": [False] * 8}}

    def solve(target, seed, **kwargs):
        targets.append(target.copy())
        accepted = np.linalg.norm(target[:3, 3]) <= accept_distance
        return SimpleNamespace(accepted=accepted, reason="accepted" if accepted else "collision",
                               qpos=seed.copy())

    result = object.__new__(CommonCartesianExecutor)
    result.robot = "panda"
    result.env = SimpleNamespace(ee_pose=lambda: np.eye(4), arm_qpos=lambda: np.zeros(7),
        max_opening=.08, gripper_opening=lambda: .08, step=step,
        limits=np.tile([-1., 1.], (7, 1)))
    result.guard = SimpleNamespace(solve=solve, forward=lambda *_args, **_kwargs: np.eye(4))
    return result, commands, targets


def test_backtracking_rechecks_each_smaller_step_and_keeps_its_direction():
    controller, commands, targets = executor(.005)
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 1.]))
    assert [target[0, 3] for target in targets] == [.03, .015, .0075, .00375]
    assert result.detail["feasibility_fraction"] == .125
    assert not result.ik_rejected
    assert result.detail["common_action"][0] == .00375
    assert result.detail["requested_common_action"][0] == .03
    assert commands[0][-1] == .08


def test_failed_candidates_hold_the_arm_and_remain_rejections():
    controller, commands, targets = executor(0.)
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 0.]))
    assert len(targets) == 5 and result.ik_rejected
    assert result.detail["feasibility_fraction"] == 0.
    np.testing.assert_array_equal(commands[0], np.zeros(8))


@pytest.mark.parametrize("accepted", [True, False])
def test_feedback_distinguishes_requested_issued_applied_and_measured_motion(accepted):
    from real2sim.panda.model import POINTING_DOWN

    controller, _, _ = executor(.04 if accepted else 0.)
    measured = np.eye(4)
    measured[:3, :3] = POINTING_DOWN
    controller.env.ee_pose = lambda: measured.copy()

    def forward(q, **kwargs):
        pose = np.eye(4)
        pose[:3, :3] = POINTING_DOWN
        pose[0, 3] = q[0]
        return pose

    def solve(target, seed, **kwargs):
        q = np.zeros(7)
        q[0] = target[0, 3]
        return SimpleNamespace(accepted=accepted, reason="accepted" if accepted else "collision", qpos=q)

    def step(command):
        # Servo rate limiting and measured motion need not equal either the
        # policy request or the native IK endpoint. Rejecting an arm request
        # also cannot erase actual motion caused by tracking/contact dynamics.
        measured[0, 3] = .001
        applied = command.copy()
        applied[0] = .004 if accepted else 0.
        return {}, 0., False, False, {"control": {
            "limit_clipped": [False] * 8, "applied": applied.tolist()}}

    controller.guard = SimpleNamespace(forward=forward, solve=solve)
    controller.env.step = step
    detail = controller.step(np.array([.03, 0., 0., .2, 0., 0., .4])).detail
    assert detail["requested_common_action"][3] == .2
    assert detail["issued_common_action"][0] == (.03 if accepted else 0.)
    assert detail["applied_common_action"][0] == (.004 if accepted else 0.)
    assert detail["achieved_common_action"][0] == .001
    assert detail["issued_common_action"][3:6] == [0., 0., 0.]
    assert detail["achieved_common_action"][3:6] == [0., 0., 0.]
    assert detail["achieved_common_action"][6] == .4


def test_soft_stop_projection_only_accepts_small_measured_seed_error():
    measured = np.array([1.0001, -.2])
    limits = np.array([[-1., 1.], [-1., 1.]])
    np.testing.assert_array_equal(soft_stop_seed(measured, limits), [1., -.2])
    assert measured[0] == 1.0001
    with pytest.raises(ValueError, match="soft-stop"):
        soft_stop_seed(np.array([1.002, 0.]), limits)


def source_constrained_executor(representable):
    controller, commands, _ = executor(.04)

    def forward(q, **_kwargs):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        return pose

    def solve(target, _seed, **_kwargs):
        q = np.zeros(7)
        q[0] = target[0, 3]
        return SimpleNamespace(accepted=True, reason="accepted", qpos=q)

    def project_pose(pose):
        error = 0. if representable(pose[0, 3]) else .01
        return SimpleNamespace(error_m=error, tolerance_m=.003, accepted=error <= .003,
            diagnostics=lambda: {"accepted": error <= .003, "position_error_m": error,
                                 "position_tolerance_m": .003})

    controller.guard = SimpleNamespace(solve=solve, forward=forward)
    controller.source = SimpleNamespace(project_pose=project_pose)
    controller.require_source_mapping = True
    return controller, commands


def test_act_input_requirement_backtracks_native_feasible_unrepresentable_pose():
    controller, commands = source_constrained_executor(lambda x: x <= .005)
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 1.]))
    assert result.detail["feasibility_fraction"] == .125
    assert result.detail["requires_virtual_source_mapping"]
    assert commands[0][0] == .00375
    attempts = result.detail["feasibility_attempts"]
    assert [attempt["reason"] for attempt in attempts] == [
        "virtual_source_unrepresentable"] * 3 + ["accepted"]
    assert attempts[0]["source_mapping"]["position_tolerance_m"] == .003
    assert attempts[-1]["source_mapping"]["path_samples"] == 5


def test_source_constraint_checks_interior_of_native_joint_path():
    controller, commands = source_constrained_executor(lambda x: not .002 < x < .008)
    # The full-step endpoint is representable but the native path crosses a
    # source-unrepresentable interval; endpoint-only checks would accept it.
    result = controller.step(np.array([.01, 0., 0., 0., 0., 0., 1.]))
    assert result.detail["feasibility_fraction"] == .125
    assert commands[0][0] == .00125
    assert result.detail["feasibility_attempts"][0]["reason"] == "virtual_source_unrepresentable"


def test_no_act_executor_keeps_native_feasible_workspace():
    controller, commands = source_constrained_executor(lambda x: x <= .005)
    controller.require_source_mapping = False
    result = controller.step(np.array([.03, 0., 0., 0., 0., 0., 1.]))
    assert result.detail["feasibility_fraction"] == 1.
    assert not result.detail["requires_virtual_source_mapping"]
    assert commands[0][0] == .03


def so101_rejected_arm_executor(opening, *, unsafe=lambda _q: False):
    """Run the actual native path checker with a jaw-dependent collision map."""
    current = np.r_[np.zeros(5), opening]
    commands, probes = [], []

    def step(command):
        commands.append(command.copy())
        applied = current + np.clip(command - current, -.2, .2)
        return {}, 0., False, False, {"control": {
            "limit_clipped": [False] * 6, "applied_radians": applied.tolist()}}

    guard = SimpleNamespace(limits=np.array([[-1., 1.]] * 5 + [[0., 1.]]),
        data=SimpleNamespace(qpos=current.copy()),
        config={"workspace_min": [-2.] * 3, "workspace_max": [2.] * 3})

    def forward(q):
        guard.current = np.array(q)
        probes.append(guard.current.copy())
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    guard.forward = forward
    guard._collision = lambda: unsafe(guard.current)
    guard.solve = lambda _target, seed, **_kwargs: SimpleNamespace(
        accepted=False, reason="residual_rejected", qpos=seed.copy())
    controller = object.__new__(CommonCartesianExecutor)
    controller.robot, controller.guard = "so101", guard
    controller.env = SimpleNamespace(ee_pose=lambda: np.eye(4), sim_state=lambda: current.copy(),
        mapping=SimpleNamespace(gripper_real_to_sim_rad=lambda value: value / 90.,
                                gripper_sim_to_real_percent=lambda value: value * 90.),
        limits=guard.limits, data=SimpleNamespace(qpos=current.copy(), ctrl=current.copy()),
        actuator_ids=np.arange(6), max_velocity=np.full(6, 5.), control_hz=25., step=step)
    return controller, commands, probes


@pytest.mark.parametrize("current_opening,requested_opening", [(.8, 0.), (.1, 1.)])
def test_rejected_so101_arm_can_issue_independently_guarded_close_or_open(
        current_opening, requested_opening):
    controller, commands, probes = so101_rejected_arm_executor(current_opening)
    result = controller.step(np.r_[.03, np.zeros(5), requested_opening])
    np.testing.assert_array_equal(commands[0][:5], np.zeros(5))
    assert commands[0][5] == requested_opening
    assert result.ik_rejected and result.detail["feasibility_fraction"] == 0.
    assert result.detail["ik_reason"] == "residual_rejected"
    assert len(result.detail["feasibility_attempts"]) == 5
    assert result.detail["jaw_only_fallback"]["accepted"] is True
    assert result.detail["executor_semantics"] == "native_common_cartesian_v2_guarded_so101_jaw_fallback"
    assert result.detail["common_action"] == [0.] * 6 + [requested_opening]
    assert result.detail["issued_common_action"][6] == requested_opening
    # Check the separately rate-limited jaw command, rather than pretending the
    # requested endpoint was immediately achieved by the actuator.
    expected_applied = current_opening + np.clip(requested_opening - current_opening, -.2, .2)
    assert result.detail["applied_common_action"][6] == pytest.approx(expected_applied)
    assert any(q[5] == pytest.approx(requested_opening) for q in probes)
    assert len(probes) >= 35  # endpoint and both 17-sample native path checks


@pytest.mark.parametrize("unsafe,expected_reason", [
    (lambda q: .64 < q[5] < .66, "path_collision"),
    (lambda q: q[5] < .01, "target_collision"),
    (lambda q: q[5] > .79, "actual_pose_workspace_or_collision"),
])
def test_rejected_so101_arm_does_not_bypass_unsafe_jaw_endpoint_path_or_actual_pose(
        unsafe, expected_reason):
    controller, commands, _ = so101_rejected_arm_executor(.8, unsafe=unsafe)
    result = controller.step(np.r_[.03, np.zeros(6)])
    np.testing.assert_array_equal(commands[0], np.r_[np.zeros(5), .8])
    assert result.ik_rejected
    assert result.detail["jaw_only_fallback"]["accepted"] is False
    assert result.detail["jaw_only_fallback"]["reason"] == expected_reason
    assert result.detail["requested_common_action"][6] == 0.
    assert result.detail["common_action"][6] == pytest.approx(.8)
    assert result.detail["issued_common_action"][6] == pytest.approx(.8)
    assert result.detail["achieved_common_action"][6] == pytest.approx(.8)


def test_so101_jaw_fallback_checks_previous_control_path_when_arm_lags():
    controller, commands, probes = so101_rejected_arm_executor(
        .8, unsafe=lambda q: .45 < q[0] < .48)
    # The requested held arm equals measured zero, but servo rate limiting
    # starts from the previous target .6 and would next apply .4. Only the
    # previous-control path crosses this collision; the measured-to-applied
    # path and both endpoint poses are clear.
    controller.env.data.ctrl[0] = .6
    result = controller.step(np.r_[.03, np.zeros(6)])
    assert result.detail["jaw_only_fallback"]["reason"] == "path_collision"
    assert result.detail["jaw_only_fallback"]["accepted"] is False
    assert result.ik_rejected
    np.testing.assert_array_equal(commands[0], np.r_[np.zeros(5), .8])
    assert any(.45 < q[0] < .48 for q in probes)
