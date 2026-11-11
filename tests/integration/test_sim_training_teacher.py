"""Simulator-only demonstrations complete both arms without rendering or RGB detection."""

import math
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation.metrics import TrialTracker
from real2sim.sim.training_teacher import SimStateTrainingTeacher
from scripts.benchmark_frozen_act import SharedEpisode

pytestmark = pytest.mark.integration
SCENE = Path("configs/v2/weight_space_20261111/sim_so101.yaml")


@pytest.fixture(params=["so101", "panda"])
def episode(request, tmp_path, monkeypatch):
    shared = SharedEpisode(request.param, tmp_path / request.param, source_scene=SCENE)

    def forbid_render(*args, **kwargs):
        raise AssertionError("Simulator-state training teacher must never request an RGB render")

    monkeypatch.setattr(shared.env, "render", forbid_render)
    try:
        yield shared
    finally:
        shared.close()


def make_tracker(env):
    return TrialTracker(success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
                        approach_distance_m=.03, table_top_z=env.table_top,
                        ball_radius_m=env.task["ball"]["radius"])


def record_tick(tracker, result):
    assert result.detail["controller_source"] == "SIM_STATE_TRAINING_TEACHER"
    assert result.detail["object_source"] == "privileged_simulator_state_for_training_only"
    return tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                          ik_rejected=result.ik_rejected)


def test_nominal_training_teacher_completes_held_lift_and_released_placement_without_rgb(episode):
    env = episode.env
    assert env.control_hz == 25.
    obs, _ = episode.reset(20000)
    teacher = SimStateTrainingTeacher(env, robot=episode.robot)
    tracker = make_tracker(env)
    unchanged = {name: np.asarray(getattr(env.model, name)).copy() for name in (
        "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "geom_friction")}
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    assert teacher._acquire()
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    for _ in range(env.max_episode_steps):
        result = teacher.step(obs, env)
        obs = result.obs
        if record_tick(tracker, result) or result.truncated:
            break
    assert tracker.pick_place_success, (teacher.failure, teacher.phase, tracker.phase_failed())
    assert tracker.lifted and tracker.released and tracker.delivered
    assert teacher.perception_log
    assert all(row["source"] == "SIM_STATE_TRAINING_TEACHER" for row in teacher.perception_log)
    assert teacher.plan_log
    assert all(plan["uses_object_ground_truth"] is True for plan in teacher.plan_log)
    for name, array in unchanged.items():
        np.testing.assert_array_equal(getattr(env.model, name), array)
    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()


def test_corrective_teacher_preserves_held_object_instead_of_reopening_at_handoff(episode):
    env = episode.env
    obs, _ = episode.reset(20000)
    original = SimStateTrainingTeacher(env, robot=episode.robot)
    tracker = make_tracker(env)
    for _ in range(env.max_episode_steps):
        result = original.step(obs, env)
        obs = result.obs
        record_tick(tracker, result)
        metrics = result.info["privileged_metrics"]
        if metrics["ball_held"] and metrics["lifted"]:
            break
        assert not result.truncated
    assert metrics["ball_held"] and metrics["lifted"]
    replacement = SimStateTrainingTeacher(env, robot=episode.robot)
    result = replacement.step(obs, env)
    assert result.detail["phase"] == "TRANSPORT"
    assert result.detail["requested_native"][-1] == pytest.approx(replacement._grip(False))
    record_tick(tracker, result)
    obs = result.obs
    for _ in range(env.max_episode_steps - tracker.ticks):
        result = replacement.step(obs, env)
        obs = result.obs
        if record_tick(tracker, result) or result.truncated:
            break
    assert tracker.pick_place_success, (replacement.failure, replacement.phase, tracker.phase_failed())
