"""Panda scene, environment and 7-D adapter tests (plan M5 steps 1-2, gate G5 smoke).

Diagnostic measurements stay in pytest's temporary directory. Publish retained
evidence through an explicit diagnostic run. No policy claim is made.
"""

import json

import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.data.labels import PROPRIO_NAMES
from real2sim.geometry.so3 import log_so3, make_T
from real2sim.panda.adapter import PandaCartesianActionAdapter, PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import (
    DEFAULT_CONFIG,
    POINTING_DOWN,
    ROOT,
    TCP_SITE,
    load_scene,
    sha256_of,
    verify_panda_assets,
)
from real2sim.panda.validate_ik import DEFAULT_OUT, validate

pytestmark = pytest.mark.integration

SIM = load_yaml(DEFAULT_CONFIG)
TASK = load_yaml(ROOT / SIM["task_config"])
DEPTHS = [float(d) for d in TASK["bowl"]["depth_sweep"]]
IMAGE_KEY = TASK["camera_iphone"]["key"]
MEASUREMENTS = ROOT / "results/metrics/panda_ik/scene_measurements.json"


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "MEASUREMENTS", tmp_path / "scene_measurements.json")
    monkeypatch.setitem(globals(), "DEFAULT_OUT", tmp_path / "ik_validation.json")


def record(key: str, value) -> None:
    MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    data[key] = value
    MEASUREMENTS.write_text(json.dumps(data, indent=2))


@pytest.fixture
def env():
    with PandaPickPlaceEnv(render_images=False) as value:
        yield value


def hold(env, ticks=30):
    cmd = np.append(env.initial_qpos, env.initial_opening)
    for _ in range(ticks):
        _, _, _, _, info = env.step(cmd)
    return info


def test_upstream_panda_xml_unchanged():
    row = verify_panda_assets(SIM["manifest"], SIM["model"])
    assert row["sha256"] == SIM["model_sha256"] == sha256_of(SIM["model"])
    assert row["sha256"] == "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273"


def test_scene_loads_headless_with_task_objects_and_tcp_site():
    model, config, task = load_scene()
    assert config["task_config"] == "configs/task.yaml"
    assert model.nkey == 0  # upstream keyframe dropped (nq changed by the free ball)
    assert model.opt.timestep == pytest.approx(1 / config["physics_hz"])
    for name in ("table", "ball_geom", "bowl_base", "mat_visual"):
        assert model.geom(name).id >= 0
    assert model.camera("iphone").id >= 0
    assert model.site(TCP_SITE).id >= 0
    assert model.body("link0").pos[2] == pytest.approx(task["table"]["height"])
    bowl = model.body("bowl")
    np.testing.assert_allclose(bowl.pos[:2], task["bowl"]["center_task_xy"])
    assert model.geom("ball_geom").size[0] == pytest.approx(task["ball"]["radius"])
    walls = [g for g in range(model.ngeom) if model.geom(g).name.startswith("bowl_wall_")]
    assert len(walls) == task["bowl"]["collision"]["wall_boxes"]


def test_tcp_site_offset_and_axis_convention(env):
    env.reset(seed=0)
    hand = env.model.body("hand").id
    hand_z = env.data.xmat[hand].reshape(3, 3)[:, 2]
    tcp = env.data.site_xpos[env.site_id] - env.data.xpos[hand]
    np.testing.assert_allclose(tcp, SIM["tcp_offset_m"] * hand_z, atol=1e-9)
    T = env.ee_pose()
    # Start pose points straight down in the shared SO-101 convention: x approach = -z task.
    assert np.linalg.norm(log_so3(POINTING_DOWN @ T[:3, :3].T)) < SIM["ik"]["orientation_tolerance_rad"]
    np.testing.assert_allclose(T[:3, 3], SIM["initial_tcp_task_xyz"], atol=SIM["ik"]["position_tolerance_m"])


def test_reset_is_deterministic_and_seeded(env):
    obs_a, info_a = env.reset(seed=3)
    ball_a = np.array(info_a["privileged_metrics"]["ball_xyz_task"])
    hold(env, 5)
    obs_b, info_b = env.reset(seed=3)
    np.testing.assert_array_equal(obs_a["observation.state"], obs_b["observation.state"])
    np.testing.assert_array_equal(ball_a, info_b["privileged_metrics"]["ball_xyz_task"])
    _, info_c = env.reset(seed=4)
    ball_c = np.array(info_c["privileged_metrics"]["ball_xyz_task"])
    assert not np.allclose(ball_a, ball_c)
    region = TASK["ball"]["start_region"]
    for ball in (ball_a, ball_c):
        assert np.linalg.norm(ball[:2] - region["center_task_xy"]) <= region["radius"] + 1e-9


def test_observation_schema_and_no_privileged_keys():
    with PandaPickPlaceEnv(render_images=True) as env:
        obs, info = env.reset(seed=0)
        assert set(obs) == {IMAGE_KEY, "observation.state", SIM["observation"]["joint_key"]}
        img = obs[IMAGE_KEY]
        assert img.shape == (3, 192, 256) and img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0 and img.std() > 0.01
        state = obs["observation.state"]
        assert state.shape == (len(PROPRIO_NAMES),) and state.dtype == np.float32
        np.testing.assert_array_equal(state[9:12], 0.0)  # zero velocity at the first tick
        assert state[12] == pytest.approx(1.0)  # fully open start
        for key in obs:
            assert not any(word in key for word in ("ball", "bowl", "privileged"))
        assert "ball_xyz" in info["privileged_metrics"]
        obs2, *_ = env.step(np.append(env.initial_qpos, 0.0))
        assert obs2["observation.state"][12] < 1.0  # gripper closing is observable
        obs3 = env.observation()
        np.testing.assert_array_equal(obs2["observation.state"], obs3["observation.state"])


def test_ball_rests_on_table(env):
    _, info0 = env.reset(seed=0)
    z0 = info0["privileged_metrics"]["ball_xyz_task"][2]
    info = hold(env, 30)
    m = info["privileged_metrics"]
    assert m["ball_touching_table"] and m["ball_resting"] and not m["lifted"]
    sink = z0 - 0.001 - m["ball_xyz_task"][2]
    assert abs(sink) < 0.002
    record("table_rest_sink_m", float(sink))


@pytest.mark.parametrize("depth", DEPTHS[:2])
def test_ball_settles_in_bowl_for_depth_sweep(depth):
    with PandaPickPlaceEnv(render_images=False, bowl_inner_depth=depth) as env:
        bowl_xy = TASK["bowl"]["center_task_xy"]
        env.reset(seed=0, scenario={"ball_xy": bowl_xy, "ball_drop_height": 0.05})
        info = hold(env, 40)
        m = info["privileged_metrics"]
        record(f"bowl_depth_{depth:.4f}_contains_ball", {
            "ball_in_bowl": m["ball_in_bowl"], "ball_resting": m["ball_resting"],
            "ball_xyz_task": m["ball_xyz_task"]})
        if depth == DEPTHS[1]:
            assert m["ball_in_bowl"] and m["ball_resting"]
        else:
            # Supplied shallow depth: reported, not asserted (may not contain a 33.5 mm ball).
            assert m["ball_resting"]


def test_ik_validation_accepts_at_least_ten_poses():
    report = validate(n=12, seed=0)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(report, indent=2))
    s = report["summary"]
    assert s["n_accepted"] >= 10
    assert s["ik_residual_position_m"]["max"] < SIM["ik"]["position_tolerance_m"]
    assert s["ik_residual_orientation_rad"]["max"] < SIM["ik"]["orientation_tolerance_rad"]
    assert s["executed_position_error_m"]["max"] < 0.01


def test_ik_rejects_outside_workspace_and_unreachable(env):
    ik = PandaIK(env.model, env.config, task_origin=env.offset)
    far = make_T(POINTING_DOWN, [1.5, 0.0, 0.1])
    assert ik.solve(far, env.initial_qpos).reason == "outside_workspace"
    unreachable = make_T(POINTING_DOWN, [0.69, 0.44, 0.59])  # inside the box, beyond reach
    result = ik.solve(unreachable, env.initial_qpos)
    assert not result.accepted and result.reason in ("residual_rejected", "joint_limit")
    np.testing.assert_array_equal(result.qpos, env.initial_qpos)  # seed returned, never applied


def test_adapter_zero_delta_holds_tcp(env):
    obs, _ = env.reset(seed=0)
    adapter = PandaCartesianActionAdapter(env)
    start = env.ee_pose()[:3, 3].copy()
    for _ in range(5):
        result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.0]), obs)
        assert not result.rejected
        obs = result.observation
    drift = float(np.linalg.norm(env.ee_pose()[:3, 3] - start))
    record("adapter_zero_delta_drift_m_5_ticks", drift)
    assert drift < 0.002
    assert adapter.stats["ticks"] == 5 and adapter.stats["rejected"] == 0


def test_gravity_compensation_removes_servo_sag_drift():
    """Without compensation the upstream servos sag and the relative contract integrates it."""
    out = {}
    for comp in (False, True):
        with PandaPickPlaceEnv(render_images=False) as env:
            env.gravity_compensation = comp
            env.reset(seed=0)
            start = env.ee_pose()[:3, 3].copy()
            sag = float(np.linalg.norm(hold(env, 30)["privileged_metrics"]["tool_xyz_task"] - start))
            obs, _ = env.reset(seed=0)
            adapter = PandaCartesianActionAdapter(env)
            for _ in range(5):
                obs = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.0]), obs).observation
            drift = float(np.linalg.norm(env.ee_pose()[:3, 3] - start))
            out[f"gravity_compensation_{comp}"] = {"hold_sag_m_30_ticks": sag,
                                                   "zero_delta_drift_m_5_ticks": drift}
    record("gravity_compensation_ablation", out)
    assert SIM["control"]["gravity_compensation"] is True
    assert out["gravity_compensation_False"]["hold_sag_m_30_ticks"] > 0.001  # real servo sag
    assert out["gravity_compensation_True"]["hold_sag_m_30_ticks"] < 1e-4
    assert out["gravity_compensation_True"]["zero_delta_drift_m_5_ticks"] < 1e-4


def test_adapter_clamps_and_moves_in_task_frame(env):
    obs, _ = env.reset(seed=0)
    adapter = PandaCartesianActionAdapter(env)
    start = env.ee_pose()[:3, 3].copy()
    result = adapter.apply(np.array([0.2, 0, 0, 0, 0, 0, 1.0]), obs)  # clamped to 0.03 m
    assert result.translation_clamped and not result.rejected
    # The observed pose is rebuilt from the float32 state, hence the micrometre tolerance.
    np.testing.assert_allclose(result.target_pose[:3, 3] - start, [0.03, 0, 0], atol=1e-6)
    one_tick = env.ee_pose()[:3, 3] - start
    record("adapter_30mm_step_reached_after_one_tick_m", float(one_tick[0]))
    assert 0.0 < one_tick[0] <= 0.03 + 1e-6 and abs(one_tick[1]) < 0.003 and abs(one_tick[2]) < 0.003
    # A zero delta re-targets the *observed* pose (contract), so finish the motion by holding
    # the accepted joint command and check the TCP reaches the 30 mm target along task x.
    for _ in range(5):
        obs, *_ = env.step(result.joint_cmd)
    moved = env.ee_pose()[:3, 3] - start
    record("adapter_30mm_step_reached_after_hold_m", moved.tolist())
    assert abs(moved[0] - 0.03) < 0.003 and abs(moved[1]) < 0.003 and abs(moved[2]) < 0.003
    result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.0]), obs)
    g_before = result.observation["observation.state"][12]
    for _ in range(10):
        result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 0.0]), result.observation)
    assert result.observation["observation.state"][12] < g_before - 0.5


def test_adapter_rejected_tick_holds_observed_arm_pose(env):
    """A rejected IK target is a hold of the observed joints, not a replay of the old target."""
    obs, _ = env.reset(seed=0)
    adapter = PandaCartesianActionAdapter(env)
    first = adapter.apply(np.array([0.03, 0, 0, 0, 0, 0, 1.0]), obs)  # accepted 30 mm step
    assert not first.rejected
    # The rate-limited servos reach only part of the step within one tick; the old target is
    # still "live" in data.ctrl, which is exactly what a rejected tick must not re-send.
    tcp_before = env.ee_pose()[:3, 3].copy()
    q_before = env.arm_qpos()
    lo, hi = adapter.ik.workspace
    adapter.ik.workspace = (lo, np.array([tcp_before[0] + 0.005, hi[1], hi[2]]))  # +30 mm is out
    rejected = adapter.apply(np.array([0.03, 0, 0, 0, 0, 0, 1.0]), first.observation)
    assert rejected.rejected and rejected.reason == "outside_workspace"
    assert rejected.info["adapter"]["rejected"] and adapter.stats["reasons"] == {
        "accepted": 1, "outside_workspace": 1}
    np.testing.assert_allclose(rejected.joint_cmd[:7], q_before, atol=1e-12)  # the hold command
    np.testing.assert_allclose(env.data.ctrl[env.actuator_ids], q_before, atol=1e-9)
    moved = float(np.linalg.norm(env.ee_pose()[:3, 3] - tcp_before))
    # Residual motion in the rejected tick is servo momentum plus env.step's in-tick ramp from
    # the previous ctrl to the hold (measured ~5 mm); re-sending the old target gave 12.5 mm.
    assert moved < 0.008
    # The old target is no longer pursued: further rejected ticks settle instead of completing
    # the 30 mm motion.
    per_tick = []
    for _ in range(5):
        p = env.ee_pose()[:3, 3].copy()
        rejected = adapter.apply(np.array([0.03, 0, 0, 0, 0, 0, 1.0]), rejected.observation)
        assert rejected.rejected
        per_tick.append(float(np.linalg.norm(env.ee_pose()[:3, 3] - p)))
    total = float(np.linalg.norm(env.ee_pose()[:3, 3] - tcp_before))
    record("adapter_rejected_tick_tcp_motion_m", {"rejected_tick": moved,
                                                   "subsequent_hold_ticks": per_tick,
                                                   "after_6_rejected_ticks": total})
    assert max(per_tick) < 0.002 and total < 0.008


def test_adapter_rotation_delta_follows_left_multiplied_contract(env):
    """dr is applied as exp(dr) @ R_obs (task-frame rotation), as in geometry.so3.apply_delta."""
    from real2sim.geometry.so3 import exp_so3

    obs, _ = env.reset(seed=0)
    adapter = PandaCartesianActionAdapter(env)
    R_obs = adapter.pose_from_state(obs["observation.state"])[:3, :3]
    dr = np.array([0.0, 0.0, 0.25])  # yaw about task z (not the tool axis: TCP points down)
    result = adapter.apply(np.array([0, 0, 0, *dr, 1.0]), obs)
    assert not result.rejected and not result.rotation_clamped
    expected = exp_so3(dr) @ R_obs
    wrong_order = R_obs @ exp_so3(dr)
    np.testing.assert_allclose(result.target_pose[:3, :3], expected, atol=1e-6)
    assert np.linalg.norm(log_so3(expected @ wrong_order.T)) > 0.2  # the orders are distinct
    for _ in range(8):  # finish the rate-limited motion on the accepted joint command
        env.step(result.joint_cmd)
    R = env.ee_pose()[:3, :3]
    err_expected = float(np.linalg.norm(log_so3(expected @ R.T)))
    err_wrong = float(np.linalg.norm(log_so3(wrong_order @ R.T)))
    record("adapter_yaw_0p25rad_reached_error_rad", {"contract": err_expected, "wrong_order": err_wrong})
    assert err_expected < SIM["ik"]["orientation_tolerance_rad"] + 0.02
    assert err_wrong > 0.15
    # Clamping: a 1 rad request is scaled onto the 0.3 rad ball, direction kept.
    clamped = adapter.apply(np.array([0, 0, 0, 0, 0, 1.0, 1.0]), result.observation)
    assert clamped.rotation_clamped
    R_now = adapter.pose_from_state(result.observation["observation.state"])[:3, :3]
    np.testing.assert_allclose(log_so3(clamped.target_pose[:3, :3] @ R_now.T),
                               [0, 0, SIM["action_limits"]["max_rotation_rad"]], atol=1e-5)


def test_ik_seed_a_few_ulp_past_a_joint_limit_is_clipped_not_fatal(env):
    ik = PandaIK(env.model, env.config, task_origin=env.offset)
    seed = env.initial_qpos.copy()
    seed[3] = ik.limits[3, 0] - 2.2e-15  # what arm_qpos() returns after holding at the limit
    T = ik.forward(env.initial_qpos)
    result = ik.solve(T, seed)  # must not raise
    assert result.qpos[3] >= ik.limits[3, 0]
    with pytest.raises(ValueError):
        ik.solve(T, np.where(np.arange(7) == 3, ik.limits[3, 0] - 0.01, seed))
    # End to end: drive joint4 onto its lower limit, as the IK does for near-base targets, then
    # tick the adapter with a zero delta; it must count (or accept) rather than raise.
    env.reset(seed=0)
    cmd = np.append(env.initial_qpos, env.initial_opening)
    cmd[3] = env.limits[3, 0]
    for _ in range(33):
        env.step(cmd)
    q3 = env.arm_qpos()[3]
    record("joint4_qpos_minus_lower_limit_after_hold_rad", float(q3 - env.limits[3, 0]))
    adapter = PandaCartesianActionAdapter(env)
    result = adapter.apply(np.array([0, 0, 0, 0, 0, 0, 1.0]), env.observation())
    assert result.reason in ("accepted", "joint_limit", "residual_rejected")
    assert adapter.stats["ticks"] == 1


def test_ik_penetration_check_sees_the_commanded_finger_opening(env):
    """Open fingers (0.08 m) inside the 0.05 m-radius bowl hit the wall; closed ones do not."""
    env.reset(seed=0)
    ik = PandaIK(env.model, env.config, task_origin=env.offset)
    bowl_xy = TASK["bowl"]["center_task_xy"]
    inside = make_T(POINTING_DOWN, [*bowl_xy, 0.02])  # TCP below the 0.03 m rim
    closed = ik.solve(inside, env.initial_qpos, finger_opening=0.0)
    opened = ik.solve(inside, env.initial_qpos, finger_opening=env.max_opening)
    assert closed.accepted
    assert not opened.accepted and opened.reason == "collision"
    np.testing.assert_array_equal(opened.qpos, env.initial_qpos)
    # Through the adapter: start 30 mm above the bowl with open fingers and descend.
    above = ik.solve(make_T(POINTING_DOWN, [*bowl_xy, 0.05]), env.initial_qpos,
                     finger_opening=env.max_opening)
    assert above.accepted
    adapter = PandaCartesianActionAdapter(env)
    obs, _ = env.reset(seed=0, scenario={"joint_positions": above.qpos,
                                         "gripper_opening": env.max_opening})
    result = adapter.apply(np.array([0, 0, -0.03, 0, 0, 0, 1.0]), obs)
    assert result.rejected and result.reason == "collision"
    assert adapter.stats["reasons"] == {"collision": 1}
    # Observed open fingers are still checked when the command closes them within the tick.
    result = adapter.apply(np.array([0, 0, -0.03, 0, 0, 0, 0.0]), result.observation)
    assert result.rejected and result.reason == "collision"
    obs, _ = env.reset(seed=0, scenario={"joint_positions": above.qpos, "gripper_opening": 0.0})
    adapter.reset_stats()
    result = adapter.apply(np.array([0, 0, -0.03, 0, 0, 0, 0.0]), obs)
    assert not result.rejected
