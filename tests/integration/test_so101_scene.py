"""Actual MuJoCo dynamics/render tests; no policy or hardware claims.

Numbers measured here (contact sink, containment) are written to
``results/audit/sim_bowl_physics.json`` so docs can cite them.
"""

import copy
import hashlib
import json

import mujoco
import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.model import (
    DEFAULT_CONFIG,
    ROOT,
    base_z_offset,
    camera_xyaxes,
    scene_xml,
    task_frame_offset,
    verify_assets,
)

pytestmark = pytest.mark.integration

SIM = load_yaml(DEFAULT_CONFIG)
TASK = load_yaml(ROOT / SIM["task_config"])
DEPTHS = [float(d) for d in TASK["bowl"]["depth_sweep"]]
FRICTIONS = [float(f) for f in SIM["contacts"]["sweep_sliding_friction"]]
DAMPINGS = [float(d) for d in SIM["contacts"]["sweep_damping_ratio"]]
SPEEDS = [0.03, 0.06, 0.09, 0.12]
# Single source for the resting-ball sink bound (configs/sim_so101.yaml documents the measured
# values); MuJoCo's default solimp transition width is 1 mm and a ball at rest must sit inside it.
STATIC_SINK_TOLERANCE_M = float(SIM["contacts"]["static_sink_tolerance_m"])
assert 0 < STATIC_SINK_TOLERANCE_M <= 0.001
MEASUREMENTS = ROOT / "results/audit/sim_bowl_physics.json"


def table_top_task(env: So101PickPlaceEnv) -> float:
    """Table-top height in task (= base) coordinates, derived from the env rather than assumed 0:
    D-014 places the base ``base_z_offset`` above the table top, so ``z_task = -base_z_offset``."""
    return float(env.table_top - env.offset[2])


@pytest.fixture
def env():
    with So101PickPlaceEnv(render_images=False) as value:
        yield value


def hold(env, ticks=30):
    for _ in range(ticks):
        _, _, _, _, info = env.step(env.initial_qpos)
    return np.array(info["privileged_metrics"]["ball_xyz_task"])


def test_pinned_assets():
    manifest = verify_assets("assets/menagerie/manifest.json")
    assert manifest["revision"] == "0059d4335f8156206f63a35662313385f7ad6d74"
    assert len(manifest["files"]) == 104


def test_scene_uses_task_yaml_single_source(env):
    assert env.config["task_config"] == "configs/task.yaml"
    lift = base_z_offset(TASK)
    np.testing.assert_allclose(env.offset, [0, 0, TASK["table"]["height"] + lift])
    # Robot base sits base_z_offset above the table top (D-014) and is the task-frame origin;
    # the table top and the bowl base stay at task.table.height in world coordinates.
    np.testing.assert_allclose(env.model.body_pos[env.base_body], env.offset)
    assert env.model.geom("table").pos[2] + env.model.geom("table").size[2] == pytest.approx(env.table_top)
    assert env.model.body("bowl").pos[2] == pytest.approx(env.table_top)
    assert table_top_task(env) == pytest.approx(-lift)
    _, info = env.reset(seed=0)
    pm = info["privileged_metrics"]
    np.testing.assert_allclose(pm["bowl_xyz_task"][:2], TASK["bowl"]["center_task_xy"])
    region = TASK["ball"]["start_region"]
    assert np.linalg.norm(np.array(pm["ball_xyz_task"][:2]) - region["center_task_xy"]) <= region["radius"]
    assert env.max_episode_steps == TASK["episode"]["timeout_s"] * SIM["control_hz"]
    assert env.bowl["wall_boxes"] == TASK["bowl"]["collision"]["wall_boxes"]
    assert env.bowl["height"] == pytest.approx(TASK["bowl"]["outer_height"])
    cam = env.model.camera("iphone")
    np.testing.assert_allclose(cam.pos, np.array(TASK["camera_iphone"]["approx_position_task"]) + env.offset)
    assert cam.fovy[0] == pytest.approx(TASK["camera_iphone"]["fovy_deg"])
    # The rendered view must use the fitted axes (with roll), not the roll-free look-at form,
    # otherwise task.yaml's PnP provenance does not describe what the policy sees.
    fitted = np.asarray(TASK["camera_iphone"]["xyaxes_task"], float)
    compiled = cam.mat0.reshape(3, 3)  # columns are the camera x, y, z axes in world
    np.testing.assert_allclose(compiled[:, 0], fitted[:3] / np.linalg.norm(fitted[:3]), atol=2e-3)
    np.testing.assert_allclose(compiled[:, 1], fitted[3:] / np.linalg.norm(fitted[3:]), atol=2e-3)
    look = np.array(TASK["camera_iphone"]["look_at_task"]) - TASK["camera_iphone"]["approx_position_task"]
    assert compiled[:, 2] @ look < 0  # MuJoCo cameras look along -z
    roll_free = camera_xyaxes({k: v for k, v in TASK["camera_iphone"].items() if k != "xyaxes_task"},
                              cam.pos, np.array(TASK["camera_iphone"]["look_at_task"]) + env.offset)
    roll_deg = np.degrees(np.arccos(np.clip(roll_free[:3] @ compiled[:, 0], -1, 1)))
    assert 5 < roll_deg < 20  # the fit's -10.9 deg roll is really applied


def test_camera_axes_validation():
    camera = dict(TASK["camera_iphone"])
    pos = np.array(camera["approx_position_task"])
    at = np.array(camera["look_at_task"])
    for bad in ([1, 0, 0, 0, 1], [2, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]):
        with pytest.raises(ValueError):
            camera_xyaxes({**camera, "xyaxes_task": bad}, pos, at)
    with pytest.raises(ValueError, match="look"):
        camera_xyaxes({**camera, "xyaxes_task": [-v for v in camera["xyaxes_task"][:3]]
                       + list(camera["xyaxes_task"][3:])}, pos, at)
    assert camera_xyaxes({**camera, "xyaxes_task": None}, pos, at).shape == (6,)


def _start_pose_contacts(task: dict) -> tuple[int, float, float]:
    """(robot-table contacts, deepest penetration m, tool height above table m) at initial_qpos."""
    model = mujoco.MjModel.from_xml_string(scene_xml(SIM, task))
    data = mujoco.MjData(model)
    for name, q in zip(SIM["joint_names"], SIM["initial_qpos"], strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
    mujoco.mj_forward(model, data)
    table, ball = model.geom("table").id, model.geom("ball_geom").id
    count, deepest = 0, 0.0
    for contact in data.contact[:data.ncon]:
        geoms = (int(contact.geom1), int(contact.geom2))
        if table in geoms and ball not in geoms:
            count += 1
            deepest = max(deepest, -float(contact.dist))
    tool_z = float(data.site("gripperframe").xpos[2]) - float(task["table"]["height"])
    return count, deepest, tool_z


def test_start_pose_penetration_is_reported_not_hidden(env):
    """D-014 (approved, 1.2 cm base lift): the clamped rest pose still penetrates the table.
    The env must expose this in privileged_metrics so trial statistics carry the confound."""
    _, info = env.reset(seed=0)
    start = info["privileged_metrics"]["start_pose"]
    assert set(start) == {"robot_table_contacts", "robot_table_penetration_m",
                          "tool_height_above_table_m"}
    assert start["robot_table_penetration_m"] == info["privileged_metrics"]["robot_table_penetration_m"]
    lift = base_z_offset(TASK)
    assert lift > 0, "D-014 approved: the base is lifted above the table top"
    # The lift reduces the identity-placement penetration (~2.3 cm, 29 contacts) by exactly the
    # offset: measured 1.14 cm over 21 contacts at 1.2 cm; a lift >= 2.4 cm would be contact-free.
    flat = copy.deepcopy(TASK)
    del flat["robot_so101"]["base_z_offset"]
    _, pen_flat, tool_flat = _start_pose_contacts(flat)
    assert start["robot_table_contacts"] >= 10
    assert start["robot_table_penetration_m"] == pytest.approx(pen_flat - lift, abs=1e-6)
    assert start["robot_table_penetration_m"] > 0.005
    assert start["tool_height_above_table_m"] == pytest.approx(tool_flat + lift, abs=1e-6)
    assert start["tool_height_above_table_m"] < 0.01  # tool site still within 1 cm of the table
    before = env.sim_state().copy()
    _, _, _, _, info = env.step(env.initial_qpos)
    jump_deg = np.rad2deg(env.sim_state() - before)
    assert abs(jump_deg[1]) > 30  # contact-driven shoulder_lift jump while holding the pose
    assert info["privileged_metrics"]["start_pose"] == start  # reset snapshot is kept


def test_base_z_offset_plumbing_moves_base_frame_and_reduces_penetration():
    """task.yaml robot_so101.base_z_offset (D-014 option A, approved at 1.2 cm) moves the base, the
    task frame and the camera together; the table stays at task.table.height. Absent key == 0 ==
    base on the table top. Measured: the 1.2 cm lift leaves 1.14 cm penetration (21 contacts; the
    deepest gripper geom sits 2.34 cm below the table top); a lift >= 2.4 cm is contact-free."""
    top = float(TASK["table"]["height"])
    lift = base_z_offset(TASK)
    assert lift > 0  # D-014 approved
    flat = copy.deepcopy(TASK)
    del flat["robot_so101"]["base_z_offset"]
    assert base_z_offset(flat) == 0.0
    assert task_frame_offset(flat)[2] == pytest.approx(top)
    np.testing.assert_allclose(task_frame_offset(TASK), [0, 0, top + lift])
    model = mujoco.MjModel.from_xml_string(scene_xml(SIM, TASK))
    np.testing.assert_allclose(model.body("base").pos, [0, 0, top + lift])
    np.testing.assert_allclose(model.camera("iphone").pos,
                               np.array(TASK["camera_iphone"]["approx_position_task"]) + [0, 0, top + lift])
    assert model.geom("table").pos[2] + model.geom("table").size[2] == pytest.approx(top)
    assert model.body("bowl").pos[2] == pytest.approx(top)
    n0, pen0, tool0 = _start_pose_contacts(flat)
    n1, pen1, tool1 = _start_pose_contacts(TASK)
    assert pen1 == pytest.approx(pen0 - lift, abs=1e-6) and n1 < n0
    assert pen1 > 0.01  # 1.2 cm alone is not contact-free; documented in D-014
    assert tool1 == pytest.approx(tool0 + lift, abs=1e-9)
    free = copy.deepcopy(TASK)
    free["robot_so101"]["base_z_offset"] = 0.025
    assert _start_pose_contacts(free)[:2] == (0, 0.0)
    for bad in (-0.01, 0.2, float("nan")):
        broken = copy.deepcopy(TASK)
        broken["robot_so101"]["base_z_offset"] = bad
        with pytest.raises(ValueError, match="base_z_offset"):
            task_frame_offset(broken)


def test_initial_pose_matches_dataset_start_pose_within_model_limits(env):
    env.reset(seed=0)
    real = env.real_state()
    target = np.array(SIM["initial_qpos_real_deg"], float)
    clamped = np.abs(real - target) > 1e-3
    # Only shoulder_lift is clamped (-104.7 deg -> -100 deg model limit); the rest are exact.
    assert clamped.tolist() == [False, True, False, False, False, False]
    assert real[1] == pytest.approx(np.rad2deg(env.limits[1, 0]), abs=1e-3)


def test_reset_and_transition_determinism(env):
    action = env.initial_qpos + [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
    results = []
    for _ in range(2):
        obs, info = env.reset(seed=123)
        initial = info["privileged_metrics"]["qpos"]
        for _ in range(5):
            obs, _, _, _, info = env.step(action)
        results.append((initial, info["privileged_metrics"]["qpos"]))
    np.testing.assert_array_equal(results[0], results[1])
    env.reset(seed=124)
    assert not np.array_equal(results[0][0], env.data.qpos)


def test_ball_start_jitter_stays_in_start_disc(env):
    region = TASK["ball"]["start_region"]
    radii = []
    for seed in range(40):
        _, info = env.reset(seed=seed)
        xy = np.array(info["privileged_metrics"]["ball_xyz_task"][:2])
        radii.append(np.linalg.norm(xy - region["center_task_xy"]))
    assert max(radii) <= region["radius"] + 1e-9
    assert min(radii) < region["radius"] / 2  # not pinned to the rim of the disc


def test_reset_restores_scenario_and_time(env):
    env.reset(seed=0, scenario={"bowl_xy": [0.3, 0.1]})
    hold(env, 3)
    _, info = env.reset(seed=0)
    assert info["time_s"] == 0
    np.testing.assert_allclose(info["privileged_metrics"]["bowl_xyz_task"][:2],
                               TASK["bowl"]["center_task_xy"])


def test_observation_schema_and_no_privileged_state():
    with So101PickPlaceEnv() as env:
        obs, info = env.reset(seed=0)
        assert set(obs) == {"observation.images.iphone", "observation.state", "sim.joint_position"}
        state = obs["observation.state"]
        assert state.shape == (6,) and state.dtype == np.float32
        np.testing.assert_allclose(state, env.real_state())
        assert obs["sim.joint_position"].dtype == np.float32
        img = obs["observation.images.iphone"]
        assert img.shape == (3, 192, 256) and img.dtype == np.float32
        assert 0 <= img.min() < img.max() <= 1
        assert img[:, 24:168].std() > 0.1
        np.testing.assert_allclose(img[:, :24], 114 / 255)
        # Same seed produces exactly the same pixels in this offscreen renderer.
        again, _ = env.reset(seed=0)
        np.testing.assert_array_equal(img, again["observation.images.iphone"])
        pm = info["privileged_metrics"]
        for key in ("ball_xyz", "ball_in_bowl", "ball_resting", "ball_held", "lifted",
                    "ee_ball_distance_m", "ee_bowl_distance_m"):
            assert key in pm
        assert not any(k.startswith("ball") or k.startswith("bowl") for k in obs)


def test_real_unit_adapters_round_trip(env):
    env.reset(seed=0)
    real = env.real_state()
    q = env.sim_state()
    np.testing.assert_allclose(np.deg2rad(real[:5]), q[:5], atol=1e-5)
    lo, hi = env.mapping.gripper_sim_range_rad
    assert real[5] == pytest.approx((q[5] - lo) / (hi - lo) * 100, abs=1e-4)
    target = real + [5, 0, -5, 0, 0, 20]
    _, _, _, _, info = env.step_real_joint_targets(target)
    expected = np.r_[np.deg2rad(target[:5]), lo + target[5] / 100 * (hi - lo)]
    np.testing.assert_allclose(info["control"]["requested_radians"], expected, atol=1e-6)
    proprio = env.canonical_proprio(None)
    assert proprio.shape == (13,) and proprio.dtype == np.float32
    T = env.ee_pose()
    np.testing.assert_allclose(proprio[:3], T[:3, 3], atol=1e-6)
    np.testing.assert_allclose(proprio[3:9], np.r_[T[:3, 0], T[:3, 1]], atol=1e-6)
    np.testing.assert_array_equal(proprio[9:12], 0)
    assert proprio[12] == pytest.approx(env.real_state()[5] / 100, abs=1e-5)
    prev = T[:3, 3].copy()
    env.step_real_joint_targets(target)
    after = env.canonical_proprio(prev)
    np.testing.assert_allclose(after[9:12], (after[:3] - prev) * env.control_hz, atol=1e-5)


def test_ball_falls_rests_and_rolls(env):
    rest = table_top_task(env) + TASK["ball"]["radius"]  # ball centre resting on the table top
    env.reset(seed=0, scenario={"ball_xy": [0.1, -0.3], "ball_drop_height": 0.15})
    ball = hold(env)
    assert abs(ball[2] - rest) < STATIC_SINK_TOLERANCE_M
    pm = env.info()["privileged_metrics"]
    assert pm["ball_resting"] and pm["ball_touching_table"] and not pm["lifted"]
    env.reset(seed=0, scenario={"ball_xy": [0.1, -0.3], "ball_velocity": [0.15, 0, 0]})
    ball = hold(env, 10)
    assert ball[0] > 0.14
    assert abs(ball[2] - rest) < STATIC_SINK_TOLERANCE_M


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("friction", FRICTIONS)
@pytest.mark.parametrize("damping", DAMPINGS)
def test_bowl_settles_without_tunnelling(depth, friction, damping):
    contacts = {"friction": [friction, 0.005, 0.0001], "solref": [0.01, damping]}
    with So101PickPlaceEnv(render_images=False, bowl_inner_depth=depth, contacts=contacts) as env:
        center = np.array(env.bowl["center_xy"])
        env.reset(seed=0, scenario={"ball_xy": center, "ball_drop_height": 0.12})
        minimum = float("inf")
        for _ in range(30):
            ball = hold(env, 1)
            minimum = min(minimum, ball[2])
        expected = table_top_task(env) + env.bowl["floor"] + TASK["ball"]["radius"]
        # Impact transient: the softest sweep point (dampratio 1.3) sinks 1.9 mm on a 12 cm
        # drop; tunnelling would mean passing the floor slab, so the bound is half its thickness.
        assert minimum >= expected - env.bowl["floor"] / 2
        assert abs(ball[2] - expected) < STATIC_SINK_TOLERANCE_M
        assert np.linalg.norm(ball[:2] - center) < 0.002
        assert env.info()["privileged_metrics"]["ball_in_bowl"]


def free_play_radius(bowl: dict, radius: float) -> float:
    """Radial travel before a floor-resting ball touches the frustum wall (straight-wall model)."""
    slope = (bowl["rim_radius"] - bowl["base_radius"]) / bowl["depth"]
    return bowl["base_radius"] + slope * radius - radius * np.sqrt(1 + slope * slope)


@pytest.mark.parametrize("depth", DEPTHS)
def test_bowl_wall_contains_lateral_velocity(depth):
    """Containment is physically expected at every sweep depth: the frustum wall (base radius
    0.03 m -> rim 0.05 m) stops a floor-resting ball after ``free_play_radius`` (13.7 mm at
    0.0254 m, 5.9 mm at 0.06 m) and escaping over the rim needs a centre rise of ``inner_depth``
    (> 0.7 m/s at 0.0254 m). Measured speeds up to 0.5 m/s are documented in
    results/audit/sim_bowl_physics.json rather than asserted here."""
    with So101PickPlaceEnv(render_images=False, bowl_inner_depth=depth) as env:
        center = np.array(env.bowl["center_xy"])
        free = free_play_radius(env.bowl, TASK["ball"]["radius"])
        reached = []
        for speed in SPEEDS:
            env.reset(seed=0, scenario={"ball_xy": center, "ball_drop_height": 0.005,
                                       "ball_velocity": [speed, 0, 0]})
            excursion = 0.0
            for _ in range(20):
                ball = hold(env, 1)
                excursion = max(excursion, float(np.linalg.norm(ball[:2] - center)))
            # The wall turns the ball back within a few mm of the geometric contact radius.
            assert excursion < free + 0.004, f"ball climbed the wall at {speed} m/s"
            assert env.info()["privileged_metrics"]["ball_in_bowl"]
            reached.append(excursion > 0.9 * free)
        assert reached[-1], "ball never reached the wall at the fastest probe speed"


def test_record_bowl_physics_measurements():
    """Writes measured numbers (sink, containment) for the docs; asserts only sanity."""
    record = {"sink_mm": {}, "largest_contained_speed_mps": {}, "escape_probe_speeds": []}
    speeds = [0.03, 0.06, 0.09, 0.12, 0.3, 0.5]
    for damping in DAMPINGS:
        with So101PickPlaceEnv(render_images=False, contacts={"solref": [0.01, damping]}) as env:
            center = np.array(env.bowl["center_xy"])
            env.reset(seed=0, scenario={"ball_xy": center, "ball_drop_height": 0.12})
            minimum = float("inf")
            for _ in range(30):
                ball = hold(env, 1)
                minimum = min(minimum, ball[2])
            expected = table_top_task(env) + env.bowl["floor"] + TASK["ball"]["radius"]
            record["sink_mm"][str(damping)] = {"impact_transient": round(1e3 * (expected - minimum), 3),
                                               "static": round(1e3 * (expected - ball[2]), 3)}
    for depth in DEPTHS:
        with So101PickPlaceEnv(render_images=False, bowl_inner_depth=depth) as env:
            center = np.array(env.bowl["center_xy"])
            largest = None
            for speed in speeds:
                env.reset(seed=0, scenario={"ball_xy": center, "ball_drop_height": 0.005,
                                           "ball_velocity": [speed, 0, 0]})
                contained = all(np.linalg.norm(hold(env, 1)[:2] - center) < env.bowl["rim_radius"]
                                for _ in range(30))
                if contained:
                    largest = speed
                else:
                    break
            record["largest_contained_speed_mps"][str(depth)] = largest
    record["escape_probe_speeds"] = speeds
    record["static_sink_tolerance_m"] = STATIC_SINK_TOLERANCE_M
    with So101PickPlaceEnv(render_images=False) as env:
        _, info = env.reset(seed=0)
        start = dict(info["privileged_metrics"]["start_pose"])
        before = env.sim_state().copy()
        env.step(env.initial_qpos)
        start["first_tick_joint_jump_deg"] = np.rad2deg(env.sim_state() - before).round(2).tolist()
        start["base_z_offset_m"] = float(env.offset[2] - env.table_top)
        start["table_top_z_task_m"] = table_top_task(env)
        start["note"] = ("D-014 approved: clamped dataset rest pose with the base lifted "
                         "base_z_offset_m above the table top (sink measured from the table top)")
    record["start_pose"] = start
    record["note"] = ("frustum bowl from task.yaml (base_diameter PROVISIONAL-STANDARD); "
                      "sink measured on a 0.12 m drop with solref timeconst 0.01")
    MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(json.dumps(record, indent=2) + "\n")
    assert all(v["static"] < 1e3 * STATIC_SINK_TOLERANCE_M for v in record["sink_mm"].values())
    assert all(v is not None for v in record["largest_contained_speed_mps"].values())


def test_action_limits_rate_limit_and_invalid_inputs(env):
    env.reset(seed=0)
    before = env.data.ctrl.copy()
    # Drive every joint towards the far side of its range so all are clipped and rate-limited.
    direction = np.where(env.initial_qpos > 0, -1.0, 1.0)
    _, _, _, _, info = env.step(100.0 * direction)
    assert all(info["control"]["limit_clipped"])
    assert all(info["control"]["rate_limited"])
    assert np.all(env.data.ctrl <= env.limits[:, 1]) and np.all(env.data.ctrl >= env.limits[:, 0])
    np.testing.assert_allclose(env.data.ctrl - before, direction * env.max_velocity / 10)
    saved = env.data.qpos.copy()
    for bad in ([0] * 7, [np.nan] * 6, [np.inf] * 6):
        with pytest.raises(ValueError):
            env.step(bad)
        np.testing.assert_array_equal(env.data.qpos, saved)
    with pytest.raises(ValueError):
        env.step_real_joint_targets([0] * 5)


def test_truncation_requires_reset(env):
    env.max_episode_steps = 1
    env.reset(seed=0)
    assert env.step(env.initial_qpos)[3] is True
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.initial_qpos)


def test_upstream_actuators_preserved(env):
    import mujoco
    source = mujoco.MjModel.from_xml_path(str(ROOT / env.config["model"]))
    for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                 "actuator_ctrlrange"):
        np.testing.assert_array_equal(getattr(env.model, name), getattr(source, name))
    manifest = json.loads((ROOT / env.config["manifest"]).read_text())
    row = next(r for r in manifest["files"] if r["path"] == "robotstudio_so101/so101.xml")
    assert hashlib.sha256((ROOT / env.config["model"]).read_bytes()).hexdigest() == row["sha256"]
