"""The explicit support proxy must spawn, contact and reset like actual geometry."""

import copy

import mujoco
import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.sim.ball_support import spawn_center_above_table, support_contact_metrics


def task_with_support():
    task = copy.deepcopy(load_yaml("configs/task.yaml"))
    task["ball"]["support"] = {"type": "box", "height": .01, "size_xy": [.10, .08],
                                "center_task_xy": [.3557, .2047], "yaw_deg": 20}
    return task


def test_sphere_spawn_clears_box_top_and_edge_instead_of_intersecting_support():
    task = task_with_support()
    support = task["ball"]["support"]
    support["yaw_deg"] = 0.
    center = np.asarray(support["center_task_xy"])
    radius = task["ball"]["radius"]
    assert spawn_center_above_table(task, center) == pytest.approx(radius + .01)
    # Ten millimetres outside a side still intersects the box unless its height is raised.
    point = center + [support["size_xy"][0] / 2 + .01, 0.]
    height = spawn_center_above_table(task, point)
    assert np.hypot(.01, height - .01) == pytest.approx(radius)
    assert height > radius
    assert spawn_center_above_table(task, center + [.5, 0.]) == radius


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_support_is_physical_and_resets_independently_of_ball_and_bowl(robot, tmp_path):
    if robot == "so101":
        from real2sim.sim.env import So101PickPlaceEnv as Env
    else:
        from real2sim.panda.env import PandaPickPlaceEnv as Env
    task = task_with_support()
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    config = load_yaml(f"configs/sim_{robot}.yaml")
    config["task_config"] = str(task_path)
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with Env(config_path, render_images=False) as env:
        center = np.array(task["ball"]["support"]["center_task_xy"])
        _, initial = env.reset(seed=0, scenario={"ball_xy": center})
        expected = env.table_top + .01 + task["ball"]["radius"] + .001
        assert initial["privileged_metrics"]["ball_xyz"][2] == pytest.approx(expected)
        # Passive settling isolates the new surface from any scripted grasp behavior.
        for _ in range(500):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        info = env.info()["privileged_metrics"]
        assert info["ball_touching_support"]
        assert not info["ball_touching_table"]
        assert info["ball_xyz"][2] == pytest.approx(expected - .001, abs=.0005)
        moved = center + [.08, -.06]
        env.reset(seed=0, scenario={"ball_xy": moved, "ball_support_xy": moved})
        support = env.model.geom("ball_support")
        np.testing.assert_allclose(support.pos[:2], moved + env.offset[:2])
        env.reset(seed=0, scenario={"ball_xy": center})
        np.testing.assert_allclose(support.pos[:2], center + env.offset[:2])


def test_absent_support_preserves_default_spawn_and_rejects_ineffective_override():
    from real2sim.sim.env import So101PickPlaceEnv

    with So101PickPlaceEnv(render_images=False) as env:
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_support") == -1
        _, info = env.reset(seed=0)
        assert info["privileged_metrics"]["ball_xyz"][2] == pytest.approx(
            env.table_top + env.task["ball"]["radius"] + .001)
        assert "ball_touching_support" not in info["privileged_metrics"]
        with pytest.raises(ValueError, match="requires an enabled"):
            env.reset(seed=0, scenario={"ball_support_xy": [0, 0]})


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_resting_on_tall_support_does_not_count_as_a_lift(robot, tmp_path):
    if robot == "so101":
        from real2sim.sim.env import So101PickPlaceEnv as Env
    else:
        from real2sim.panda.env import PandaPickPlaceEnv as Env
    task = task_with_support()
    task["ball"]["support"]["height"] = .04
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    config = load_yaml(f"configs/sim_{robot}.yaml")
    config["task_config"] = str(task_path)
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with Env(config_path, render_images=False) as env:
        _, info = env.reset(seed=0, scenario={
            "ball_xy": task["ball"]["support"]["center_task_xy"],
        })
        metrics = info["privileged_metrics"]
        assert metrics["ball_xyz"][2] > env.table_top + task["ball"]["radius"] + .03
        assert not metrics["lifted"]
        env.data.qpos[env._ball_q + 2] = env._ball_rest_center_z + .031
        mujoco.mj_forward(env.model, env.data)
        assert env.info()["privileged_metrics"]["lifted"]


def test_support_contact_metrics_exclude_nonrobot_objects_and_follow_robot_descendants():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="ball_support" type="box" size=".6 .2 .05" pos="0 0 .05"/>
          <body name="robot" pos="-.3 0 .14"><body name="robot_link">
            <joint name="robot_slide" type="slide" axis="0 0 1"/>
            <geom name="robot_geom" type="sphere" size=".05" mass="1"/>
          </body></body>
          <body name="foreign_object" pos=".3 0 .13"><freejoint/>
            <geom name="foreign_geom" type="sphere" size=".05" mass="1"/>
          </body>
          <body name="ball" pos="0 0 .145"><freejoint name="ball_free"/>
            <geom name="ball_geom" type="sphere" size=".05" mass="1"/>
          </body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    root = model.body("robot").id
    mujoco.mj_forward(model, data)
    # The foreign object penetrates 20mm, deeper than the robot's 10mm: including
    # it would inflate both the robot contact count and the reset rejection depth.
    assert data.ncon == 3
    metrics = support_contact_metrics(model, data, root)
    assert metrics["ball_touching_support"]
    assert metrics["robot_support_contacts"] == 1
    assert metrics["robot_support_penetration_m"] == pytest.approx(.01)
    data.qpos[model.jnt_qposadr[model.joint("robot_slide").id]] = .2
    mujoco.mj_forward(model, data)
    assert data.ncon == 2  # foreign object and ball remain in contact
    metrics = support_contact_metrics(model, data, root)
    assert metrics["robot_support_contacts"] == 0
    assert metrics["robot_support_penetration_m"] == 0
    assert metrics["ball_touching_support"]
    data.qpos[model.jnt_qposadr[model.joint("ball_free").id] + 2] = .4
    mujoco.mj_forward(model, data)
    assert data.ncon == 1  # foreign object alone cannot become ball/robot contact
    assert support_contact_metrics(model, data, root) == {
        "ball_touching_support": False, "robot_support_contacts": 0,
        "robot_support_penetration_m": 0,
    }
