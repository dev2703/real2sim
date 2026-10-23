"""The native IK gates must include optional support geometry after it moves."""

import copy

import numpy as np
import pytest
import yaml

from real2sim.config import load_yaml
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_native_ik_rejects_robot_support_penetration_but_tolerates_ball_contact(robot, tmp_path):
    task = copy.deepcopy(load_yaml("configs/task.yaml"))
    task["ball"]["support"] = {
        "type": "box", "height": .20, "size_xy": [.10, .10],
        "center_task_xy": [.5, .3],
    }
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    config = load_yaml(f"configs/sim_{robot}.yaml")
    config["task_config"] = str(task_path)
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cls = So101PickPlaceEnv if robot == "so101" else PandaPickPlaceEnv
    with cls(config_path, render_images=False) as env:
        env.reset(seed=0)
        if robot == "so101":
            seed = np.array([0., -.8, .8, .8, 0., .3])
            kin = NativeKinematics(env.model, env.config)
        else:
            seed = env.initial_qpos.copy()
            kin = PandaIK(env.model, env.config, task_origin=env.offset)
            kin.set_finger_opening(.08)
        target = kin.forward(seed)
        assert kin.solve(target, seed).accepted
        support = env.model.geom("ball_support")
        support_xy = support.pos[:2].copy()
        # The controller's separate MjData must observe a support moved after IK
        # construction, as happens through the environment's scenario reset.
        support.pos[:2] = target[:2, 3] + env.offset[:2]
        kin.forward(seed)
        penetrating = [c for c in kin.data.contact[:kin.data.ncon]
                       if support.id in (c.geom1, c.geom2)
                       and c.dist < -config["ik"]["penetration_tolerance_m"]]
        assert penetrating  # real narrow-phase contacts, not a mocked rejection
        rejected = kin.solve(target, seed)
        assert not rejected.accepted and rejected.reason == "collision"
        np.testing.assert_array_equal(rejected.qpos, seed)
        # Ball contact is intentional during grasp and must remain exempt even
        # when its opposing geometry is the newly introduced support.
        support.pos[:2] = support_xy
        ball_qadr = env.model.jnt_qposadr[env.model.joint("ball_free").id]
        kin.data.qpos[ball_qadr:ball_qadr + 3] = [
            *support_xy, env.table_top + .20 + task["ball"]["radius"] - .01,
        ]
        kin.forward(seed)
        ball = env.model.geom("ball_geom").id
        assert any({support.id, ball} == {int(c.geom1), int(c.geom2)} and c.dist < -.005
                   for c in kin.data.contact[:kin.data.ncon])
        assert kin.solve(target, seed).accepted
