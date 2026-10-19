"""Contact diagnosis must preserve physics and distinguish load support from table contact."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from real2sim.sim.env import So101PickPlaceEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("diag_so101_contact_retention")


def test_ball_normal_sign_is_invariant_to_geom_order(diagnostic):
    first = SimpleNamespace(geom1=5, geom2=7, frame=np.eye(3).ravel())
    second = SimpleNamespace(geom1=7, geom2=5, frame=(-np.eye(3)).ravel())
    np.testing.assert_array_equal(diagnostic.ball_contact_frame(first, 7),
                                  diagnostic.ball_contact_frame(second, 7))
    with pytest.raises(ValueError, match="does not contain ball"):
        diagnostic.ball_contact_frame(first, 9)


def test_hold_rule_requires_consecutive_bilateral_lift(diagnostic):
    good = {"bilateral": True, "height_m": .031}
    assert not diagnostic.hold_ready([good] * 4)
    assert diagnostic.hold_ready([good] * 5)
    for bad in ({"bilateral": False, "height_m": .1}, {"bilateral": True, "height_m": .029}):
        assert not diagnostic.hold_ready([good] * 4 + [bad])
        assert diagnostic.hold_ready([bad] + [good] * 5)


def test_snapshot_excludes_table_and_preserves_integration_state(diagnostic):
    with So101PickPlaceEnv(render_images=False) as env:
        env.reset(seed=0, scenario={"ball_xy": [.4, -.2]})
        for _ in range(5):
            env.step(env.initial_qpos)
        assert env.table_geom in env._ball_contacts()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(env.model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(env.model, env.data, before, spec)
        snapshot = diagnostic.snapshot(env)
        mujoco.mj_getState(env.model, env.data, after, spec)
        np.testing.assert_array_equal(before, after)
        assert not snapshot["bilateral"]
        assert not snapshot["contacts"]
        assert snapshot["jaw_normal_opposition_deg"] is None
        json.dumps(snapshot, allow_nan=False)


def test_contact_force_and_surface_slip_include_correct_body_frame(diagnostic):
    # A stationary fixed jaw supports a ball; a moving jaw and table are far away.
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody>
        <site name="tool" pos="0 0 0"/>
        <geom name="table" type="box" size=".1 .1 .01" pos="2 0 0"/>
        <body name="gripper">
          <geom name="support" type="box" size=".1 .1 .01" pos="0 0 -.01"/>
          <geom name="fixed_jaw_sph_tip1" size=".001" pos=".05 0 0" contype="0" conaffinity="0"/>
        </body>
        <body name="moving_jaw_so101_v1" pos="1 0 0">
          <geom name="moving_jaw_sph_tip1" size=".001" contype="0" conaffinity="0"/>
        </body>
        <body name="ball" pos="0 0 .02"><freejoint/>
          <geom name="ball_geom" size=".02" mass=".05"/>
        </body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data, ball_geom=model.geom("ball_geom").id,
        ball_body=model.body("ball").id, table_geom=model.geom("table").id,
        site_id=model.site("tool").id, table_top=0., task={"ball": {"radius": .02}})
    result = diagnostic.snapshot(env)
    contact, = result["contacts"]
    assert contact["jaw"] == "gripper"
    assert contact["normal_onto_ball_tool"] == [0., 0., 1.]
    assert contact["force_onto_ball_world_n"][2] == pytest.approx(.05 * 9.81, rel=.001)
    assert not result["bilateral"]
    data.qvel[:] = 0.
    data.qvel[0] = .01
    mujoco.mj_forward(model, data)
    translating = diagnostic.snapshot(env)["contacts"][0]
    assert translating["tangential_surface_slip_mps"] == pytest.approx(.01)
    # Add rotation that cancels translation at the actual contact point.
    radius_at_contact = data.xpos[env.ball_body, 2] - data.contact[0].pos[2]
    data.qvel[4] = .01 / radius_at_contact
    mujoco.mj_forward(model, data)
    rolling = diagnostic.snapshot(env)["contacts"][0]
    assert rolling["tangential_surface_slip_mps"] < 1e-10
