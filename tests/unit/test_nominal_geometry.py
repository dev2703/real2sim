"""Standard CAD support must touch the table without hiding reference collisions."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics
from scripts.diag_v2_geometry_collision import apply_static_candidate, visual_meshes

CANDIDATE = Path("configs/v2/geometry.nominal_so101_v1.json")


def test_standard_no_spacer_support_plane_and_encoder_mapping_preserve_native_bounds():
    with So101PickPlaceEnv(render_images=False) as env:
        limits = env.model.jnt_range.copy()
        actuators = env.model.actuator_ctrlrange.copy()
        masks = env.model.geom_contype.copy(), env.model.geom_conaffinity.copy()
        table = env.model.geom_pos[env.table_geom].copy()
        camera = env.model.cam_pos.copy()
        original = copy.deepcopy(env.task)
        provenance = apply_static_candidate(env, CANDIDATE)
        kin = NativeKinematics(env.model, env.config)
        kin.forward(np.zeros(6))
        meshes = visual_meshes(env.model, {env.base_body})
        minimum = min(np.min(vertices @ kin.data.geom_xmat[g].reshape(3, 3)[2]
                             + kin.data.geom_xpos[g, 2]) for g, vertices in meshes.items())
        assert minimum - env.table_top == pytest.approx(0, abs=1e-8)
        assert env.model.body_pos[env.base_body, 2] - env.table_top == pytest.approx(.0024)
        assert env.mapping.gripper_real_to_sim_rad(30) == pytest.approx(.5030713386203921)
        assert env.mapping.gripper_real_to_sim_rad(100) > env.limits[5, 1]  # no silent clamp
        np.testing.assert_array_equal(env.model.jnt_range, limits)
        np.testing.assert_array_equal(env.model.actuator_ctrlrange, actuators)
        np.testing.assert_array_equal(env.model.geom_contype, masks[0])
        np.testing.assert_array_equal(env.model.geom_conaffinity, masks[1])
        np.testing.assert_array_equal(env.model.geom_pos[env.table_geom], table)
        np.testing.assert_array_equal(env.model.cam_pos, camera)
        assert env.task == original
        assert provenance["definition"]["physical_calibration_accepted"] is False


@pytest.mark.parametrize("change,match", [
    (lambda x: x.update(native_model_sha256="wrong"), "hash mismatch"),
    (lambda x: x.update(physical_calibration_accepted=True), "cannot claim"),
    (lambda x: x["changes"].update(arm_offset_deg=[0]*5), "only base"),
    (lambda x: x["changes"].update(base_origin_above_table_m=-1), "height"),
    (lambda x: x["changes"]["gripper_percent_to_joint"].update(scale_rad_per_percent=0), "nonzero"),
])
def test_invalid_static_candidate_cannot_partially_mutate_scene(tmp_path, change, match):
    candidate = json.loads(CANDIDATE.read_text())
    change(candidate)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    with So101PickPlaceEnv(render_images=False) as env:
        positions, mapping = env.model.body_pos.copy(), env.mapping
        with pytest.raises(ValueError, match=match):
            apply_static_candidate(env, path)
        np.testing.assert_array_equal(env.model.body_pos, positions)
        assert env.mapping is mapping
