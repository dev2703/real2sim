"""SO-101 FK: MuJoCo (real2sim.geometry) vs the sim owner's independent XML-chain FK."""

import numpy as np
import pytest

from real2sim.geometry.so101_fk import ARM_JOINTS, DEFAULT_MJCF, So101FK

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DEFAULT_MJCF.is_file(), reason="Menagerie SO-101 not fetched"),
]


def test_mujoco_fk_matches_independent_xml_chain():
    xml_fk = pytest.importorskip("real2sim.sim.kinematics").xml_forward_kinematics
    fk = So101FK()
    rng = np.random.default_rng(0)
    lo, hi = fk.joint_range[:, 0], fk.joint_range[:, 1]
    for q in rng.uniform(lo, hi, size=(25, 5)):
        T_mj = fk.fk_sim(q)
        T_xml = xml_fk(DEFAULT_MJCF, dict(zip(ARM_JOINTS, q, strict=True)))
        np.testing.assert_allclose(T_mj[:3, 3], T_xml[:3, 3], atol=1e-6)
        np.testing.assert_allclose(T_mj[:3, :3], T_xml[:3, :3], atol=1e-6)


def test_batch_shapes_and_degree_mapping():
    fk = So101FK()
    q_deg = np.zeros((3, 6))
    T = fk.fk_real(q_deg)
    assert T.shape == (3, 4, 4)
    np.testing.assert_allclose(T[0], fk.fk_sim(np.zeros(5)), atol=1e-12)
