"""Calibration must generalize without fitting verification points or hiding limits."""

import cv2
import numpy as np
import pytest

from real2sim.geometry.registration import fit_camera_registration, fit_joint_zero_offsets
from real2sim.geometry.so101_fk import So101FK

K = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])


def camera_problem():
    rng = np.random.default_rng(4)
    points = rng.uniform([-.3, -.2, 0], [.3, .2, .3], (24, 3))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([.3, -.1, .2]))[0]
    T[:3, 3] = [.04, -.02, 1]
    pixels = cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
    kwargs = dict(K=K, fit_groups=[0, 1], verification_groups=[2],
                  geometry_provenance="synthetic independent metric anchors",
                  independent_geometry=True, sigma_px=1.)
    return points, pixels, np.repeat([0, 1, 2], 8), T, kwargs


def test_camera_fit_recovers_independent_geometry_and_does_not_fit_verification():
    points, pixels, groups, T, kwargs = camera_problem()
    report = fit_camera_registration(points, pixels, groups, **kwargs)
    np.testing.assert_allclose(report["T_optical_task"], T, atol=1e-8)
    assert report["verification"]["p95_px"] < 1e-6
    assert report["identifiability"]["full_rank"]
    assert not report["automatically_adopted"]
    pixels[groups == 2] += [80, -40]
    changed = fit_camera_registration(points, pixels, groups, **kwargs)
    np.testing.assert_array_equal(report["T_optical_task"], changed["T_optical_task"])
    assert changed["verification"]["p95_px"] > 80


@pytest.mark.parametrize("override,match", [
    ({"independent_geometry": False}, "independent"),
    ({"reserved_groups": [2]}, "reserved"),
    ({"verification_groups": [1, 2]}, "disjoint"),
    ({"fit_groups": [0]}, "every supplied"),
])
def test_rejects_circular_geometry_or_split_leakage(override, match):
    points, pixels, groups, _, kwargs = camera_problem()
    with pytest.raises(ValueError, match=match):
        fit_camera_registration(points, pixels, groups, **(kwargs | override))


def test_collinear_camera_anchors_rejected_and_planarity_explicit():
    points, _, groups, T, kwargs = camera_problem()
    points[:, 2] = 0
    pixels = cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
    planar = fit_camera_registration(points, pixels, groups, **kwargs)
    assert planar["planar_ambiguity_requires_independent_check"]
    points[:, 1] = 0
    with pytest.raises(ValueError, match="collinear"):
        fit_camera_registration(points, pixels, groups, **kwargs)


def joint_problem():
    fk = So101FK()
    rng = np.random.default_rng(8)
    q = rng.uniform(-.6, .6, (24, 5))
    offsets = np.deg2rad([2., -3., 4., -5., 3.])
    T = np.eye(4)
    T[2, 3] = 1.

    def points_for_q(joints, indices):
        poses = fk.fk_sim(joints)
        # An independently located, off-axis marker makes roll observable.
        return poses[:, :3, 3] + poses[:, :3, :3] @ np.array([.02, .01, .005])

    points = points_for_q(q + offsets, np.arange(len(q)))
    pixels = cv2.projectPoints(points, np.zeros(3), T[:3, 3], K, None)[0].reshape(-1, 2)
    kwargs = dict(K=K, T_optical_task=T, joint_limits_rad=np.tile([-2., 2.], (5, 1)),
                  offset_bounds_rad=np.tile([-.2, .2], (5, 1)), fit_groups=[0, 1],
                  verification_groups=[2], sigma_px=1.)
    return q, pixels, np.repeat([0, 1, 2], 8), points_for_q, offsets, kwargs


def test_joint_zero_fit_recovers_offsets_without_camera_compensation():
    q, pixels, groups, points, offsets, kwargs = joint_problem()
    result = fit_joint_zero_offsets(q, pixels, groups, points, **kwargs)
    np.testing.assert_allclose(result["joint_zero_offsets_rad"], offsets, atol=1e-6)
    assert result["identifiability"]["full_rank"]
    assert result["verification"]["p95_px"] < 1e-5
    assert not result["camera_fitted"]
    assert not result["reference_clipping_applied"]

    # Verification coordinates/observations cannot change feasible bounds or fitting.
    q[groups == 2, 0] += 4
    pixels[groups == 2] += 50
    changed = fit_joint_zero_offsets(q, pixels, groups, points, **kwargs)
    np.testing.assert_array_equal(result["joint_zero_offsets_rad"], changed["joint_zero_offsets_rad"])
    assert changed["verification_joint_limit_violation_rows"] == 8
    assert changed["verification_max_joint_limit_violation_rad"] > 1


def test_joint_fit_reports_unobservable_parameters_instead_of_prior_confidence():
    q, pixels, groups, _, _, kwargs = joint_problem()

    def unobservable(joints, indices):
        return np.column_stack([joints[:, 0] * .1, joints[:, 1] * .1, joints[:, 2] * .1])

    points = unobservable(q, np.arange(len(q)))
    pixels = cv2.projectPoints(points, np.zeros(3), np.array([0., 0., 1.]), K, None)[0].reshape(-1, 2)
    result = fit_joint_zero_offsets(q, pixels, groups, unobservable, **kwargs)
    assert result["identifiability"]["rank"] == 3
    assert result["identifiability"]["conditional_parameter_std"] is None


def test_impossible_joint_states_are_never_clipped_into_feasibility():
    q, pixels, groups, points, _, kwargs = joint_problem()
    q[0, 0] = -3
    with pytest.raises(ValueError, match="without clipping"):
        fit_joint_zero_offsets(q, pixels, groups, points, **kwargs)
