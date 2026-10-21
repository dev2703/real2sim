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
    assert report["all_landmarks_in_front_of_camera"]
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
    null_directions = np.asarray(result["identifiability"]["scaled_null_directions"])
    assert null_directions.shape == (2, 5)
    np.testing.assert_allclose(null_directions[:, :3], 0, atol=1e-8)
    np.testing.assert_allclose(null_directions @ null_directions.T, np.eye(2), atol=1e-8)


def test_fewer_pixel_constraints_than_joints_reports_complete_nullspace():
    q, _, _, _, _, kwargs = joint_problem()
    q = q[:2]
    groups = [0, 1]

    def points(joints, indices):
        # One fit point provides only two independent image coordinates. The
        # compensating joint sums and the absent third coordinate leave three
        # directions unobservable, even if the residual happens to be zero.
        return np.column_stack([.1 * (joints[:, 0] + joints[:, 3]),
                                .1 * (joints[:, 1] + joints[:, 4]),
                                np.zeros(len(joints))])

    def project(joints):
        return cv2.projectPoints(points(joints, np.arange(len(joints))), np.zeros(3),
                                 np.array([0., 0., 1.]), K, None)[0].reshape(-1, 2)

    result = fit_joint_zero_offsets(q, project(q), groups, points,
                                   **(kwargs | {"fit_groups": [0], "verification_groups": [1]}))
    ident = result["identifiability"]
    assert ident["rank"] == 2
    assert not ident["full_rank"]
    assert ident["conditional_parameter_std"] is None
    assert len(ident["scaled_singular_values"]) == 5
    assert np.asarray(ident["scaled_right_singular_vectors"]).shape == (5, 5)
    null_directions = np.asarray(ident["scaled_null_directions"])
    assert null_directions.shape == (3, 5)
    for direction in null_directions * np.asarray(ident["parameter_scales"]):
        np.testing.assert_allclose(project(q + direction * 1e-3), project(q), atol=1e-7)


@pytest.mark.parametrize("behind_groups,expected_fit,expected_verify", [
    ([2], 0, 8), ([0, 1, 2], 16, 8),
])
def test_joint_fit_exposes_behind_camera_landmarks_despite_low_pixel_error(
    behind_groups, expected_fit, expected_verify,
):
    q, _, groups, real_points, offsets, kwargs = joint_problem()
    behind = np.isin(groups, behind_groups)

    def points(joints, indices):
        xyz = real_points(joints, indices)
        xyz[behind[indices], 2] -= 2.
        return xyz

    xyz = points(q + offsets, np.arange(len(q)))
    pixels = cv2.projectPoints(xyz, np.zeros(3), np.array([0., 0., 1.]), K, None)[0].reshape(-1, 2)
    result = fit_joint_zero_offsets(q, pixels, groups, points, **kwargs)
    assert result["verification"]["p95_px"] < 1e-5
    assert result["fit_points_behind_camera"] == expected_fit
    assert result["verification_points_behind_camera"] == expected_verify
    assert result["points_behind_camera"] == expected_fit + expected_verify
    assert not result["all_landmarks_in_front_of_camera"]
    np.testing.assert_array_equal(np.asarray(result["per_row_depth_m"]) <= 0, behind)
    assert not result["automatically_adopted"]


def test_impossible_joint_states_are_never_clipped_into_feasibility():
    q, pixels, groups, points, _, kwargs = joint_problem()
    q[0, 0] = -3
    with pytest.raises(ValueError, match="without clipping"):
        fit_joint_zero_offsets(q, pixels, groups, points, **kwargs)
