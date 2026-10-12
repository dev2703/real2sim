import numpy as np
import pytest

from real2sim.geometry.so3 import (
    apply_delta,
    delta_action,
    exp_so3,
    inv_T,
    is_rotation,
    log_so3,
    make_T,
    matrix_from_rot6d,
    rot6d_from_matrix,
)

RNG = np.random.default_rng(0)


def random_rotvecs(n, max_angle=np.pi):
    axes = RNG.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes * RNG.uniform(0, max_angle, size=(n, 1))


@pytest.mark.parametrize("w", list(random_rotvecs(50, np.pi - 1e-3)) + [np.zeros(3), np.array([1e-9, 0, 0])])
def test_exp_log_roundtrip(w):
    R = exp_so3(w)
    assert is_rotation(R)
    np.testing.assert_allclose(log_so3(R), w, atol=1e-7)


def test_log_near_pi():
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 1.0]) / np.sqrt(2), np.array([1.0, -2.0, 0.5])):
        axis = axis / np.linalg.norm(axis)
        w = axis * (np.pi - 1e-6)
        np.testing.assert_allclose(exp_so3(log_so3(exp_so3(w))), exp_so3(w), atol=1e-6)


def test_exp_matches_known_rotation():
    R = exp_so3(np.array([0, 0, np.pi / 2]))
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)  # right-handed z


def test_rot6d_roundtrip_and_gram_schmidt():
    for w in random_rotvecs(20):
        R = exp_so3(w)
        np.testing.assert_allclose(matrix_from_rot6d(rot6d_from_matrix(R)), R, atol=1e-12)
    noisy = rot6d_from_matrix(exp_so3(np.array([0.3, -0.2, 0.1]))) + RNG.normal(0, 0.01, 6)
    assert is_rotation(matrix_from_rot6d(noisy))


def test_se3_inverse_and_composition():
    T = make_T(exp_so3(np.array([0.2, 0.5, -0.4])), np.array([0.1, -0.3, 0.25]))
    np.testing.assert_allclose(T @ inv_T(T), np.eye(4), atol=1e-12)
    p_b = np.array([0.01, 0.02, 0.03, 1.0])
    np.testing.assert_allclose(inv_T(T) @ (T @ p_b), p_b, atol=1e-12)


def test_delta_action_roundtrip_units_and_frame():
    T0 = make_T(exp_so3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.0, 0.1]))
    d = np.array([0.01, -0.02, 0.005, 0.05, 0.0, -0.1])
    T1 = apply_delta(T0, d)
    np.testing.assert_allclose(delta_action(T0, T1), d, atol=1e-10)
    # pure translation delta does not change orientation and is expressed in the task frame
    T2 = apply_delta(T0, np.array([0.03, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(T2[:3, 3] - T0[:3, 3], [0.03, 0, 0], atol=1e-12)
    np.testing.assert_allclose(T2[:3, :3], T0[:3, :3], atol=1e-12)
