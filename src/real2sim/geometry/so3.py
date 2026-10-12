"""SO(3)/SE(3) utilities used by the canonical 7-D action contract (plan §3.3).

Conventions: rotation matrices act on column vectors; ``T_a_b`` maps points in frame ``b`` to
frame ``a``; rotation vectors are axis * angle in radians with angle in [0, pi].
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def hat(w: np.ndarray) -> np.ndarray:
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector -> rotation matrix."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    K = hat(w)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * K @ K


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (numerically safe near 0 and pi)."""
    R = np.asarray(R, dtype=np.float64)
    anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    # atan2 keeps full precision near 0 and pi, where arccos of the trace loses ~1e-8.
    theta = float(np.arctan2(np.linalg.norm(anti) / 2.0, (np.trace(R) - 1.0) / 2.0))
    if theta < 1e-6:
        # first-order: R ≈ I + hat(w)
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        # near pi: axis from the symmetric part (R + R^T)/2 = I + (1 - cos) k k^T, sign from the
        # antisymmetric residue. Using R itself mixes in sin(theta) K and loses ~1e-4 accuracy.
        B = ((R + R.T) / 2.0 + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        i = int(np.argmax(axis))
        axis = B[:, i] / max(axis[i], _EPS)
        axis /= np.linalg.norm(axis)
        anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(anti, axis) < 0:
            axis = -axis
        return axis * theta
    return theta / (2.0 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def is_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(R.T @ R, np.eye(3), atol=atol) and abs(np.linalg.det(R) - 1.0) < atol)


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """Continuous 6-D representation: first two columns, column-major [c0, c1] (Zhou et al. 2019)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def matrix_from_rot6d(x: np.ndarray) -> np.ndarray:
    a1, a2 = np.asarray(x[:3], np.float64), np.asarray(x[3:6], np.float64)
    n1 = np.linalg.norm(a1)
    if not np.isfinite(n1) or n1 < 1e-9:
        raise ValueError("rot6d first column is zero or non-finite")
    b1 = a1 / n1
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2)
    if not np.isfinite(n2) or n2 < 1e-9:
        raise ValueError("rot6d columns are collinear or non-finite")
    b2 = a2 / n2
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def delta_action(T_cur: np.ndarray, T_next: np.ndarray) -> np.ndarray:
    """6-D task-frame delta ``[dx, dy, dz, d_rx, d_ry, d_rz]`` from current to next EE pose.

    Translation is the task-frame difference of positions; rotation is ``log(R_next R_cur^T)``
    (a left/world-frame increment), exactly as specified in plan §3.3.
    """
    dp = T_next[:3, 3] - T_cur[:3, 3]
    dr = log_so3(T_next[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([dp, dr])


def apply_delta(T_cur: np.ndarray, delta6: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_action`: target pose from the current pose and a 6-D delta."""
    R = exp_so3(delta6[3:6]) @ T_cur[:3, :3]
    return make_T(R, T_cur[:3, 3] + delta6[:3])
