"""Explicit grasp-relative RGB progress from causal recovery observations.

The fixed tool-to-grasp offsets are the same declared geometric assumptions as
the existing native controllers. They are not dynamic object measurements or
physical calibration. Only the relative RGB target channels change; robot pose,
visibility, age, and action history retain their original meanings.
"""

from __future__ import annotations

import numpy as np

from real2sim.policy.common_recovery import (
    ACHIEVED_OBSERVATION_NAMES,
    OBSERVATION_NAMES,
    TASK_EVENT_OBSERVATION_NAMES,
    TRACKED_ACHIEVED_OBSERVATION_NAMES,
    TRACKED_OBSERVATION_NAMES,
    TRACKED_TASK_EVENT_OBSERVATION_NAMES,
)

SOURCE_GRASP_OFFSET_M = np.array([-.010, 0., .030])
PANDA_GRASP_OFFSET_M = np.array([-.008, 0., 0.])
SOURCE_GRASP_OFFSET_M.setflags(write=False)
PANDA_GRASP_OFFSET_M.setflags(write=False)


def _grasp_names(names):
    return tuple(name.replace("_rgb_ball_relative_", "_rgb_ball_grasp_relative_")
                 .replace("_rgb_bowl_relative_", "_rgb_bowl_grasp_relative_") for name in names)


_CONTRACTS = {}
for _names in (OBSERVATION_NAMES, TRACKED_OBSERVATION_NAMES,
               ACHIEVED_OBSERVATION_NAMES, TRACKED_ACHIEVED_OBSERVATION_NAMES,
               TASK_EVENT_OBSERVATION_NAMES, TRACKED_TASK_EVENT_OBSERVATION_NAMES):
    _base = tuple(_names)
    _CONTRACTS[_base] = ("tcp", _base)
    _CONTRACTS[_grasp_names(_base)] = ("grasp", _base)


def _contract(names):
    try:
        return _CONTRACTS[tuple(names)]
    except (KeyError, TypeError) as exc:
        raise ValueError("Known 99D recovery observation names required for reference geometry") from exc


def observation_geometry_names(names, geometry="tcp"):
    """Return a versioned 99D contract for either TCP or grasp-relative goals."""
    if geometry not in ("tcp", "grasp"):
        raise ValueError("Recovery reference geometry must be tcp or grasp")
    _, base = _contract(names)
    return list(base if geometry == "tcp" else _grasp_names(base))


def transform_observation_geometry(observations, names, geometry="grasp"):
    """Convert live or batched recovery observations without reading object truth.

    For each of the three causal frames, a TCP-relative target ``g - p`` becomes
    ``g - (p + R @ offset)``. The encoded robot one-hot selects its declared
    offset, so a source-schema Panda observation naturally uses the SO-101
    offset. Source geometry is inferred from the explicit names, preventing an
    accidental second subtraction when a converted pack is reused.
    """
    if geometry not in ("tcp", "grasp"):
        raise ValueError("Recovery reference geometry must be tcp or grasp")
    source, _ = _contract(names)
    values = np.asarray(observations)
    if (values.shape[-1:] != (99,) or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise ValueError("Recovery reference geometry requires finite (...,99) observations")
    result = np.array(values, dtype=np.result_type(values.dtype, np.float32), copy=True)
    frames = result.reshape(*result.shape[:-1], 3, 33)
    robots = frames[..., 30:32]
    binary = np.isclose(robots, 0., rtol=0., atol=1e-6) | np.isclose(robots, 1., rtol=0., atol=1e-6)
    if not binary.all() or not np.allclose(robots.sum(axis=-1), 1., rtol=0., atol=1e-6):
        raise ValueError("Each recovery frame requires a valid SO-101/Panda robot one-hot")
    # Match matrix_from_rot6d's column-major Gram-Schmidt convention, while
    # evaluating all rows/history frames together for dataset conversion.
    first = frames[..., 3:6]
    second = frames[..., 6:9]
    norm_first = np.linalg.norm(first, axis=-1, keepdims=True)
    if not np.isfinite(norm_first).all() or np.any(norm_first < 1e-9):
        raise ValueError("Recovery reference geometry has a zero rotation6d column")
    axis0 = first / norm_first
    orthogonal = second - (axis0 * second).sum(axis=-1, keepdims=True) * axis0
    norm_second = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if not np.isfinite(norm_second).all() or np.any(norm_second < 1e-9):
        raise ValueError("Recovery reference geometry has collinear rotation6d columns")
    if source == geometry:
        return result
    axis1 = orthogonal / norm_second
    axis2 = np.cross(axis0, axis1)
    offsets = (robots[..., :1] * SOURCE_GRASP_OFFSET_M
               + robots[..., 1:] * PANDA_GRASP_OFFSET_M)
    task_offset = (axis0 * offsets[..., :1] + axis1 * offsets[..., 1:2]
                   + axis2 * offsets[..., 2:3])
    sign = -1. if geometry == "grasp" else 1.
    frames[..., 13:16] += sign * task_offset
    frames[..., 16:19] += sign * task_offset
    return result
