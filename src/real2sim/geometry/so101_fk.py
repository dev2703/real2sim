"""SO-101 end-effector forward kinematics from the pinned Menagerie MJCF (via MuJoCo).

Real joint readings (LeRobot degrees, gripper %) are mapped to MJCF joint angles with the
provisional mapping in ``configs/task.yaml`` (``robot_so101.real_to_sim``; sign +1, offset 0 until
M2 verifies it). The end-effector frame ``ee`` is the Menagerie ``gripperframe`` site. Poses are
returned as ``T_so101base_ee`` (4x4, metres); with the provisional identity
``T_task_so101base`` this is also the task-frame pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import REPO_ROOT

REPO = REPO_ROOT
DEFAULT_MJCF = REPO / "assets" / "menagerie" / "robotstudio_so101" / "so101.xml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EE_SITE = "gripperframe"


@dataclass(frozen=True)
class JointMapping:
    sign: tuple[float, ...] = (1, 1, 1, 1, 1)
    offset_deg: tuple[float, ...] = (0, 0, 0, 0, 0)
    gripper_sim_range_rad: tuple[float, float] = (-0.1745, 1.7454)
    # Explicit percent-to-joint calibration is independent of native joint limits.
    # Legacy checkpoints retain the endpoint-based mapping when this is absent.
    gripper_affine_rad: tuple[float, float] | None = None  # offset, radians per percent

    def __post_init__(self) -> None:
        if len(self.sign) != 5 or len(self.offset_deg) != 5:
            raise ValueError("JointMapping needs exactly five signs and five offsets (arm joints)")
        if any(s not in (1.0, -1.0) for s in map(float, self.sign)):
            raise ValueError(f"joint signs must be +1 or -1, got {self.sign}")
        if not all(np.isfinite(self.offset_deg)):
            raise ValueError("joint offsets must be finite")
        lo, hi = self.gripper_sim_range_rad
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"gripper range must be finite with lo < hi, got {self.gripper_sim_range_rad}")
        if self.gripper_affine_rad is not None:
            if len(self.gripper_affine_rad) != 2:
                raise ValueError("gripper affine mapping needs offset and radians per percent")
            offset, scale = self.gripper_affine_rad
            if not (np.isfinite(offset) and np.isfinite(scale) and scale != 0):
                raise ValueError("gripper affine mapping must be finite with nonzero scale")

    @classmethod
    def from_task_config(cls, task_cfg: dict) -> JointMapping:
        r = task_cfg["robot_so101"]
        affine = r["gripper"].get("percent_to_joint")
        return cls(
            sign=tuple(float(s) for s in r["real_to_sim"]["sign"]),
            offset_deg=tuple(float(o) for o in r["real_to_sim"]["offset_deg"]),
            gripper_sim_range_rad=tuple(float(x) for x in r["gripper"].get("sim_range_rad", (-0.1745, 1.7454))),
            gripper_affine_rad=None if affine is None else (
                float(affine["offset_rad"]), float(affine["scale_rad_per_percent"])),
        )

    def arm_real_to_sim_rad(self, q_real_deg: np.ndarray) -> np.ndarray:
        q = np.asarray(q_real_deg, np.float64)[..., :5]
        return np.deg2rad(q * np.asarray(self.sign) + np.asarray(self.offset_deg))

    def gripper_real_to_sim_rad(self, gripper_pct: np.ndarray) -> np.ndarray:
        if self.gripper_affine_rad is not None:
            offset, scale = self.gripper_affine_rad
            return offset + np.asarray(gripper_pct, np.float64) * scale
        lo, hi = self.gripper_sim_range_rad
        return lo + np.asarray(gripper_pct, np.float64) / 100.0 * (hi - lo)

    def gripper_sim_to_real_percent(self, gripper_rad: np.ndarray) -> np.ndarray:
        """Inverse calibration, without concealing out-of-range observed positions."""
        q = np.asarray(gripper_rad, np.float64)
        if self.gripper_affine_rad is not None:
            offset, scale = self.gripper_affine_rad
            return (q - offset) / scale
        lo, hi = self.gripper_sim_range_rad
        return (q - lo) / (hi - lo) * 100.0


class So101FK:
    """Batch FK for the five arm joints (the gripper does not move the ``gripperframe`` site)."""

    def __init__(self, mjcf: str | Path = DEFAULT_MJCF, mapping: JointMapping | None = None):
        self.mjcf = Path(mjcf)
        if not self.mjcf.is_file():
            raise FileNotFoundError(f"{self.mjcf} missing; run `uv run python scripts/fetch_menagerie.py`")
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf))
        self.data = mujoco.MjData(self.model)
        self.mapping = mapping or JointMapping()
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in ARM_JOINTS])
        self.dofadr = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in ARM_JOINTS])
        self.site_id = self.model.site(EE_SITE).id
        self.joint_range = np.array([self.model.jnt_range[self.model.joint(j).id] for j in ARM_JOINTS])

    def fk_sim(self, q_sim_rad: np.ndarray) -> np.ndarray:
        """``[..., 5]`` MJCF joint angles -> ``[..., 4, 4]`` gripperframe poses in the base frame."""
        q = np.asarray(q_sim_rad, np.float64)
        flat = q.reshape(-1, 5)
        out = np.empty((len(flat), 4, 4))
        for i, qi in enumerate(flat):
            self.data.qpos[:] = 0.0
            self.data.qpos[self.qadr] = qi
            mujoco.mj_kinematics(self.model, self.data)
            out[i] = np.eye(4)
            out[i, :3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
            out[i, :3, 3] = self.data.site_xpos[self.site_id]
        return out.reshape(*q.shape[:-1], 4, 4)

    def fk_real(self, q_real_deg: np.ndarray) -> np.ndarray:
        """LeRobot state/action vectors (degrees; 6th entry gripper ignored) -> EE poses."""
        return self.fk_sim(self.mapping.arm_real_to_sim_rad(q_real_deg))

    def point_kinematics_sim(self, q_sim_rad: np.ndarray, offset: np.ndarray):
        """Position and analytic arm Jacobian of an EE-local point.

        This includes motion from the point's rotational offset, which matters
        when reconstructing the source grasp point from a different embodiment.
        No dynamics, contacts, or dynamic-object measurements are evaluated.
        """
        q, offset = np.asarray(q_sim_rad, float), np.asarray(offset, float)
        if q.shape != (5,) or offset.shape != (3,) or not np.isfinite(np.r_[q, offset]).all():
            raise ValueError("Point kinematics requires five finite joints and a finite 3D offset")
        pose = self.fk_sim(q)
        point = pose[:3, 3] + pose[:3, :3] @ offset
        # mj_jac requires the subtree COM quantities as well as kinematics.
        mujoco.mj_comPos(self.model, self.data)
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacobian, None, point,
                      int(self.model.site_bodyid[self.site_id]))
        return point, jacobian[:, self.dofadr].copy()

    def out_of_range(self, q_real_deg: np.ndarray, margin_rad: float = 0.0) -> np.ndarray:
        q = self.mapping.arm_real_to_sim_rad(q_real_deg)
        lo, hi = self.joint_range[:, 0] - margin_rad, self.joint_range[:, 1] + margin_rad
        return (q < lo) | (q > hi)
