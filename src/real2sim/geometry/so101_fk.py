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

    @classmethod
    def from_task_config(cls, task_cfg: dict) -> JointMapping:
        r = task_cfg["robot_so101"]
        return cls(
            sign=tuple(float(s) for s in r["real_to_sim"]["sign"]),
            offset_deg=tuple(float(o) for o in r["real_to_sim"]["offset_deg"]),
            gripper_sim_range_rad=tuple(float(x) for x in r["gripper"]["sim_range_rad"]),
        )

    def arm_real_to_sim_rad(self, q_real_deg: np.ndarray) -> np.ndarray:
        q = np.asarray(q_real_deg, np.float64)[..., :5]
        return np.deg2rad(q * np.asarray(self.sign) + np.asarray(self.offset_deg))

    def gripper_real_to_sim_rad(self, gripper_pct: np.ndarray) -> np.ndarray:
        lo, hi = self.gripper_sim_range_rad
        return lo + np.asarray(gripper_pct, np.float64) / 100.0 * (hi - lo)


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

    def out_of_range(self, q_real_deg: np.ndarray, margin_rad: float = 0.0) -> np.ndarray:
        q = self.mapping.arm_real_to_sim_rad(q_real_deg)
        lo, hi = self.joint_range[:, 0] - margin_rad, self.joint_range[:, 1] + margin_rad
        return (q < lo) | (q > hi)
