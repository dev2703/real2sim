"""FK cross-check from the upstream XML's rigid transforms, independent of mj_forward.

This validates the native model chain only. It does not calibrate real motor signs/zeros,
verify an independent URDF, or provide the deferred Placo inverse-kinematics adapter.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _pose(element: ET.Element) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    wxyz = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    result[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    if any(key in element.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("this independent FK supports the pinned quaternion-based model only")
    return result


def xml_forward_kinematics(model_path: str | Path, joints: dict[str, float],
                          site_name: str = "gripperframe") -> np.ndarray:
    root = ET.parse(model_path).getroot()

    def visit(body, parent):
        transform = parent @ _pose(body)
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") != "hinge":
                raise ValueError("expected a hinge joint in robot chain")
            pivot = np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
            axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
            axis /= np.linalg.norm(axis)
            angle = joints[joint.attrib["name"]] - float(joint.get("ref", "0"))
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            motion[:3, 3] = pivot - motion[:3, :3] @ pivot
            transform = transform @ motion
        for site in body.findall("site"):
            if site.get("name") == site_name:
                return transform @ _pose(site)
        for child in body.findall("body"):
            found = visit(child, transform)
            if found is not None:
                return found
        return None

    for body in root.find("worldbody").findall("body"):
        result = visit(body, np.eye(4))
        if result is not None:
            return result
    raise ValueError(f"site {site_name!r} not found")


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    accepted: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    reason: str


class NativeKinematics:
    """Position-prioritised diagnostic IK for the five arm joints, separate gripper hinge.

    Independent MjData keeps solver probes from modifying the running environment. Failed
    targets return the original seed and must never be applied as accepted control targets.
    Poses are ``T_base_ee`` (robot base = task frame); the base body is a static translation
    of the world frame in the assembled scene.
    """

    def __init__(self, model: mujoco.MjModel, config: dict):
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config["ik"]
        # Optional numerical convergence is separate from hard residual acceptance.
        # Legacy configs retain their exact early-stop behavior for frozen benchmarks.
        self.polish = any(k in self.config for k in
                          ("position_convergence_m", "orientation_convergence_rad"))
        self.position_convergence = float(self.config.get(
            "position_convergence_m", self.config["position_tolerance_m"]))
        self.orientation_convergence = float(self.config.get(
            "orientation_convergence_rad", self.config["orientation_tolerance_rad"]))
        for value, limit in ((self.position_convergence, self.config["position_tolerance_m"]),
                             (self.orientation_convergence, self.config["orientation_tolerance_rad"])):
            if not np.isfinite(value) or not 0 < value <= limit:
                raise ValueError("IK convergence must be positive and no looser than acceptance")
        ids = np.array([model.joint(name).id for name in config["joint_names"]])
        self.qadr = model.jnt_qposadr[ids]
        self.vadr = model.jnt_dofadr[ids]
        self.limits = model.jnt_range[ids].copy()
        self.site = model.site(config["tool_site"]).id
        base = model.body("base").id
        if not np.allclose(model.body_quat[base], [1, 0, 0, 0]):
            raise ValueError("base-frame kinematics assume an unrotated robot base")
        self.base_pos = model.body_pos[base].copy()

    def forward(self, qpos) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError("FK expects six finite native joint radians")
        self.data.qpos[self.qadr] = qpos
        mujoco.mj_forward(self.model, self.data)
        pose = np.eye(4)
        pose[:3, 3] = self.data.site_xpos[self.site] - self.base_pos
        pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        return pose

    def _collision(self) -> bool:
        # Contact with a ball is expected during grasp; jaws may contact each other at closure.
        tolerated = {"ball", "gripper", "moving_jaw_so101_v1"}
        for contact in self.data.contact:
            names = {self.model.body(self.model.geom_bodyid[g]).name
                     for g in (contact.geom1, contact.geom2)}
            if "ball" in names or names <= tolerated:
                continue
            if contact.dist < -self.config["penetration_tolerance_m"]:
                return True
        return False

    def solve(self, target: np.ndarray, seed: np.ndarray, *, gripper: float | None = None) -> IKResult:
        target, seed = np.asarray(target, dtype=float), np.asarray(seed, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK target must be a finite 4x4 transform")
        rot = target[:3, :3]
        if (not np.allclose(target[3], [0, 0, 0, 1])
                or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rot), 1, atol=1e-6)):
            raise ValueError("IK target must contain a proper rotation and homogeneous last row")
        self.forward(seed)
        if np.any(seed < self.limits[:, 0]) or np.any(seed > self.limits[:, 1]):
            raise ValueError("IK seed exceeds native joint limits")
        grip = float(seed[5] if gripper is None else gripper)
        if not np.isfinite(grip) or not self.limits[5, 0] <= grip <= self.limits[5, 1]:
            raise ValueError("gripper target exceeds native joint limits")
        q = seed.copy()
        q[5] = grip
        cfg = self.config
        outside = np.any(target[:3, 3] < cfg["workspace_min"]) or np.any(
            target[:3, 3] > cfg["workspace_max"])
        weight = cfg["orientation_weight_m_per_rad"]

        def error(pose):
            translation = target[:3, 3] - pose[:3, 3]
            rotation = Rotation.from_matrix(rot @ pose[:3, :3].T).as_rotvec()
            return translation, rotation

        for iteration in range(cfg["max_iterations"] + 1):
            pose = self.forward(q)
            dp, dr = error(pose)
            pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if outside:
                return IKResult(seed.copy(), False, pe, re, iteration, "outside_workspace")
            if pe <= self.position_convergence and re <= self.orientation_convergence:
                if self._collision():
                    return IKResult(seed.copy(), False, pe, re, iteration, "collision")
                return IKResult(q.copy(), True, pe, re, iteration, "accepted")
            if iteration == cfg["max_iterations"]:
                break
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.vadr[:5]], weight * jr[:, self.vadr[:5]]])
            residual = np.concatenate([dp, weight * dr])
            delta = np.linalg.solve(jac.T @ jac + cfg["damping"] ** 2 * np.eye(5),
                                    jac.T @ residual)
            delta *= min(1.0, cfg["max_joint_step_rad"] / max(np.max(np.abs(delta)), 1e-12))
            improved = False
            for scale in (1, 0.5, 0.25, 0.1):
                candidate = q.copy()
                candidate[:5] = np.clip(q[:5] + scale * delta, self.limits[:5, 0], self.limits[:5, 1])
                cp, cr = error(self.forward(candidate))
                if np.linalg.norm(np.r_[cp, weight * cr]) < np.linalg.norm(residual):
                    q, improved = candidate, True
                    break
            if not improved:
                break
        # Residual refers to the final candidate, never to a silently applied fallback target.
        dp, dr = error(self.forward(q))
        pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        # A 5-DOF arm may not reach numerical convergence in all six pose dimensions.
        # Accept a polished result only under the original residual AND collision limits.
        if self.polish and pe <= cfg["position_tolerance_m"] and re <= cfg["orientation_tolerance_rad"]:
            if self._collision():
                return IKResult(seed.copy(), False, pe, re, iteration, "collision")
            return IKResult(q.copy(), True, pe, re, iteration, "accepted")
        return IKResult(seed.copy(), False, float(np.linalg.norm(dp)), float(np.linalg.norm(dr)),
                        iteration, "residual_rejected")
