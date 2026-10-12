"""FK cross-check from the upstream XML's rigid transforms, independent of mj_forward.

This validates the native model chain only. It does not calibrate real motor signs/zeros,
verify an independent URDF, or provide the deferred Placo inverse-kinematics adapter.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

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
