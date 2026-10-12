"""Assemble an explicitly provisional task around the unchanged pinned robot model."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from real2sim.config import load_yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/sim_so101.yaml"


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def verify_assets(manifest_path: str | Path) -> dict:
    path = project_path(manifest_path)
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        asset = path.parent / row["path"]
        payload = asset.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"asset hash mismatch: {asset}")
    return manifest


def _numbers(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def _camera_axes(position, target) -> np.ndarray:
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(z, x)])


def scene_xml(config: dict, task: dict) -> str:
    source = project_path(config["model"])
    root = ET.parse(source).getroot()
    root.set("model", "so101_provisional_pickplace_diagnostic")
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    option = root.find("option")
    option.set("timestep", str(1 / config["physics_hz"]))
    option.set("gravity", "0 0 -9.81")
    camera = task["camera"]
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(camera["width"]),
                  offheight=str(camera["height"]))
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.6 0.6 0.6")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 1.5", dir="0 0 -1", directional="true")
    ET.SubElement(world, "camera", name=camera["name"], pos=_numbers(camera["position"]),
                  xyaxes=_numbers(_camera_axes(camera["position"], camera["look_at"])),
                  fovy=str(camera["fovy_degrees"]))
    table = task["table"]
    top = table["top_z"]
    contacts = dict(friction=_numbers(task["contacts"]["friction"]),
                    solref=_numbers(task["contacts"]["solref"]), condim="6")
    ET.SubElement(world, "geom", name="table", type="box", size=_numbers(table["half_size"]),
                  pos=_numbers([0, 0, top - table["half_size"][2]]),
                  rgba="0.65 0.50 0.34 1", **contacts)
    ET.SubElement(world, "geom", name="mat_visual", type="box", size="0.29 0.26 0.0001",
                  pos=_numbers([0.22, 0, top + 0.0001]), rgba="0.08 0.09 0.1 1",
                  contype="0", conaffinity="0")
    bowl = task["bowl"]
    radius, wall, height = bowl["inner_radius"], bowl["wall_thickness"], bowl["outer_height"]
    bottom = height - bowl["inner_depth"]
    if not 0 < bottom < height or radius <= 0 or wall <= 0 or bowl["wall_segments"] < 16:
        raise ValueError("invalid bowl dimensions")
    if bowl["fixed"] is not True:
        raise ValueError("dynamic bowl requires measured mass/inertia and a new decision")
    body = ET.SubElement(world, "body", name="bowl", pos=_numbers([*bowl["center_xy"], top]))
    ET.SubElement(body, "geom", name="bowl_base", type="cylinder",
                  size=_numbers([radius + wall, bottom / 2]),
                  pos=_numbers([0, 0, bottom / 2]), rgba="0.85 0.86 0.9 1", **contacts)
    n = bowl["wall_segments"]
    tangent_half = (radius + wall) * np.tan(np.pi / n) * 1.01
    for i in range(n):
        angle = 2 * np.pi * i / n
        ET.SubElement(body, "geom", name=f"bowl_wall_{i:02d}", type="box",
                      size=_numbers([wall / 2, tangent_half, height / 2]),
                      pos=_numbers([(radius + wall / 2) * np.cos(angle),
                                    (radius + wall / 2) * np.sin(angle), height / 2]),
                      quat=_numbers([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
                      rgba="0.85 0.86 0.9 1", **contacts)
    ball = task["ball"]
    body = ET.SubElement(world, "body", name="ball",
                         pos=_numbers([*ball["start_xy"], top + ball["radius"] + 0.001]))
    ET.SubElement(body, "freejoint", name="ball_free")
    ET.SubElement(body, "geom", name="ball_geom", type="sphere", size=str(ball["radius"]),
                  mass=str(ball["mass"]), rgba="0.72 0.9 0.06 1", **contacts)
    return ET.tostring(root, encoding="unicode")


def load_scene(config_path: str | Path = DEFAULT_CONFIG) -> tuple[mujoco.MjModel, dict, dict]:
    config = load_yaml(config_path)
    task = load_yaml(project_path(config["task_config"]))
    model = mujoco.MjModel.from_xml_string(scene_xml(config, task))
    return model, config, task
