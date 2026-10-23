"""Assemble the provisional pick-and-place scene around the unchanged pinned robot model.

Single source of truth for geometry and physical parameters is ``configs/task.yaml`` (provenance
labels live there and in ``docs/calibration.md``). ``configs/sim_so101.yaml`` holds only the
simulator-specific knobs: physics rate, control interface, contact softness, render size, IK.

Frames (docs/architecture.md section 4.3):

* ``world``  MuJoCo world, z up. The table top is at ``z = task.table.height`` and the robot base
  body is placed at ``(0, 0, table.height + base_z_offset)`` (offset 0 until D-014 is approved).
* ``task``   origin at the robot base (``T_task_so101base`` is identity in task.yaml), so
  ``p_task = p_world - (0, 0, table.height + base_z_offset)``. Everything policy-relevant is
  expressed here.

The upstream ``so101.xml`` is parsed and extended in memory only; the pinned file keeps its SHA.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from real2sim.config import REPO_ROOT, load_yaml, project_path

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/sim_so101.yaml"
MIN_WALL_BOXES = 12




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
    """Roll-free look-at ``xyaxes`` (x horizontal); fallback when no fitted axes exist."""
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(z, x)])


def camera_xyaxes(camera: dict, cam_pos, cam_at) -> np.ndarray:
    """MuJoCo ``xyaxes`` of the iphone camera.

    The PnP fit (``real2sim.sim.camera_fit``) writes the full axes to task.yaml
    ``camera_iphone.xyaxes_task`` because the roll-free look-at form drops the roll about the
    optical axis (results/audit/camera_fit/report.json: 61.6 px RMS roll-free vs 17.2 px with
    the fitted axes). The fitted axes are used whenever present; the look-at form is only a
    fallback for a task.yaml without a fit. The task frame is a pure translation of the world
    frame, so the axes need no re-expression.
    """
    if (xyaxes := camera.get("xyaxes_task")) is not None:
        axes = np.asarray(xyaxes, dtype=float)
        if axes.shape != (6,) or not np.all(np.isfinite(axes)):
            raise ValueError("camera_iphone.xyaxes_task must be six finite numbers")
        x, y = axes[:3], axes[3:]
        if (abs(np.linalg.norm(x) - 1) > 1e-3 or abs(np.linalg.norm(y) - 1) > 1e-3
                or abs(float(x @ y)) > 1e-3):
            raise ValueError("camera_iphone.xyaxes_task must be orthonormal unit axes")
        look_dir = np.asarray(cam_at, float) - np.asarray(cam_pos, float)
        if float(np.cross(x, y) @ look_dir) >= 0:  # MuJoCo cameras look along -z
            raise ValueError("camera_iphone.xyaxes_task does not look towards look_at_task")
        return axes
    return _camera_axes(cam_pos, cam_at)


def base_z_offset(task: dict) -> float:
    """Height of the robot base origin above the table top (m), ``robot_so101.base_z_offset``.

    Absent or zero until docs/decisions.md D-014 is approved (the key is read here so that the
    approved value is a one-line task.yaml change). The task frame stays equal to the base frame,
    so with a non-zero offset the table top lies at ``z_task = -base_z_offset``.
    """
    value = float(task.get("robot_so101", {}).get("base_z_offset", 0.0))
    if not np.isfinite(value) or value < 0 or value > 0.05:
        raise ValueError("robot_so101.base_z_offset must be a finite height in [0, 0.05] m")
    return value


def task_frame_offset(task: dict) -> np.ndarray:
    """``p_world = p_task + offset``; task.yaml fixes T_task_so101base = identity.

    The offset is the world position of the robot base origin: table top plus
    :func:`base_z_offset`.
    """
    frame = task["frames"]["T_task_so101base"]
    if np.any(np.asarray(frame["xyz"], float) != 0) or np.any(np.asarray(frame["rpy"], float) != 0):
        raise ValueError("scene assembly assumes the identity T_task_so101base from task.yaml")
    return np.array([0.0, 0.0, float(task["table"]["height"]) + base_z_offset(task)])


def bowl_dimensions(task: dict, inner_depth: float | None = None) -> dict:
    """Bowl geometry in metres; ``inner_depth`` overrides task.yaml for the depth sweep.

    The floor thickness ``outer_height - inner_depth`` of the operator-supplied bowl is kept
    constant when the depth is swept, so a deeper bowl is also taller.
    """
    bowl = task["bowl"]
    floor = float(bowl["outer_height"]) - float(bowl["inner_depth"])
    depth = float(bowl["inner_depth"] if inner_depth is None else inner_depth)
    dims = {
        "rim_radius": float(bowl["inner_rim_radius"]),
        "base_radius": float(bowl["base_diameter"]) / 2,
        "wall": float(bowl["wall_thickness"]),
        "floor": floor,
        "depth": depth,
        "height": floor + depth,
        "wall_boxes": int(bowl["collision"]["wall_boxes"]),
        "center_xy": [float(v) for v in bowl["center_task_xy"]],
    }
    if not (0 < dims["floor"] and 0 < depth and 0 < dims["wall"]
            and 0 < dims["base_radius"] <= dims["rim_radius"]):
        raise ValueError("invalid bowl dimensions in task.yaml")
    if dims["wall_boxes"] < MIN_WALL_BOXES or bowl["collision"].get("base") != "cylinder":
        raise ValueError("bowl collision must be >= 12 wall boxes over a base cylinder")
    if bowl.get("static") is not True:
        raise ValueError("dynamic bowl requires measured mass/inertia and a new decision")
    return dims


def _add_bowl(world: ET.Element, dims: dict, origin, contacts: dict, rgba: str = "0.85 0.86 0.9 1") -> None:
    """Static bowl: base cylinder plus ``wall_boxes`` tilted boxes forming a truncated cone.

    The inner wall runs from ``base_radius`` at the floor to ``rim_radius`` at the rim, following
    the operator-supplied base diameter and rim radius (a straight-walled frustum, not a measured
    profile). Geoms in one static body never collide with each other, so overlaps are harmless.
    """
    body = ET.SubElement(world, "body", name="bowl", pos=_numbers(origin))
    slope = (dims["rim_radius"] - dims["base_radius"]) / dims["depth"]  # dr/dz of the inner wall
    phi = float(np.arctan(slope))  # tilt from vertical
    # Extend the wall segment down to the table so no gap opens at the floor/wall junction.
    r_low = dims["base_radius"] - slope * dims["floor"]
    length = float(np.hypot(dims["rim_radius"] - r_low, dims["height"]))
    ET.SubElement(body, "geom", name="bowl_base", type="cylinder",
                  size=_numbers([max(r_low, 0.0) + dims["wall"], dims["floor"] / 2]),
                  pos=_numbers([0, 0, dims["floor"] / 2]), rgba=rgba, **contacts)
    n = dims["wall_boxes"]
    tangent_half = (dims["rim_radius"] + dims["wall"]) * np.tan(np.pi / n) * 1.02
    mid_r = (r_low + dims["rim_radius"]) / 2 + dims["wall"] / 2 * np.cos(phi)
    mid_z = dims["height"] / 2 - dims["wall"] / 2 * np.sin(phi)
    for i in range(n):
        angle = 2 * np.pi * i / n
        rot = Rotation.from_euler("z", angle) * Rotation.from_euler("y", phi)
        x, y, z, w = rot.as_quat()
        ET.SubElement(body, "geom", name=f"bowl_wall_{i:02d}", type="box",
                      size=_numbers([dims["wall"] / 2, tangent_half, length / 2]),
                      pos=_numbers([mid_r * np.cos(angle), mid_r * np.sin(angle), mid_z]),
                      quat=_numbers([w, x, y, z]), rgba=rgba, **contacts)


UPSTREAM_PLASTIC_RGBA = "1 0.82 0.12 1"  # Menagerie so101.xml plastic material colour
# Pre-2026-10-13 appearance (kept as the default when sim config has no ``appearance`` section).
LEGACY_APPEARANCE = {
    "robot_plastic_rgba": None,
    "table_rgba": [0.65, 0.50, 0.34, 1],
    "mat_rgba": [0.08, 0.09, 0.1, 1],
    "bowl_rgba": [0.85, 0.86, 0.9, 1],
    "ball_rgba": [0.72, 0.9, 0.06, 1],
    "background_boxes": [],
    "textures": {},
    "lighting": {"headlight_ambient": [0.3, 0.3, 0.3], "headlight_diffuse": [0.6, 0.6, 0.6],
                 "headlight_specular": [0.3, 0.3, 0.3], "sun_diffuse": [0.8, 0.8, 0.8], "shadows": True},
}


def _appearance(config: dict) -> dict:
    """Visual-only scene appearance (colours, lighting, scenery); never a physics parameter."""
    look = {**LEGACY_APPEARANCE, **(config.get("appearance") or {})}
    look["lighting"] = {**LEGACY_APPEARANCE["lighting"], **(look.get("lighting") or {})}
    look["textures"] = dict(look.get("textures") or {})
    for key in ("table_rgba", "mat_rgba", "bowl_rgba", "ball_rgba"):
        if len(look[key]) != 4 or not all(0 <= float(v) <= 1 for v in look[key]):
            raise ValueError(f"appearance.{key} must be four values in [0, 1]")
    if look["robot_plastic_rgba"] is not None and len(look["robot_plastic_rgba"]) != 4:
        raise ValueError("appearance.robot_plastic_rgba must be four values or null")
    for box in look["background_boxes"]:
        if set(box) - {"name", "pos_task", "half_size", "rgba"} or len(box["half_size"]) != 3:
            raise ValueError(f"invalid appearance.background_boxes entry: {box}")
    return look


def _textured(root: ET.Element, look: dict, name: str, rgba) -> dict:
    """Return geom attributes for ``name``: a speckled builtin texture if configured, else plain rgba.

    Visual only. The texture is a flat colour with random marks (MuJoCo builtin), which gives the
    camera image the fine-grained detail that real surfaces have and flat shading lacks.
    """
    spec = look["textures"].get(name)
    if not spec:
        return {"rgba": _numbers(rgba)}
    asset = root.find("asset")
    rgb = [float(v) for v in rgba[:3]]
    ET.SubElement(asset, "texture", name=f"tex_{name}", type="2d", builtin="flat", rgb1=_numbers(rgb),
                  rgb2=_numbers(rgb), mark="random", markrgb=_numbers(spec.get("markrgb", [0, 0, 0])),
                  random=str(float(spec.get("random", 0.02))), width="512", height="512")
    ET.SubElement(asset, "material", name=f"mat_{name}", texture=f"tex_{name}",
                  texrepeat=_numbers(spec.get("repeat", [1, 1])), texuniform="true", rgba="1 1 1 1")
    return {"material": f"mat_{name}"}


def scene_xml(config: dict, task: dict, *, bowl_inner_depth: float | None = None,
              contacts: dict | None = None) -> str:
    source = project_path(config["model"])
    root = ET.parse(source).getroot()
    root.set("model", "so101_provisional_pickplace")
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    option = root.find("option")
    option.set("timestep", str(1 / config["physics_hz"]))
    option.set("gravity", "0 0 -9.81")
    offset = task_frame_offset(task)
    camera = task["camera_iphone"]
    render = config["render"]
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(render["width"]), offheight=str(render["height"]))
    look = _appearance(config)
    light = look["lighting"]
    ET.SubElement(visual, "headlight", ambient=_numbers(light["headlight_ambient"]),
                  diffuse=_numbers(light["headlight_diffuse"]), specular=_numbers(light["headlight_specular"]))
    if look["robot_plastic_rgba"] is not None:
        # Visual only: recolour the upstream plastic materials (yellow in Menagerie, white on the
        # recorded robot). The asset block lives in the parsed tree; the pinned so101.xml file and
        # every physics field are untouched (tests/integration/test_so101_scene.py).
        for material in root.find("asset").findall("material"):
            if material.get("rgba", "").split() == UPSTREAM_PLASTIC_RGBA.split():
                material.set("rgba", _numbers(look["robot_plastic_rgba"]))
    world = root.find("worldbody")
    base = next(b for b in world.findall("body") if b.get("name") == "base")
    base.set("pos", _numbers(offset))  # robot base on the table top; task frame origin
    ET.SubElement(world, "light", pos=_numbers(offset + [0.3, -0.4, 1.2]), dir="0 0 -1",
                  directional="true", castshadow="true" if light["shadows"] else "false",
                  diffuse=_numbers(light["sun_diffuse"]))
    cam_pos = np.asarray(camera["approx_position_task"], float) + offset
    cam_at = np.asarray(camera["look_at_task"], float) + offset
    from real2sim.sim.cameras import camera_projection_attributes

    ET.SubElement(world, "camera", name="iphone", pos=_numbers(cam_pos),
                  xyaxes=_numbers(camera_xyaxes(camera, cam_pos, cam_at)),
                  **camera_projection_attributes(camera, render))
    table = task["table"]
    top = float(table["height"])
    physics = dict(config["contacts"])
    physics.update(contacts or {})
    contact_attrs = dict(friction=_numbers(physics["friction"]), solref=_numbers(physics["solref"]),
                         condim="6")
    half = config["table_half_size"]
    table_rgba = list(look["table_rgba"])
    # alpha 0 means "hide the collision box" (the desk is drawn by appearance.background_boxes). Hiding
    # by geom group 3 (not rendered by the default MjvOption) avoids MuJoCo's transparency pass, which
    # produced 1/255 pixel flicker between identical resets. Physics is unaffected by group.
    table_extra = {"group": "3"} if float(table_rgba[3]) == 0.0 else {}
    if table_extra:
        table_rgba[3] = 1.0
    ET.SubElement(world, "geom", name="table", type="box", size=_numbers(half),
                  pos=_numbers([*config["table_center_task_xy"], top - half[2]]),
                  rgba=_numbers(table_rgba), **table_extra, **contact_attrs)
    mat = table["mat"]
    mat_yaw = np.deg2rad(float(mat.get("yaw_deg", 0.0)))
    if not np.isfinite(mat_yaw):
        raise ValueError("table.mat.yaw_deg must be finite")
    ET.SubElement(world, "geom", name="mat_visual", type="box",
                  size=_numbers([mat["size_xy"][0] / 2, mat["size_xy"][1] / 2, 0.0001]),
                  pos=_numbers([*mat["center_task_xy"], top + 0.0001]), contype="0", conaffinity="0",
                  quat=_numbers([np.cos(mat_yaw / 2), 0, 0, np.sin(mat_yaw / 2)]),
                  **_textured(root, look, "mat", look["mat_rgba"]))
    for i, box in enumerate(look["background_boxes"]):
        # Visual-only scenery (desk extension, wall, furniture) seen by the camera; no collisions.
        ET.SubElement(world, "geom", name=f"scenery_{i:02d}_{box.get('name', 'box')}", type="box",
                      size=_numbers(box["half_size"]), pos=_numbers(np.asarray(box["pos_task"], float) + offset),
                      contype="0", conaffinity="0", group="2",
                      **_textured(root, look, box.get("name", f"box{i}"), box["rgba"]))
    dims = bowl_dimensions(task, bowl_inner_depth)
    _add_bowl(world, dims, [*dims["center_xy"], top], contact_attrs, rgba=_numbers(look["bowl_rgba"]))
    ball = task["ball"]
    start = ball["start_region"]["center_task_xy"]
    from real2sim.sim.ball_support import add_ball_support, spawn_center_above_table

    add_ball_support(world, task, contact_attrs, offset)
    body = ET.SubElement(world, "body", name="ball",
                         pos=_numbers([*start, top + spawn_center_above_table(task, start) + 0.001]))
    ET.SubElement(body, "freejoint", name="ball_free")
    ET.SubElement(body, "geom", name="ball_geom", type="sphere", size=str(ball["radius"]),
                  mass=str(ball["mass"]), rgba=_numbers(look["ball_rgba"]), **contact_attrs)
    from real2sim.sim.cameras import add_link_cameras

    add_link_cameras(root, task, "so101", render)
    return ET.tostring(root, encoding="unicode")


def load_scene(config_path: str | Path = DEFAULT_CONFIG, *, bowl_inner_depth: float | None = None,
               contacts: dict | None = None) -> tuple[mujoco.MjModel, dict, dict]:
    config = load_yaml(config_path)
    task = load_yaml(project_path(config["task_config"]))
    xml = scene_xml(config, task, bowl_inner_depth=bowl_inner_depth, contacts=contacts)
    model = mujoco.MjModel.from_xml_string(xml)
    if task["ball"].get("inertia") == "thin_shell":
        # Tennis balls are hollow: I = 2/3 m r^2 (docs/calibration.md), not the solid 2/5.
        geom = model.geom("ball_geom").id
        body = model.geom_bodyid[geom]
        m, r = float(task["ball"]["mass"]), float(task["ball"]["radius"])
        model.body_inertia[body] = 2 / 3 * m * r * r
    return model, config, task
