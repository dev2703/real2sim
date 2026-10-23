"""Assemble the Panda pick-and-place scene around the unchanged pinned Menagerie ``panda.xml``.

The task objects (table top, black mat, ball, bowl), their placement and the ``iphone`` camera
come from ``configs/task.yaml`` so the Panda sees the same task frame as the SO-101 scene
(plan M5 step 1). ``configs/sim_panda.yaml`` holds simulator-only knobs.

Frames (docs/architecture.md section 4.3):

* ``world``  MuJoCo world, z up. The table top is at ``z = task.table.height``; the Panda base
  body ``link0`` is placed on it at ``(0, 0, table.height)`` with identity orientation, so the
  Panda's forward axis (+x of ``link0``) is the task +x (away from the robot) and task +y is
  left, exactly the SO-101 convention. ``p_task = p_world - (0, 0, table.height)``.
* ``ee``     site ``panda_tcp`` injected into the ``hand`` body at ``+0.1034 m`` along hand z
  (hand origin = flange; fingers mount at 0.0584 and the fingertip pads span hand z
  0.094-0.111, so 0.1034 is the fingertip-pad centre, the standard Franka TCP). The site axes
  are permuted so that the canonical rot6d carries the same meaning as the SO-101
  ``gripperframe``: site x = approach (hand z, toward the fingertips), site y = finger-closing
  hinge equivalent (hand x), site z = finger opening direction (hand y, the finger slide axis).

The upstream ``panda.xml`` is parsed and extended in memory only; the pinned file keeps its SHA.
The upstream ``home`` keyframe is dropped from the assembled model because the free ball joint
changes ``nq``; its arm values are reproduced in ``configs/sim_panda.yaml``. Visually the Panda
differs from the SO-101 in every frame (white/black industrial arm versus a small printed
orange arm): this is a real domain gap for the frozen policy and is reported as such.
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
from real2sim.sim.model import camera_xyaxes as so101_camera_xyaxes
from real2sim.sim.model import task_frame_offset as so101_task_frame_offset

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/sim_panda.yaml"
TCP_SITE = "panda_tcp"
# Site x = hand z, site y = hand x, site z = hand y (rotation of -120 deg about (1,1,1)).
TCP_SITE_QUAT_WXYZ = (0.5, -0.5, -0.5, -0.5)
# Task-frame TCP rotation of a top-down grasp in the shared convention (SO-101 gripperframe at
# grasp: x = -z_task approach, y = +y_task hinge, z = +x_task opening); equals the Menagerie
# home-keyframe hand orientation through the site permutation above.
POINTING_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
BASE_BODY = "link0"
HAND_BODY = "hand"
FINGER_BODIES = ("left_finger", "right_finger")
MIN_WALL_BOXES = 12


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(project_path(path).read_bytes()).hexdigest()


def verify_panda_assets(manifest_path: str | Path, model_path: str | Path) -> dict:
    """Check every manifest row under the Panda directory and return the panda.xml row."""
    manifest_file = project_path(manifest_path)
    manifest = json.loads(manifest_file.read_text())
    model_file = project_path(model_path)
    relative = model_file.relative_to(manifest_file.parent).as_posix()
    rows = [r for r in manifest["files"] if r["path"].startswith(relative.split("/")[0] + "/")]
    if not rows:
        raise ValueError(f"manifest has no rows for {relative}")
    for row in rows:
        asset = manifest_file.parent / row["path"]
        payload = asset.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"asset hash mismatch: {asset}")
    return next(r for r in rows if r["path"] == relative)


def _numbers(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def task_frame_offset(task: dict) -> np.ndarray:
    """``p_world = p_task + offset``: the SAME task frame as the SO-101 scene (the SO-101 base,
    ``base_z_offset`` above the table top, D-014). Shared so the two embodiments cannot drift."""
    return so101_task_frame_offset(task)


def panda_base_world(task: dict) -> np.ndarray:
    """World position of the Panda base body: on the table top under the task origin.

    The SO-101 mounting lift (``base_z_offset``) is a property of that robot's base plate, not of
    the task frame, so the Panda stands on the table and sits ``-base_z_offset`` below the origin.
    """
    frame = task["frames"]["T_task_so101base"]
    if np.any(np.asarray(frame["xyz"], float) != 0) or np.any(np.asarray(frame["rpy"], float) != 0):
        raise ValueError("scene assembly assumes the identity T_task_so101base from task.yaml")
    return np.array([0.0, 0.0, float(task["table"]["height"])])


def bowl_dimensions(task: dict, inner_depth: float | None = None) -> dict:
    """Bowl geometry in metres; ``inner_depth`` overrides task.yaml for the depth sweep.

    The floor thickness ``outer_height - inner_depth`` is kept when the depth is swept.
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
    if not (dims["floor"] > 0 and depth > 0 and dims["wall"] > 0
            and 0 < dims["base_radius"] <= dims["rim_radius"]):
        raise ValueError("invalid bowl dimensions in task.yaml")
    if dims["wall_boxes"] < MIN_WALL_BOXES or bowl["collision"].get("base") != "cylinder":
        raise ValueError("bowl collision must be >= 12 wall boxes over a base cylinder")
    if bowl.get("static") is not True:
        raise ValueError("dynamic bowl requires measured mass/inertia and a new decision")
    return dims


def _add_bowl(world: ET.Element, dims: dict, origin, contacts: dict) -> None:
    """Static bowl: base cylinder plus ``wall_boxes`` tilted boxes forming a truncated cone.

    The inner wall is a straight frustum from ``base_radius`` at the floor to ``rim_radius`` at
    the rim; wall boxes run down to the table so the floor/wall junction has no gap. Geoms of
    one static body do not collide with each other, so the overlaps are harmless.
    """
    body = ET.SubElement(world, "body", name="bowl", pos=_numbers(origin))
    rgba = "0.85 0.86 0.9 1"
    slope = (dims["rim_radius"] - dims["base_radius"]) / dims["depth"]
    tilt = float(np.arctan(slope))  # wall tilt from vertical
    r_low = dims["base_radius"] - slope * dims["floor"]
    wall_len = float(np.hypot(dims["rim_radius"] - r_low, dims["height"]))
    ET.SubElement(body, "geom", name="bowl_base", type="cylinder",
                  size=_numbers([max(r_low, 0.0) + dims["wall"], dims["floor"] / 2]),
                  pos=_numbers([0, 0, dims["floor"] / 2]), rgba=rgba, **contacts)
    n = dims["wall_boxes"]
    half_tangent = (dims["rim_radius"] + dims["wall"]) * np.tan(np.pi / n) * 1.02
    mid_r = (r_low + dims["rim_radius"]) / 2 + dims["wall"] / 2 * np.cos(tilt)
    mid_z = dims["height"] / 2 - dims["wall"] / 2 * np.sin(tilt)
    for i in range(n):
        angle = 2 * np.pi * i / n
        x, y, z, w = (Rotation.from_euler("z", angle) * Rotation.from_euler("y", tilt)).as_quat()
        ET.SubElement(body, "geom", name=f"bowl_wall_{i:02d}", type="box",
                      size=_numbers([dims["wall"] / 2, half_tangent, wall_len / 2]),
                      pos=_numbers([mid_r * np.cos(angle), mid_r * np.sin(angle), mid_z]),
                      quat=_numbers([w, x, y, z]), rgba=rgba, **contacts)


def scene_xml(config: dict, task: dict, *, bowl_inner_depth: float | None = None,
              contacts: dict | None = None) -> str:
    source = project_path(config["model"])
    root = ET.parse(source).getroot()
    root.set("model", "panda_provisional_pickplace")
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(1 / config["physics_hz"]))
    option.set("gravity", "0 0 -9.81")
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)  # nq changes once the ball's free joint is added
    render = config["render"]
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(render["width"]), offheight=str(render["height"]))
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.6 0.6 0.6")
    offset = task_frame_offset(task)
    world = root.find("worldbody")
    base = next(b for b in world.findall("body") if b.get("name") == BASE_BODY)
    base.set("pos", _numbers(panda_base_world(task)))
    if base.get("quat", "1 0 0 0").split() != ["1", "0", "0", "0"]:
        raise ValueError("expected an unrotated upstream base body")
    hand = next(b for b in base.iter("body") if b.get("name") == HAND_BODY)
    ET.SubElement(hand, "site", name=TCP_SITE, pos=_numbers([0, 0, config["tcp_offset_m"]]),
                  quat=_numbers(TCP_SITE_QUAT_WXYZ), size="0.004", group="3",
                  rgba="1 0 0 1")
    ET.SubElement(world, "light", pos=_numbers(offset + [0.3, -0.4, 1.2]), dir="0 0 -1",
                  directional="true")
    camera = task["camera_iphone"]
    cam_pos = np.asarray(camera["approx_position_task"], float) + offset
    cam_at = np.asarray(camera["look_at_task"], float) + offset
    from real2sim.sim.cameras import camera_projection_attributes

    ET.SubElement(world, "camera", name="iphone", pos=_numbers(cam_pos),
                  xyaxes=_numbers(so101_camera_xyaxes(camera, cam_pos, cam_at)),
                  **camera_projection_attributes(camera, config["render"]))
    table = task["table"]
    top = float(table["height"])
    physics = dict(config["contacts"])
    physics.update(contacts or {})
    contact_attrs = dict(friction=_numbers(physics["friction"]), solref=_numbers(physics["solref"]),
                         condim="6")
    half = config["table_half_size"]
    ET.SubElement(world, "geom", name="table", type="box", size=_numbers(half),
                  pos=_numbers([*config["table_center_task_xy"], top - half[2]]),
                  rgba="0.65 0.50 0.34 1", **contact_attrs)
    mat = table["mat"]
    ET.SubElement(world, "geom", name="mat_visual", type="box",
                  size=_numbers([mat["size_xy"][0] / 2, mat["size_xy"][1] / 2, 0.0001]),
                  pos=_numbers([*mat["center_task_xy"], top + 0.0001]), rgba="0.08 0.09 0.1 1",
                  contype="0", conaffinity="0")
    dims = bowl_dimensions(task, bowl_inner_depth)
    _add_bowl(world, dims, [*dims["center_xy"], top], contact_attrs)
    ball = task["ball"]
    start = ball["start_region"]["center_task_xy"]
    from real2sim.sim.ball_support import add_ball_support, spawn_center_above_table

    add_ball_support(world, task, contact_attrs, offset)
    body = ET.SubElement(world, "body", name="ball",
                         pos=_numbers([*start, top + spawn_center_above_table(task, start) + 0.001]))
    ET.SubElement(body, "freejoint", name="ball_free")
    ET.SubElement(body, "geom", name="ball_geom", type="sphere", size=str(ball["radius"]),
                  mass=str(ball["mass"]), rgba="0.72 0.9 0.06 1", **contact_attrs)
    from real2sim.sim.cameras import add_link_cameras

    add_link_cameras(root, task, "panda", config["render"])
    return ET.tostring(root, encoding="unicode")


def load_scene(config_path: str | Path = DEFAULT_CONFIG, *, bowl_inner_depth: float | None = None,
               contacts: dict | None = None) -> tuple[mujoco.MjModel, dict, dict]:
    config = load_yaml(config_path)
    task = load_yaml(project_path(config["task_config"]))
    row = verify_panda_assets(config["manifest"], config["model"])
    if config.get("model_sha256") and config["model_sha256"] != row["sha256"]:
        raise ValueError("configs/sim_panda.yaml model_sha256 disagrees with the manifest")
    xml = scene_xml(config, task, bowl_inner_depth=bowl_inner_depth, contacts=contacts)
    model = mujoco.MjModel.from_xml_string(xml)
    if task["ball"].get("inertia") == "thin_shell":
        geom = model.geom("ball_geom").id
        body = model.geom_bodyid[geom]
        m, r = float(task["ball"]["mass"]), float(task["ball"]["radius"])
        model.body_inertia[body] = 2 / 3 * m * r * r  # hollow tennis ball
    return model, config, task
