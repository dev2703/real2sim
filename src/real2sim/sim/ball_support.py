"""Optional rigid box proxy for the visible material supporting the recorded ball.

Nothing is added unless the task explicitly declares ball.support. This represents
geometry only; cloth compliance and dimensions require separate physical verification.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def support_spec(task):
    value = task["ball"].get("support")
    if value is None:
        return None
    if value.get("type") != "box":
        raise ValueError("ball.support.type must be box")
    height = float(value["height"])
    size, center = np.asarray(value["size_xy"], float), np.asarray(value["center_task_xy"], float)
    yaw = np.deg2rad(float(value.get("yaw_deg", 0)))
    color = np.asarray(value.get("rgba", [.92, .92, .90, 1.]), float)
    if (size.shape != (2,) or center.shape != (2,) or color.shape != (4,)
            or not np.isfinite(np.r_[height, size, center, yaw, color]).all()
            or height < 0 or np.any(size <= 0) or np.any(color < 0) or np.any(color > 1)):
        raise ValueError("ball support requires finite nonnegative height, positive footprint and valid color")
    return {"height": height, "size_xy": size, "center_task_xy": center, "yaw": yaw, "rgba": color}


def spawn_center_above_table(task, ball_xy, support_xy=None):
    """Minimum sphere height clearing the table and support, including its edges."""
    spec = support_spec(task)
    radius = float(task["ball"]["radius"])
    if spec is None or spec["height"] == 0:
        if support_xy is not None:
            raise ValueError("ball_support_xy requires an enabled ball.support")
        return radius
    ball = np.asarray(ball_xy, float)
    center = spec["center_task_xy"] if support_xy is None else np.asarray(support_xy, float)
    if ball.shape != (2,) or center.shape != (2,) or not np.isfinite(np.r_[ball, center]).all():
        raise ValueError("ball and support centers must be finite two-vectors")
    c, s = np.cos(spec["yaw"]), np.sin(spec["yaw"])
    local = np.array([[c, s], [-s, c]]) @ (ball - center)
    distance = np.linalg.norm(np.maximum(np.abs(local) - spec["size_xy"] / 2, 0))
    if distance >= radius:
        return radius
    return max(radius, float(spec["height"] + np.sqrt(radius**2 - distance**2)))


def add_ball_support(world, task, contacts, offset):
    spec = support_spec(task)
    if spec is None or spec["height"] == 0:
        return
    def numbers(values):
        return " ".join(f"{float(v):.12g}" for v in values)
    ET.SubElement(world, "geom", name="ball_support", type="box",
                  size=numbers([*(spec["size_xy"] / 2), spec["height"] / 2]),
                  pos=numbers([*(spec["center_task_xy"] + np.asarray(offset)[:2]),
                               task["table"]["height"] + spec["height"] / 2]),
                  quat=numbers([np.cos(spec["yaw"] / 2), 0, 0, np.sin(spec["yaw"] / 2)]),
                  rgba=numbers(spec["rgba"]), **contacts)


def reset_ball_support(env, ball_xy, support_xy=None):
    """Reset fixed support location independently of ball XY; return spawn height."""
    spec = support_spec(env.task)
    height = spawn_center_above_table(env.task, ball_xy, support_xy)
    if spec is not None and spec["height"] > 0:
        center = spec["center_task_xy"] if support_xy is None else np.asarray(support_xy, float)
        env.model.geom_pos[env.model.geom("ball_support").id, :2] = center + env.offset[:2]
    return height


def support_contact_metrics(model, data):
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_support")
    if geom < 0:
        return {}
    ball = model.geom("ball_geom").id
    count, depth, touching = 0, 0., False
    for contact in data.contact[:data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if geom not in pair:
            continue
        if ball in pair:
            touching = True
        else:
            count += 1
            depth = max(depth, -float(contact.dist))
    return {"ball_touching_support": touching, "robot_support_contacts": count,
            "robot_support_penetration_m": depth}
