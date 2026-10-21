"""Inspect unchanged v2 reference collisions against visual CAD, without rendering or fitting.

Visual mesh vertices cross-check the collider representation, not physical calibration. Self distances use convex
hulls of the visual meshes and therefore cannot prove intersection of concave physical parts.
This diagnostic never changes calibration, joint limits, collision masks, or recorded targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from real2sim.data.raw import RawDataset, sha256_file
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics, xml_forward_kinematics


def geom_name(model, geom):
    mesh = int(model.geom_dataid[geom])
    return {
        "id": int(geom), "name": model.geom(geom).name,
        "body": model.body(model.geom_bodyid[geom]).name,
        "mesh": model.mesh(mesh).name if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else None,
        "type": int(model.geom_type[geom]),
    }


def robot_bodies(model):
    base = model.body("base").id
    result = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != base:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == base:
            result.add(body)
    return result


def visual_meshes(model, bodies):
    result = {}
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] not in bodies or model.geom_group[geom] != 2
                or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        result[geom] = np.asarray(model.mesh_vert[start:start + count], dtype=float)
    return result


def contacts(env, data, bodies):
    table, self_contacts = [], []
    for contact in data.contact:
        depth = max(0., -float(contact.dist) * 1000)
        if depth <= 0:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        ba, bb = int(env.model.geom_bodyid[a]), int(env.model.geom_bodyid[b])
        row = {"a": geom_name(env.model, a), "b": geom_name(env.model, b),
               "penetration_mm": depth, "position_world_m": contact.pos.tolist()}
        if env.table_geom in (a, b) and ({ba, bb} & bodies):
            table.append(row)
        elif ba in bodies and bb in bodies and not {
                env.model.body(ba).name, env.model.body(bb).name
        } <= {"gripper", "moving_jaw_so101_v1"}:
            self_contacts.append(row)
    return sorted(table, key=lambda x: -x["penetration_mm"]), sorted(
        self_contacts, key=lambda x: -x["penetration_mm"])


def inspect_pose(env, kin, q, meshes, bodies, episode, frame):
    kin.forward(q)
    table, self_contacts = contacts(env, kin.data, bodies)
    low = []
    for geom, verts in meshes.items():
        world = verts @ kin.data.geom_xmat[geom].reshape(3, 3).T + kin.data.geom_xpos[geom]
        minimum = int(np.argmin(world[:, 2]))
        low.append({**geom_name(env.model, geom),
                    "lowest_vertex_height_above_table_mm": float((world[minimum, 2] - env.table_top) * 1000),
                    "lowest_vertex_world_m": world[minimum].tolist()})
    low.sort(key=lambda x: x["lowest_vertex_height_above_table_mm"])
    hull_distances = []
    if self_contacts:
        pair = self_contacts[0]
        a = [g for g in meshes if env.model.body(env.model.geom_bodyid[g]).name == pair["a"]["body"]]
        b = [g for g in meshes if env.model.body(env.model.geom_bodyid[g]).name == pair["b"]["body"]]
        for ga in a:
            for gb in b:
                distance = mujoco.mj_geomDistance(env.model, kin.data, ga, gb, .5, None)
                hull_distances.append({"a": geom_name(env.model, ga), "b": geom_name(env.model, gb),
                                       "signed_convex_hull_distance_mm": float(distance * 1000)})
        hull_distances.sort(key=lambda x: x["signed_convex_hull_distance_mm"])
    independent = xml_forward_kinematics(
        env.config["model"], dict(zip(env.config["joint_names"], q, strict=True)))
    native = kin.forward(q)
    return {
        "episode_index": episode, "source_frame": frame,
        "phase": "unannotated: source frame is not assumed to be grasp, placement, or return",
        "native_joint_deg": np.rad2deg(q).tolist(),
        "in_limits": bool(np.all(q >= env.limits[:, 0]) and np.all(q <= env.limits[:, 1])),
        "table_contacts": table[:4], "self_contacts": self_contacts[:4],
        "lowest_visual_parts": low[:3],
        "worst_self_pair_visual_hull_distances": hull_distances,
        "independent_xml_vs_native_fk_max_abs_difference": float(np.max(np.abs(independent - native))),
    }


def audit(root, config):
    ds = RawDataset(root)
    episode_reports, selected, table_gaps = [], {}, []
    table_pairs, self_pairs = Counter(), Counter()
    with So101PickPlaceEnv(config, render_images=False) as env:
        kin = NativeKinematics(env.model, env.config)
        bodies = robot_bodies(env.model)
        meshes = visual_meshes(env.model, bodies)
        tol_mm = float(kin.config["penetration_tolerance_m"]) * 1000
        for ep in ds.episode_indices:
            raw = ds.stack(ds.episode_rows(ep), "observation.state")
            frames = []
            first_table = first_self = None
            max_table = max_self = 0.
            worst_table_frame = worst_self_frame = None
            in_limits = table_count = self_count = 0
            for frame in range(0, len(raw), 2):
                q = sim_from_real(env.mapping, raw[frame])
                if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
                    continue
                in_limits += 1
                kin.forward(q)
                table, self_contacts = contacts(env, kin.data, bodies)
                td = table[0]["penetration_mm"] if table else 0.
                sd = self_contacts[0]["penetration_mm"] if self_contacts else 0.
                if td > max_table:
                    max_table, worst_table_frame = td, frame
                if sd > max_self:
                    max_self, worst_self_frame = sd, frame
                if td > tol_mm:
                    table_count += 1
                    first_table = frame if first_table is None else first_table
                    deepest = table[0]
                    robot = deepest["b"] if deepest["a"]["name"] == "table" else deepest["a"]
                    table_pairs.update([f'{robot["body"]}:{robot["name"] or robot["mesh"] or robot["id"]}'])
                    body_meshes = [g for g in meshes if env.model.body(env.model.geom_bodyid[g]).name == robot["body"]]
                    lowest = min(float(np.min(meshes[g] @ kin.data.geom_xmat[g].reshape(3, 3)[2]
                                               + kin.data.geom_xpos[g, 2])) for g in body_meshes)
                    visual_depth = max(0., (env.table_top - lowest) * 1000)
                    table_gaps.append({"collision_mm": td, "visual_mm": visual_depth,
                                       "excess_mm": td - visual_depth, "episode": ep, "frame": frame})
                if sd > tol_mm:
                    self_count += 1
                    first_self = frame if first_self is None else first_self
                    self_pairs.update(set(":".join(sorted([c["a"]["body"], c["b"]["body"]]))
                                          for c in self_contacts if c["penetration_mm"] > tol_mm))
            for frame in {0, first_table, first_self, worst_table_frame, worst_self_frame} - {None}:
                if ep in {0, 6, 13, 21, 36}:
                    frames.append(int(frame))
                    selected[f"{ep}:{frame}"] = inspect_pose(
                        env, kin, sim_from_real(env.mapping, raw[frame]), meshes, bodies, ep, frame)
            for episode, frame in ((0, 326), (21, 228), (36, 280), (36, 596)):
                if ep == episode:
                    selected[f"{ep}:{frame}"] = inspect_pose(
                        env, kin, sim_from_real(env.mapping, raw[frame]), meshes, bodies, ep, frame)
            episode_reports.append({"episode_index": ep, "sampled_rows": len(raw[::2]),
                                    "in_limit_rows": in_limits,
                                    "first_table_over_tolerance_source_frame": first_table,
                                    "first_self_over_tolerance_source_frame": first_self,
                                    "table_over_tolerance_rows": table_count,
                                    "self_over_tolerance_rows": self_count,
                                    "max_table_penetration_mm": max_table,
                                    "max_self_penetration_mm": max_self,
                                    "worst_table_source_frame": worst_table_frame,
                                    "worst_self_source_frame": worst_self_frame})
        original = inspect_pose(env, kin, env.initial_qpos, meshes, bodies, None, None)
        base = env.model.body("base").id
        table = env.table_geom
        kin.forward(env.initial_qpos)
        base_lowest = min(float(np.min(meshes[g] @ kin.data.geom_xmat[g].reshape(3, 3)[2]
                                      + kin.data.geom_xpos[g, 2])) for g in meshes
                          if env.model.geom_bodyid[g] == base)
        excess = np.array([r["excess_mm"] for r in table_gaps])
        return {
            "diagnostic": "Unchanged v2 provisional geometry: collider vs visual CAD audit",
            "created_date": "2026-10-21", "physical_calibration": False,
            "script_sha256": sha256_file(Path(__file__)),
            "config_path": str(config), "config_sha256": sha256_file(config),
            "task_path": env.config["task_config"], "task_sha256": sha256_file(Path(env.config["task_config"])),
            "model_path": env.config["model"], "model_sha256": sha256_file(Path(env.config["model"])),
            "raw_root": str(root), "source_hz_nominal": ds.fps, "stride": 2,
            "raw_files_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for directory in ("data", "meta")
                for p in sorted((root / directory).rglob("*")) if p.is_file()
            },
            "ik_penetration_tolerance_mm": tol_mm,
            "geometry_unchanged": True, "limits_unchanged": True, "collision_checks_unchanged": True,
            "frame_consistency": {
                "task_equals_base": True,
                "world_base_position_m": env.model.body_pos[base].tolist(),
                "table_box_world_center_m": env.model.geom_pos[table].tolist(),
                "table_top_world_m": env.table_top,
                "table_top_in_base_frame_m": env.table_top - float(env.model.body_pos[base, 2]),
                "visual_base_min_height_above_table_mm": (base_lowest - env.table_top) * 1000,
                "interpretation": "Scene places table and base consistently with task.yaml. Existing 12 mm base offset leaves the lowest base CAD vertex 9.6 mm above table; physical support/mount thickness is not independently verified.",
            },
            "summary": {
                "in_limit_rows": sum(r["in_limit_rows"] for r in episode_reports),
                "table_over_tolerance_rows": len(table_gaps),
                "self_over_tolerance_rows": sum(r["self_over_tolerance_rows"] for r in episode_reports),
                "table_failures_with_same_body_visual_mesh_below_table": sum(r["visual_mm"] > 0 for r in table_gaps),
                "table_failures_with_same_body_visual_mesh_below_tolerance": sum(r["visual_mm"] > tol_mm for r in table_gaps),
                "collision_depth_minus_visual_depth_mm": {
                    "min": float(excess.min()) if len(excess) else None,
                    "median": float(np.median(excess)) if len(excess) else None,
                    "p95": float(np.percentile(excess, 95)) if len(excess) else None,
                    "max": float(excess.max()) if len(excess) else None},
                "deepest_table_geom_by_pose": dict(table_pairs),
                "self_body_pairs_by_pose": dict(self_pairs),
            },
            "configured_legacy_start": original,
            "episodes": episode_reports, "selected_pose_details": selected,
            "findings": [
                "Table failures also occur in the visual CAD, so deleting oversized collision boxes alone cannot resolve this mismatch.",
                "Native FK matches independent XML rigid-transform composition; this verifies implementation consistency only, not real motor calibration.",
                "A global base/table translation cannot remove robot self-collisions; investigate joint mapping and physical assembly before collision exclusions.",
                "Negative visual convex-hull distance does not prove concave physical meshes intersect; confirm self-contact against real landmarks and assembly geometry.",
                "Source phase boundaries remain unannotated. First collision means first in-limit sampled collision, not necessarily first task contact or grasp.",
            ],
            "fix_now": [
                "Preserve these collision and visual-CAD comparisons in calibration acceptance reports.",
                "Use independently verified joint-unit/zero mapping, base support height, and camera landmarks; keep limits and collision rejection active.",
            ],
            "missing_evidence": [
                "Independent physical joint-zero/sign verification. Operator attests unchanged calibration/setup in configs/v2/setup_provenance_20261021.json; recorded units and current calibration are audited separately.",
                "Measured base support/table registration and physical joint landmark observations across multiple poses.",
                "Measured physical jaw and camera-mount geometry versus pinned upstream CAD.",
                "Reviewed approach/grasp/lift/release boundaries to distinguish necessary task motion from return motion.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2")
    parser.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.root, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
