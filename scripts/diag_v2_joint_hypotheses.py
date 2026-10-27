"""Compare joint-coordinate hypotheses without changing any calibration or references.

Native limits remain fixed. Train extrema define one deliberately nonphysical feasibility
control; validation is scored afterward, and reserved episodes are never inspected.
Static contacts are descriptive CAD checks, not an objective for selecting physical zeros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from real2sim.data.raw import sha256_file
from real2sim.sim.env import So101PickPlaceEnv

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def hypotheses(calibration: dict, limits_deg: np.ndarray, train: np.ndarray) -> dict:
    """Return explicitly labelled alternatives; none is a calibration recommendation."""
    bounds = np.array([[calibration[n]["range_min"], calibration[n]["range_max"]]
                       for n in ARM], dtype=float)
    half_span = np.diff(bounds, axis=1).ravel() * 180 / 4095
    midpoint = bounds.mean(axis=1)
    lower = limits_deg[:, 0] - train.min(axis=0)
    upper = limits_deg[:, 1] - train.max(axis=0)
    if np.any(lower > upper):
        raise ValueError("train recording span cannot fit native limits with any constant offset")
    return {
        "identity_degrees": {"scale": np.ones(5), "offset_deg": np.zeros(5),
                             "rationale": "Current provisional mapping; output encoder degrees."},
        "homing_pose_as_zero": {
            "scale": np.ones(5), "offset_deg": (midpoint - 2047) * 360 / 4095,
            "rationale": "Hypothesis only: operator homing pose equals exact CAD zero."
                         " Saved homing offsets must not be added again."},
        "calibration_lower_endpoint_to_native_lower": {
            "scale": np.ones(5), "offset_deg": limits_deg[:, 0] + half_span,
            "rationale": "Hypothesis only: recorded calibration lower stop equals native lower stop."},
        "calibration_upper_endpoint_to_native_upper": {
            "scale": np.ones(5), "offset_deg": limits_deg[:, 1] - half_span,
            "rationale": "Hypothesis only: recorded calibration upper stop equals native upper stop."},
        "endpoint_affine_control": {
            "scale": np.diff(limits_deg, axis=1).ravel() / (2 * half_span),
            "offset_deg": limits_deg.mean(axis=1),
            "rationale": "Nonphysical control: rescales angular encoder degrees to fit CAD bounds."
                         " Incorrect unless an independent transmission ratio establishes the scale."},
        "train_range_feasible_offset_control": {
            "scale": np.ones(5), "offset_deg": np.clip(np.zeros(5), lower, upper),
            "rationale": "Nonphysical control: closest-to-zero offsets satisfying train bounds."
                         " No image, physical-zero or contact evidence enters this construction."},
    }


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def score(env, real, episodes, frames, scale, offset):
    model, data = env.model, mujoco.MjData(env.model)
    q = np.column_stack([np.deg2rad(real[:, :5] * scale + offset),
                         env.mapping.gripper_real_to_sim_rad(real[:, 5])])
    # Avoid calling a <1e-10 rad arithmetic roundoff at a constructed bound invalid.
    tolerance = 1e-10
    outside = (q < env.limits[:, 0] - tolerance) | (q > env.limits[:, 1] + tolerance)
    base = model.body("base").id
    robot_bodies = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != base:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == base:
            robot_bodies.add(body)
    table_depth, self_depth, clamp_error, clamp_angle = [], [], [], []
    worst = []
    for index, qi in enumerate(q):
        data.qpos[env.qadr] = qi
        # No dynamics or reset: raw references are never clipped before static evaluation.
        mujoco.mj_forward(model, data)
        p = data.site_xpos[env.site_id].copy()
        r = data.site_xmat[env.site_id].reshape(3, 3).copy()
        table, self_collision = 0., 0.
        for contact in data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            names = {model.body(b).name for b in (b1, b2)}
            depth = max(0., -float(contact.dist)) * 1000
            if env.table_geom in (g1, g2) and ({b1, b2} & robot_bodies):
                table = max(table, depth)
            elif b1 in robot_bodies and b2 in robot_bodies and not names <= {
                    "gripper", "moving_jaw_so101_v1"}:
                self_collision = max(self_collision, depth)
        table_depth.append(table)
        self_depth.append(self_collision)
        data.qpos[env.qadr] = np.clip(qi, env.limits[:, 0], env.limits[:, 1])
        mujoco.mj_kinematics(model, data)
        clamp_error.append(float(np.linalg.norm(data.site_xpos[env.site_id] - p) * 1000))
        relative = r.T @ data.site_xmat[env.site_id].reshape(3, 3)
        sine = np.linalg.norm([relative[2, 1] - relative[1, 2],
                               relative[0, 2] - relative[2, 0],
                               relative[1, 0] - relative[0, 1]]) / 2
        clamp_angle.append(float(np.rad2deg(np.arctan2(sine, (np.trace(relative) - 1) / 2))))
        worst.append((table, int(episodes[index]), int(frames[index])))
    starts = np.r_[True, episodes[1:] != episodes[:-1]]
    return {"rows": len(q), "episodes": sorted(map(int, np.unique(episodes))),
            "limit_roundoff_tolerance_rad": tolerance,
            "out_of_limits_rows": int(outside.any(axis=1).sum()),
            "out_of_limits_by_joint": outside.sum(axis=0).tolist(),
            "starts_out_of_limits": int(outside[starts].any(axis=1).sum()),
            "table_penetration_over_2mm_rows": int(np.sum(np.array(table_depth) > 2)),
            "self_penetration_over_2mm_rows": int(np.sum(np.array(self_depth) > 2)),
            "table_penetration_mm": distribution(np.array(table_depth)),
            "self_penetration_mm": distribution(np.array(self_depth)),
            "native_clipping_ee_position_error_mm": distribution(np.array(clamp_error)),
            "native_clipping_ee_rotation_error_deg": distribution(np.array(clamp_angle)),
            "start_clipping_ee_position_error_mm": distribution(np.array(clamp_error)[starts]),
            "worst_table_reference": {"penetration_mm": max(worst)[0],
                                      "episode": max(worst)[1], "source_frame": max(worst)[2]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=Path(
        ".cache/v2_exploratory_20261022/cartesian_iphone_top/arrays.npz"))
    parser.add_argument("--split", type=Path, default=Path("results/v2/split.json"))
    parser.add_argument("--calibration", type=Path, default=Path.home() /
                        ".cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    arrays = np.load(args.arrays)
    split = json.loads(args.split.read_text())
    calibration = json.loads(args.calibration.read_text())
    for motor in calibration.values():
        if motor["drive_mode"] != 0:
            raise ValueError("only calibrated drive_mode=0 is supported")
    selected = np.isin(arrays["episode_index"], split["train"] + split["val"])
    real = np.asarray(arrays["joint_state"][selected], dtype=float)
    episodes, frames = arrays["episode_index"][selected], arrays["frame_index"][selected]
    train = np.isin(episodes, split["train"])
    with So101PickPlaceEnv("configs/sim_so101.yaml", render_images=False) as env:
        # Operator confirmed stock support, with CAD origin 2.4 mm above the table.
        # Change only this diagnostic's model instance. Task/camera positions are not fitted.
        env.model.body_pos[env.model.body("base").id, 2] += (
            .0024 - float(env.task["robot_so101"]["base_z_offset"]))
        native = np.rad2deg(env.limits[:5])
        alternatives = hypotheses(calibration, native, real[train, :5])
        report = {
            "kind": "JOINT_HYPOTHESES_DIAGNOSTIC_NOT_PHYSICAL_CALIBRATION",
            "physical_calibration_accepted": False, "production_files_changed": False,
            "script_sha256": sha256_file(Path(__file__)),
            "input_sha256": {str(p): sha256_file(p) for p in (
                args.arrays, args.split, args.calibration,
                Path("assets/menagerie/robotstudio_so101/so101.xml"), Path("configs/task.yaml"),
                Path("configs/sim_so101.yaml"), Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/base_env.py"))},
            "official_calibration_conventions":
                "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/"
                "608122e9ac330a753735f2e18aee73338e9ac407/Simulation/SO101/README.md",
            "excluded_reserved_episodes": split["reserved"],
            "joint_names": [*ARM, "gripper"], "native_arm_bounds_deg": native.tolist(),
            "calibration_arm_span_deg": [
                (calibration[n]["range_max"] - calibration[n]["range_min"]) * 360 / 4095
                for n in ARM],
            "scope": "Fixed native limits, no-spacer 2.4 mm diagnostic base height, legacy"
                     " provisional gripper mapping. All selected poses evaluated, including those"
                     " beyond native bounds; contacts at those poses are descriptive FK only."
                     " No dynamics, images, task-success measurement or zero acceptance.",
            "hypotheses": {},
        }
        for name, candidate in alternatives.items():
            print(f"Scoring {name}", flush=True)
            report["hypotheses"][name] = {
                "scale": candidate["scale"].tolist(),
                "offset_deg": candidate["offset_deg"].tolist(),
                "rationale": candidate["rationale"],
                **{label: score(env, real[mask], episodes[mask], frames[mask],
                               candidate["scale"], candidate["offset_deg"])
                   for label, mask in (("train", train), ("val", ~train))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
