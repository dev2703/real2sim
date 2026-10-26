"""Frozen native-joint ACT through an explicit virtual SO-101 -> Panda adapter.

The live mode feeds actual Panda RGB and measured Panda pose/opening projected
into virtual SO-101 joint observations. The trajectory mode instead consumes a
previously recorded *learned ACT* SO-101 rollout and is labelled separately: it
tests trajectory retargeting, not Panda-image closed-loop policy transfer.
Neither mode calls a waypoint controller or reads object state to choose actions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from scipy.optimize import least_squares

from real2sim.config import load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.geometry.so101_fk import So101FK
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import POINTING_DOWN
from real2sim.panda.retarget import classify_contacts
from real2sim.policy.act import load_checkpoint
from real2sim.policy.adapters import normalize_image_chw
from real2sim.policy.bounded_joint import bound_gripper_command
from real2sim.sim.env import So101PickPlaceEnv, real_from_sim

SOURCE_OFFSET = np.array([-.010, 0., .030])
PANDA_OFFSET = np.array([-.008, 0., 0.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n")


def validate_learned_trace(trace_path, checkpoint_sha256):
    """Pin a trace to a matching frozen, unassisted learned-source evaluation."""
    report_path = trace_path.parent.parent / "report.json"
    report = json.loads(report_path.read_text())
    if (report.get("checkpoint_sha256") != checkpoint_sha256
            or report.get("teacher_or_ik_corrections") is not False):
        raise ValueError("source trace must be from the same frozen ACT without teacher corrections")
    matches = [trial for trial in report["trials"]
               if Path(trial["trace"]).resolve() == trace_path.resolve()]
    if len(matches) != 1 or matches[0].get("teacher_goals_or_ik_corrections") is not False:
        raise ValueError("source trace is not uniquely identified as an unassisted ACT trial")
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if not trace or len(trace) != matches[0]["ticks"]:
        raise ValueError("source trace row count differs from its scored trial")
    return trace, {"trace": str(trace_path), "trace_sha256": digest(trace_path),
        "evaluation_report": str(report_path), "evaluation_report_sha256": digest(report_path),
        "source_seed": matches[0]["seed"],
        "source_pick_place_success": matches[0]["pick_place_success"],
        "checkpoint_sha256": checkpoint_sha256}


class VirtualSource:
    """Robot-only pose/opening projection; no object goals or task phase input."""

    def __init__(self, scene, *, panda_offset=PANDA_OFFSET):
        self.source = So101PickPlaceEnv(scene, render_images=False)
        self.source.reset(seed=0)
        self.mapping = self.source.mapping
        self.fk = So101FK(mapping=self.mapping)
        self.limits = self.source.limits.copy()
        self.q = self.source.initial_qpos.copy()
        self.last_target = self.q.copy()
        self.panda_offset = np.asarray(panda_offset, float).copy()
        self.percent = np.linspace(0., 100., 1001)
        fixed = self.source.model.geom("fixed_jaw_sph_tip1").id
        moving = self.source.model.geom("moving_jaw_sph_tip1").id
        self.gaps = []
        for percent in self.percent:
            self.source.data.qpos[self.source.qadr] = self.q
            self.source.data.qpos[self.source.qadr[5]] = self.mapping.gripper_real_to_sim_rad(percent)
            mujoco.mj_forward(self.source.model, self.source.data)
            # CAD fingertip separation; physical calibration remains unverified.
            gap = np.linalg.norm(self.source.data.geom_xpos[fixed] - self.source.data.geom_xpos[moving])
            gap -= self.source.model.geom_size[fixed, 0] + self.source.model.geom_size[moving, 0]
            self.gaps.append(gap)
        self.gaps = np.asarray(self.gaps)
        if np.any(np.diff(self.gaps) <= 0):
            raise ValueError("SO-101 jaw-gap projection requires monotonic CAD separation")

    def close(self):
        self.source.close()

    def ball_point(self, q):
        pose = self.fk.fk_sim(np.asarray(q)[:5])
        return pose[:3, 3] + pose[:3, :3] @ SOURCE_OFFSET

    def source_state(self, panda):
        pose = panda.ee_pose()
        equivalent_point = pose[:3, 3] + pose[:3, :3] @ self.panda_offset
        previous = self.q[:5].copy()

        def residual(q):
            return np.r_[self.ball_point(q) - equivalent_point, .0001 * (q - previous)]

        fitted = least_squares(residual, np.clip(self.last_target[:5], *self.limits[:5].T),
            bounds=tuple(self.limits[:5].T), max_nfev=60,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = float(np.linalg.norm(self.ball_point(fitted.x) - equivalent_point))
        if not np.isfinite(fitted.x).all() or error > .003:
            raise RuntimeError(f"Panda pose has no virtual SO-101 representation within3mm: {error:.6f}m")
        opening = panda.gripper_opening()
        # Panda cannot open as far as the SO-101. At the fully-open stop the
        # command history disambiguates the equivalent source opening; below it
        # the measured pad separation maps through explicit source CAD geometry.
        grip_percent = (float(self.mapping.gripper_sim_to_real_percent(self.last_target[5]))
            if opening >= panda.max_opening - .001 else
            float(np.interp(opening, self.gaps, self.percent)))
        self.q = np.r_[fitted.x, self.mapping.gripper_real_to_sim_rad(grip_percent)]
        return real_from_sim(self.mapping, self.q).astype(np.float32), error

    def source_action(self, action):
        requested, bounded = bound_gripper_command(self.mapping, self.limits, action)
        if np.any(bounded[:5] < self.limits[:5, 0]) or np.any(bounded[:5] > self.limits[:5, 1]):
            raise RuntimeError("Learned source arm command exceeds unchanged SO-101 limits")
        self.last_target = bounded.copy()
        return requested, bounded


def target_from_source(source, q):
    target = np.eye(4)
    target[:3, :3] = POINTING_DOWN
    target[:3, 3] = source.ball_point(q) - POINTING_DOWN @ source.panda_offset
    return target


def panda_opening_from_percent(percent, maximum):
    """Map the source teacher's 0=closed/90=open commands to Panda endpoints.

    Source jaw CAD width can exceed Panda's maximum opening. Clipping that metric
    width delays closure; command intent therefore maps endpoint-to-endpoint,
    while measured readbacks remain a separately declared geometric projection.
    """
    if not np.isfinite([percent, maximum]).all() or maximum <= 0:
        raise ValueError("finite source percentage and positive Panda opening required")
    return float(np.clip(percent / 90., 0., 1.) * maximum)


def run(checkpoint, source_scene, panda_scene, out, mode, seeds, trace_path=None, *, seat_mm=8.):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    policy, norm, contract = load_checkpoint(checkpoint, "cpu")
    if contract["state_names"] != contract["action_names"] or len(contract["state_names"]) != 6:
        raise ValueError("This diagnostic requires the native six-value SO-101 ACT contract")
    hz = float(contract["policy_hz"])
    config = copy.deepcopy(load_yaml(panda_scene))
    config.update(control_hz=hz, data_config="configs/v2/data.waypoint_20261026.yaml")
    config["action_limits"].update(max_translation_m=.024, max_rotation_rad=.24)
    config_path = out / "sim_panda.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if not np.isfinite(seat_mm) or not 0 <= seat_mm <= 30:
        raise ValueError("Panda seating offset must be in0..30mm")
    panda_offset = np.array([-seat_mm / 1000., 0., 0.])
    source = VirtualSource(source_scene, panda_offset=panda_offset)
    rows = []
    sources = [Path(__file__), source_scene, panda_scene, checkpoint / "model.safetensors",
        checkpoint / "real2sim_contract.json", project_path(config["task_config"])]
    trace, provenance = None, None
    if mode == "trajectory":
        if trace_path is None:
            raise ValueError("Trajectory mode needs a learned-source trace")
        trace, provenance = validate_learned_trace(trace_path, digest(checkpoint / "model.safetensors"))
        sources.extend([trace_path, trace_path.parent.parent / "report.json"])
    hashes = {str(p): digest(p) for p in sources}
    report = {"schema": "learned_panda_retarget_20261027_v1", "status": "running", "mode": mode,
        "controller_source": "FROZEN_ACT_PANDA_RGB_VIRTUAL_JOINTS" if mode == "live" else "FROZEN_ACT_TRAJECTORY_RETARGET",
        "policy_images_from_panda": mode == "live", "waypoint_or_teacher_actions": False,
        "panda_orientation_adapter": "Fixed top-down Panda grasp; SO-101 rotational motion is not copied",
        "grasp_point_offsets_m": {"so101": SOURCE_OFFSET, "panda": panda_offset},
        "source_joint_projection": "Position-only virtual SO-101 IK plus CAD jaw-gap/readback mapping",
        "panda_gripper_command_mapping": "Source 0percent closed,90percent open maps linearly to Panda0..maximum opening; metric jaw-gap clipping is not used for commands",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashes[str(checkpoint / "model.safetensors")],
        "physical_calibration_accepted": False, "source_sha256_before": hashes, "trials": []}
    report["learned_source_trajectory_provenance"] = provenance
    write_json(out / "report.json", report)
    try:
        with closing(PandaPickPlaceEnv(config_path)) as env:
            ik = PandaIK(env.model, env.config, task_origin=env.offset)
            start_target = target_from_source(source, source.source.initial_qpos)
            source_open = np.interp(float(source.source.mapping.gripper_sim_to_real_percent(
                source.source.initial_qpos[5])), source.percent, source.gaps)
            # Robot-only mounting adaptation: choose the first declared base
            # translation that represents the source start under native gates.
            # No ball/bowl pose, held state or trial outcome chooses the mount.
            original_base = env.model.body("link0").pos.copy()
            mount_attempts = []
            for base_dx in (0., -.15, -.20, -.10, -.25):
                env.model.body_pos[env.model.body("link0").id] = original_base + [base_dx, 0., 0.]
                mujoco.mj_forward(env.model, env.data)
                start = ik.solve(start_target, env.initial_qpos, finger_opening=float(min(source_open, env.max_opening)))
                mount_attempts.append({"base_dx_task_m": base_dx, "accepted": start.accepted,
                    "reason": start.reason, "position_error_m": start.position_error_m})
                if start.accepted:
                    break
            report["panda_base_mount_attempts"] = mount_attempts
            report["panda_base_dx_task_m"] = base_dx
            if not start.accepted:
                raise RuntimeError(f"Mapped Panda start rejected: {start.reason}")
            for seed in seeds:
                directory = out / f"seed_{seed:03d}"
                directory.mkdir()
                source.q = source.source.initial_qpos.copy()
                source.last_target = source.q.copy()
                obs, initial_info = env.reset(seed=seed, scenario={"joint_positions": start.qpos,
                    "gripper_opening": float(min(source_open, env.max_opening))})
                policy.reset()
                tracker = TrialTracker(success_hold_ticks=math.ceil(hz * env.task["episode"]["success_hold_s"]),
                    approach_distance_m=.03, table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
                reasons, durations, error_max, abort, max_penetration = {}, [], 0., None, 0.
                writer = CaptionedVideo(directory / "rollout.mp4", hz, (384, 512))
                with (directory / "ticks.jsonl").open("w") as handle:
                    try:
                        for tick in range(env.max_episode_steps):
                            started = time.perf_counter()
                            if mode == "live":
                                virtual, projection_error = source.source_state(env)
                                batch = {"observation.state": torch.from_numpy(norm.state(virtual))[None],
                                    "observation.images.iphone": torch.from_numpy(normalize_image_chw(norm,
                                        obs["observation.images.iphone"]))[None]}
                                with torch.inference_mode():
                                    predicted = norm.unaction(policy.predict_action_chunk(batch))[0, 0].numpy()
                                raw_source, bounded_source = source.source_action(predicted)
                            else:
                                entry = trace[min(tick, len(trace) - 1)]
                                raw_source = np.asarray(entry["control"]["applied_radians"], float)
                                bounded_source = np.clip(raw_source, *source.limits.T)
                                source.last_target = bounded_source.copy()
                                virtual, projection_error = None, None
                            desired = target_from_source(source, bounded_source)
                            observed = env.ee_pose()
                            delta = desired[:3, 3] - observed[:3, 3]
                            if np.linalg.norm(delta) > .024:
                                desired[:3, 3] = observed[:3, 3] + delta * (.024 / np.linalg.norm(delta))
                            source_pct = float(source.mapping.gripper_sim_to_real_percent(bounded_source[5]))
                            width = panda_opening_from_percent(source_pct, env.max_opening)
                            solved = ik.solve(desired, env.arm_qpos(), finger_opening=max(width, env.gripper_opening()))
                            reason = solved.reason
                            command = np.r_[solved.qpos if solved.accepted else env.arm_qpos(), width]
                            obs, _, _, truncated, info = env.step(command)
                            durations.append((time.perf_counter() - started) * 1000)
                            reasons[reason] = reasons.get(reason, 0) + 1
                            pm = info["privileged_metrics"]
                            max_penetration = max(max_penetration,
                                classify_contacts(pm["contacts"])["robot_table_penetration_m"])
                            if projection_error is not None:
                                error_max = max(error_max, projection_error)
                            done = tracker.update(pm, saturation=float(np.mean(info["control"]["limit_clipped"])),
                                ik_rejected=not solved.accepted)
                            record = {"tick": tick, "virtual_source_state": virtual,
                                "virtual_projection_error_m": projection_error,
                                "raw_source_target_rad": raw_source, "bounded_source_target_rad": bounded_source,
                                "requested_panda_pose": desired, "panda_ik_reason": reason,
                                "panda_ik_position_error_m": solved.position_error_m,
                                "panda_action_control": info["control"], "panda_readback": env.sim_state(),
                                "scoring_only": pm}
                            handle.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                            writer.write(policy_view_to_bgr(obs["observation.images.iphone"], 2),
                                "LEARNED ACT: Panda RGB + virtual SO101" if mode == "live" else
                                "LEARNED ACT SOURCE TRAJECTORY: Panda retarget (no Panda policy images)")
                            if done or truncated:
                                break
                    except (RuntimeError, ValueError) as exc:
                        abort = f"{type(exc).__name__}: {exc}"
                    finally:
                        writer.close()
                row = tracker.row(experiment_id="LEARNED_PANDA_RETARGET_20261027", controller_source=report["controller_source"],
                    robot="panda", seed=seed, checkpoint_sha256=report["checkpoint_sha256"], checkpoint_step=2000,
                    mean_latency_ms=float(np.mean(durations)) if durations else None,
                    video_path=str(directory / "rollout.mp4"), abort=abort)
                rows.append(row)
                report["trials"].append({**row, "ik_reasons": reasons,
                    "virtual_projection_error_max_m": error_max,
                    "max_robot_table_penetration_m": max_penetration,
                    "configured_start_replaced_by_mapped_source_start": True,
                    "mapped_start_panda_joints": start.qpos,
                    "initial_scoring_only": initial_info["privileged_metrics"],
                    "trace": str(directory / "ticks.jsonl")})
                report["summary"] = summarize(rows)
                write_json(out / "report.json", report)
                print(json.dumps({"mode": mode, "seed": seed, "success": row["pick_place_success"],
                    "lifted": row["lifted"], "phase_failed": row["phase_failed"], "abort": abort,
                    "ik_reasons": reasons}), flush=True)
        report["status"] = "complete"
    finally:
        source.close()
        report["source_sha256_after"] = {p: digest(p) for p in hashes}
        report["runtime_source_changes"] = [p for p in hashes if hashes[p] != report["source_sha256_after"][p]]
        write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v2/matched_scene_act_20261026/training/checkpoints/002000"))
    parser.add_argument("--source-scene", type=Path, default=Path("configs/v2/sim_so101.waypoint_20261026.yaml"))
    parser.add_argument("--panda-scene", type=Path, default=Path("configs/v2/sim_panda.waypoint_20261026.yaml"))
    parser.add_argument("--mode", choices=("live", "trajectory"), required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--panda-seat-mm", type=float, default=8.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.source_scene, args.panda_scene, args.out, args.mode, args.seeds,
        args.trace, seat_mm=args.panda_seat_mm)
