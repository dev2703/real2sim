"""Provisional unloaded R1/K2 comparisons on recorded poses; never a placement gate.

Compare holds and the same contiguous in-limit recorded windows without clipping references.
Ball and bowl are moved away from the arm. Mapping/geometry remain explicitly provisional.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from real2sim.config import load_yaml
from real2sim.data.raw import RawDataset
from real2sim.geometry.so3 import delta_action, rot6d_from_matrix
from real2sim.policy.adapters import AbsolutePoseActionAdapter, CartesianActionAdapter
from real2sim.policy.kinematic_actions import KinematicDeltaAdapter
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

MODES = ("joint_reference", "current_delta", "persistent_delta", "absolute_pose",
         "tool_persistent_delta", "kinematic_persistent_delta")


def contacts(kin):
    tolerated = {"ball", "gripper", "moving_jaw_so101_v1"}
    return [
        {"geom1": kin.model.geom(c.geom1).name, "geom2": kin.model.geom(c.geom2).name,
         "body1": kin.model.body(kin.model.geom_bodyid[c.geom1]).name,
         "body2": kin.model.body(kin.model.geom_bodyid[c.geom2]).name,
         "penetration_mm": float(-c.dist * 1000)}
        for c in kin.data.contact if c.dist < -kin.config["penetration_tolerance_m"]
        and "ball" not in (names := {kin.model.body(kin.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)})
        and not names <= tolerated
    ]


def probe(config, states, commands, mode, *, hold_ticks=None):
    if mode not in MODES:
        raise ValueError(mode)
    with So101PickPlaceEnv(config, render_images=False) as env:
        if env.control_hz != 12.5 or env.config["physics_hz"] != 500:
            raise ValueError("diagnostic requires 12.5 Hz control and 500 Hz physics")
        q = np.array([sim_from_real(env.mapping, state) for state in states])
        if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
            raise ValueError("reference window must be in range; clipping is not equivalent")
        ticks = hold_ticks if hold_ticks is not None else len(q) - 1
        env.max_episode_steps = ticks + 1
        _, info = env.reset(seed=0, scenario={
            "joint_positions": q[0], "ball_xy": [.5, .3], "bowl_xy": [.5, -.3],
        })
        if info["privileged_metrics"]["robot_table_penetration_m"] > 1e-6:
            raise ValueError("initial recorded pose penetrates table")
        kin = NativeKinematics(env.model, env.config)
        reference_kin = NativeKinematics(env.model, env.config)
        poses = np.array([kin.forward(v) for v in q])
        native = CartesianActionAdapter(
            env, kin, max_step_m=.024, max_step_rad=.24,
            target_reference="accepted_target" if "persistent" in mode else "current_pose",
        )
        adapter = native
        if mode == "absolute_pose":
            adapter = AbsolutePoseActionAdapter(native)
        elif mode in ("tool_persistent_delta", "kinematic_persistent_delta"):
            adapter = KinematicDeltaAdapter(
                native, tool_frame=mode == "tool_persistent_delta",
                projection={"rotation_weight": .1, "damping": .001}
                if mode == "kinematic_persistent_delta" else None,
            )
        trace = []
        discarded_translation = np.zeros(3)
        initial_pose = env.ee_pose().copy()
        for tick in range(ticks):
            k, next_k = (0, 0) if hold_ticks is not None else (tick, tick + 1)
            target_pose = initial_pose if hold_ticks is not None else poses[next_k]
            grip = float(states[0, 5] / 100 if hold_ticks is not None else commands[k, 5] / 100)
            if not 0 <= grip <= 1:
                raise ValueError("recorded gripper intent is out of range")
            reference_q = q[next_k].copy()
            reference_q[5] = env.mapping.gripper_real_to_sim_rad(grip * 100)
            reference_kin.forward(reference_q)
            reference_contacts = contacts(reference_kin)
            started = time.perf_counter()
            if mode == "joint_reference":
                target = q[next_k].copy()
                target[5] = env.mapping.gripper_real_to_sim_rad(grip * 100)
                _, _, _, _, info = env.step(target)
                control = info["control"]
                rejected, saturation = False, float(np.mean(control["limit_clipped"]))
                detail = {
                    "limit_clipped": control["limit_clipped"],
                    "rate_limited": control["rate_limited"],
                    "requested_gripper_rad": float(target[5]),
                    "bounded_gripper_rad": float(np.clip(target[5], *env.limits[5])),
                    "native_gripper_clipped": bool(control["limit_clipped"][5]),
                }
            else:
                delta = np.zeros(6) if hold_ticks is not None else delta_action(poses[k], poses[next_k])
                if mode == "absolute_pose":
                    action = np.r_[target_pose[:3, 3], rot6d_from_matrix(target_pose[:3, :3]), grip]
                else:
                    if mode == "tool_persistent_delta":
                        delta = np.r_[poses[k, :3, :3].T @ delta[:3], poses[k, :3, :3].T @ delta[3:]]
                    action = np.r_[delta, grip]
                result = adapter.apply(action)
                info, rejected, detail, saturation = result.info, result.ik_rejected, result.detail, result.saturation
            elapsed = time.perf_counter() - started
            error = delta_action(env.ee_pose(), target_pose)
            accepted = native.accepted_target
            if mode == "persistent_delta" and rejected:
                discarded_translation += action[:3]
            metrics, control = info["privileged_metrics"], info["control"]
            trace.append({
                "tick": tick, "tracking_error_mm": float(np.linalg.norm(error[:3]) * 1000),
                "rotation_error_rad": float(np.linalg.norm(error[3:])),
                "ik_rejected": rejected, "ik_reason": detail.get("ik_reason"),
                "action_saturation": saturation,
                "adapter_detail": detail,
                "joint_limit_clipped": control["limit_clipped"],
                "joint_rate_limited": control["rate_limited"],
                "robot_table_penetration_m": metrics["robot_table_penetration_m"],
                "ball_non_table_contact": bool(env._ball_contacts() - {env.table_geom}),
                "control_and_physics_ms": elapsed * 1000,
                "projection": detail.get("kinematic_delta"),
                "reference_contacts": reference_contacts,
                "rejected_candidate_contacts": contacts(kin) if rejected else [],
                "accepted_reference_error_mm": float(np.linalg.norm(
                    accepted[:3, 3] - target_pose[:3, 3]) * 1000) if accepted is not None else None,
                "discarded_translation_norm_mm": float(np.linalg.norm(discarded_translation) * 1000)
                if mode == "persistent_delta" else None,
            })
        errors = [row["tracking_error_mm"] for row in trace]
        projections = [r["projection"] for r in trace if r["projection"] and "projection_applied" in r["projection"]]
        summary = {
            "mode": mode, "probe": "20s_hold" if hold_ticks is not None else "recorded_window",
            "ticks": ticks, "seconds": ticks / 12.5, "tracking_p95_mm": float(np.quantile(errors, .95)),
            "tracking_max_mm": max(errors), "tracking_end_mm": errors[-1],
            "rotation_p95_rad": float(np.quantile([r["rotation_error_rad"] for r in trace], .95)),
            "ik_rejections": sum(r["ik_rejected"] for r in trace),
            "ik_reasons": dict(Counter(r["ik_reason"] for r in trace)),
            "limit_clipped_ticks": sum(any(r["joint_limit_clipped"]) for r in trace),
            "rate_limited_ticks": sum(any(r["joint_rate_limited"]) for r in trace),
            "native_gripper_clipped_ticks": sum(
                bool(r["adapter_detail"].get("native_gripper_clipped")) for r in trace),
            "max_action_saturation": max(r["action_saturation"] for r in trace),
            "max_table_penetration_mm": max(r["robot_table_penetration_m"] for r in trace) * 1000,
            "ball_contact_ticks": sum(r["ball_non_table_contact"] for r in trace),
            "control_and_physics_p95_ms": float(np.quantile([r["control_and_physics_ms"] for r in trace], .95)),
            "projection_failed_ticks": sum(not p["projection_applied"] for p in projections),
            "projection_position_change_p95_mm": float(np.quantile(
                [p.get("projection_position_change_m", 0) * 1000 for p in projections], .95,
            )) if projections else None,
            "reference_penetration_max_mm": max(
                (c["penetration_mm"] for r in trace for c in r["reference_contacts"]), default=0.,
            ),
            "reference_contact_pairs": dict(Counter(
                f"{c['body1']}:{c['body2']}" for r in trace for c in r["reference_contacts"]
            )),
            "rejected_candidate_contact_pairs": dict(Counter(
                f"{c['body1']}:{c['body2']}" for r in trace for c in r["rejected_candidate_contacts"]
            )),
            "accepted_reference_p95_error_mm": float(np.quantile([
                r["accepted_reference_error_mm"] for r in trace], .95,
            )) if native.accepted_target is not None else None,
        }
        if mode == "persistent_delta":
            first_rejection = next((r["tick"] for r in trace if r["ik_rejected"]), len(trace))
            summary["pre_first_rejection_tracking_p95_mm"] = (
                float(np.quantile(errors[:first_rejection], .95)) if first_rejection else None)
            summary["max_reference_offset_discarded_translation_discrepancy_mm"] = max(
                abs(r["accepted_reference_error_mm"] - r["discarded_translation_norm_mm"]) for r in trace)
            summary["action_or_feedback_clipped_ticks"] = sum(r["action_saturation"] > 0 for r in trace)
            summary["rejected_ticks"] = [r["tick"] for r in trace if r["ik_rejected"]]
        return summary, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2")
    p.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    p.add_argument("--poses", nargs="+", default=["0:156", "21:126", "36:142"])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--probe", choices=("hold", "motion", "both"), default="both")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("reuse completed artifacts; select a new version for a changed experiment")
    raw = RawDataset(a.root)
    if raw.fps != 25:
        raise ValueError("25 FPS source required for exact stride-two diagnostic")
    a.out.mkdir(parents=True)
    cfg = copy.deepcopy(load_yaml(a.config))
    data = load_yaml(cfg["data_config"])
    data["policy_hz"] = 12.5
    dp = a.out / "clock.yaml"
    dp.write_text(yaml.safe_dump(data))
    cfg.update(control_hz=12.5, physics_hz=500, data_config=str(dp.resolve()))
    cfg["ik"].update(position_convergence_m=1e-5, orientation_convergence_rad=1e-4)
    cp = a.out / "sim.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    report = {
        "kind": "provisional unloaded R1/K2 diagnostic", "valid_for_learning_gate": False,
        "limitations": "Unconfirmed conditions; provisional mapping/calibration; contiguous valid windows only, not complete tasks. Ball and bowl moved away. No learned policy.",
        "source_hashes": {str(path): file_sha256(path) for path in (
            a.config, Path(cfg["task_config"]), Path(__file__),
            Path("src/real2sim/policy/adapters.py"), Path("src/real2sim/policy/kinematic_actions.py"),
            Path("src/real2sim/sim/kinematics.py"), Path("src/real2sim/sim/env.py"),
        )},
        "raw_data_hashes": {str(path.relative_to(a.root)): file_sha256(path)
                           for path in sorted((a.root / "data").rglob("*.parquet"))},
        "cases": [], "rows": [],
    }
    snapshots = a.out / "sources"
    snapshots.mkdir()
    report["source_snapshots"] = {}
    for i, source in enumerate(report["source_hashes"]):
        destination = snapshots / f"{i}_{Path(source).name}"
        destination.write_bytes(Path(source).read_bytes())
        report["source_snapshots"][source] = str(destination.relative_to(a.out))
    for spec in a.poses:
        ep, start = map(int, spec.split(":"))
        if start < 0 or start % 2:
            raise ValueError("source start must be nonnegative and on the stride-two grid")
        rows = raw.episode_rows(ep)
        states = raw.stack(rows, "observation.state")[start::2]
        commands = raw.stack(rows, "action")[start::2]
        with So101PickPlaceEnv(cp, render_images=False) as env:
            q = np.array([sim_from_real(env.mapping, state) for state in states])
            invalid = np.flatnonzero(((q < env.limits[:, 0]) | (q > env.limits[:, 1])).any(axis=1))
        stop = int(invalid[0]) if len(invalid) else len(states)
        if stop < 2:
            raise ValueError(f"{spec} has no valid motion window")
        report["cases"].append({
            "source_pose": spec, "start_source_frame": start, "end_source_frame": start + 2 * (stop - 1),
            "first_invalid_source_frame": start + 2 * stop if len(invalid) else None,
            "remaining_source_rows": len(rows) - start, "used_source_rows": stop,
        })
        for hold in ((250,) if a.probe == "hold" else (None,) if a.probe == "motion" else (250, None)):
            for mode in a.modes:
                summary, trace = probe(cp, states[:stop], commands[:stop], mode, hold_ticks=hold)
                summary["source_pose"] = spec
                trace_path = a.out / f"ep{ep}_{summary['probe']}_{mode}.json"
                trace_path.write_text(json.dumps(trace, separators=(",", ":"), allow_nan=False) + "\n")
                summary["trace_sha256"] = file_sha256(trace_path)
                report["rows"].append(summary)
                (a.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"{spec} {summary['probe']} {mode}: p95 {summary['tracking_p95_mm']:.3f} mm, "
                      f"max {summary['tracking_max_mm']:.3f} mm, rejects {summary['ik_rejections']}", flush=True)
    (a.out / "COMPLETE.json").write_text(json.dumps({"report_sha256": file_sha256(a.out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()
