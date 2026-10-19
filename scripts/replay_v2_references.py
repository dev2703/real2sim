"""R1/R2: matched recorded trajectories and optional loaded fixed-target holds.

Requires measured scene layouts and reviewed pre-approach starts. These are diagnostic
reference controllers, never learned-policy successes. No initial clipping is allowed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import RawDataset
from real2sim.evaluation.metrics import TrialTracker
from real2sim.geometry.so3 import delta_action, rot6d_from_matrix
from real2sim.policy.adapters import AbsolutePoseActionAdapter, CartesianActionAdapter
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics

MODES = ("joint_reference", "current_delta", "persistent_delta", "absolute_pose")


def replay(raw, config, case, mode):
    ep = int(case["episode_index"])
    rows = raw.episode_rows(ep)
    start, end = int(case["start_source_frame"]), int(case["end_source_frame"])
    if start % 2 or end % 2 or not 0 <= start < end < len(rows):
        raise ValueError("reviewed replay window must use valid even source rows")
    if case.get("preserves_complete_task") is not True:
        raise ValueError(
            "review that this start preserves the entire approach, grasp and placement"
        )
    indices = np.arange(start, end + 1, 2)
    states = raw.stack(rows, "observation.state")[indices]
    commands = raw.stack(rows, "action")[indices]
    with So101PickPlaceEnv(config, render_images=False) as env:
        if env.control_hz != 12.5 or raw.fps != 25:
            raise ValueError("v2 replay requires 25/12.5 Hz")
        q = np.array([sim_from_real(env.mapping, s) for s in states])
        outside = ((q < env.limits[:, 0]) | (q > env.limits[:, 1])).any(axis=1)
        if outside[0]:
            raise ValueError("recorded start is out of range; no reference clipping allowed")
        hold = case.get("fixed_target_hold")
        hold_ticks = int(np.ceil(float(hold["duration_s"]) * 12.5)) if hold else 0
        env.max_episode_steps = len(q) + hold_ticks
        _, info = env.reset(
            seed=0,
            scenario={
                "joint_positions": q[0],
                "ball_xy": case["ball_xy"],
                "bowl_xy": case["bowl_xy"],
            },
        )
        if info["privileged_metrics"]["robot_table_penetration_m"] > 1e-6:
            raise ValueError("recorded start penetrates table")
        kin = NativeKinematics(env.model, env.config)
        poses = np.array([kin.forward(v) for v in q])
        adapter = CartesianActionAdapter(
            env,
            kin,
            max_step_m=0.024,
            max_step_rad=0.24,
            target_reference="accepted_target" if mode == "persistent_delta" else "current_pose",
        )
        if mode == "absolute_pose":
            adapter = AbsolutePoseActionAdapter(adapter)
        tracker = TrialTracker(13, 0.03, env.table_top, float(env.task["ball"]["radius"]))
        trace = []
        hold_result = None
        phases = case["phases_source_frame"]
        if any(phases.get(name) is None for name in ("grasp", "lift", "release")):
            raise ValueError("reviewed grasp/lift/release boundaries required")
        for k in range(len(q) - 1):
            grip = float(np.clip(commands[k, 5] / 100, 0, 1))
            if mode == "joint_reference":
                target = q[k + 1].copy()
                target[5] = env.mapping.gripper_real_to_sim_rad(grip * 100)
                _, _, _, _, info = env.step(target)
                rejection = False
            else:
                action = (
                    np.r_[poses[k + 1, :3, 3], rot6d_from_matrix(poses[k + 1, :3, :3]), grip]
                    if mode == "absolute_pose"
                    else np.r_[delta_action(poses[k], poses[k + 1]), grip]
                )
                applied = adapter.apply(action)
                info, rejection = applied.info, applied.ik_rejected
            privileged = info["privileged_metrics"]
            tracker.update(privileged, ik_rejected=rejection)
            error = delta_action(env.ee_pose(), poses[k + 1])
            trace.append(
                {
                    "source_frame": int(indices[k]),
                    "reference_outside_limits": bool(outside[k + 1]),
                    "tool_reference_error_mm": float(np.linalg.norm(error[:3]) * 1000),
                    "rotation_error_rad": float(np.linalg.norm(error[3:])),
                    "closure_transport": phases["grasp"] <= indices[k] < phases["release"],
                    "ik_rejected": rejection,
                    "control": info["control"],
                    "contacts": privileged["contacts"],
                    "ball_held": privileged["ball_held"],
                    "pick_place_success": tracker.pick_place_success,
                }
            )
            if hold and int(indices[k + 1]) == int(hold["source_frame"]):
                if not privileged["ball_held"]:
                    raise ValueError(
                        "loaded retention probe starts without an established simulated grasp"
                    )
                target = env.data.ctrl[env.actuator_ids].copy()
                pose = env.ee_pose().copy()
                first_lost = None
                drift = []
                for j in range(hold_ticks):
                    _, _, _, _, held_info = env.step(target)
                    if not held_info["privileged_metrics"]["ball_held"] and first_lost is None:
                        first_lost = (j + 1) / 12.5
                    drift.append(float(np.linalg.norm(env.ee_pose()[:3, 3] - pose[:3, 3]) * 1000))
                hold_result = {
                    "requested_s": hold["duration_s"],
                    "first_loss_s": first_lost,
                    "retained_for_entire_hold": first_lost is None,
                    "max_tool_drift_mm": max(drift),
                    "fixed_native_targets": target.tolist(),
                }
                break  # retention is a separate probe; never call this interrupted replay a placement.
        phase_errors = [r["tool_reference_error_mm"] for r in trace if r["closure_transport"]]
        return {
            "episode_index": ep,
            "condition": case["condition"],
            "mode": mode,
            "controller_source": "RECORDED_REFERENCE_DIAGNOSTIC",
            "pick_place_success": tracker.pick_place_success,
            "historical_success": tracker.success,
            "reference_outside_limit_ticks": int(outside.sum()),
            "closure_transport_error_p95_mm": float(np.quantile(phase_errors, 0.95))
            if phase_errors
            else None,
            "ik_rejections": tracker.ik_rejections,
            "retention_probe": hold_result,
            "trace": trace,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2",
    )
    p.add_argument("--config", type=Path, default=Path("configs/v2/sim_so101.yaml"))
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    raw = RawDataset(a.root)
    results = []
    for case in json.loads(a.cases.read_text())["cases"]:
        for mode in a.modes:
            results.append(replay(raw, a.config, case, mode))
            (a.out / "progress.json").write_text(json.dumps({"completed": len(results)}))
    (a.out / "report.json").write_text(
        json.dumps(
            {
                "config_sha256": file_sha256(a.config),
                "cases_sha256": file_sha256(a.cases),
                "learned_policy_success": False,
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
