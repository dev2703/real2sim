"""Replay frozen ACT commands and inspect every physics substep; no control changes."""

import json
from pathlib import Path

import mujoco
import numpy as np

from real2sim.sim.env import So101PickPlaceEnv


def main():
    root = Path("results/metrics/camera_refit_verified_20261014/camera_objects")
    out = Path("results/audit/grasp_contacts_20261014")
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((root / "paired_response/summary.json").read_text())
    traces = json.loads((root / "paired_response/response_traces_0.json").read_text())
    report = {"kind": "PRIVILEGED_PHYSICS_SUBSTEP_AUDIT_OF_SAVED_ACT_COMMANDS", "trials": []}
    original_step = mujoco.mj_step
    with So101PickPlaceEnv(root / "configs/sim_so101.yaml", render_images=False) as env:
        for trial, trace in zip(summary["trials"], traces, strict=True):
            env.reset(seed=trial["seed"])
            events = []
            counts = {}
            deepest = {}
            first_seen = set()
            first_movement = None
            origin = env.data.xpos[env.ball_body].copy()
            max_error = 0.

            def inspect_step(model, data, *, counts=counts, deepest=deepest,
                             first_seen=first_seen, events=events, origin=origin):
                nonlocal first_movement
                # Contact state used by the imminent physics step; no extra FK or
                # force calls that could alter solver state or warm-start behavior.
                for contact in data.contact[:data.ncon]:
                    pair = (int(contact.geom1), int(contact.geom2))
                    if env.ball_geom not in pair:
                        continue
                    other = pair[1] if pair[0] == env.ball_geom else pair[0]
                    if other == env.table_geom:
                        continue
                    name = model.geom(other).name or f"geom_{other}"
                    counts[name] = counts.get(name, 0) + 1
                    deepest[name] = min(deepest.get(name, 0.), float(contact.dist))
                    if name not in first_seen:
                        first_seen.add(name)
                        T = env.ee_pose()
                        ball_task = data.xpos[env.ball_body] - env.offset
                        events.append({"time_s": float(data.time), "geom": name,
                            "body": model.body(int(model.geom_bodyid[other])).name,
                            "penetration_m": max(0., -float(contact.dist)),
                            "ball_in_tool": (T[:3, :3].T @ (ball_task - T[:3, 3])).tolist(),
                            "state": env.real_state().tolist()})
                if first_movement is None and np.linalg.norm(data.xpos[env.ball_body, :2] - origin[:2]) > .01:
                    first_movement = float(data.time)
                return original_step(model, data)

            mujoco.mj_step = inspect_step
            try:
                for row, history in zip(trace, trial["history"], strict=True):
                    _, _, _, _, info = env.step_real_joint_targets(row["applied_target"])
                    max_error = max(max_error, float(np.max(abs(
                        np.array(info["privileged_metrics"]["ball_xyz_task"]) - history["ball_xyz_task"]))))
            finally:
                mujoco.mj_step = original_step
            assert max_error < 1e-8, max_error
            result = {"seed": trial["seed"], "first_contacts": events,
                      "contact_substeps_by_geom": counts, "minimum_contact_distance_by_geom": deepest,
                      "ball_moves_1cm_at_s": first_movement, "max_ball_replay_error_m": max_error}
            report["trials"].append(result)
            print(trial["seed"], [(round(e["time_s"], 3), e["geom"]) for e in events], flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
