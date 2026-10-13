"""SCRIPTED replay of saved ACT-adapter targets, for contact diagnosis only."""

import json
from pathlib import Path

import numpy as np

from real2sim.sim.env import So101PickPlaceEnv

ROOT = Path("results/metrics/act_response_20261013")


def main():
    out = ROOT / "contact_audit.json"
    if out.exists():
        raise FileExistsError(out)
    summary = json.loads((ROOT / "summary.json").read_text())
    traces = json.loads((ROOT / "response_traces_0.json").read_text())
    env = So101PickPlaceEnv("results/metrics/geometry_3988577/configs/sim_so101.yaml", render_images=False)
    report = {"kind": "SCRIPTED_SAVED_ACT_COMMAND_REPLAY", "trials": [],
              "note": "No new policy inference, no training. Privileged contacts are inspected only "
                      "after saved trajectories were generated. Original benchmark rows are unchanged."}
    try:
        for trial, trace in zip(summary["trials"], traces, strict=True):
            env.reset(seed=trial["seed"])
            contacts = []
            closest = None
            max_state_error, max_ball_error = 0., 0.
            for row, history in zip(trace, trial["history"], strict=True):
                max_state_error = max(max_state_error,
                    float(np.max(abs(env.real_state() - row["observed_state"]))))
                _, _, _, _, info = env.step_real_joint_targets(row["applied_target"])
                metrics = info["privileged_metrics"]
                max_ball_error = max(max_ball_error,
                    float(np.max(abs(np.asarray(metrics["ball_xyz_task"]) - history["ball_xyz_task"]))))
                bodies = sorted({env.model.body(int(env.model.geom_bodyid[g])).name
                                 for g in env._ball_contacts() if g != env.table_geom})
                if bodies:
                    contacts.append({"tick": history["tick"], "bodies": bodies,
                        "held": metrics["ball_held"], "policy_gripper": row["policy_target"][5]})
                if closest is None or metrics["ee_ball_distance_m"] < closest["distance_m"]:
                    closest = {"tick": history["tick"], "distance_m": metrics["ee_ball_distance_m"],
                        "tool_minus_ball_m": (np.asarray(metrics["tool_xyz_task"]) - metrics["ball_xyz_task"]).tolist(),
                        "policy_gripper": row["policy_target"][5], "observed_gripper": env.real_state()[5].item(),
                        "response_active": row["response_active"]}
            if max_state_error > 1e-5 or max_ball_error > 1e-8:
                raise RuntimeError("saved command replay diverged from original trajectory")
            report["trials"].append({"seed": trial["seed"], "closest": closest, "robot_ball_contacts": contacts,
                "max_state_replay_error": max_state_error, "max_ball_replay_error_m": max_ball_error})
        out.write_text(json.dumps(report, indent=2) + "\n")
        for r in report["trials"]:
            print(r["seed"], "contacts", len(r["robot_ball_contacts"]), "closest", r["closest"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
