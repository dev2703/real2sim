"""DIAGNOSTIC: does removing the first-tick table-contact kick change the joint ACT rollout?

Condition ``settle0`` reproduces the primary E1 start (policy infers on the pre-kick observation).
Condition ``settleN`` commands the reset pose for N hold ticks first, so the arm has left the
table before the first inference. Per-tick trace: predicted command, achieved state, EE-ball
distance. Not a benchmark result; never joins the primary tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TrialTracker
from real2sim.evaluation.run import build_learned_controller, make_env

J = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/act_joint_so101/checkpoints/020000")
    )
    p.add_argument(
        "--out", type=Path, default=Path("results/metrics/debug_so101_joint/settled_start")
    )
    p.add_argument("--settle", type=int, nargs="+", default=[0, 5])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--max-ticks", type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = make_env("so101", Path("configs/sim_so101.yaml"))
    eval_cfg = load_yaml(Path("configs/eval_so101.yaml"))
    ctrl, contract, meta = build_learned_controller(env, a.checkpoint, eval_cfg, "cpu")
    task = env.task
    report = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "note": __doc__,
        "conditions": {},
    }
    for settle in a.settle:
        name = f"settle{settle}"
        rows_summary, trace = [], []
        for seed in range(a.trials):
            obs, info = env.reset(seed=seed)
            q0 = env.sim_state()
            for _ in range(settle):
                obs, _, _, _, info = env.step(q0)
            ctrl.reset()
            tracker = TrialTracker(
                success_hold_ticks=10,
                approach_distance_m=float(eval_cfg["phase"]["approach_distance_m"]),
                table_top_z=env.table_top,
                ball_radius_m=float(task["ball"]["radius"]),
            )
            start_state = env.real_state().tolist()
            for tick in range(1, a.max_ticks + 1):
                res = ctrl.step(obs, env)
                obs = res.obs
                pm = res.info["privileged_metrics"]
                done = tracker.update(pm, saturation=res.saturation, ik_rejected=res.ik_rejected)
                if tick <= 60 or tick % 10 == 0:
                    trace.append(
                        {
                            "condition": name,
                            "seed": seed,
                            "tick": tick,
                            **{
                                f"cmd_{j}": round(float(v), 2)
                                for j, v in zip(J, res.detail["action"], strict=False)
                            },
                            **{
                                f"state_{j}": round(float(v), 2)
                                for j, v in zip(J, env.real_state(), strict=False)
                            },
                            "clipped": int(sum(res.detail["limit_clipped"])),
                            "ee_ball_m": round(float(pm["ee_ball_distance_m"]), 4),
                            "tool_z_task": round(float(pm["tool_xyz_task"][2]), 4),
                        }
                    )
                if done or res.truncated:
                    break
            r = tracker.row(
                experiment_id=f"DIAGNOSTIC_{name}",
                controller_source="LEARNED_ACT",
                robot="so101",
                seed=seed,
                checkpoint_sha256=meta["checkpoint_sha256"],
                checkpoint_step=meta["checkpoint_step"],
                mean_latency_ms=float(np.mean(ctrl.latencies_ms)),
                video_path="",
            )
            r.update({"start_state_real": [round(x, 1) for x in start_state]})
            rows_summary.append(r)
            print(
                name,
                seed,
                {
                    k: r[k]
                    for k in r
                    if k in ("success", "phase_failed", "min_ee_ball_distance_m", "lifted", "ticks")
                },
            )
        report["conditions"][name] = {"settle_ticks": settle, "trials": rows_summary}
        with (a.out / f"trace_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)
    env.close()
    (a.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
