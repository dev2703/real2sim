"""Plan M5 step 2: validate >= 10 reachable 7-D target poses through the Panda IK.

Targets are sampled deterministically in the SO-101 workspace seen in the recordings (task-frame
TCP x 0.25-0.40, y -0.15..0.15, z 0.03-0.25, pointing down with small tilt/yaw jitter), solved
with the full-pose DLS IK, then sent through the environment (rate-limited upstream actuators)
to measure the pose actually reached. Every residual and rejection is written to
``results/metrics/panda_ik/ik_validation.json``. Nothing here is a policy result.

    uv run python -m real2sim.panda.validate_ik [--n 12] [--seed 0] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.geometry.so3 import exp_so3, log_so3, make_T
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import DEFAULT_CONFIG, POINTING_DOWN, ROOT, sha256_of

DEFAULT_OUT = ROOT / "results/metrics/panda_ik/ik_validation.json"
SAMPLE_MIN = np.array([0.25, -0.15, 0.03])
SAMPLE_MAX = np.array([0.40, 0.15, 0.25])
TILT_RAD = 0.15
YAW_RAD = 0.30
SETTLE_TICKS = 15


def sample_targets(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = []
    for _ in range(n):
        p = rng.uniform(SAMPLE_MIN, SAMPLE_MAX)
        tilt = rng.uniform(-TILT_RAD, TILT_RAD, size=2)  # about task x and y
        yaw = rng.uniform(-YAW_RAD, YAW_RAD)  # about task z
        R = exp_so3([tilt[0], tilt[1], yaw]) @ POINTING_DOWN
        targets.append(make_T(R, p))
    return targets


def validate(n: int = 12, seed: int = 0, config_path=DEFAULT_CONFIG) -> dict:
    env = PandaPickPlaceEnv(config_path, render_images=False)
    ik = PandaIK(env.model, env.config)
    cfg = env.config["ik"]
    rows = []
    for i, target in enumerate(sample_targets(n, seed)):
        env.reset(seed=seed)
        result = ik.solve(target, env.initial_qpos, finger_opening=env.initial_opening)
        row = {
            "index": i,
            "target_task_xyz": target[:3, 3].round(6).tolist(),
            "target_rotvec_from_down": log_so3(target[:3, :3] @ POINTING_DOWN.T).round(6).tolist(),
            "accepted": result.accepted,
            "reason": result.reason,
            "ik_residual_position_m": result.position_error_m,
            "ik_residual_orientation_rad": result.orientation_error_rad,
            "ik_iterations": result.iterations,
            "joint_solution_rad": result.qpos.round(6).tolist() if result.accepted else None,
        }
        if result.accepted:
            # Execute through the rate-limited upstream actuators and measure the reached pose.
            cmd = np.append(result.qpos, env.initial_opening)
            for _ in range(SETTLE_TICKS):
                env.step(cmd)
            T = env.ee_pose()
            row["executed_position_error_m"] = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
            row["executed_orientation_error_rad"] = float(
                np.linalg.norm(log_so3(target[:3, :3] @ T[:3, :3].T)))
            row["executed_ticks"] = SETTLE_TICKS
        rows.append(row)
    accepted = [r for r in rows if r["accepted"]]
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summary = {
        "n_targets": n,
        "n_accepted": len(accepted),
        "rejections_by_reason": {k: v for k, v in reasons.items() if k != "accepted"},
        "thresholds": {"position_m": cfg["position_tolerance_m"],
                       "orientation_rad": cfg["orientation_tolerance_rad"]},
        "ik_residual_position_m": _stats([r["ik_residual_position_m"] for r in accepted]),
        "ik_residual_orientation_rad": _stats([r["ik_residual_orientation_rad"] for r in accepted]),
        "executed_position_error_m": _stats([r["executed_position_error_m"] for r in accepted]),
        "executed_orientation_error_rad": _stats(
            [r["executed_orientation_error_rad"] for r in accepted]),
    }
    env.close()
    return {
        "controller_source": "IK_VALIDATION_DIAGNOSTIC",
        "note": "Plan M5 step 2 retargeting evidence only; no policy, no transfer claim.",
        "robot": "panda",
        "model": env.config["model"],
        "model_sha256": sha256_of(env.config["model"]),
        "sample_box_task_xyz": {"min": SAMPLE_MIN.tolist(), "max": SAMPLE_MAX.tolist(),
                                "tilt_rad": TILT_RAD, "yaw_rad": YAW_RAD},
        "seed": seed,
        "action_limits": env.config["action_limits"],
        "execution": "IK solutions sent directly as native joint commands for SETTLE_TICKS ticks; "
                     "the 7-D adapter (clamped deltas) is exercised in tests/integration/test_panda.py",
        "summary": summary,
        "targets": rows,
    }


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "max": float(arr.max()),
            "min": float(arr.min())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    report = validate(args.n, args.seed, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    s = report["summary"]
    print(f"accepted {s['n_accepted']}/{s['n_targets']}; rejections {s['rejections_by_reason']}")
    print(f"IK residual pos {s['ik_residual_position_m']} rot {s['ik_residual_orientation_rad']}")
    print(f"executed pos {s['executed_position_error_m']} rot {s['executed_orientation_error_rad']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
