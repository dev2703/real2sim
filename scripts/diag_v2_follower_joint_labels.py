"""Static original-gate feasibility of both target definitions on shared validation rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from real2sim.data.follower_joint_targets import VARIANTS, native_joint_rows
from real2sim.data.raw import sha256_file
from real2sim.policy.follower_joint import check_joint_request
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics


def diagnose(data: Path, sim: Path):
    manifests = {name: json.loads((data / name / "manifest.json").read_text()) for name in VARIANTS}
    arrays = {name: dict(np.load(data / name / "arrays.npz", allow_pickle=False)) for name in VARIANTS}
    first = arrays[VARIANTS[0]]
    selected = np.flatnonzero(np.isin(first["episode_index"], manifests[VARIANTS[0]]["split"]["val"]) & ~first["action_is_invalid"])
    report = {"kind": "SHARED_VALIDATION_JOINT_LABEL_STATIC_FEASIBILITY", "n": len(selected),
              "physical_calibration_accepted": False, "reserved_episodes_used": [], "variants": {},
              "sim_config_sha256": sha256_file(sim), "script_sha256": sha256_file(Path(__file__)),
              "assumptions": "Same provisional scene for all recorded poses; teacher-forced command ramp starts at observed current native joints because executed servo targets were not recorded.17static samples per path, original workspace/native/collision tolerances. No physics steps or path certification; native-only data mask remains immutable.",
              "rows": []}
    with So101PickPlaceEnv(sim, render_images=False) as env:
        env.reset(seed=0)
        k = NativeKinematics(env.model, env.config)
        current = native_joint_rows(first["joint_state"][selected], env.mapping)
        poses = []
        for q in native_joint_rows(first["joint_state"][selected + 1], env.mapping):
            poses.append(k.forward(q).copy())
        poses = np.asarray(poses)
        for name in VARIANTS:
            reasons, errors, rows = Counter(), [], []
            targets = native_joint_rows(arrays[name]["action"][selected], env.mapping)
            for i, tick in enumerate(selected):
                k.data.qpos[:] = env.data.qpos
                accepted, reason, command = check_joint_request(k, current[i], current[i], targets[i], env.max_velocity / env.control_hz, 17)
                position = k.forward(command)[:3, 3].copy()
                error = float(np.linalg.norm(position - poses[i, :3, 3]) * 1000)
                reasons[reason] += 1
                errors.append(error)
                rows.append({"tick": int(tick), "episode": int(first["episode_index"][tick]),
                             "source_frame": int(first["frame_index"][tick]), "accepted": accepted,
                             "reason": reason, "bounded_command_next_follower_fk_error_mm": error})
            report["variants"][name] = {"accepted": reasons["accepted"], "reason_counts": dict(reasons),
                                         "bounded_command_next_follower_fk_error_mm_mean": float(np.mean(errors)),
                                         "bounded_command_next_follower_fk_error_mm_p95": float(np.quantile(errors, .95))}
            report["rows"].append({"variant": name, "ticks": rows})
        report["original_ik_gates"] = env.config["ik"]
        report["native_joint_limits_rad"] = env.limits.tolist()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(".cache/v2_follower_joint_20261024"))
    p.add_argument("--sim", type=Path, default=Path("outputs/v2/inference_sim_smoke_20261023/sim.yaml"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = diagnose(args.data, args.sim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["variants"], indent=2))
