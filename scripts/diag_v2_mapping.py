"""R0: read-only v2 mapping/reset audit. Never clamp references or invent valid starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import RawDataset, sha256_file
from real2sim.geometry.so3 import delta_action
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics


def audit(root: Path, config: Path):
    ds = RawDataset(root)
    rows = []
    with So101PickPlaceEnv(config, render_images=False) as env:
        kin = NativeKinematics(env.model, env.config)
        for episode in ds.episode_indices:
            g = ds.episode_rows(episode)
            states = ds.stack(g, "observation.state")[::2]
            q = np.array([sim_from_real(env.mapping, s) for s in states])
            outside = (q < env.limits[:, 0]) | (q > env.limits[:, 1])
            candidates = np.flatnonzero(~outside.any(axis=1))
            first_valid = None
            for index in candidates:
                _, info = env.reset(
                    seed=0, scenario={"joint_positions": q[index], "ball_xy": [0.5, 0.3]}
                )
                pen = info["privileged_metrics"]["robot_table_penetration_m"]
                if pen <= 1e-6:
                    first_valid = int(index)
                    break
            reconstruction = []
            if first_valid is not None:
                for index in sorted(
                    set([first_valid, *candidates[:: max(1, len(candidates) // 8)].tolist()])
                ):
                    target = kin.forward(q[index])
                    # A different bounded seed prevents a trivial identity reconstruction.
                    seed = q[index].copy()
                    seed[:5] += np.where(seed[:5] < env.limits[:5].mean(axis=1), 0.01, -0.01)
                    result = kin.solve(target, seed, gripper=float(q[index, 5]))
                    reconstructed = kin.forward(result.qpos)
                    error = delta_action(reconstructed, target)
                    reconstruction.append(
                        {
                            "source_frame": int(index) * 2,
                            "accepted": result.accepted,
                            "position_error_mm": float(np.linalg.norm(error[:3]) * 1000),
                            "rotation_error_rad": float(np.linalg.norm(error[3:])),
                            "reason": result.reason,
                        }
                    )
            rows.append(
                {
                    "episode_index": episode,
                    "sampled_rows": len(q),
                    "start_out_of_limits": bool(outside[0].any()),
                    "start_outside_by_joint": outside[0].tolist(),
                    "outside_fraction_by_joint": outside.mean(axis=0).tolist(),
                    "first_nonpenetrating_in_limit_source_frame": None
                    if first_valid is None
                    else first_valid * 2,
                    "candidate_is_approved_task_start": False,
                    "fk_ik_reconstruction": reconstruction,
                }
            )
        _, original = env.reset(seed=0)
        configured_real = sim_from_real(env.mapping, env.config["initial_qpos_real_deg"])
        default_clipping = configured_real - env.initial_qpos
        task_path = Path(env.config["task_config"])
        return {
            "diagnostic": "R0 provisional mapping, not physical calibration",
            "raw_root": str(root),
            "raw_files_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for directory in ("data", "meta")
                for p in sorted((root / directory).rglob("*"))
                if p.is_file()
            },
            "config_sha256": sha256_file(config),
            "task_sha256": sha256_file(task_path),
            "source_hz": ds.fps,
            "sample_stride": 2,
            "nominal_policy_hz": ds.fps / 2,
            "physical_cadence_verified": False,
            "joint_names": env.config["joint_names"],
            "original_start": {
                "native_difference_from_requested": default_clipping.tolist(),
                "metrics": original["privileged_metrics"]["start_pose"],
            },
            "episodes": rows,
            "starts_outside_limits": sum(r["start_out_of_limits"] for r in rows),
            "warning": "A later valid pose is only a candidate. Phase annotations must establish that the full grasp-and-place sequence remains.",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2",
    )
    p.add_argument("--config", type=Path, default=Path("configs/sim_so101.yaml"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = audit(a.root, a.config)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("starts_outside_limits", "source_hz", "nominal_policy_hz")}
        )
    )


if __name__ == "__main__":
    main()
