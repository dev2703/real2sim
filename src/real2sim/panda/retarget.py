"""SO-101 -> Panda retargeting engineering baseline (plan M5 step 3).

Replays the canonical 7-D achieved-motion labels of one recorded SO-101 episode
(``real2sim.data.labels.episode_labels``: pose deltas from ``So101FK`` of the achieved follower
joints, gripper = leader command per D-013) on the Panda through
``PandaCartesianActionAdapter.apply``. Every delta is applied relative to the Panda's **observed**
TCP pose of the tick, exactly as a learned Cartesian policy's output would be, starting from the
Panda start pose of ``configs/sim_panda.yaml``.

The SO-101 and Panda start poses differ (the Panda cannot reach the SO-101 folded rest pose,
D-015). The initial TCP offset ``FK(real start state)`` -> Panda TCP start is recorded and
reported, not corrected: the raw position error between the Panda TCP and the SO-101 FK pose at
the same tick therefore contains that constant offset plus whatever the relative execution
accumulates (rejected ticks hold the arm, so the error never resets). The
``offset_corrected`` columns subtract the start offset and compare the *relative* motion only;
that is the actual retargeting tracking error.

Every artefact is labelled ``SCRIPTED REPLAY`` (controller_source ``SCRIPTED_REPLAY``). This is
controller feasibility evidence only and never an ACT rollout (AGENTS.md rule 4). The ball starts
at the seeded task.yaml start region, which is independent of the real episode's unknown ball
position, so the privileged ball outcome is reported but is not a success claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from real2sim.data.labels import episode_labels
from real2sim.data.raw import resolve_dataset_root
from real2sim.evaluation.replay import _Video, draw_line_plot, load_episode_ticks
from real2sim.geometry.so3 import log_so3
from real2sim.geometry.so101_fk import JointMapping, So101FK
from real2sim.panda.adapter import PandaCartesianActionAdapter
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.panda.model import DEFAULT_CONFIG, ROOT, project_path

CONTROLLER_SOURCE = "SCRIPTED_REPLAY"
DELTA_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz")
DEFAULT_OUT = ROOT / "results/metrics/panda_retarget"


def caption_for(episode: int) -> str:
    """Burned-in caption; contains 'SCRIPTED REPLAY' and never 'LEARNED'."""
    return f"SCRIPTED REPLAY (SO-101 -> Panda retarget) ep{episode}"


def classify_contacts(contacts: list[dict]) -> dict:
    """Split the env's privileged contact list by what the robot touches.

    Robot geoms of the Menagerie Panda are unnamed (``geom_N``); task geoms are named. A contact
    is robot-table / robot-bowl / robot-ball when exactly one side is the named task geom; the
    ball resting on the table or in the bowl is not a robot contact.
    """
    def kind(name: str) -> str:
        if name == "table":
            return "table"
        if name.startswith("bowl_"):
            return "bowl"
        if name == "ball_geom":
            return "ball"
        return "robot" if name.startswith("geom_") else "other"

    counts = {"robot_table": 0, "robot_bowl": 0, "robot_ball": 0, "robot_self": 0}
    deepest_table = 0.0
    for c in contacts:
        a, b = kind(c["geom1"]), kind(c["geom2"])
        pair = {a, b}
        if pair == {"robot"}:
            counts["robot_self"] += 1
        elif pair == {"robot", "table"}:
            counts["robot_table"] += 1
            deepest_table = max(deepest_table, -float(c["distance_m"]))
        elif pair == {"robot", "bowl"}:
            counts["robot_bowl"] += 1
        elif pair == {"robot", "ball"}:
            counts["robot_ball"] += 1
    counts["robot_table_penetration_m"] = deepest_table
    return counts


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def run_retarget(episode: int, local_root: Path, out_dir: Path, video_dir: Path | None, *,
                 max_ticks: int | None = None, video_stride: int = 1,
                 config_path: str | Path = DEFAULT_CONFIG, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    env = PandaPickPlaceEnv(config_path, render_images=False)
    data_cfg = env.data_config
    hz = float(data_cfg["policy_hz"])
    ep = load_episode_ticks(local_root, episode, hz)
    S, A = ep["state"], ep["action"]
    n_total = len(S)
    n = n_total if max_ticks is None else min(n_total, int(max_ticks))
    caption = caption_for(episode)
    render = video_dir is not None

    # SO-101 FK of the achieved real joints (task frame = SO-101 base, T_task_so101base identity).
    fk = So101FK(mapping=JointMapping.from_task_config(env.task))
    real_poses = fk.fk_real(S)
    lcfg = data_cfg["labels"]
    gsrc = lcfg["cartesian_gripper_source"]
    lead = int(lcfg.get("cartesian_lead_ticks", 1))
    labels = episode_labels(real_poses, S[:, 5], A[:, 5], hz=hz, gripper_source=gsrc, lead=lead,
                            leader_poses=fk.fk_real(A))

    adapter = PandaCartesianActionAdapter(env)
    obs, info = env.reset(seed=seed)
    env_cap = env.max_episode_steps
    episode_cap_lifted = n > env_cap
    if episode_cap_lifted:
        env.max_episode_steps = n + 1  # diagnostic: cover the whole recording
    T_panda0 = adapter.pose_from_state(obs[adapter.state_key])
    T_real0 = real_poses[0]
    start_offset_m = T_panda0[:3, 3] - T_real0[:3, 3]
    start_rot_offset = log_so3(T_panda0[:3, :3] @ T_real0[:3, :3].T)
    ball_start = np.asarray(info["privileged_metrics"]["ball_xyz_task"], float)

    video = _Video(video_dir / f"panda_retarget_ep{episode}.mp4" if render else None,
                   hz / video_stride, caption)
    rows: list[dict] = []
    truncated = False
    any_table = False
    joint_names = list(env.config["joint_names"])
    for k in range(n):
        action = labels.action[k]
        # Diagnostic only (not a policy input): which arm joint the IK seed is closest to a
        # limit of, so joint_limit rejections can be attributed (D-015 reach caveat).
        q_seed = env.arm_qpos()
        margins = np.minimum(q_seed - env.limits[:, 0], env.limits[:, 1] - q_seed)
        nearest = int(np.argmin(margins))
        res = adapter.apply(action, obs)
        obs, info, truncated = res.observation, res.info, res.truncated
        pm = info["privileged_metrics"]
        T_panda = adapter.pose_from_state(obs[adapter.state_key])
        j = min(k + lead, n_total - 1)
        T_ref = real_poses[j]  # pose the label points at
        p_err_raw = T_panda[:3, 3] - T_ref[:3, 3]
        p_err_corr = p_err_raw - start_offset_m
        # Relative rotation since the start on each robot; equal if the deltas were tracked.
        R_rel_panda = T_panda[:3, :3] @ T_panda0[:3, :3].T
        R_rel_real = T_ref[:3, :3] @ T_real0[:3, :3].T
        rot_err_corr = float(np.linalg.norm(log_so3(R_rel_panda @ R_rel_real.T)))
        contacts = classify_contacts(pm["contacts"])
        any_table |= contacts["robot_table"] > 0
        row = {"tick": k, "timestamp_s": float(ep["timestamp"][k]),
               "ik_accepted": int(not res.rejected), "ik_reason": res.reason,
               "ik_pos_residual_m": res.residual_position_m,
               "ik_rot_residual_rad": res.residual_orientation_rad,
               "ik_iterations": res.ik_iterations,
               "translation_clamped": int(res.translation_clamped),
               "rotation_clamped": int(res.rotation_clamped),
               "label_extrapolated": int(labels.label_is_extrapolated[k]),
               "label_jump": int(labels.jump_flags[k]),
               "seed_nearest_limit_joint": joint_names[nearest],
               "seed_limit_margin_rad": float(margins[nearest])}
        for name, v in zip(DELTA_NAMES, action[:6], strict=True):
            row[f"cmd_{name}"] = float(v)
        row["cmd_gripper_target"] = float(action[6])
        for name, v in zip("xyz", res.target_pose[:3, 3], strict=True):
            row[f"target_tcp_{name}_m"] = float(v)
        for name, v in zip("xyz", T_panda[:3, 3], strict=True):
            row[f"panda_tcp_{name}_m"] = float(v)
        for name, v in zip("xyz", T_ref[:3, 3], strict=True):
            row[f"so101_fk_{name}_m"] = float(v)
        row["pos_err_raw_m"] = float(np.linalg.norm(p_err_raw))
        row["pos_err_offset_corrected_m"] = float(np.linalg.norm(p_err_corr))
        row["rot_err_offset_corrected_rad"] = rot_err_corr
        row["panda_gripper_opening_m"] = float(pm["gripper_opening_m"])
        row["panda_gripper_semantic"] = float(obs[adapter.state_key][12])
        row["so101_gripper_obs_pct"] = float(S[k, 5])
        row["robot_table_contacts"] = contacts["robot_table"]
        row["robot_table_penetration_m"] = contacts["robot_table_penetration_m"]
        row["robot_bowl_contacts"] = contacts["robot_bowl"]
        row["robot_ball_contacts"] = contacts["robot_ball"]
        row["ball_held"] = int(pm["ball_held"])
        row["lifted"] = int(pm["lifted"])
        row["ball_in_bowl"] = int(pm["ball_in_bowl"])
        row["ee_ball_distance_m"] = float(pm["ee_ball_distance_m"])
        rows.append(row)
        if render and k % video_stride == 0:
            ik_txt = "ok" if not res.rejected else res.reason
            video.add(env.render(),
                      f"tick {k}/{n}  ik={ik_txt}  rel_err {row['pos_err_offset_corrected_m'] * 100:.1f} cm"
                      f"  raw_err {row['pos_err_raw_m'] * 100:.1f} cm  controller={CONTROLLER_SOURCE}")
        if truncated:
            break
    video.close()
    final_metrics = env.info()["privileged_metrics"]
    env.close()
    if not rows:
        raise RuntimeError("no ticks replayed")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"panda_retarget_ep{episode}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    acc = np.array([r["ik_accepted"] for r in rows], bool)
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["ik_reason"]] = reasons.get(r["ik_reason"], 0) + 1
    pos_res = np.array([r["ik_pos_residual_m"] for r in rows])
    rot_res = np.array([r["ik_rot_residual_rad"] for r in rows])
    err_raw = np.array([r["pos_err_raw_m"] for r in rows])
    err_corr = np.array([r["pos_err_offset_corrected_m"] for r in rows])
    rot_corr = np.array([r["rot_err_offset_corrected_rad"] for r in rows])
    iters = np.array([r["ik_iterations"] for r in rows])

    def stats(v: np.ndarray) -> dict:
        return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}

    summary = {
        "controller_source": CONTROLLER_SOURCE, "caption": caption, "episode": episode,
        "robot": "panda", "dataset_root": str(local_root), "policy_hz": hz, "stride": ep["stride"],
        "source_fps": ep["fps"], "n_rows": ep["n_rows"], "n_ticks_available": n_total,
        "n_ticks_replayed": len(rows), "truncated_by_env": bool(truncated),
        "episode_cap_lifted": {"lifted": bool(episode_cap_lifted), "env_max_episode_steps": env_cap},
        "labels": {"source": "achieved motion FK(q_obs[k+lead]) - FK(q_obs[k]) (D-003)",
                   "gripper_source": gsrc, "lead_ticks": lead,
                   "jump_flags": int(labels.jump_flags[:len(rows)].sum()),
                   "extrapolated_ticks": int(labels.label_is_extrapolated[:len(rows)].sum())},
        "start_pose": {
            "so101_fk_start_task_xyz_m": T_real0[:3, 3].tolist(),
            "panda_tcp_start_task_xyz_m": T_panda0[:3, 3].tolist(),
            "offset_panda_minus_so101_m": start_offset_m.tolist(),
            "offset_norm_m": float(np.linalg.norm(start_offset_m)),
            "rotation_offset_rad": float(np.linalg.norm(start_rot_offset)),
            "note": "not corrected: deltas are executed relative to the Panda's own observed TCP pose "
                    "(D-015: the Panda cannot reach the SO-101 folded rest pose)",
        },
        "ik": {
            "acceptance_rate": float(acc.mean()), "accepted": int(acc.sum()), "ticks": len(rows),
            "reasons": reasons,
            "pos_residual_m": stats(pos_res),
            "pos_residual_accepted_m": stats(pos_res[acc]) if acc.any() else None,
            "rot_residual_rad": stats(rot_res),
            "rot_residual_accepted_rad": stats(rot_res[acc]) if acc.any() else None,
            "iterations": stats(iters.astype(float)),
            "translation_clamped_ticks": int(sum(r["translation_clamped"] for r in rows)),
            "rotation_clamped_ticks": int(sum(r["rotation_clamped"] for r in rows)),
            "rejected_ticks": [r["tick"] for r in rows if not r["ik_accepted"]],
            "rejected_nearest_limit_joint": _count(r["seed_nearest_limit_joint"] for r in rows
                                                  if not r["ik_accepted"]),
            "lost_motion_on_rejected_ticks_m": float(sum(
                float(np.linalg.norm([r[f"cmd_{d}"] for d in DELTA_NAMES[:3]]))
                for r in rows if not r["ik_accepted"])),
            "clamp": {"max_translation_m": adapter.max_translation, "max_rotation_rad": adapter.max_rotation},
            "note": "rejected ticks hold the arm joints observed at the start of the tick (gripper still "
                    "applied), so the relative-motion error accumulates rather than resets",
        },
        "position_error_cm": {
            "raw_vs_so101_fk": {k: v * 100 for k, v in stats(err_raw).items()},
            "offset_corrected": {k: v * 100 for k, v in stats(err_corr).items()},
            "offset_corrected_first_tick_cm": float(err_corr[0] * 100),
            "offset_corrected_last_tick_cm": float(err_corr[-1] * 100),
            "note": "raw = |Panda TCP - FK(real state at tick+lead)|: contains the constant start-pose "
                    "offset and grows with relative execution; offset_corrected subtracts the start "
                    "offset and measures the relative-motion tracking only",
        },
        "rotation_error_offset_corrected_rad": stats(rot_corr),
        "gripper": {
            "panda_opening_m": stats(np.array([r["panda_gripper_opening_m"] for r in rows])),
            "min_opening_m": float(min(r["panda_gripper_opening_m"] for r in rows)),
            "cmd_target_min": float(min(r["cmd_gripper_target"] for r in rows)),
            "cmd_target_max": float(max(r["cmd_gripper_target"] for r in rows)),
        },
        "contacts": {
            "robot_table_any_tick": bool(any_table),
            "robot_table_ticks": int(sum(r["robot_table_contacts"] > 0 for r in rows)),
            "robot_table_max_penetration_m": float(max(r["robot_table_penetration_m"] for r in rows)),
            "robot_bowl_ticks": int(sum(r["robot_bowl_contacts"] > 0 for r in rows)),
            "robot_ball_ticks": int(sum(r["robot_ball_contacts"] > 0 for r in rows)),
            "note": "privileged contact list; a robot-table contact means the retargeted path hit the table",
        },
        "ball_outcome": {
            "ball_start_task_xyz": ball_start.tolist(),
            "ball_in_bowl_final": bool(final_metrics["ball_in_bowl"]),
            "ball_held_any_tick": bool(any(r["ball_held"] for r in rows)),
            "lifted_any_tick": bool(any(r["lifted"] for r in rows)),
            "min_ee_ball_distance_m": float(min(r["ee_ball_distance_m"] for r in rows)),
            "ball_xyz_task_final": final_metrics["ball_xyz_task"],
            "note": "privileged metric; the ball starts at the seeded task.yaml start region, not at the "
                    "real episode's unknown ball position, so no grasp is expected and this is not a "
                    "success claim",
        },
        "video": {"path": str(video.path) if video.path else None, "frames": video.frames,
                  "video_stride": video_stride},
        "wall_time_s": time.perf_counter() - t0,
        "not_learned": "Scripted replay of recorded SO-101 labels retargeted onto the Panda. It is not "
                       "an ACT rollout and not a transfer result.",
    }
    (out_dir / f"panda_retarget_ep{episode}.json").write_text(json.dumps(summary, indent=2))
    draw_line_plot({"raw_err_cm": err_raw * 100, "offset_corrected_err_cm": err_corr * 100,
                    "ik_rejected": (~acc).astype(float)},
                   f"{caption}: TCP position error (cm) and IK rejections", "cm / flag",
                   out_dir / f"panda_retarget_ep{episode}.png")
    summary["csv"] = str(csv_path)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="SO-101 -> Panda retargeting baseline: scripted replay, never a learned rollout")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--local-root", type=Path, default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--video", type=Path, default=None, help="directory for the mp4; omitted = no video")
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--video-stride", type=int, default=1, help="render every n-th tick")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = resolve_dataset_root(a.repo_id, a.revision,
                                a.local_root.expanduser() if a.local_root else None)
    s = run_retarget(a.episode, root, project_path(a.out), project_path(a.video) if a.video else None,
                     max_ticks=a.max_ticks, video_stride=a.video_stride, config_path=a.config,
                     seed=a.seed)
    brief = {k: s[k] for k in ("caption", "controller_source", "n_ticks_replayed", "start_pose", "ik",
                               "position_error_cm", "rotation_error_offset_corrected_rad", "contacts",
                               "ball_outcome", "wall_time_s")}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
