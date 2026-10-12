"""E0 replay diagnostic (plan M2 step 5, section 7 row E0 ``raw_replay``).

Reproduces one recorded SO-101 episode in the MuJoCo scene without any learned inference:

* ``joint``:     the achieved real joints (``observation.state``, LeRobot degrees / gripper %)
                 are sent tick by tick as position setpoints (``env.step_real_joint_targets``).
* ``cartesian``: the canonical 7-D achieved-motion labels (``real2sim.data.labels``) are applied
                 relative to the *current simulated* EE pose through the diagnostic IK
                 (``NativeKinematics``), with the architecture section 4.2 per-tick clamps.

Every artefact is labelled ``SCRIPTED REPLAY`` (controller_source ``SCRIPTED_REPLAY``); this is
calibration/control feasibility evidence only and never an ACT rollout (AGENTS.md rule 4).
The ball starts at the task.yaml start region, which is independent of the real episode, so the
privileged ``ball_in_bowl`` outcome is reported but is not a success claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from real2sim.config import load_yaml
from real2sim.data.labels import episode_labels
from real2sim.data.raw import RawDataset, resolve_dataset_root
from real2sim.data.resample import episode_tick_rows, policy_stride
from real2sim.geometry.so3 import apply_delta
from real2sim.geometry.so101_fk import So101FK
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real
from real2sim.sim.kinematics import NativeKinematics
from real2sim.sim.model import DEFAULT_CONFIG, ROOT

CONTROLLER_SOURCE = "SCRIPTED_REPLAY"
MODES = ("joint", "cartesian")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_UNITS = ("deg", "deg", "deg", "deg", "deg", "pct")
# Architecture section 4.2 provisional per-tick action clamp (shared with the ACT controller).
DEFAULT_MAX_TRANSLATION_M = 0.03
DEFAULT_MAX_ROTATION_RAD = 0.3
# Ticks excluded from the labelled "after_settle" statistics (3 s at 10 Hz).
SETTLE_TICKS = 30


def caption_for(mode: str, episode: int) -> str:
    """Burned-in caption; contains 'SCRIPTED REPLAY' and never 'LEARNED'."""
    detail = {"joint": "joint setpoints", "cartesian": "FK->IK"}[mode]
    return f"SCRIPTED REPLAY ({detail}) ep{episode}"


def load_episode_ticks(root: Path, episode: int, policy_hz: float) -> dict:
    """Raw rows of one episode subsampled to the policy clock (every ``stride``-th row)."""
    ds = RawDataset(root)
    rows = ds.episode_rows(episode)
    if len(rows) == 0:
        raise KeyError(f"episode {episode} has no rows")
    stride = policy_stride(ds.fps, policy_hz)
    idx = episode_tick_rows(len(rows), stride)
    return {
        "state": ds.stack(rows, "observation.state")[idx],
        "action": ds.stack(rows, "action")[idx],
        "timestamp": rows["timestamp"].to_numpy(np.float64)[idx],
        "stride": stride,
        "fps": ds.fps,
        "n_rows": len(rows),
        "state_names": ds.state_names,
    }


def _start_scenario(env: So101PickPlaceEnv, first_state_real: np.ndarray) -> tuple[np.ndarray, list]:
    """Native start pose = clamped mapping of the episode's first real state."""
    q = sim_from_real(env.mapping, first_state_real)
    clamped = np.clip(q, env.limits[:, 0], env.limits[:, 1])
    clamps = [
        {"joint": JOINTS[i], "requested_rad": float(q[i]), "applied_rad": float(clamped[i]),
         "clamp_deg": float(np.rad2deg(abs(q[i] - clamped[i])))}
        for i in range(6) if abs(q[i] - clamped[i]) > 1e-9
    ]
    return clamped, clamps


def _clamp_delta(delta6: np.ndarray, max_t: float, max_r: float) -> tuple[np.ndarray, bool]:
    d = np.asarray(delta6, np.float64).copy()
    clipped = False
    nt, nr = np.linalg.norm(d[:3]), np.linalg.norm(d[3:6])
    if nt > max_t:
        d[:3] *= max_t / nt
        clipped = True
    if nr > max_r:
        d[3:6] *= max_r / nr
        clipped = True
    return d, clipped


class _Video:
    def __init__(self, path: Path | None, fps: float, caption: str):
        self.path, self.fps, self.caption, self.writer = path, fps, caption, None
        self.frames = 0

    def add(self, rgb: np.ndarray, line2: str) -> None:
        if self.path is None:
            return
        frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter could not open {self.path}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(frame, self.caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def draw_line_plot(series: dict[str, np.ndarray], title: str, ylabel: str, path: Path,
                   size: tuple[int, int] = (1100, 420)) -> None:
    """Minimal multi-series line plot with cv2 (matplotlib is not a dependency)."""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    left, right, top, bottom = 70, 220, 40, 50
    pw, ph = w - left - right, h - top - bottom
    n = max(len(v) for v in series.values())
    finite = np.concatenate([v[np.isfinite(v)] for v in series.values()]) if series else np.zeros(1)
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    lo, hi = min(lo, 0.0), max(hi, lo + 1e-6)
    colours = [(200, 60, 60), (60, 140, 60), (60, 60, 200), (0, 140, 200), (160, 40, 160), (40, 40, 40)]

    def xy(i, v):
        return (int(left + pw * i / max(n - 1, 1)), int(top + ph * (1 - (v - lo) / (hi - lo))))

    cv2.rectangle(img, (left, top), (left + pw, top + ph), (0, 0, 0), 1)
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = xy(0, v)[1]
        cv2.line(img, (left, y), (left + pw, y), (220, 220, 220), 1)
        cv2.putText(img, f"{v:.1f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for k in range(5):
        i = int((n - 1) * k / 4)
        x = xy(i, lo)[0]
        cv2.putText(img, str(i), (x - 8, h - bottom + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for j, (name, v) in enumerate(series.items()):
        c = colours[j % len(colours)]
        pts = [xy(i, float(x)) for i, x in enumerate(v) if np.isfinite(x)]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            cv2.line(img, a, b, c, 1, cv2.LINE_AA)
        cv2.putText(img, name, (left + pw + 10, top + 16 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    cv2.putText(img, title, (left, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "tick (10 Hz)", (left + pw // 2 - 30, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, ylabel, (4, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def run_replay(episode: int, mode: str, local_root: Path, out_dir: Path, video_dir: Path | None, *,
               max_ticks: int | None = None, video_stride: int = 1, config_path=DEFAULT_CONFIG,
               max_translation_m: float = DEFAULT_MAX_TRANSLATION_M,
               max_rotation_rad: float = DEFAULT_MAX_ROTATION_RAD, seed: int = 0) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    sim_cfg = load_yaml(config_path)
    data_cfg = load_yaml(ROOT / sim_cfg["data_config"])
    hz = float(data_cfg["policy_hz"])
    ep = load_episode_ticks(local_root, episode, hz)
    S, A = ep["state"], ep["action"]
    n_total = len(S)
    n = n_total if max_ticks is None else min(n_total, int(max_ticks))
    caption = caption_for(mode, episode)
    render = video_dir is not None
    env = So101PickPlaceEnv(config_path, render_images=False)
    fk = So101FK(mapping=env.mapping)
    real_poses = fk.fk_real(S)  # T_base_ee of the achieved real joints, all ticks
    q0, start_clamps = _start_scenario(env, S[0])
    env.reset(seed=seed, scenario={"joint_positions": q0})
    # Diagnostic only: a recorded episode may be longer than the task.yaml timeout that bounds
    # policy rollouts (30 s = 300 ticks); the replay must cover the whole recording.
    env_cap = env.max_episode_steps
    episode_cap_lifted = n > env_cap
    if episode_cap_lifted:
        env.max_episode_steps = n + 1
    labels = None
    ik = None
    if mode == "cartesian":
        gsrc = data_cfg["labels"]["cartesian_gripper_source"]
        labels = episode_labels(real_poses, S[:, 5], A[:, 5], hz=hz, gripper_source=gsrc,
                                lead=int(data_cfg["labels"].get("cartesian_lead_ticks", 1)),
                                leader_poses=fk.fk_real(A))
        ik = NativeKinematics(env.model, env.config)

    video = _Video(video_dir / f"replay_ep{episode}_{mode}.mp4" if render else None, hz / video_stride, caption)
    rows: list[dict] = []
    limit_clip_counts = np.zeros(6, int)
    rate_limit_counts = np.zeros(6, int)
    last_target = q0.copy()
    truncated = False
    info = env.info()
    for k in range(n):
        cmd_real = S[k]
        row = {"tick": k, "timestamp_s": float(ep["timestamp"][k])}
        if mode == "joint":
            # Target for this tick = the real achieved state at this tick (one-tick lag expected).
            _, _, _, truncated, info = env.step_real_joint_targets(cmd_real)
            T_ref = real_poses[k]
        else:
            T_cur = env.ee_pose()
            delta, clipped = _clamp_delta(labels.action[k, :6], max_translation_m, max_rotation_rad)
            target = apply_delta(T_cur, delta)
            grip = float(np.clip(env.mapping.gripper_real_to_sim_rad(labels.action[k, 6] * 100.0),
                                 env.limits[5, 0], env.limits[5, 1]))
            seed_q = np.clip(env.sim_state(), env.limits[:, 0], env.limits[:, 1])
            res = ik.solve(target, seed_q, gripper=grip)
            if res.accepted:
                last_target = res.qpos.copy()
            else:
                last_target = last_target.copy()
                last_target[5] = grip  # the gripper is not part of the IK problem
            _, _, _, truncated, info = env.step(last_target)
            T_ref = real_poses[min(k + 1, n_total - 1)]  # pose the label points at
            row.update({"ik_accepted": int(res.accepted), "ik_reason": res.reason,
                        "ik_pos_residual_m": res.position_error_m,
                        "ik_rot_residual_rad": res.orientation_error_rad,
                        "ik_iterations": res.iterations, "delta_clipped": int(clipped),
                        "label_extrapolated": int(labels.label_is_extrapolated[k])})
            for name, v in zip(("dx", "dy", "dz", "drx", "dry", "drz"), delta, strict=True):
                row[f"cmd_{name}"] = float(v)
            for name, v in zip(("x", "y", "z"), target[:3, 3], strict=True):
                row[f"target_ee_{name}_m"] = float(v)
        ach_real = env.real_state().astype(np.float64)
        T_sim = env.ee_pose()
        ee_err = float(np.linalg.norm(T_sim[:3, 3] - T_ref[:3, 3]))
        ctrl = info.get("control", {})
        limit_clip_counts += np.asarray(ctrl.get("limit_clipped", [0] * 6), int)
        rate_limit_counts += np.asarray(ctrl.get("rate_limited", [0] * 6), int)
        pm = info["privileged_metrics"]
        for i, j in enumerate(JOINTS):
            row[f"cmd_{j}"] = float(cmd_real[i])
            row[f"sim_{j}"] = float(ach_real[i])
            row[f"err_{j}"] = float(cmd_real[i] - ach_real[i])
        for name, v in zip(("x", "y", "z"), T_ref[:3, 3], strict=True):
            row[f"real_ee_{name}_m"] = float(v)
        for name, v in zip(("x", "y", "z"), T_sim[:3, 3], strict=True):
            row[f"sim_ee_{name}_m"] = float(v)
        row["ee_pos_err_m"] = ee_err
        row["ball_in_bowl"] = int(pm["ball_in_bowl"])
        row["ball_held"] = int(pm["ball_held"])
        row["lifted"] = int(pm["lifted"])
        rows.append(row)
        if render and k % video_stride == 0:
            ik_txt = "" if mode == "joint" else f" ik={'ok' if row['ik_accepted'] else row['ik_reason']}"
            video.add(env.render(), f"tick {k}/{n}  ee_err {ee_err * 100:.1f} cm{ik_txt}  controller={CONTROLLER_SOURCE}")
        if truncated:
            break
    video.close()
    final_metrics = env.info()["privileged_metrics"]
    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"replay_ep{episode}_{mode}.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    err = np.array([[r[f"err_{j}"] for j in JOINTS] for r in rows])
    ee = np.array([r["ee_pos_err_m"] for r in rows])
    abs_err = np.abs(err)
    settle = min(SETTLE_TICKS, max(len(rows) - 1, 0))
    summary = {
        "controller_source": CONTROLLER_SOURCE, "caption": caption, "mode": mode, "episode": episode,
        "dataset_root": str(local_root), "policy_hz": hz, "stride": ep["stride"], "source_fps": ep["fps"],
        "n_rows": ep["n_rows"], "n_ticks_available": n_total, "n_ticks_replayed": len(rows),
        "truncated_by_env": bool(truncated),
        "episode_cap_lifted": {"lifted": bool(episode_cap_lifted), "env_max_episode_steps": env_cap,
                               "note": "diagnostic replay covers the full recording; policy rollouts keep the task.yaml timeout"},
        "start_pose_clamps": start_clamps,
        "limit_clipped_ticks_per_joint": dict(zip(JOINTS, limit_clip_counts.tolist(), strict=True)),
        "rate_limited_ticks_per_joint": dict(zip(JOINTS, rate_limit_counts.tolist(), strict=True)),
        "tracking_error": {
            j: {"unit": u, "median_abs": float(np.median(abs_err[:, i])),
                "p95_abs": float(np.percentile(abs_err[:, i], 95)), "max_abs": float(abs_err[:, i].max())}
            for i, (j, u) in enumerate(zip(JOINTS, JOINT_UNITS, strict=True))
        },
        "ee_position_error_cm": {"median": float(np.median(ee) * 100), "p95": float(np.percentile(ee, 95) * 100),
                                 "max": float(ee.max() * 100)},
        "after_settle": {
            "from_tick": settle,
            "note": "excludes the initial transient in which contact forces push the clamped start pose out of the "
                    "table (the real start pose penetrates the sim table; see configs/sim_so101.yaml)",
            "tracking_error_median_abs": {j: float(np.median(abs_err[settle:, i])) for i, j in enumerate(JOINTS)},
            "tracking_error_p95_abs": {j: float(np.percentile(abs_err[settle:, i], 95)) for i, j in enumerate(JOINTS)},
            "ee_position_error_cm": {"median": float(np.median(ee[settle:]) * 100),
                                     "p95": float(np.percentile(ee[settle:], 95) * 100)},
        },
        "ee_error_reference": ("FK(real state at the same tick)" if mode == "joint"
                               else "FK(real state at tick+lead), the pose the label points at"),
        "ball_outcome": {
            "ball_in_bowl_final": bool(final_metrics["ball_in_bowl"]),
            "ball_held_any_tick": bool(any(r["ball_held"] for r in rows)),
            "lifted_any_tick": bool(any(r["lifted"] for r in rows)),
            "ball_xyz_task_final": final_metrics["ball_xyz_task"],
            "note": "privileged metric; the ball starts at the task.yaml start region (seeded), not at the real "
                    "episode's unknown ball position, so this is not a success claim",
        },
        "video": {"path": str(video.path) if video.path else None, "frames": video.frames,
                  "video_stride": video_stride, "render_wh": [sim_cfg["render"]["width"], sim_cfg["render"]["height"]]},
        "wall_time_s": time.perf_counter() - t0,
        "not_learned": "This is a scripted replay of recorded data (E0). It is not an ACT rollout.",
    }
    if mode == "cartesian":
        acc = np.array([r["ik_accepted"] for r in rows], bool)
        reasons = {}
        for r in rows:
            reasons[r["ik_reason"]] = reasons.get(r["ik_reason"], 0) + 1
        pos_res = np.array([r["ik_pos_residual_m"] for r in rows])
        rot_res = np.array([r["ik_rot_residual_rad"] for r in rows])
        summary["ik"] = {
            "acceptance_rate": float(acc.mean()), "accepted": int(acc.sum()), "reasons": reasons,
            "pos_residual_m": {"median": float(np.median(pos_res)), "p95": float(np.percentile(pos_res, 95))},
            "pos_residual_accepted_m": {"median": float(np.median(pos_res[acc])) if acc.any() else None,
                                        "p95": float(np.percentile(pos_res[acc], 95)) if acc.any() else None},
            "rot_residual_rad": {"median": float(np.median(rot_res)), "p95": float(np.percentile(rot_res, 95))},
            "delta_clipped_ticks": int(sum(r["delta_clipped"] for r in rows)),
            "clamp": {"max_translation_m": max_translation_m, "max_rotation_rad": max_rotation_rad},
            "gripper_source": data_cfg["labels"]["cartesian_gripper_source"],
            "label_jump_flags": int(labels.jump_flags[:len(rows)].sum()),
            "note": "deltas are applied relative to the current simulated EE pose (plan 3.3); rejected ticks "
                    "hold the last accepted joint target, so errors accumulate rather than reset",
        }
    (out_dir / f"replay_ep{episode}_{mode}.json").write_text(json.dumps(summary, indent=2))
    series = {j: err[:, i] for i, j in enumerate(JOINTS[:5])}
    series["ee_err_cm"] = ee * 100
    draw_line_plot(series, f"{caption}: commanded - simulated (deg), EE error (cm)", "deg / cm",
                   out_dir / f"replay_ep{episode}_{mode}.png")
    summary["csv"] = str(csv_path)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="E0 scripted replay diagnostic (never a learned rollout)")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--mode", choices=MODES + ("both",), default="both")
    p.add_argument("--local-root", type=Path, default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "results/metrics/replay")
    p.add_argument("--video", type=Path, default=None, help="directory for the mp4; omitted = no video")
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--video-stride", type=int, default=1, help="render every n-th tick")
    p.add_argument("--max-translation-m", type=float, default=DEFAULT_MAX_TRANSLATION_M)
    p.add_argument("--max-rotation-rad", type=float, default=DEFAULT_MAX_ROTATION_RAD)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = resolve_dataset_root(a.repo_id, a.revision, a.local_root.expanduser() if a.local_root else None)
    modes = MODES if a.mode == "both" else (a.mode,)
    for mode in modes:
        s = run_replay(a.episode, mode, root, a.out, a.video, max_ticks=a.max_ticks, video_stride=a.video_stride,
                       max_translation_m=a.max_translation_m, max_rotation_rad=a.max_rotation_rad, seed=a.seed)
        brief = {k: s[k] for k in ("caption", "n_ticks_replayed", "truncated_by_env", "tracking_error",
                                   "ee_position_error_cm", "after_settle", "ball_outcome", "start_pose_clamps",
                                   "limit_clipped_ticks_per_joint", "wall_time_s")}
        if "ik" in s:
            brief["ik"] = s["ik"]
        print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
