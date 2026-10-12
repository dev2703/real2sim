"""Seeded closed-loop trials of a frozen ACT checkpoint in the SO-101 MuJoCo scene (plan M4.4).

    uv run python -m real2sim.evaluation.run --robot so101 \
        --checkpoint outputs/act_joint_so101/checkpoints/002000 \
        --config configs/sim_so101.yaml --eval-config configs/eval_so101.yaml \
        --n-trials 3 --seed-start 0 --out results/metrics/debug_so101_joint --video

One row per trial goes to ``<out>/trials.csv`` (failures and aborts included); ``summary.json``
holds ``successes/N``, the per-trial list, the resolved configs and the checkpoint contract.
``--controller hold`` runs the HOLD_POSE null baseline (initial pose held, no policy); it is
labelled as such and is never an ACT result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml
from real2sim.evaluation.metrics import TRIAL_COLUMNS, TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.policy.adapters import (
    IPHONE_KEY,
    ApplyResult,
    CartesianActionAdapter,
    ChunkExecutor,
    ContractError,
    JointActionAdapter,
    ObservationAdapter,
)
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics
from real2sim.sim.model import project_path

CONTROLLER_SOURCES = {"learned": "LEARNED_ACT", "hold": "HOLD_POSE"}
EXPERIMENT_IDS = {"joint": "E1_joint_ACT", "cartesian": "E2_cart_ACT_so101", "hold": "E0_hold_pose_null"}


class HoldPoseController:
    """Null baseline: re-commands the initial native pose every tick. Not a policy."""

    def __init__(self, env):
        self.env = env
        self.latencies_ms: list[float] = []
        self.n_policy_calls = 0

    def reset(self) -> None:
        pass

    def step(self, obs: dict, env) -> ApplyResult:
        obs, _, _, truncated, info = env.step(env.initial_qpos)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_learned_controller(env, checkpoint: Path, eval_cfg: dict, device: str) -> tuple[ChunkExecutor, dict, dict]:
    from real2sim.policy.act import load_checkpoint

    policy, normalizer, contract = load_checkpoint(checkpoint, device)
    obs_adapter = ObservationAdapter(contract, normalizer, device=device)
    if obs_adapter.kind == "joint":
        action_adapter = JointActionAdapter(env)
    else:
        cart = eval_cfg["cartesian"]
        action_adapter = CartesianActionAdapter(
            env, NativeKinematics(env.model, env.config),
            max_step_m=cart["max_step_m"], max_step_rad=cart["max_step_rad"])
    executor = ChunkExecutor(policy, normalizer, obs_adapter, action_adapter)
    run_info = json.loads((checkpoint / "run_info.json").read_text()) if (checkpoint / "run_info.json").is_file() else {}
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_step": run_info.get("step"),
        "kind": obs_adapter.kind,
        "chunk_size": int(policy.config.chunk_size),
        "n_action_steps": executor.n_action_steps,
        "device": device,
    }
    return executor, contract, meta


def run_trial(env, controller, *, seed: int, max_ticks: int, tracker: TrialTracker,
              caption_prefix: str, video: CaptionedVideo | None = None,
              upscale: int = 2) -> TrialTracker:
    obs, info = env.reset(seed=seed)
    controller.reset()
    if video is not None:
        video.write(policy_view_to_bgr(obs[IPHONE_KEY], upscale), f"{caption_prefix} seed={seed} tick=0")
    for tick in range(1, max_ticks + 1):
        result = controller.step(obs, env)
        obs = result.obs
        done = tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                              ik_rejected=result.ik_rejected)
        if video is not None:
            video.write(policy_view_to_bgr(obs[IPHONE_KEY], upscale),
                        f"{caption_prefix} seed={seed} tick={tick}")
        if done or result.truncated:
            break
    return tracker


def run_trials(*, robot: str, checkpoint: Path | None, config: Path, eval_config: Path, n_trials: int,
               seed_start: int, out: Path, video: bool, controller: str, device: str,
               max_ticks: int | None = None, experiment_id: str | None = None) -> dict:
    if robot != "so101":
        raise ValueError("only robot=so101 is implemented (Panda is M5)")
    if controller not in CONTROLLER_SOURCES:
        raise ValueError(f"controller must be one of {sorted(CONTROLLER_SOURCES)}")
    eval_cfg = load_yaml(eval_config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = So101PickPlaceEnv(config)
    task = env.task
    hold_ticks = int(math.ceil(task["episode"]["success_hold_s"] * env.control_hz))
    ticks_cap = env.max_episode_steps if max_ticks is None else min(int(max_ticks), env.max_episode_steps)
    contract: dict | None = None
    if controller == "learned":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for the learned controller")
        ctrl, contract, meta = build_learned_controller(env, Path(checkpoint), eval_cfg, device)
        caption = "LEARNED ACT (sim)"
        exp_id = experiment_id or EXPERIMENT_IDS[meta["kind"]]
    else:
        ctrl = HoldPoseController(env)
        meta = {"checkpoint": None, "checkpoint_sha256": None, "checkpoint_step": None, "kind": "hold"}
        caption = "HOLD POSE (null baseline, no policy)"
        exp_id = experiment_id or EXPERIMENT_IDS["hold"]
    source = CONTROLLER_SOURCES[controller]
    vcfg = eval_cfg["video"]
    rows, trials = [], []
    t_start = time.perf_counter()
    for i in range(n_trials):
        seed = seed_start + i
        tracker = TrialTracker(success_hold_ticks=hold_ticks,
                               approach_distance_m=float(eval_cfg["phase"]["approach_distance_m"]),
                               table_top_z=env.table_top, ball_radius_m=float(task["ball"]["radius"]))
        writer, video_path = None, ""
        if video:
            h, w = env.data_config["image"]["height"], env.data_config["image"]["width"]
            video_path = str(out / f"trial_seed{seed:04d}_{source}.mp4")
            writer = CaptionedVideo(video_path, vcfg["fps"], (h * vcfg["upscale"], w * vcfg["upscale"]),
                                    vcfg["codec"])
        t0 = time.perf_counter()
        try:
            run_trial(env, ctrl, seed=seed, max_ticks=ticks_cap, tracker=tracker,
                      caption_prefix=caption, video=writer, upscale=int(vcfg["upscale"]))
            abort = None
        except ContractError:  # a checkpoint/observation contract mismatch is a setup bug: stop
            raise
        except (RuntimeError, ValueError) as exc:  # physics/IK/action abort = failed trial, never dropped
            abort = str(exc)
        finally:
            if writer is not None:
                writer.close()
        lat = ctrl.latencies_ms
        row = tracker.row(
            experiment_id=exp_id, controller_source=source, robot=robot, seed=seed,
            checkpoint_sha256=meta["checkpoint_sha256"], checkpoint_step=meta["checkpoint_step"],
            mean_latency_ms=(round(float(np.mean(lat)), 2) if lat else None), video_path=video_path)
        rows.append(row)
        trials.append({**row, "abort": abort, "wall_s": round(time.perf_counter() - t0, 2),
                       "policy_calls": getattr(ctrl, "n_policy_calls", 0),
                       "max_latency_ms": (round(float(np.max(lat)), 2) if lat else None),
                       "history": tracker.history})
        print(f"[trial seed={seed}] source={source} success={row['success']} phase_failed="
              f"{row['phase_failed']} ticks={row['ticks']} ik_rej={row['ik_rejections']} "
              f"latency_ms={row['mean_latency_ms']} wall_s={trials[-1]['wall_s']}"
              + (f" ABORT={abort}" if abort else ""), flush=True)
    env.close()
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "experiment_id": exp_id, "controller_source": source, "robot": robot,
        "summary": summarize(rows), "trials": trials,
        "checkpoint": meta,
        "contract": contract,
        "resolved": {"sim_config": env.config, "task_config": task, "eval_config": eval_cfg,
                     "data_config": env.data_config, "max_ticks": ticks_cap,
                     "success_hold_ticks": hold_ticks},
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.system()},
        "wall_s_total": round(time.perf_counter() - t_start, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"successes {summary['summary']['successes']}/{n_trials} -> {out}", flush=True)
    return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="so101", choices=["so101"])
    p.add_argument("--checkpoint", type=Path, default=None, help="checkpoint dir with model.safetensors")
    p.add_argument("--config", type=Path, default=project_path("configs/sim_so101.yaml"))
    p.add_argument("--eval-config", type=Path, default=project_path("configs/eval_so101.yaml"))
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true", help="write a captioned mp4 per trial")
    p.add_argument("--controller", default="learned", choices=sorted(CONTROLLER_SOURCES))
    p.add_argument("--device", default=None, help="inference device (default: eval config inference.device)")
    p.add_argument("--max-ticks", type=int, default=None, help="debug cap below task.yaml timeout")
    p.add_argument("--experiment-id", default=None)
    args = p.parse_args(argv)
    device = args.device or load_yaml(args.eval_config)["inference"]["device"]
    run_trials(robot=args.robot, checkpoint=args.checkpoint, config=args.config,
               eval_config=args.eval_config, n_trials=args.n_trials, seed_start=args.seed_start,
               out=args.out, video=args.video, controller=args.controller, device=device,
               max_ticks=args.max_ticks, experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()
