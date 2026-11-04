"""Seeded closed-loop trials of a frozen ACT checkpoint in the SO-101 or Panda MuJoCo scene
(plan M4.4 / M5 E3).

    uv run python -m real2sim.evaluation.run --robot so101 \
        --checkpoint outputs/act_joint_so101/checkpoints/002000 \
        --config configs/sim_so101.yaml --eval-config configs/eval_so101.yaml \
        --n-trials 3 --seed-start 0 --out results/metrics/debug_so101_joint --video

    uv run python -m real2sim.evaluation.run --robot panda \
        --checkpoint outputs/act_cartesian/checkpoints/006000 \
        --config configs/sim_panda.yaml --eval-config configs/eval_panda.yaml \
        --n-trials 3 --out results/metrics/debug_panda_cart --video

``--robot panda`` serves the SAME frozen cartesian checkpoint through the Panda adapters
(``real2sim.policy.adapters.PandaObservationAdapter`` / ``PandaActionAdapter``); a joint checkpoint
is refused. The trials.csv / summary.json schema is identical across robots.

One row per trial goes to ``<out>/trials.csv`` (failures and aborts included); ``summary.json``
holds ``successes/N``, the per-trial list, the resolved configs and the checkpoint contract.
``--controller hold`` runs the HOLD_POSE null baseline (initial pose held, no policy); it is
labelled as such and is never an ACT result.

``cartesian.target_reference: accepted_target`` opts into a SO-101 Cartesian execution-interface
ablation. The checkpoint, training labels and policy inputs remain unchanged; only the runtime
target integration changes. Both robots support this interface; joint policies and hold baselines refuse it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from real2sim.artifacts import sha256_file as sha256_file
from real2sim.config import json_default, load_yaml, project_path
from real2sim.evaluation.metrics import TRIAL_COLUMNS, TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo, policy_view_to_bgr
from real2sim.policy.adapters import (
    IPHONE_KEY,
    AbsolutePoseActionAdapter,
    ApplyResult,
    CartesianActionAdapter,
    ChunkExecutor,
    ContractError,
    JointActionAdapter,
    ObservationAdapter,
    PandaActionAdapter,
    PandaObservationAdapter,
)
from real2sim.sim.contact_solver import contact_solver_settings
from real2sim.sim.env import So101PickPlaceEnv
from real2sim.sim.kinematics import NativeKinematics

ROBOTS = ("so101", "panda")
CONTROLLER_SOURCES = {"learned": "LEARNED_ACT", "hold": "HOLD_POSE"}
EXPERIMENT_IDS = {"joint": "E1_joint_ACT", "cartesian": "E2_cart_ACT_so101",
                  "cartesian_tool": "K1_tool_delta_ACT_so101",
                  "cartesian_persistent_target": "DIAGNOSTIC_E2_cart_ACT_so101_persistent_target",
                  "cartesian_absolute": "E2_absolute_ACT_so101", "canonical_joint": "DIAGNOSTIC_canonical_joint_ACT",
                  "panda": "E3_cart_ACT_panda", "hold": "E0_hold_pose_null",
                  "hold_panda": "E0_hold_pose_null_panda"}
DEFAULT_CONFIGS = {"so101": ("configs/sim_so101.yaml", "configs/eval_so101.yaml"),
                   "panda": ("configs/sim_panda.yaml", "configs/eval_panda.yaml")}
CAPTIONS = {("so101", "learned"): "LEARNED ACT (sim)",
            ("panda", "learned"): "PANDA FROZEN ACT (sim)",
            ("so101", "hold"): "HOLD POSE (null baseline, no policy)",
            ("panda", "hold"): "PANDA HOLD POSE (null baseline, no policy)"}


def make_env(robot: str, config: Path):
    if robot == "so101":
        return So101PickPlaceEnv(config)
    if robot == "panda":
        from real2sim.panda.env import PandaPickPlaceEnv

        return PandaPickPlaceEnv(config)
    raise ValueError(f"robot must be one of {ROBOTS}")


def hold_action(robot: str, env) -> np.ndarray:
    """Native action that holds the reset pose (SO-101: 6 qpos; Panda: 7 rad + opening m)."""
    if robot == "panda":
        return np.append(env.initial_qpos, env.initial_opening)
    return np.asarray(env.initial_qpos, np.float64)


class HoldPoseController:
    """Null baseline: re-commands the initial native pose every tick. Not a policy."""

    def __init__(self, env, action: np.ndarray | None = None):
        self.env = env
        self.action = np.asarray(env.initial_qpos if action is None else action, np.float64)
        self.latencies_ms: list[float] = []
        self.n_policy_calls = 0

    def reset(self) -> None:
        pass

    def step(self, obs: dict, env) -> ApplyResult:
        obs, _, _, truncated, info = env.step(self.action)
        return ApplyResult(obs, info, truncated, float(np.mean(info["control"]["limit_clipped"])))


def check_contract_timing_and_image(contract: dict, env) -> None:
    """The checkpoint's clock and image transform must equal the simulator's (plan section 7).

    A 30 Hz checkpoint at 10 Hz control would mis-scale the velocity feature and action timing; a
    different letterbox would change the image geometry. Both pass the shape checks silently.
    """
    if "camera_keys" in contract and hasattr(env, "camera_keys") and contract["camera_keys"] != env.camera_keys:
        raise ContractError("checkpoint camera order differs from simulator cameras")
    if contract.get("contract_version"):
        task_path = project_path(env.config["task_config"])
        from real2sim.sim.cameras import shared_task_hash

        panda_mount_extension = (env.config.get("robot") == "panda"
            and contract.get("transfer_shared_task_sha256") == shared_task_hash(env.task))
        if contract.get("calibration_sha256") != sha256_file(task_path) and not panda_mount_extension:
            raise ContractError("checkpoint calibration differs from simulator task registration")
    hz = contract.get("policy_hz")
    if hz is None or abs(float(hz) - float(env.control_hz)) > 1e-9:
        raise ContractError(f"checkpoint policy_hz {hz} != simulator control_hz {env.control_hz}")
    lb = contract.get("letterbox") or {}
    img = env.data_config["image"]
    if (lb.get("dst_h"), lb.get("dst_w")) != (img["height"], img["width"]):
        raise ContractError(f"checkpoint letterbox {lb.get('dst_h')}x{lb.get('dst_w')} != simulator "
                            f"{img['height']}x{img['width']}")
    src_aspect = lb.get("src_w", 0) / max(lb.get("src_h", 1), 1)
    if abs(src_aspect - 16 / 9) > 1e-3:
        raise ContractError(f"checkpoint was trained on a {src_aspect:.3f} aspect source; the simulator renders 16:9")


def _cartesian_target_reference(eval_cfg: dict, *, robot: str, controller: str) -> str:
    reference = eval_cfg.get("cartesian", {}).get("target_reference", "current_pose")
    if reference not in ("current_pose", "accepted_target"):
        raise ValueError("cartesian.target_reference must be 'current_pose' or 'accepted_target'")
    if reference == "accepted_target" and controller != "learned":
        raise ContractError("accepted_target requires a learned Cartesian controller; it cannot be applied to the hold baseline")
    return reference


def build_learned_controller(env, checkpoint: Path, eval_cfg: dict, device: str,
                             robot: str = "so101") -> tuple[ChunkExecutor, dict, dict]:
    target_reference = _cartesian_target_reference(eval_cfg, robot=robot, controller="learned")
    from real2sim.policy.act import load_checkpoint

    policy, normalizer, contract = load_checkpoint(checkpoint, device)
    check_contract_timing_and_image(contract, env)
    cart = eval_cfg["cartesian"]
    if robot == "panda":
        from real2sim.panda.adapter import PandaCartesianActionAdapter

        # Refuses a joint contract (ContractError) before any Panda machinery is built.
        obs_adapter = PandaObservationAdapter(contract, normalizer, device=device)
        panda = PandaCartesianActionAdapter(env, target_reference=target_reference)
        # The Panda adapter clamps with sim_panda.yaml action_limits; the eval config mirrors the
        # SO-101 thresholds and must agree so both robots see the same per-tick clamp.
        if not (math.isclose(panda.max_translation, float(cart["max_step_m"]))
                and math.isclose(panda.max_rotation, float(cart["max_step_rad"]))):
            raise ValueError(
                f"eval config cartesian clamp ({cart['max_step_m']} m, {cart['max_step_rad']} rad) "
                f"!= sim config action_limits ({panda.max_translation}, {panda.max_rotation})")
        action_adapter = PandaActionAdapter(panda)
    else:
        obs_adapter = ObservationAdapter(contract, normalizer, device=device)
        if obs_adapter.kind in ("joint", "canonical_joint"):
            if target_reference == "accepted_target":
                raise ContractError("accepted_target requires a Cartesian checkpoint; "
                                    "joint targets do not use a Cartesian target reference")
            if "joint_target_semantics" in contract:
                from real2sim.policy.follower_joint import GuardedJointActionAdapter

                action_adapter = GuardedJointActionAdapter(env, NativeKinematics(env.model, env.config), contract)
            else:
                action_adapter = JointActionAdapter(env)
        else:
            action_adapter = CartesianActionAdapter(
                env, NativeKinematics(env.model, env.config),
                max_step_m=cart["max_step_m"], max_step_rad=cart["max_step_rad"],
                target_reference=target_reference)
    if obs_adapter.kind == "cartesian_absolute":
        if target_reference != "current_pose":
            raise ContractError("absolute pose actions require current_pose bounded feedback")
        action_adapter = AbsolutePoseActionAdapter(action_adapter)
    projection = cart.get("kinematic_projection")
    if obs_adapter.kind == "cartesian_tool" or projection is not None:
        from real2sim.policy.kinematic_actions import KinematicDeltaAdapter

        if obs_adapter.kind not in ("cartesian", "cartesian_tool"):
            raise ContractError("kinematic delta projection requires a delta-action checkpoint")
        action_adapter = KinematicDeltaAdapter(
            action_adapter, tool_frame=obs_adapter.kind == "cartesian_tool", projection=projection,
        )
    expected_controller = contract.get("controller_semantics")
    if contract.get("contract_version") and expected_controller not in (None, target_reference):
        raise ContractError("checkpoint execution semantics differ from evaluation configuration")
    run_info = json.loads((checkpoint / "run_info.json").read_text()) if (checkpoint / "run_info.json").is_file() else {}
    inference = eval_cfg.get("inference", {})
    executor = ChunkExecutor(policy, normalizer, obs_adapter, action_adapter,
        n_action_steps=inference.get("n_action_steps"),
        temporal_ensemble_coeff=inference.get("temporal_ensemble_coeff"),
        supervised_prefix_steps=run_info.get("reconstruction_loss", {}).get("supervised_prefix_steps"))
    execution_ablation = (executor.n_action_steps != int(policy.config.n_action_steps)
        or executor.temporal_ensemble_coeff != getattr(policy.config, "temporal_ensemble_coeff", None))
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_step": run_info.get("step"),
        "kind": obs_adapter.kind,
        "chunk_size": int(policy.config.chunk_size),
        "n_action_steps": executor.n_action_steps,
        "temporal_ensemble_coeff": executor.temporal_ensemble_coeff,
        "supervised_action_steps": executor.supervised_prefix_steps,
        "device": device,
        "target_reference": target_reference if obs_adapter.kind in ("cartesian", "cartesian_absolute", "cartesian_tool") else None,
        "execution_interface_ablation": execution_ablation or projection is not None or (
            target_reference == "accepted_target" and not contract.get("contract_version")),
        "kinematic_projection": projection,
        "checkpoint_contract_modified": False,
    }
    return executor, contract, meta


def run_trial(env, controller, *, seed: int, max_ticks: int, tracker: TrialTracker,
              caption_prefix: str, video: CaptionedVideo | None = None,
              upscale: int = 2, reset_scenario: dict | None = None) -> TrialTracker:
    obs, _ = (env.reset(seed=seed) if reset_scenario is None
              else env.reset(seed=seed, scenario=reset_scenario))
    controller.reset()
    if video is not None:
        video.write(policy_view_to_bgr(obs[IPHONE_KEY if IPHONE_KEY in obs else env.camera_keys[0]], upscale), f"{caption_prefix} seed={seed} tick=0")
    for tick in range(1, max_ticks + 1):
        result = controller.step(obs, env)
        obs = result.obs
        done = tracker.update(result.info["privileged_metrics"], saturation=result.saturation,
                              ik_rejected=result.ik_rejected)
        if video is not None:
            video.write(policy_view_to_bgr(obs[IPHONE_KEY if IPHONE_KEY in obs else env.camera_keys[0]], upscale),
                        f"{caption_prefix} seed={seed} tick={tick}")
        if done or result.truncated:
            break
    return tracker


def run_trials(*, robot: str, checkpoint: Path | None, config: Path, eval_config: Path, n_trials: int,
               seed_start: int, out: Path, video: bool, controller: str, device: str,
               max_ticks: int | None = None, experiment_id: str | None = None,
               reset_scenarios: list[dict] | None = None) -> dict:
    if robot not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}")
    if controller not in CONTROLLER_SOURCES:
        raise ValueError(f"controller must be one of {sorted(CONTROLLER_SOURCES)}")
    if reset_scenarios is not None and len(reset_scenarios) != n_trials:
        raise ValueError("one reset scenario is required for every diagnostic trial")
    eval_cfg = load_yaml(eval_config)
    _cartesian_target_reference(eval_cfg, robot=robot, controller=controller)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Check the scene yaml's robot field BEFORE building the env: the robot-specific env class
    # fails inside its constructor (opaque StopIteration) when handed the other robot's scene.
    config_robot = str(load_yaml(config).get("robot", "so101"))
    if config_robot != robot:
        raise ValueError(f"--config is a {config_robot} scene but --robot is {robot}")
    with closing(make_env(robot, config)) as env:
        task = env.task
        hold_ticks = int(math.ceil(task["episode"]["success_hold_s"] * env.control_hz))
        ticks_cap = env.max_episode_steps if max_ticks is None else min(int(max_ticks), env.max_episode_steps)
        contract: dict | None = None
        if controller == "learned":
            if checkpoint is None:
                raise ValueError("--checkpoint is required for the learned controller")
            ctrl, contract, meta = build_learned_controller(env, Path(checkpoint), eval_cfg, device,
                                                           robot=robot)
            experiment_kind = ("panda" if robot == "panda" else
                               "cartesian_persistent_target" if meta["execution_interface_ablation"]
                               else meta["kind"])
            exp_id = experiment_id or EXPERIMENT_IDS[experiment_kind]
        else:
            ctrl = HoldPoseController(env, hold_action(robot, env))
            meta = {"checkpoint": None, "checkpoint_sha256": None, "checkpoint_step": None,
                    "kind": "hold", "target_reference": None, "execution_interface_ablation": False,
                    "checkpoint_contract_modified": False}
            exp_id = experiment_id or EXPERIMENT_IDS["hold_panda" if robot == "panda" else "hold"]
        caption = CAPTIONS[(robot, controller)]
        if meta["execution_interface_ablation"]:
            caption += " configured controller (execution ablation)"
        source = CONTROLLER_SOURCES[controller]
        if controller == "learned" and getattr(ctrl.policy.config, "type", "") == "small_chunk_head":
            source = "LEARNED_FLOW" if ctrl.policy.config.objective == "flow" else "LEARNED_REGRESSION"
            caption = f"{source} ({robot} sim)"
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
                          caption_prefix=caption, video=writer, upscale=int(vcfg["upscale"]),
                          reset_scenario=None if reset_scenarios is None else reset_scenarios[i])
                abort = None
            except ContractError:  # a checkpoint/observation contract mismatch is a setup bug: stop
                raise
            except (RuntimeError, ValueError) as exc:  # physics/IK/action abort = failed trial, never dropped
                abort = str(exc)
            finally:
                if writer is not None:
                    writer.close()
            lat = ctrl.latencies_ms
            adapter_stats = getattr(getattr(ctrl, "action_adapter", None), "stats", None)
            if adapter_stats is not None and adapter_stats["rejected"] != tracker.ik_rejections:
                raise RuntimeError("Panda adapter rejection tally disagrees with the trial tracker")
            # The first inference includes one-off warm-up (allocation, kernel selection); report it
            # separately so mean_latency_ms describes steady-state control latency.
            steady = lat[1:] if len(lat) > 1 else lat
            step_lat = getattr(ctrl, "control_step_latencies_ms", [])
            row = tracker.row(
                experiment_id=exp_id, controller_source=source, robot=robot, seed=seed,
                checkpoint_sha256=meta["checkpoint_sha256"], checkpoint_step=meta["checkpoint_step"],
                mean_latency_ms=(round(float(np.mean(steady)), 2) if steady else None), video_path=video_path,
                abort=abort)
            rows.append(row)
            trials.append({**row, "wall_s": round(time.perf_counter() - t0, 2),
                           "warmup_latency_ms": (round(float(lat[0]), 2) if lat else None),
                           "policy_calls": getattr(ctrl, "n_policy_calls", 0),
                           "max_latency_ms": (round(float(np.max(lat)), 2) if lat else None),
                           "control_step_wall_ms": ({
                               "mean": float(np.mean(step_lat)),
                               "p95": float(np.quantile(step_lat, .95)),
                               "max": float(np.max(step_lat)),
                               "scope": "host simulation step including next rendered observation; physical camera/motor timestamps unavailable",
                           } if step_lat else None),
                           "adapter_stats": _compact_stats(adapter_stats),
                           "history": tracker.history})
            print(f"[trial seed={seed}] source={source} success={row['success']} phase_failed="
                  f"{row['phase_failed']} ticks={row['ticks']} ik_rej={row['ik_rejections']} "
                  f"latency_ms={row['mean_latency_ms']} wall_s={trials[-1]['wall_s']}"
                  + (f" ABORT={abort}" if abort else ""), flush=True)
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "experiment_id": exp_id, "controller_source": source, "robot": robot,
        "summary": summarize(rows), "trials": trials,
        "checkpoint": meta,
        "contract": contract,
        "execution_interface": {
            "target_reference": meta["target_reference"],
            "n_action_steps": meta.get("n_action_steps"),
            "temporal_ensemble_coeff": meta.get("temporal_ensemble_coeff"),
            "supervised_action_steps": meta.get("supervised_action_steps"),
            "ablation": meta["execution_interface_ablation"],
            "checkpoint_contract_modified": False,
            "note": ("Execution configuration ablation; frozen checkpoint, labels "
                     "and policy observations unchanged. This is not a trained contract change."
                     if meta["execution_interface_ablation"] else "Default execution interface."),
        },
        "reset_scenarios": reset_scenarios,
        "resolved": {"sim_config": env.config, "task_config": task, "eval_config": eval_cfg,
                     "data_config": env.data_config, "max_ticks": ticks_cap,
                     "success_hold_ticks": hold_ticks,
                     "contact_solver": contact_solver_settings(env.model)},
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.system()},
        "wall_s_total": round(time.perf_counter() - t_start, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    print(f"successes {summary['summary']['successes']}/{n_trials} -> {out}", flush=True)
    return summary


def _compact_stats(stats: dict | None) -> dict | None:
    """Panda adapter tallies for summary.json (residual lists reduced to summary numbers)."""
    if stats is None:
        return None
    out = {k: v for k, v in stats.items() if not isinstance(v, list)}
    for key in ("residual_position_m", "residual_orientation_rad"):
        vals = stats.get(key) or []
        out[f"{key}_mean"] = (float(np.mean(vals)) if vals else None)
        out[f"{key}_max"] = (float(np.max(vals)) if vals else None)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="so101", choices=list(ROBOTS))
    p.add_argument("--checkpoint", type=Path, default=None, help="checkpoint dir with model.safetensors")
    p.add_argument("--config", type=Path, default=None,
                   help="sim scene yaml (default: configs/sim_<robot>.yaml)")
    p.add_argument("--eval-config", type=Path, default=None,
                   help="evaluation yaml (default: configs/eval_<robot>.yaml)")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true", help="write a captioned mp4 per trial")
    p.add_argument("--controller", default="learned", choices=sorted(CONTROLLER_SOURCES))
    p.add_argument("--device", default=None, help="inference device (default: eval config inference.device)")
    p.add_argument("--max-ticks", type=int, default=None, help="debug cap below task.yaml timeout")
    p.add_argument("--experiment-id", default=None)
    args = p.parse_args(argv)
    sim_default, eval_default = DEFAULT_CONFIGS[args.robot]
    args.config = args.config or project_path(sim_default)
    args.eval_config = args.eval_config or project_path(eval_default)
    device = args.device or load_yaml(args.eval_config)["inference"]["device"]
    run_trials(robot=args.robot, checkpoint=args.checkpoint, config=args.config,
               eval_config=args.eval_config, n_trials=args.n_trials, seed_start=args.seed_start,
               out=args.out, video=args.video, controller=args.controller, device=device,
               max_ticks=args.max_ticks, experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()
