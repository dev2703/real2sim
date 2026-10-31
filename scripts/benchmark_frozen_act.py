"""Matched, resumable frozen-ACT benchmark with explicit assistance accounting.

All conditions share 25 Hz control, native starts, cameras and seeded object
layouts. Panda uses the same declared -150 mm base adapter in every condition.
Completed failures remain in the denominator. Incomplete attempts are preserved
and retried in a new directory; stale, edited or incomplete reports are not
trusted as completed trials. Lighting and frozen HF weights never vary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.hf_recovery import ApproachMonitor
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.kinematics import NativeKinematics

if __package__:
    from scripts import evaluate_hf_act_panda_20261028 as transfer
    from scripts.evaluate_hf_recovery import MEASURED_SCENE, StaticBowlRecovery, lighting
else:
    # Direct-file invocation puts scripts/, rather than the project root, on
    # sys.path. Factories use the same scripts.module names as module invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import evaluate_hf_act_panda_20261028 as transfer
    from scripts.evaluate_hf_recovery import MEASURED_SCENE, StaticBowlRecovery, lighting

CONDITIONS = ("raw", "rgb", "hf_rgb", "learned", "no_act", "hf_learned")
CUSTOM_CONDITIONS = ("learned", "no_act", "hf_learned")


def digest_json(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default,
                                    allow_nan=False).encode()).hexdigest()


def runtime_hashes(snapshot, source_scene, extra_paths=()):
    """Hash executable inputs, including saved preprocessing and static assets."""
    from real2sim.config import load_yaml, project_path

    paths = {Path(__file__), Path(transfer.__file__), Path(transfer.native.__file__),
             Path(transfer.retarget.__file__), Path("scripts/evaluate_hf_recovery.py"),
             Path("physical_measurements.md"), Path(source_scene)}
    paths |= {p for directory in ("src/real2sim/sim", "src/real2sim/panda", "src/real2sim/geometry")
              for p in Path(directory).glob("*.py")}
    paths |= {Path("src/real2sim/policy") / name for name in (
        "native_lerobot_act.py", "hf_recovery.py", "visual_waypoint.py", "visual_targets.py",
        "follower_joint.py", "adapters.py", "common_recovery.py", "recovery_mlp.py",
        "phase_recovery_mlp.py", "rgb_reference_tracker.py", "grasp_feedback.py")}
    paths.add(Path("src/real2sim/policy/source_recovery_frame.py"))
    paths.add(Path("src/real2sim/policy/progress_geometry.py"))
    paths.add(Path("src/real2sim/policy/frozen_prefix_recovery.py"))
    paths |= {Path("src/real2sim/evaluation") / name for name in ("metrics.py", "video.py")}
    paths |= {Path("src/real2sim") / name for name in ("config.py", "serialization.py", "artifacts.py")}
    paths |= set(Path(snapshot).glob("*.json")) | set(Path(snapshot).glob("*.safetensors"))
    for config in (source_scene, Path("configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml")):
        cfg = load_yaml(config)
        paths |= {project_path(cfg[key]) for key in ("task_config", "data_config", "model", "manifest")}
    paths |= {Path(p) for p in extra_paths}
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


class SharedEpisode:
    """Identical embodiment setup for every compared control condition."""

    def __init__(self, robot, directory, source_scene=MEASURED_SCENE, *, native_contract=None):
        directory.mkdir(parents=True)
        self.robot = robot
        scene, task = transfer.prepare_source(directory, (0.,) * 5,
            source_scene=source_scene, initial_gripper_percent=1.2,
            native_contract=native_contract)
        self.source = transfer.retarget.VirtualSource(scene)
        self.env = self.source.source
        self.start = self.opening = None
        self.base_dx = -.15 if robot == "panda" else 0.
        if robot == "panda":
            from real2sim.panda.env import PandaPickPlaceEnv

            panda_scene, _ = transfer.prepare_panda(directory, self.source, task, self.base_dx)
            self.env = PandaPickPlaceEnv(panda_scene, render_images=False)
            self.start, self.opening = transfer.panda_start(self.env, self.source, self.base_dx)
            from real2sim.panda.adapter import PandaIK

            self.guard = PandaIK(self.env.model, self.env.config, task_origin=self.env.offset)
        elif robot != "so101":
            self.close()
            raise ValueError("Unknown embodiment")
        else:
            self.guard = NativeKinematics(self.env.model, self.env.config)

    def reset(self, seed, scenario=None):
        scenario = dict(scenario or {})
        if self.robot == "panda":
            scenario.update(joint_positions=self.start, gripper_opening=self.opening)
        obs, info = self.env.reset(seed=seed, scenario=scenario or None)
        self.source.q = self.source.source.initial_qpos.copy()
        self.source.last_target = self.source.q.copy()
        return obs, info

    def close(self):
        if self.env is not self.source.source:
            self.env.close()
        self.source.close()


class FrozenDriver:
    def __init__(self, shared, policy):
        self.env, self.source, self.robot = shared.env, shared.source, shared.robot
        self.policy = policy
        self.guard = shared.guard
        self.reset()

    def reset(self):
        self.policy.reset()
        self.queue = []
        self.inference_ms = []

    def step(self, obs, env):
        del obs
        raw_state, inference_ms = None, 0.
        if not self.queue:
            t0 = time.perf_counter()
            raw_state = env.real_state() if self.robot == "so101" else self.source.source_state(env)[0]
            if self.robot == "panda":
                raw_state[5] = np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.
            images = {key: env.render(env.camera_bindings[key]) for key in self.policy.camera_keys}
            self.queue = list(self.policy.predict_action_chunk(raw_state, images)[:self.policy.config.n_action_steps])
            inference_ms = (time.perf_counter() - t0) * 1000
            self.inference_ms.append(inference_ms)
        raw = self.queue.pop(0)
        if self.robot == "so101":
            _, bounded = transfer.native.saturate_gripper_only(raw, env)
            current = env.sim_state()
            self.guard.data.qpos[:] = env.data.qpos
            accepted, reason, _ = check_joint_request(self.guard, current,
                env.data.ctrl[env.actuator_ids], bounded, env.max_velocity / env.control_hz, samples=17)
            command = bounded if accepted else current
        else:
            _, bounded = self.source.source_action(raw)
            desired = transfer.retarget.target_from_source(self.source, bounded)
            observed = env.ee_pose()
            delta = desired[:3, 3] - observed[:3, 3]
            distance = np.linalg.norm(delta)
            if distance > .024:
                desired[:3, 3] = observed[:3, 3] + delta * .024 / distance
            width = transfer.retarget.panda_opening_from_percent(raw[5], env.max_opening)
            solved = self.guard.solve(desired, env.arm_qpos(),
                finger_opening=max(width, env.gripper_opening()))
            accepted, reason = solved.accepted, solved.reason
            command = np.r_[solved.qpos if accepted else env.arm_qpos(), width]
        observed, _, _, truncated, info = env.step(command)
        return ApplyResult(observed, info, truncated,
            float(np.mean(info["control"]["limit_clipped"])), ik_rejected=not accepted,
            detail={"uses_act": True, "intervention": False, "raw_hf_action": raw,
                    "native_input": raw_state, "inference_ms": inference_ms,
                    "guard_reason": reason, "requested_native": command})


def initial_fingerprint(shared, info):
    pm = info["privileged_metrics"]
    return {
        "robot": shared.robot, "base_dx_task_m": shared.base_dx,
        "native_state": shared.env.sim_state().tolist(),
        "ball_xyz_task": list(pm["ball_xyz_task"]), "bowl_xyz_task": list(pm["bowl_xyz_task"]),
        "control_hz": shared.env.control_hz, "max_ticks": shared.env.max_episode_steps,
        "lighting": lighting(shared.env),
    }


def run_episode(shared, condition, seed, directory, policy=None, *, scenario=None,
                controller_factory=None, options=None):
    """Custom controllers use reset()/step(obs, env)->ApplyResult, with no scorer input."""
    env = shared.env
    tracker = TrialTracker(success_hold_ticks=math.ceil(
        env.task["episode"]["success_hold_s"] * env.control_hz),
        approach_distance_m=.03, table_top_z=env.table_top,
        ball_radius_m=env.task["ball"]["radius"])
    directory.mkdir(parents=True)
    (directory / "ticks.jsonl").touch()
    started, abort, initial = time.perf_counter(), None, None
    latencies, inference_latencies = [], []
    act_ticks = assistance_ticks = intervention_count = 0
    mode_before = False
    trigger, controller = None, None
    perception_log = []
    try:
        obs, info = shared.reset(seed, scenario)
        initial = initial_fingerprint(shared, info)
        if condition in CUSTOM_CONDITIONS:
            if controller_factory is None:
                raise ValueError("Learned conditions require a controller factory")
            controller = controller_factory(env=env, source=shared.source, robot=shared.robot,
                snapshot=Path(options["snapshot"]), condition=condition,
                options={**options, "_frozen_act_policy": policy})
            controller.reset()
        else:
            geometric = StaticBowlRecovery(env, robot=shared.robot)
            geometric.bowl_memory = None
            geometric.reset()
            acquired = geometric._acquire() if condition == "hf_rgb" else False
            initial_ball = geometric.ball.copy() if acquired else None
            monitor = ApproachMonitor(env.control_hz)
            frozen = FrozenDriver(shared, policy) if condition != "rgb" else None
        with (directory / "ticks.jsonl").open("w") as stream:
            for tick in range(env.max_episode_steps):
                t0 = time.perf_counter()
                if condition in CUSTOM_CONDITIONS:
                    result = controller.step(obs, env)
                    uses_act = bool(result.detail.get("uses_act", condition == "learned"))
                    assisted = bool(result.detail.get("intervention", True))
                else:
                    if condition == "hf_rgb" and trigger is None:
                        distance = None
                        if initial_ball is not None:
                            pose = env.ee_pose()
                            grasp = pose[:3, 3] + pose[:3, :3] @ geometric.ball_in_tool
                            distance = float(np.linalg.norm(grasp - initial_ball))
                        reason = monitor.update(tick, distance)
                        if reason:
                            trigger = {"tick": tick, "reason": reason, "distance_m": distance}
                            geometric.reset()
                    assisted = condition == "rgb" or trigger is not None
                    uses_act = not assisted
                    result = geometric.step(obs, env) if assisted else frozen.step(obs, env)
                latency = (time.perf_counter() - t0) * 1000
                latencies.append(latency)
                infer_ms = float(result.detail.get("inference_ms", 0.))
                if infer_ms > 0:
                    inference_latencies.append(infer_ms)
                act_ticks += uses_act
                assistance_ticks += assisted
                intervention_count += assisted and not mode_before
                mode_before = assisted
                obs = result.obs
                done = tracker.update(result.info["privileged_metrics"],
                    saturation=result.saturation, ik_rejected=result.ik_rejected)
                stream.write(json.dumps({"tick": tick, "time_s": tick / env.control_hz,
                    "uses_act": uses_act, "assisted": assisted, "latency_ms": latency,
                    "detail": result.detail, "control": result.info["control"],
                    "scoring_only": result.info["privileged_metrics"]}, default=json_default,
                    allow_nan=False) + "\n")
                if done or result.truncated:
                    break
        if condition not in CUSTOM_CONDITIONS:
            perception_log = geometric.perception_log
        else:
            trigger = getattr(controller, "trigger", None)
    except (RuntimeError, ValueError, OSError) as exc:
        abort = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(controller, "close", None)
        if close is not None:
            try:
                close()
            except (RuntimeError, ValueError, OSError) as exc:
                message = f"Controller cleanup {type(exc).__name__}: {exc}"
                abort = f"{abort}; {message}" if abort else message
    checkpoint_hash = getattr(policy, "weights_sha256", None)
    if checkpoint_hash is None and options and options.get("snapshot"):
        # Factory controllers own their native policy, while no-ACT still binds
        # the same reference checkpoint for its paired ablation.
        checkpoint_hash = sha256_file(Path(options["snapshot"]) / "model.safetensors")
    row = tracker.row(experiment_id="MATCHED_FROZEN_ACT_BENCHMARK", controller_source=condition,
        robot=shared.robot, seed=seed, checkpoint_sha256=checkpoint_hash,
        checkpoint_step=None, mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        video_path="", abort=abort)
    row.update(condition=condition, initial_state=initial,
        initial_state_sha256=digest_json(initial) if initial is not None else None,
        act_ticks=act_ticks, assistance_ticks=assistance_ticks,
        act_fraction=act_ticks / tracker.ticks if tracker.ticks else 0.,
        assistance_fraction=assistance_ticks / tracker.ticks if tracker.ticks else 0.,
        intervention_count=intervention_count, intervention_required=bool(assistance_ticks),
        simulated_time_s=tracker.ticks / env.control_hz, wall_s=time.perf_counter() - started,
        control_latency_p95_ms=float(np.quantile(latencies, .95)) if latencies else None,
        inference_latency_mean_ms=float(np.mean(inference_latencies)) if inference_latencies else None,
        recovery_trigger=trigger, perception_log=perception_log,
        trace=str(directory / "ticks.jsonl"))
    return row


def completed_trial(path, protocol_id, *, robot=None, condition=None, seed=None):
    """Only explicit completed, untampered, matching attempts may be resumed."""
    try:
        data = json.loads(path.read_text())
        if (data.get("status") != "complete" or data.get("protocol_id") != protocol_id
                or data.get("runtime_source_changes") or data.get("lighting_unchanged") is not True
                or data.get("checkpoint_unchanged") is not True):
            return None
        row = data["trial"]
        if data.get("trial_sha256") != digest_json(row):
            return None
        if any(expected is not None and row.get(name) != expected
               for name, expected in (("robot", robot), ("condition", condition), ("seed", seed))):
            return None
        trace = Path(row["trace"])
        if not trace.is_file() or sha256_file(trace) != data.get("trace_sha256"):
            return None
        if (not isinstance(row.get("pick_place_success"), bool)
                or type(row.get("ticks")) is not int or row["ticks"] < 0):
            return None
        with trace.open() as stream:
            if sum(1 for _ in stream) != row["ticks"]:
                return None
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return None


def aggregate(rows):
    result = summarize(rows)
    n = len(rows)
    result.update(intervention_rate=sum(r["intervention_required"] for r in rows) / n if n else None,
        mean_act_fraction=float(np.mean([r["act_fraction"] for r in rows])) if n else None,
        mean_assistance_fraction=float(np.mean([r["assistance_fraction"] for r in rows])) if n else None,
        mean_simulated_time_s=float(np.mean([r["simulated_time_s"] for r in rows])) if n else None,
        mean_wall_s=float(np.mean([r["wall_s"] for r in rows])) if n else None)
    return result


def run(out, robots, conditions, seeds, *, snapshot=transfer.native.SNAPSHOT,
        source_scene=MEASURED_SCENE, plan_only=False, scenario_plan=None,
        scenario_condition="nominal", controller_factory=None, controller_options=None):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply unique nonnegative trial seeds")
    if not robots or not set(robots) <= {"so101", "panda"} or len(set(robots)) != len(robots):
        raise ValueError("Supply unique supported robots")
    if not conditions or not set(conditions) <= set(CONDITIONS) or len(set(conditions)) != len(conditions):
        raise ValueError("Supply unique supported conditions")
    scenarios, model_effects = {}, {}
    extra_paths = []
    if scenario_plan:
        descriptor = json.loads(Path(scenario_plan).read_text())["conditions"][scenario_condition]
        source_scene = Path(descriptor["source_scene"])
        scenarios = descriptor.get("scenarios", {})
        model_effects = descriptor.get("model_effects", {})
        if not set(map(str, seeds)) <= set(scenarios):
            raise ValueError("Explicit layout plan does not cover all requested seeds")
        extra_paths += [Path(scenario_plan)]
        if model_effects:
            extra_paths += [Path("scripts/prepare_fixed_lighting_robustness.py")]
    factory = None
    if controller_factory:
        module, name = controller_factory.split(":", 1)
        imported = importlib.import_module(module)
        factory = getattr(imported, name)
        extra_paths += [Path(imported.__file__)]
    options = dict(controller_options or {})
    options["snapshot"] = str(snapshot)
    head_paths = [options.get(key) for key in ("head", "head_dir", "checkpoint", "model")]
    # A single matched run can compare ACT-conditioned and no-ACT heads. Both
    # artifacts must enter the protocol, so changing either invalidates resume.
    heads = options.get("heads", {})
    if not isinstance(heads, dict) or set(heads) - set(CUSTOM_CONDITIONS):
        raise ValueError("heads must map learned/no_act/hf_learned conditions to head directories")
    head_paths.extend(heads.values())
    for path in head_paths:
        if path:
            artifact = Path(path)
            if artifact.is_dir():
                extra_paths += [p for p in artifact.iterdir() if p.is_file()]
            elif artifact.is_file():
                extra_paths.append(artifact)
    hashes = runtime_hashes(snapshot, source_scene, extra_paths)
    protocol = {"schema": "matched_frozen_act_protocol_v1", "robots": robots,
        "conditions": conditions, "seeds": seeds, "snapshot": str(snapshot),
        "checkpoint_sha256": sha256_file(Path(snapshot) / "model.safetensors"),
        "source_scene": str(source_scene), "scenario_plan": str(scenario_plan) if scenario_plan else None,
        "scenario_condition": scenario_condition, "scenarios": scenarios, "model_effects": model_effects,
        "control_hz": 25., "initial_source_gripper_percent": 1.2,
        "panda_base_dx_task_m": -.15, "lighting_varies": False,
        "score": "Configured held lift, release and success_hold_s resting placement within configured horizon",
        "controller_factory": controller_factory, "controller_options": options,
        "runtime_source_sha256": hashes,
        "terminology": {"current_rgb_fallback": "geometric controller", "recovery_mlp": "learned model",
                        "DAgger": "iterative corrective-label data collection, not a controller"}}
    protocol_id = digest_json(protocol)
    protocol["protocol_id"] = protocol_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Benchmark protocol/inputs changed; use a new output directory")
    else:
        write_json(manifest, protocol, atomic=True)
    if plan_only:
        return protocol
    if set(conditions) & set(CUSTOM_CONDITIONS) and factory is None:
        raise ValueError("Learned conditions require --controller-factory module:function")
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot) if set(conditions) - {"rgb", "no_act"} else None
    all_rows = []
    fingerprints = {}
    for robot in robots:
        for condition in conditions:
            parent = out / robot / condition
            parent.mkdir(parents=True, exist_ok=True)
            setup = parent / "setups" / f"setup_{time.time_ns()}"
            shared = SharedEpisode(robot, setup, source_scene,
                                   native_contract=getattr(policy, "scene_contract", None))
            try:
                if model_effects:
                    from scripts.prepare_fixed_lighting_robustness import apply_model_condition

                    apply_model_condition(shared.env, model_effects)
                for seed in seeds:
                    trial_root = parent / f"seed_{seed:06d}"
                    trial_root.mkdir(exist_ok=True)
                    existing = [completed_trial(p, protocol_id, robot=robot, condition=condition, seed=seed)
                                for p in sorted(trial_root.glob("attempt_*/report.json"))]
                    row = next((r for r in reversed(existing) if r is not None), None)
                    if row is None:
                        attempts = [int(p.name.split("_")[-1]) for p in trial_root.glob("attempt_*") if p.is_dir()]
                        attempt = trial_root / f"attempt_{max(attempts, default=0) + 1:04d}"
                        before_lighting = lighting(shared.env)
                        row = run_episode(shared, condition, seed, attempt, policy,
                            scenario=scenarios.get(str(seed)), controller_factory=factory, options=options)
                        after = {p: sha256_file(Path(p)) for p in hashes}
                        changed = [p for p in hashes if hashes[p] != after[p]]
                        trial = {"status": "complete", "protocol_id": protocol_id,
                            "trial": row, "trial_sha256": digest_json(row), "runtime_source_changes": changed,
                            "lighting_unchanged": before_lighting == lighting(shared.env),
                            "checkpoint_unchanged": sha256_file(Path(snapshot) / "model.safetensors") == protocol["checkpoint_sha256"],
                            "trace_sha256": sha256_file(Path(row["trace"])) if Path(row["trace"]).exists() else None}
                        write_json(attempt / "report.json", trial, atomic=True)
                        if changed or not trial["lighting_unchanged"] or not trial["checkpoint_unchanged"]:
                            raise RuntimeError("Benchmark inputs changed during a trial; results cannot be pooled")
                    key = (robot, seed)
                    prior = fingerprints.setdefault(key, row["initial_state_sha256"])
                    if prior is not None and row["initial_state_sha256"] is not None and prior != row["initial_state_sha256"]:
                        raise RuntimeError(f"Conditions differ at reset for {robot} seed {seed}")
                    all_rows.append(row)
                    report = {"schema": "matched_frozen_act_results_v1", "protocol_id": protocol_id,
                        "status": "running", "trials": all_rows,
                        "conditions": {f"{r}/{c}": aggregate([x for x in all_rows if x["robot"] == r and x["condition"] == c])
                                       for r in robots for c in conditions}}
                    write_json(out / "report.json", report, atomic=True)
                    print(json.dumps({"robot": robot, "condition": condition, "seed": seed,
                        "success": row["pick_place_success"], "intervention": row["intervention_required"],
                        "abort": row["abort"]}), flush=True)
            finally:
                shared.close()
    report["status"] = "complete"
    report["matched_initial_states_verified"] = all(r["initial_state_sha256"] is not None for r in all_rows)
    write_json(out / "report.json", report, atomic=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robots", nargs="+", choices=("so101", "panda"), default=["so101", "panda"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["raw", "rgb", "hf_rgb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(13000, 13100)))
    parser.add_argument("--snapshot", type=Path, default=transfer.native.SNAPSHOT)
    parser.add_argument("--source-scene", type=Path, default=MEASURED_SCENE)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--scenario-plan", type=Path)
    parser.add_argument("--scenario-condition", default="nominal")
    parser.add_argument("--controller-factory")
    parser.add_argument("--controller-options", type=json.loads, default={})
    args = parser.parse_args()
    run(args.out, args.robots, args.conditions, args.seeds, snapshot=args.snapshot,
        source_scene=args.source_scene, plan_only=args.plan_only, scenario_plan=args.scenario_plan,
        scenario_condition=args.scenario_condition, controller_factory=args.controller_factory,
        controller_options=args.controller_options)


if __name__ == "__main__":
    main()
