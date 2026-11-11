"""Unassisted native ACT execution and a fixed, auditable 20-trial gate.

The only action source is the loaded native policy. Robot-only FK/IK adapters
and rejection guards are retained; object state is consumed by scoring only.
This module never imports or constructs a geometric image controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.config import load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.policy.adapters import ApplyResult
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.serialization import json_default, write_json
from real2sim.sim.kinematics import NativeKinematics
from scripts import evaluate_hf_act_panda_20261028 as transfer

ROBOTS = ("so101", "panda")
DEFAULT_SCENE = Path("configs/v2/weight_space_20261111/sim_so101.yaml")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default,
                                     allow_nan=False).encode()).hexdigest()


def runtime_hashes(snapshot, source_scene, extra_paths=()):
    """Hash the native policy, processors, scene builders, inputs and assets."""
    paths = {Path(__file__), Path(transfer.__file__), Path(transfer.native.__file__),
             Path(transfer.retarget.__file__), Path(source_scene), *map(Path, extra_paths)}
    for directory in ("src/real2sim/sim", "src/real2sim/panda", "src/real2sim/geometry"):
        paths.update(Path(directory).glob("*.py"))
    paths.update(Path("src/real2sim/policy") / name for name in (
        "native_lerobot_act.py", "follower_joint.py", "bounded_joint.py", "adapters.py"))
    paths.update(Path("src/real2sim") / name for name in (
        "artifacts.py", "config.py", "serialization.py"))
    paths.add(Path("src/real2sim/evaluation/metrics.py"))
    paths.update(Path(snapshot).glob("*.json"))
    paths.update(Path(snapshot).glob("*.safetensors"))
    for scene in (source_scene, Path("configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml")):
        config = load_yaml(scene)
        paths.update(project_path(config[key]) for key in ("task_config", "data_config", "model", "manifest"))
        paths.update(path for path in project_path(config["model"]).parent.rglob("*")
                     if path.is_file() and path.suffix.lower() in (".xml", ".stl", ".obj", ".png", ".mtl"))
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


class SharedEpisode:
    """Native robot setup with the same source/Panda adapter in every trial."""

    def __init__(self, robot, directory, source_scene=DEFAULT_SCENE, *, native_contract=None,
                 initial_gripper_percent=1.2):
        if robot not in ROBOTS:
            raise ValueError("Unknown embodiment")
        directory = Path(directory)
        directory.mkdir(parents=True)
        self.robot, self.base_dx = robot, -.15 if robot == "panda" else 0.
        scene, task = transfer.prepare_source(directory, source_scene=source_scene,
            initial_gripper_percent=initial_gripper_percent, native_contract=native_contract)
        self.source = transfer.retarget.VirtualSource(scene)
        self.env = self.source.source
        self.start = self.opening = None
        try:
            if robot == "panda":
                from real2sim.panda.adapter import PandaIK
                from real2sim.panda.env import PandaPickPlaceEnv

                scene, _ = transfer.prepare_panda(directory, self.source, task, self.base_dx)
                self.env = PandaPickPlaceEnv(scene, render_images=False)
                self.start, self.opening = transfer.panda_start(self.env, self.source, self.base_dx)
                self.guard = PandaIK(self.env.model, self.env.config, task_origin=self.env.offset)
            else:
                self.guard = NativeKinematics(self.env.model, self.env.config)
        except Exception:
            self.close()
            raise

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
    """Native ACT actions only; an invalid request holds the measured robot."""

    def __init__(self, shared, policy, execute_steps=None):
        self.env, self.source, self.robot = shared.env, shared.source, shared.robot
        self.policy, self.guard = policy, shared.guard
        self.execute_steps = policy.config.n_action_steps if execute_steps is None else execute_steps
        if type(self.execute_steps) is not int or not 1 <= self.execute_steps <= policy.config.chunk_size:
            raise ValueError("Execution prefix must be an integer within the saved action chunk")
        self.reset()

    def reset(self):
        self.policy.reset()
        self.queue, self.inference_ms = [], []

    def step(self, obs, env):
        del obs
        raw_state, inference_ms = None, 0.
        if not self.queue:
            started = time.perf_counter()
            raw_state = env.real_state() if self.robot == "so101" else self.source.source_state(env)[0]
            if self.robot == "panda":
                raw_state[5] = np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.
            images = {key: env.render(env.camera_bindings[key]) for key in self.policy.camera_keys}
            self.queue = list(self.policy.predict_action_chunk(raw_state, images)[:self.execute_steps])
            inference_ms = (time.perf_counter() - started) * 1000
            self.inference_ms.append(inference_ms)
        raw = self.queue.pop(0)
        source_rejection = None
        if self.robot == "so101":
            _, bounded = transfer.native.saturate_gripper_only(raw, env)
            current = env.sim_state()
            self.guard.data.qpos[:] = env.data.qpos
            accepted, reason, _ = check_joint_request(self.guard, current,
                env.data.ctrl[env.actuator_ids], bounded, env.max_velocity / env.control_hz, samples=17)
            command = bounded if accepted else current
        else:
            try:
                _, bounded = self.source.source_action(raw)
            except transfer.retarget.VirtualSourceCommandError as exc:
                accepted, reason = False, "source_target_native_limits"
                source_rejection = exc.diagnostics()
                command = np.r_[env.arm_qpos(), env.gripper_opening()]
            else:
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
                "guard_reason": reason, "requested_native": command,
                "source_command_rejection": source_rejection})


def initial_fingerprint(shared, info):
    privileged = info["privileged_metrics"]
    model, env = shared.env.model, shared.env
    return {"robot": shared.robot, "base_dx_task_m": shared.base_dx,
        "native_state": env.sim_state().tolist(),
        "ball_xyz_task": list(privileged["ball_xyz_task"]),
        "bowl_xyz_task": list(privileged["bowl_xyz_task"]),
        "control_hz": env.control_hz, "max_ticks": env.max_episode_steps,
        "near_clip_m": float(model.vis.map.znear) * float(model.stat.extent),
        "lighting": {"ambient": model.vis.headlight.ambient.copy().tolist(),
            "diffuse": model.vis.headlight.diffuse.copy().tolist(),
            "specular": model.vis.headlight.specular.copy().tolist(),
            "light_position": model.light_pos.copy().tolist(),
            "light_diffuse": model.light_diffuse.copy().tolist()}}


def validate_cohort(robots, seeds, training_seeds=(), validation_seeds=()):
    robots, seeds = tuple(robots), tuple(seeds)
    if not robots or len(set(robots)) != len(robots) or set(robots) - set(ROBOTS):
        raise ValueError("Supply unique supported robots")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Supply unique nonnegative integer evaluation seeds")
    if set(seeds) & (set(training_seeds) | set(validation_seeds)):
        raise ValueError("Evaluation seeds overlap training or validation seeds")
    return robots, seeds


def known_fitting_seeds(snapshot):
    """Read saved fit roles; never infer a held-out split from a seed's magnitude."""
    training, validation = set(), set()
    declarations = []
    for name in ("decoder_weight_edit_report.json", "weight_edit_report.json"):
        path = Path(snapshot) / name
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        for key, destination in (("train_episodes", training), ("validation_episodes", validation)):
            for identity in report.get(key, []):
                if (not isinstance(identity, list) or len(identity) != 2
                        or type(identity[1]) is not int or identity[1] < 0):
                    raise ValueError(f"Malformed saved fitting episode: {name}:{key}")
                destination.add(identity[1])
        declarations.append(str(path))
    return training, validation, declarations


def completion_gate(trials, robots, seeds, *, integrity_verified=True):
    """Require every predeclared attempt, full task success and no assistance."""
    expected = {(robot, seed) for robot in robots for seed in seeds}
    actual = [(row.get("robot"), row.get("seed")) for row in trials]
    cohort_complete = len(actual) == len(expected) and set(actual) == expected
    results = {}
    for robot in robots:
        rows = [row for row in trials if row.get("robot") == robot]
        passed = [row for row in rows if row.get("pick_place_success") is True
            and not row.get("abort") and row.get("controller_source") == "raw"
            and row.get("assistance_ticks") == 0 and row.get("intervention_count") == 0
            and type(row.get("ticks")) is int and row["ticks"] > 0
            and row.get("act_ticks") == row["ticks"]]
        results[robot] = {"attempted": len(rows), "required": 20,
            "pick_place_successes": len(passed),
            "gate_met": bool(integrity_verified and cohort_complete and len(seeds) == 20
                             and len(rows) == 20 and len(passed) == 20)}
    return {"required_successes_per_robot": 20, "cohort_complete": cohort_complete,
        "integrity_verified": bool(integrity_verified), "per_robot": results,
        "objective_met": bool(set(robots) == set(ROBOTS) and all(
            result["gate_met"] for result in results.values()))}


def run_episode(shared, policy, seed, directory, *, execute_steps=None, scenario=None,
                randomization=None):
    env = shared.env
    directory = Path(directory)
    directory.mkdir(parents=True)
    trace = directory / "ticks.jsonl"
    tracker = TrialTracker(success_hold_ticks=math.ceil(env.task["episode"]["success_hold_s"] * env.control_hz),
        approach_distance_m=.03, table_top_z=env.table_top, ball_radius_m=env.task["ball"]["radius"])
    initial, abort, started = None, None, time.perf_counter()
    latencies, act_ticks = [], 0
    try:
        obs, info = shared.reset(seed, scenario)
        initial = initial_fingerprint(shared, info)
        driver = FrozenDriver(shared, policy, execute_steps)
        with trace.open("w") as stream:
            for tick in range(env.max_episode_steps):
                start = time.perf_counter()
                result = driver.step(obs, env)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                if result.detail.get("uses_act") is not True or result.detail.get("intervention") is not False:
                    raise ValueError("Raw ACT execution cannot accept assistance")
                act_ticks += 1
                obs = result.obs
                done = tracker.update(result.info["privileged_metrics"],
                    saturation=result.saturation, ik_rejected=result.ik_rejected)
                stream.write(json.dumps({"tick": tick, "time_s": tick / env.control_hz,
                    "uses_act": True, "assisted": False, "latency_ms": latency,
                    "detail": result.detail, "control": result.info["control"],
                    "scoring_only": result.info["privileged_metrics"]},
                    default=json_default, allow_nan=False) + "\n")
                if done or result.truncated:
                    break
    except (RuntimeError, ValueError, OSError) as exc:
        abort = f"{type(exc).__name__}: {exc}"
    trace.touch(exist_ok=True)
    row = tracker.row(experiment_id="RAW_ACT_FIXED_COHORT", controller_source="raw",
        robot=shared.robot, seed=seed, checkpoint_sha256=policy.weights_sha256,
        checkpoint_step=None, mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        video_path="", abort=abort)
    row.update(condition="raw", initial_state=initial,
        initial_state_sha256=digest_json(initial) if initial else None,
        act_ticks=act_ticks, assistance_ticks=0, intervention_count=0,
        intervention_required=False, act_fraction=act_ticks / tracker.ticks if tracker.ticks else 0.,
        assistance_fraction=0., randomization=randomization,
        score_contract={"success_hold_ticks": tracker.success_hold_ticks,
            "approach_distance_m": tracker.approach_distance_m,
            "table_top_z": tracker.table_top_z, "ball_radius_m": tracker.ball_radius_m},
        wall_s=time.perf_counter() - started, trace=str(trace.resolve()))
    return row


def _setup_failure(robot, seed, directory, policy, exc):
    directory.mkdir(parents=True)
    trace = directory / "ticks.jsonl"
    trace.touch()
    tracker = TrialTracker(success_hold_ticks=1, approach_distance_m=.03,
                           table_top_z=0., ball_radius_m=.03)
    row = tracker.row(experiment_id="RAW_ACT_FIXED_COHORT", controller_source="raw",
        robot=robot, seed=seed, checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
        mean_latency_ms=None, video_path="", abort=f"Setup {type(exc).__name__}: {exc}")
    row.update(condition="raw", act_ticks=0, assistance_ticks=0, intervention_count=0,
        intervention_required=False, trace=str(trace.resolve()), randomization=None,
        score_contract={"success_hold_ticks": 1, "approach_distance_m": .03,
            "table_top_z": 0., "ball_radius_m": .03},
        initial_state=None, initial_state_sha256=None)
    return row


def audit_report(path):
    """Verify retained raw traces and recompute task scores before opening the gate."""
    path = Path(path)
    report = json.loads(path.read_text())
    protocol = json.loads((path.parent / "protocol.json").read_text())
    errors, verified = [], []
    declared_id = protocol.pop("protocol_id", None)
    if declared_id != digest_json(protocol) or report.get("protocol_id") != declared_id:
        errors.append("Protocol identity mismatch")
    if protocol.get("raw_policy_only") is not True or protocol.get("rgb_controller_enabled") is not False:
        errors.append("Protocol does not exclude assistance")
    if report.get("status") != "complete" or report.get("checkpoint_unchanged") is not True:
        errors.append("Run is incomplete or checkpoint preservation is unverified")
    if (report.get("source_sha256_before") != protocol.get("runtime_source_sha256")
            or report.get("source_sha256_before") != report.get("source_sha256_after")
            or report.get("runtime_source_changes") != []):
        errors.append("Runtime inputs changed or provenance is inconsistent")
    for row in report.get("trials", []):
        identity = f"{row.get('robot')}/{row.get('seed')}"
        try:
            trace = Path(row["trace"])
            leaf = json.loads((trace.parent / "report.json").read_text())
            if (leaf.get("status") != "complete" or leaf.get("protocol_id") != declared_id
                    or leaf.get("trial") != row or leaf.get("trial_sha256") != digest_json(row)):
                raise ValueError("trial record mismatch")
            if sha256_file(trace) != row["trace_sha256"]:
                raise ValueError("trace checksum mismatch")
            if row.get("checkpoint_sha256") != protocol.get("checkpoint_sha256"):
                raise ValueError("trial checkpoint identity mismatch")
            ticks = [json.loads(line) for line in trace.read_text().splitlines()]
            if len(ticks) != row["ticks"] or row.get("act_ticks") != len(ticks):
                raise ValueError("trace count differs from executed actions")
            if row.get("assistance_ticks") != 0 or row.get("intervention_count") != 0:
                raise ValueError("trial includes assistance")
            tracker = TrialTracker(**row["score_contract"])
            for index, tick in enumerate(ticks):
                if (tick.get("tick") != index or tick.get("uses_act") is not True
                        or tick.get("assisted") is not False
                        or tick.get("detail", {}).get("uses_act") is not True
                        or tick.get("detail", {}).get("intervention") is not False):
                    raise ValueError("trace ordering or raw-action accounting differs")
                tracker.update(tick["scoring_only"])
            if row.get("pick_place_success") is not (tracker.pick_place_success and not bool(row.get("abort"))):
                raise ValueError("task score differs from retained scoring trace")
            verified.append(row)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{identity}: {exc}")
    robots, seeds = protocol.get("robots", []), protocol.get("seeds", [])
    gate = completion_gate(verified, robots, seeds, integrity_verified=not errors)
    if not gate["cohort_complete"]:
        errors.append("Retained trials do not cover the predeclared cohort exactly once")
        gate = completion_gate(verified, robots, seeds, integrity_verified=False)
    return {"schema": "raw_act_evidence_audit_v1", "status": "verified" if not errors else "failed",
        "locally_verified": not errors, "claimed_trials": len(report.get("trials", [])),
        "locally_verified_trials": len(verified), "errors": errors, "gate": gate,
        "scope": "Retained protocol, trial/trace hashes, raw-action accounting and recomputed task scores; no physics rerun"}


def run(out, *, snapshot, source_scene=DEFAULT_SCENE, robots=ROBOTS, seeds=tuple(range(15000, 15020)),
        execute_steps=None, initial_gripper_percent=1.2, training_seeds=(), validation_seeds=(),
        domain_randomization=None, plan_only=False):
    """Evaluate each predeclared robot/seed exactly once; failed attempts stay put."""
    snapshot, out, source_scene = Path(snapshot).expanduser().resolve(), Path(out), Path(source_scene)
    fitted_train, fitted_val, declarations = known_fitting_seeds(snapshot)
    fitted_train.update(training_seeds)
    fitted_val.update(validation_seeds)
    robots, seeds = validate_cohort(robots, seeds, fitted_train, fitted_val)
    if out.exists():
        raise FileExistsError("Use a fresh directory; completed failures cannot be overwritten")
    config = json.loads((snapshot / "config.json").read_text())
    prefix = config["n_action_steps"] if execute_steps is None else execute_steps
    if type(prefix) is not int or not 1 <= prefix <= config["chunk_size"]:
        raise ValueError("Execution prefix must fit the native action chunk")
    if (not np.isfinite(initial_gripper_percent) or not 0 <= initial_gripper_percent <= 100):
        raise ValueError("Initial gripper must be a finite percentage")
    randomization_config, extras = None, []
    if domain_randomization is not None:
        path = Path(domain_randomization)
        randomization_config = json.loads(path.read_text())
        extras.extend([path, Path("src/real2sim/sim/domain_randomization.py")])
    before = runtime_hashes(snapshot, source_scene, extras)
    protocol = {"schema": "raw_act_fixed_cohort_protocol_v1", "snapshot": str(snapshot),
        "checkpoint_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_scene": str(source_scene), "robots": list(robots), "seeds": list(seeds),
        "requested_trials_per_robot": len(seeds), "execution_steps": prefix,
        "control_hz": 25., "initial_gripper_percent": initial_gripper_percent,
        "training_seeds": sorted(fitted_train), "validation_seeds": sorted(fitted_val),
        "fitting_seed_declarations": declarations,
        "domain_randomization": randomization_config,
        "runtime_source_sha256": before, "raw_policy_only": True,
        "rgb_controller_enabled": False, "scoring_only_privileged_metrics": True}
    protocol["protocol_id"] = digest_json(protocol)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol, atomic=True)
    if plan_only:
        return protocol
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot, inference_backend="vectorized")
    rows = []
    report = {"schema": "raw_act_fixed_cohort_results_v1", "status": "running",
        "protocol_id": protocol["protocol_id"], "raw_policy_only": True, "trials": rows,
        "gate": completion_gate(rows, robots, seeds, integrity_verified=False)}
    write_json(out / "report.json", report, atomic=True)
    for robot in robots:
        shared, setup_error, randomizer = None, None, None
        try:
            shared = SharedEpisode(robot, out / "setups" / robot, source_scene,
                native_contract=policy.scene_contract, initial_gripper_percent=initial_gripper_percent)
            if randomization_config is not None:
                from real2sim.sim.domain_randomization import DomainRandomizer

                randomizer = DomainRandomizer(shared.env, randomization_config)
        except (RuntimeError, ValueError, OSError) as exc:
            setup_error = exc
        try:
            for seed in seeds:
                directory = out / robot / f"seed_{seed:06d}"
                if setup_error:
                    row = _setup_failure(robot, seed, directory, policy, setup_error)
                else:
                    try:
                        draw = randomizer.apply(seed) if randomizer is not None else {}
                    except (RuntimeError, ValueError, OSError) as exc:
                        row = _setup_failure(robot, seed, directory, policy, exc)
                    else:
                        row = run_episode(shared, policy, seed, directory, execute_steps=prefix,
                            scenario=draw.get("scenario"), randomization=draw.get("parameters"))
                row["trace_sha256"] = sha256_file(Path(row["trace"]))
                write_json(directory / "report.json", {"schema": "raw_act_trial_v1", "status": "complete",
                    "protocol_id": protocol["protocol_id"], "trial": row,
                    "trial_sha256": digest_json(row)}, atomic=True)
                rows.append(row)
                report["gate"] = completion_gate(rows, robots, seeds, integrity_verified=False)
                write_json(out / "report.json", report, atomic=True)
                print(json.dumps({"robot": robot, "seed": seed,
                    "pick_place_success": row["pick_place_success"], "abort": row["abort"]}), flush=True)
        finally:
            if shared is not None:
                shared.close()
    after = {path: sha256_file(path) for path in before}
    changes = [path for path in before if before[path] != after[path]]
    report.update(status="complete", runtime_source_changes=changes,
        source_sha256_before=before, source_sha256_after=after,
        checkpoint_unchanged=policy.weights_sha256 == sha256_file(snapshot / "model.safetensors"),
        conditions={robot: summarize([row for row in rows if row["robot"] == robot]) for robot in robots})
    report["gate"] = completion_gate(rows, robots, seeds,
        integrity_verified=not changes and report["checkpoint_unchanged"])
    report["objective_met"] = report["gate"]["objective_met"]
    write_json(out / "report.json", report, atomic=True)
    audit = audit_report(out / "report.json")
    write_json(out / "evidence.json", audit, atomic=True)
    report["gate"] = audit["gate"]
    report["objective_met"] = audit["gate"]["objective_met"]
    report["evidence_verified"] = audit["locally_verified"]
    write_json(out / "report.json", report, atomic=True)
    return report
