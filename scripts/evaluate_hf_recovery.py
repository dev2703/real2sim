"""Evaluate latest frozen HF ACT followed by explicitly disclosed RGB recovery.

This measures an assisted system. It never labels geometric-controller success
as unassisted ACT transfer. The HF weights and lighting remain unchanged; object
ground truth is used only by the existing scorer and for stopping an episode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from real2sim.artifacts import sha256_file
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.panda.adapter import PandaIK
from real2sim.panda.env import PandaPickPlaceEnv
from real2sim.policy.follower_joint import check_joint_request
from real2sim.policy.hf_recovery import ApproachMonitor
from real2sim.policy.native_lerobot_act import NativeLeRobotACT
from real2sim.policy.visual_waypoint import VisualWaypointController
from real2sim.serialization import json_default, write_json
from real2sim.sim.kinematics import NativeKinematics

if __package__:
    from scripts import evaluate_hf_act_panda_20261028 as transfer
else:
    import evaluate_hf_act_panda_20261028 as transfer

MEASURED_SCENE = Path("configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml")


class StaticBowlRecovery(VisualWaypointController):
    """Remember an observed static bowl while always reacquiring the movable ball."""

    def _acquire(self):
        acquired = super()._acquire()
        targets = self.perception_log[-1]["targets"]
        if targets["bowl"] is not None:
            self.bowl_memory = np.asarray(targets["bowl"]["center_task"], float)
        if acquired:
            return True
        memory = getattr(self, "bowl_memory", None)
        if targets["ball"] is not None and memory is not None and self.env.task["bowl"]["static"]:
            self.ball = np.asarray(targets["ball"]["center_task"], float)
            self.bowl = memory.copy()
            self.perception_log[-1]["static_bowl_memory_from_initial_rgb"] = memory.tolist()
            return True
        return False


def lighting(env):
    return {"ambient": env.model.vis.headlight.ambient.copy().tolist(),
            "diffuse": env.model.vis.headlight.diffuse.copy().tolist(),
            "specular": env.model.vis.headlight.specular.copy().tolist(),
            "light_ambient": env.model.light_ambient.copy().tolist(),
            "light_diffuse": env.model.light_diffuse.copy().tolist(),
            "light_specular": env.model.light_specular.copy().tolist(),
            "light_position": env.model.light_pos.copy().tolist(),
            "light_direction": env.model.light_dir.copy().tolist()}


def run(robot, out, seeds, *, snapshot=transfer.native.SNAPSHOT,
        source_scene=MEASURED_SCENE, video=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply unique trial seeds")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    policy = NativeLeRobotACT(snapshot)
    policy.policy.eval().requires_grad_(False)
    scene, task = transfer.prepare_source(out, (0.,) * 5, source_scene=source_scene,
                                         initial_gripper_percent=1.2)
    source = transfer.retarget.VirtualSource(scene)
    env = source.source
    report = {"schema": "frozen_hf_with_rgb_recovery_v1", "status": "running",
              "robot": robot, "snapshot": str(snapshot), "seeds": seeds,
              "checkpoint_sha256": policy.weights_sha256,
              "controller_source": "FROZEN_HF_ACT_WITH_RGB_WAYPOINT_RECOVERY",
              "assisted_system": True, "unassisted_frozen_transfer_demonstrated": False,
              "privileged_dynamic_object_poses_or_held_flags_drive_control": False,
              "static_collision_map_used_by_native_guards": True,
              "lighting_conditions_changed": False,
              "recovery_rule": "Initial RGB target/live FK approach stagnation for2s, or bounded5s HF attempt; recovery reacquires the ball and may remember a previously observed static bowl",
              "trials": []}
    watched = [Path(__file__), Path(transfer.__file__), Path(transfer.native.__file__),
               Path(transfer.retarget.__file__), Path("physical_measurements.md"),
               Path("src/real2sim/policy/hf_recovery.py"),
               Path("src/real2sim/policy/visual_waypoint.py"),
               Path("src/real2sim/policy/visual_targets.py"),
               Path("src/real2sim/policy/native_lerobot_act.py"),
               Path("src/real2sim/policy/follower_joint.py"),
               Path("src/real2sim/sim/env.py"), Path("src/real2sim/sim/kinematics.py"),
               Path("src/real2sim/panda/env.py"), Path("src/real2sim/panda/adapter.py"),
               Path("src/real2sim/panda/workspace.py"),
               Path("src/real2sim/evaluation/metrics.py"),
               source_scene, scene, out / "source/task.yaml", out / "source/data.yaml"]
    watched += list(Path(snapshot).glob("*.json")) + list(Path(snapshot).glob("*.safetensors"))
    writer = None
    try:
        start, opening = None, None
        if robot == "panda":
            panda_scene, camera = transfer.prepare_panda(out, source, task, -.15)
            watched += [panda_scene, out / "panda_task.yaml", out / "panda_data.yaml"]
            env = PandaPickPlaceEnv(panda_scene, render_images=False)
            start, opening = transfer.panda_start(env, source, -.15)
            guard = PandaIK(env.model, env.config, task_origin=env.offset)
            report["panda_wrist"] = camera
        else:
            guard = NativeKinematics(env.model, env.config)
        hashes = {str(path): sha256_file(path) for path in watched}
        report["source_sha256_before"] = hashes
        report["lighting_before"] = lighting(env)
        write_json(out / "report.json", report, atomic=True)
        recovery = StaticBowlRecovery(env, robot=robot)
        for seed in seeds:
            directory = out / f"seed_{seed:03d}"
            directory.mkdir()
            tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz),
                approach_distance_m=.03, table_top_z=env.table_top,
                ball_radius_m=env.task["ball"]["radius"])
            try:
                obs, _ = env.reset(seed=seed, scenario={"joint_positions": start,
                    "gripper_opening": opening} if robot == "panda" else None)
                source.q = source.source.initial_qpos.copy()
                source.last_target = source.q.copy()
                policy.reset()
                recovery.reset()
                recovery.bowl_memory = None
                acquired = recovery._acquire()
                initial_rgb = recovery.perception_log[-1]
            except (RuntimeError, ValueError) as exc:
                row = tracker.row(experiment_id="HF_RGB_RECOVERY", controller_source=report["controller_source"],
                    robot=robot, seed=seed, checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                    mean_latency_ms=None, video_path="", abort=f"{type(exc).__name__}: {exc}")
                row.update(hf_ticks=0, recovery_ticks=0, hf_policy_calls=0,
                           recovery_trigger=None, initial_rgb=None, recovery_failure=None)
                report["trials"].append(row)
                report["summary"] = summarize(report["trials"])
                write_json(out / "report.json", report, atomic=True)
                continue
            monitor = ApproachMonitor(env.control_hz)
            queue, trigger, abort = [], None, None
            hf_ticks = recovery_ticks = calls = 0
            if video:
                writer = CaptionedVideo(directory / "rollout.mp4", env.control_hz, (360, 640))
            with (directory / "ticks.jsonl").open("w") as stream:
                try:
                    for tick in range(env.max_episode_steps):
                        distance = None
                        if acquired and trigger is None:
                            pose = env.ee_pose()
                            estimated_grasp = pose[:3, 3] + pose[:3, :3] @ recovery.ball_in_tool
                            distance = float(np.linalg.norm(estimated_grasp - recovery.ball))
                        reason = monitor.update(tick, distance) if trigger is None else None
                        if reason:
                            trigger = {"tick": tick, "time_s": tick / env.control_hz,
                                       "reason": reason, "rgb_tool_ball_distance_m": distance}
                            recovery.reset()
                            queue.clear()
                        raw_action = None
                        if trigger is not None:
                            result = recovery.step(obs, env)
                            obs, info, truncated = result.obs, result.info, result.truncated
                            rejected, saturation = result.ik_rejected, result.saturation
                            detail = result.detail
                            recovery_ticks += 1
                        else:
                            if not queue:
                                state = env.real_state() if robot == "so101" else source.source_state(env)[0]
                                if robot == "panda":
                                    state[5] = float(np.clip(env.gripper_opening() / env.max_opening, 0., 1.) * 90.)
                                images = {key: env.render(env.camera_bindings[key]) for key in policy.camera_keys}
                                queue = list(policy.predict_action_chunk(state, images)[:policy.config.n_action_steps])
                                calls += 1
                            raw_action = queue.pop(0)
                            _, bounded = source.source_action(raw_action)
                            if robot == "so101":
                                current = env.sim_state()
                                guard.data.qpos[:] = env.data.qpos
                                accepted, command_reason, _ = check_joint_request(guard, current,
                                    env.data.ctrl[env.actuator_ids], bounded,
                                    env.max_velocity / env.control_hz, samples=17)
                                command = bounded if accepted else current
                            else:
                                desired = transfer.retarget.target_from_source(source, bounded)
                                observed = env.ee_pose()
                                delta = desired[:3, 3] - observed[:3, 3]
                                length = np.linalg.norm(delta)
                                if length > .024:
                                    desired[:3, 3] = observed[:3, 3] + delta * .024 / length
                                width = transfer.retarget.panda_opening_from_percent(raw_action[5], env.max_opening)
                                solved = guard.solve(desired, env.arm_qpos(),
                                    finger_opening=max(width, env.gripper_opening()))
                                accepted, command_reason = solved.accepted, solved.reason
                                command = np.r_[solved.qpos if accepted else env.arm_qpos(), width]
                            obs, _, _, truncated, info = env.step(command)
                            rejected = not accepted
                            saturation = float(np.mean(info["control"]["limit_clipped"]))
                            detail = {"guard_reason": command_reason, "requested_native": command}
                            hf_ticks += 1
                        done = tracker.update(info["privileged_metrics"],
                                              ik_rejected=rejected, saturation=saturation)
                        stream.write(json.dumps({"tick": tick, "mode": "RGB_RECOVERY" if trigger else "HF_ACT",
                            "raw_hf_action": raw_action, "detail": detail, "control": info["control"],
                            "scoring_only": info["privileged_metrics"]}, default=json_default) + "\n")
                        if writer:
                            writer.write(env.render("iphone")[:, :, ::-1].copy(),
                                f"Frozen HF + RGB recovery; {robot}; {'RECOVERY' if trigger else 'HF ACT'}")
                        if done or truncated:
                            break
                except (RuntimeError, ValueError) as exc:
                    abort = f"{type(exc).__name__}: {exc}"
                finally:
                    if writer:
                        writer.close()
                        writer = None
            row = tracker.row(experiment_id="HF_RGB_RECOVERY", controller_source=report["controller_source"],
                robot=robot, seed=seed, checkpoint_sha256=policy.weights_sha256, checkpoint_step=None,
                mean_latency_ms=None, video_path=str(directory / "rollout.mp4") if video else "", abort=abort)
            row.update(hf_ticks=hf_ticks, recovery_ticks=recovery_ticks, hf_policy_calls=calls,
                recovery_trigger=trigger, initial_rgb=initial_rgb, recovery_failure=recovery.failure,
                final_phase=recovery.phase, trace=str(directory / "ticks.jsonl"))
            report["trials"].append(row)
            report["summary"] = summarize(report["trials"])
            write_json(out / "report.json", report, atomic=True)
            print(json.dumps({"robot": robot, "seed": seed, "placed": row["pick_place_success"],
                "hf_ticks": hf_ticks, "recovery_ticks": recovery_ticks, "abort": abort,
                "failure": recovery.failure}), flush=True)
        report.update(status="complete", lighting_after=lighting(env),
            checkpoint_unchanged=sha256_file(Path(snapshot) / "model.safetensors") == policy.weights_sha256,
            source_sha256_after={path: sha256_file(Path(path)) for path in hashes})
        report["runtime_source_changes"] = [path for path in hashes
            if hashes[path] != report["source_sha256_after"][path]]
        if report["lighting_before"] != report["lighting_after"]:
            raise RuntimeError("Lighting changed during evaluation")
        write_json(out / "report.json", report, atomic=True)
        return report
    finally:
        if writer:
            writer.close()
        if env is not source.source:
            env.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    parser.add_argument("--snapshot", type=Path, default=transfer.native.SNAPSHOT)
    parser.add_argument("--source-scene", type=Path, default=MEASURED_SCENE)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    run(args.robot, args.out, args.seeds, snapshot=args.snapshot,
        source_scene=args.source_scene, video=args.video)
