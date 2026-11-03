"""Predeclare and evaluate a frozen shared semantic ACT on paired fresh layouts.

The learned policy, perception, robot adapters, reset conventions and scorer are
the existing preview-three experiment. Only declared scene factors vary. This
does not evaluate the separate uploaded real-data image ACT checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from lerobot.policies.act.modeling_act import ACTPolicy

from real2sim.artifacts import sha256_file
from real2sim.config import REPO_ROOT, load_yaml, project_path
from real2sim.evaluation.metrics import TrialTracker, summarize
from real2sim.evaluation.video import CaptionedVideo
from real2sim.serialization import json_default, write_json

if __package__:
    from scripts import run_v2_semantic_act_20261027 as experiment
else:
    import run_v2_semantic_act_20261027 as experiment

SOURCE = Path("outputs/v2/semantic_act_transfer_20261027_v2")
ROBOTS = ("so101", "panda")
CONDITIONS = {
    "nominal": {"lighting_scale": 1., "radius_scale": 1., "sliding_scale": 1.},
    "lighting_dim": {"lighting_scale": .6, "radius_scale": 1., "sliding_scale": 1.},
    "lighting_bright": {"lighting_scale": 1.4, "radius_scale": 1., "sliding_scale": 1.},
    "ball_small": {"lighting_scale": 1., "radius_scale": .95, "sliding_scale": 1.},
    "ball_large": {"lighting_scale": 1., "radius_scale": 1.05, "sliding_scale": 1.},
    "friction_low": {"lighting_scale": 1., "radius_scale": 1., "sliding_scale": 2 / 3},
    "friction_high": {"lighting_scale": 1., "radius_scale": 1., "sliding_scale": 1.5},
}


def save(path, value):
    write_json(path, value, atomic=True)


def dependency_paths(source):
    paths = {Path(__file__).resolve(), Path(experiment.__file__).resolve()}
    paths.update((REPO_ROOT / "src/real2sim").rglob("*.py"))
    paths.update((REPO_ROOT / "configs").rglob("*.yaml"))
    paths.update((REPO_ROOT / "assets/menagerie").rglob("*.xml"))
    paths.add(REPO_ROOT / "assets/menagerie/manifest.json")
    paths.update(source / name for name in (
        "normalization.json", "semantic_data.npz", "source.json",
        "checkpoint/config.json", "checkpoint/model.safetensors"))
    return sorted(p.resolve() for p in paths)


def prepare(out, source=SOURCE, seeds=tuple(range(10000, 10020))):
    if out.exists():
        raise FileExistsError(out)
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError("the confirmatory protocol requires exactly 20 unique layout seeds")
    out.mkdir(parents=True)
    source = source.resolve()
    task = load_yaml(project_path(load_yaml(experiment.SO_SCENE)["task_config"]))
    layouts = []
    region = task["ball"]["start_region"]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        r = float(region["radius"]) * np.sqrt(rng.uniform())
        theta = rng.uniform(0, 2 * np.pi)
        xy = np.asarray(region["center_task_xy"]) + r * np.array([np.cos(theta), np.sin(theta)])
        bowl_rng = np.random.default_rng(np.random.SeedSequence([seed, 20261027]))
        bowl_r = .03 * np.sqrt(bowl_rng.uniform())
        bowl_theta = bowl_rng.uniform(0, 2 * np.pi)
        bowl_xy = np.asarray(task["bowl"]["center_task_xy"]) + bowl_r * np.array([
            np.cos(bowl_theta), np.sin(bowl_theta)])
        layouts.append({"seed": seed, "ball_xy": xy.tolist(),
                        "bowl_xy": bowl_xy.tolist()})
    hashes = {str(p): sha256_file(p) for p in dependency_paths(source)}
    audit_path = REPO_ROOT / "results/v2/generalization_layout_novelty_20261027.json"
    audit = json.loads(audit_path.read_text())
    audited_layouts = [{key: row[key] for key in ("seed", "ball_xy", "bowl_xy")}
        for row in audit["planned_layouts"]]
    if audited_layouts != layouts:
        raise ValueError("Independent novelty audit does not match the declared layouts")
    protocol = {
        "schema": "frozen_shared_semantic_act_generalization_20261027_v1",
        "prepared_utc": datetime.now(UTC).isoformat(),
        "prepared_before_rollouts": True, "source": str(source),
        "layout_novelty_audit": str(audit_path),
        "layout_novelty_audit_sha256": sha256_file(audit_path),
        "checkpoint_sha256": sha256_file(source / "checkpoint/model.safetensors"),
        "robots": list(ROBOTS), "layouts": layouts, "conditions": CONDITIONS,
        "schedule": "Complete nominal on both arms before any perturbation. Every perturbation repeats all 20 layouts independently.",
        "frozen_settings": {"joint_margin_rad": .01, "position_forecast_offset": 3,
            "intent_forecast_offset": 0, "selected_so_orientation": False,
            "panda_base_dx_m": -.15, "panda_parked_perception_then_mapped_start": True,
            "so_grasp_point_offset_m": experiment.SO_OFFSET.tolist(),
            "panda_grasp_point_offset_m": experiment.PANDA_OFFSET.tolist(),
            "max_point_step_m": .024, "nominal_perception_radius_m": task["ball"]["radius"],
            "reset": "SO configured native start; Panda RGB at configured parked pose followed by same-layout reset to existing mapped source point; no settling",
            "control_hz": 12.5, "horizon_ticks": 375, "training_updates": 0,
            "requires_grad": False, "policy_mode": "eval/inference_mode"},
        "task": {"nominal_ball_radius_m": task["ball"]["radius"],
            "ball_mass_kg_constant": task["ball"]["mass"]},
        "size_protocol": "Rebuild task geometry/inertia/spawn/scoring with changed radius and constant mass; RGB frontend retains the nominal radius assumption in every condition. The true varied radius is never supplied to perception or ACT.",
        "friction_protocol": "Multiply only sliding coefficient on all collision-enabled geoms by 2/3 or 1.5, preserving material ratios, priorities, solref, torsional and rolling coefficients. This changes effective ball contacts under MuJoCo max mixing. Other robot/static contacts also change and are disclosed.",
        "lighting_protocol": "Multiply model light ambient/diffuse/specular and headlight ambient/diffuse/specular by .6 or1.4; camera/exposure/shadows unchanged.",
        "layout_scope": "Fresh ball draws in existing 3cm disc from default_rng(seed); independent bowl jitter uniform3cm disc from default_rng(SeedSequence([seed,20261027])). Draw radial-uniform first, then angle. Same explicit XY pairs in both arms/all conditions; no workspace-extrapolation or physical registration claim.",
        "scoring": "Existing TrialTracker: held lift, release over bowl, unheld resting placement for 1s; full30s timeout and original success early stop; every failure/abort retained.",
        "reporting": "Raw numerator/20 and paired transitions; no confidence intervals for N<=20 under existing experiment protocol.",
        "static_collision_map_assumption": "Exact configured static scene including declared shifted bowl is available to unchanged robot-specific kinematic/path collision guards; RGB alone supplies policy object coordinates.",
        "physical_calibration_accepted": False, "real_hf_weights_used": False,
        "source_sha256_before": hashes,
    }
    save(out / "protocol.json", protocol)
    (out / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    return protocol


def read_protocol(out):
    return json.loads((out / "protocol.json").read_text())


def verify_sources(protocol):
    actual = {p: sha256_file(p) for p in protocol["source_sha256_before"]}
    changes = [p for p, value in actual.items() if value != protocol["source_sha256_before"][p]]
    if changes:
        raise RuntimeError(f"Frozen dependencies changed: {changes}")
    if "layout_novelty_audit" in protocol and sha256_file(protocol["layout_novelty_audit"]) != protocol["layout_novelty_audit_sha256"]:
        raise RuntimeError("Frozen layout novelty audit changed")
    return actual


def configured_scene(out, robot, condition, source):
    directory = out / condition / robot
    directory.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(experiment.SO_SCENE if robot == "so101" else experiment.PANDA_SCENE)
    if robot == "panda":
        cfg["control_hz"] = 12.5
        cfg["data_config"] = "configs/v2/data.waypoint_20261026.yaml"
    task = load_yaml(project_path(cfg["task_config"]))
    task["ball"]["radius"] *= CONDITIONS[condition]["radius_scale"]
    task_path = directory / "task.yaml"
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    cfg["task_config"] = str(task_path.resolve())
    config_path = directory / "sim.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return directory, config_path


def vary_model(env, factors):
    before = {}
    scale = factors["lighting_scale"]
    for name in ("ambient", "diffuse", "specular"):
        light = getattr(env.model, "light_" + name)
        head = getattr(env.model.vis.headlight, name)
        before["light_" + name] = light.copy()
        before["headlight_" + name] = head.copy()
        light[:] *= scale
        head[:] *= scale
    enabled = (env.model.geom_contype | env.model.geom_conaffinity) != 0
    friction_before = env.model.geom_friction.copy()
    env.model.geom_friction[enabled, 0] *= factors["sliding_scale"]
    return {"lighting_before": before, "lighting_scale": scale,
        "sliding_scale": factors["sliding_scale"],
        "geom_friction": [{"name": env.model.geom(i).name, "collision_enabled": bool(enabled[i]),
            "before": friction_before[i], "after": env.model.geom_friction[i].copy()}
            for i in range(env.model.ngeom)],
        "ball_radius_m": float(env.model.geom_size[env.ball_geom, 0]),
        "ball_mass_kg": float(env.model.body_mass[env.ball_body]),
        "ball_inertia_kg_m2": env.model.body_inertia[env.ball_body].copy()}


def picture(path, rgb):
    if not cv2.imwrite(str(path), rgb[:, :, ::-1]):
        raise OSError(f"Image write failed: {path}")


def acquire_frozen_perception(env, nominal_radius):
    """Keep the established RGB frontend's size assumption fixed under size changes."""
    actual = env.task["ball"]["radius"]
    try:
        env.task["ball"]["radius"] = nominal_radius
        return experiment.acquire(env)
    finally:
        env.task["ball"]["radius"] = actual


def run_trial(env, policy, stats, protocol, layout, robot, directory, mapped_start, video):
    seed = layout["seed"]
    trial = directory / f"seed_{seed}"
    trial.mkdir()
    tracker = TrialTracker(success_hold_ticks=math.ceil(env.control_hz),
        approach_distance_m=.03, table_top_z=env.table_top,
        ball_radius_m=env.task["ball"]["radius"])
    setup, initial, objects, writer, abort = None, None, None, None, None
    started = time.perf_counter()
    reasons, durations, samples = {}, [], {}
    last_point, last_intent, age = None, 1., 0.
    try:
        scenario = {"ball_xy": layout["ball_xy"], "bowl_xy": layout["bowl_xy"]}
        _, parked = env.reset(seed=seed, scenario=scenario)
        initial = parked["privileged_metrics"]
        picture(trial / "perception.png", env.render("iphone"))
        objects = acquire_frozen_perception(env, protocol["task"]["nominal_ball_radius_m"])
        if robot == "panda":
            _, mapped = env.reset(seed=seed, scenario={**scenario, "joint_positions": mapped_start})
            setup = {key: bool(np.allclose(parked["privileged_metrics"][key], mapped["privileged_metrics"][key]))
                for key in ("ball_xyz_task", "bowl_xyz_task")}
            if not all(setup.values()):
                raise RuntimeError("Panda parked/mapped reset changed object layout")
        picture(trial / "initial.png", env.render("iphone"))
        if objects is None:
            raise ValueError("Initial RGB ball/bowl detection unavailable")
        adapter = experiment.PointAdapter(env, robot, objects, joint_margin=.01)
        if video:
            writer = CaptionedVideo(trial / "rollout.mp4", env.control_hz, (360, 640))
        with (trial / "ticks.jsonl").open("w") as stream:
            for tick in range(env.max_episode_steps):
                p = experiment.point(env.ee_pose(), adapter.offset)
                velocity = np.zeros(3) if last_point is None else (p - last_point) * 12.5
                state = np.r_[p, velocity, last_intent, min(age, 10.)]
                batch = {"observation.state": torch.tensor(experiment.normalized(state, stats["state"]), dtype=torch.float32)[None],
                    "observation.environment_state": torch.tensor(experiment.normalized(objects, stats["objects"]), dtype=torch.float32)[None]}
                t0 = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(batch)[0].numpy()
                durations.append((time.perf_counter() - t0) * 1000)
                raw = chunk[3].copy()
                raw[3] = chunk[0, 3]
                action = raw * np.asarray(stats["action"]["std"]) + np.asarray(stats["action"]["mean"])
                intent = float(action[3] >= .5)
                _, truncated, info, accepted, reason, requested = adapter.apply(action[:3], bool(intent))
                reasons[reason] = reasons.get(reason, 0) + 1
                pm = info["privileged_metrics"]
                done = tracker.update(pm, ik_rejected=not accepted,
                    saturation=float(np.mean(info["control"]["limit_clipped"])))
                contact_samples = []
                for contact in env.data.contact[:env.data.ncon]:
                    a, b = int(contact.geom1), int(contact.geom2)
                    if env.ball_geom in (a, b):
                        other = b if a == env.ball_geom else a
                        name = env.model.geom(other).name or f"geom_{other}"
                        value = np.asarray(contact.friction).copy()
                        samples.setdefault(name, value.tolist())
                        contact_samples.append({"other": name, "friction": value})
                record = {"tick": tick, "state": state, "rgb_objects": objects,
                    "action": action, "bounded_point": requested, "intent": intent,
                    "accepted": accepted, "reason": reason, "control": info["control"],
                    "native_readback": env.sim_state(), "ball_contact_friction": contact_samples,
                    "scoring_only": {key: pm[key] for key in (
                        "ball_xyz_task", "bowl_xyz_task", "tool_xyz_task", "ee_ball_distance_m",
                        "ball_held", "lifted", "ball_over_bowl", "ball_in_bowl", "ball_resting")}}
                stream.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
                if writer:
                    writer.write(env.render("iphone")[:, :, ::-1].copy(), f"FROZEN SEMANTIC ACT {robot} seed{seed}")
                last_point = p.copy()
                age = age + 1 / 12.5 if intent == last_intent else 0.
                last_intent = intent
                if done or truncated:
                    break
    except (ValueError, RuntimeError, OSError) as exc:
        abort = f"{type(exc).__name__}: {exc}"
    finally:
        if writer:
            writer.close()
    if initial is not None:
        picture(trial / "final.png", env.render("iphone"))
    row = tracker.row(experiment_id="FROZEN_SHARED_ACT_GENERALIZATION_20261027",
        controller_source="FROZEN_SEMANTIC_ACT", robot=robot, seed=seed,
        checkpoint_sha256=protocol["checkpoint_sha256"], checkpoint_step=2000,
        mean_latency_ms=float(np.mean(durations)) if durations else None,
        video_path=str(trial / "rollout.mp4") if video else "", abort=abort)
    row.update(adapter_reasons=reasons, rgb_objects=objects, perception_setup=setup,
        initial_scoring_only=initial, held_ever=tracker.held_ever, wall_s=time.perf_counter() - started,
        policy_calls=len(durations), ball_contact_friction_samples=samples,
        trace=str(trial / "ticks.jsonl"), declared_layout=layout)
    save(trial / "result.json", row)
    return row


def run_condition(out, robot, condition, *, video=False):
    protocol = read_protocol(out)
    before = verify_sources(protocol)
    if condition != "nominal":
        for arm in ROBOTS:
            path = out / "nominal" / arm / "report.json"
            if not path.exists() or json.loads(path.read_text())["status"] != "complete":
                raise ValueError("Finish both nominal arms before perturbations")
    source = Path(protocol["source"])
    torch.set_num_threads(2)
    policy = ACTPolicy.from_pretrained(source / "checkpoint", strict=True, local_files_only=True).eval()
    policy.requires_grad_(False)
    if policy.training or any(p.requires_grad for p in policy.parameters()):
        raise RuntimeError("Policy was not frozen")
    stats = json.loads((source / "normalization.json").read_text())
    with np.load(source / "semantic_data.npz") as arrays:
        start_point = arrays["state"][0, :3].copy()
    directory, cfg_path = configured_scene(out, robot, condition, source)
    if (directory / "report.json").exists():
        raise FileExistsError(directory / "report.json")
    report = {"schema": protocol["schema"], "status": "running", "robot": robot,
        "condition": condition, "protocol_sha256": sha256_file(out / "protocol.json"),
        "checkpoint_sha256": protocol["checkpoint_sha256"], "no_retraining": True,
        "trials": [], "source_sha256_before": before}
    save(directory / "report.json", report)
    env_class = experiment.So101PickPlaceEnv if robot == "so101" else experiment.PandaPickPlaceEnv
    with env_class(cfg_path, render_images=False) as env:
        report["model_variation"] = vary_model(env, CONDITIONS[condition])
        mapped_start = None
        if robot == "panda":
            env.model.body_pos[env.model.body("link0").id] += [-.15, 0., 0.]
            ik = experiment.PandaIK(env.model, env.config, task_origin=env.offset)
            target = np.eye(4)
            target[:3, :3] = experiment.POINTING_DOWN
            target[:3, 3] = start_point - experiment.POINTING_DOWN @ experiment.PANDA_OFFSET
            solved = ik.solve(target, env.initial_qpos, finger_opening=env.max_opening)
            if not solved.accepted:
                raise RuntimeError(f"Panda mapped start rejected: {solved.reason}")
            mapped_start = solved.qpos
        report["mapped_panda_initial_joints"] = mapped_start
        report["resolved_config"] = env.config
        report["resolved_task"] = env.task
        save(directory / "report.json", report)
        for layout in protocol["layouts"]:
            row = run_trial(env, policy, stats, protocol, layout, robot, directory, mapped_start, video)
            report["trials"].append(row)
            report["summary"] = summarize(report["trials"])
            save(directory / "report.json", report)
            print(json.dumps({"robot": robot, "condition": condition, "seed": layout["seed"],
                "success": row["pick_place_success"], "lifted": row["lifted"],
                "phase": row["pick_place_phase_failed"], "abort": row["abort"],
                "wall_s": row["wall_s"]}), flush=True)
    after = verify_sources(protocol)
    report.update(status="complete", source_sha256_after=after, runtime_source_changes=[],
        checkpoint_unchanged=before[str((source / "checkpoint/model.safetensors").resolve())] == after[str((source / "checkpoint/model.safetensors").resolve())])
    save(directory / "report.json", report)
    return report


def verify_completed_report(out, protocol, report, robot, condition):
    """Refuse incomplete, differently frozen or differently scheduled reports."""
    if report["robot"] != robot or report["condition"] != condition:
        raise ValueError("Completed report identity disagrees with its path")
    if report["protocol_sha256"] != sha256_file(out / "protocol.json"):
        raise ValueError("Completed report used a different frozen protocol")
    if report["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
        raise ValueError("Completed report used a different checkpoint")
    rows = report["trials"]
    if len(rows) != 20 or [row["declared_layout"] for row in rows] != protocol["layouts"]:
        raise ValueError("Completed report does not contain the exact 20-layout schedule")
    if any(row["seed"] != layout["seed"] or row["robot"] != robot
           or row["checkpoint_sha256"] != protocol["checkpoint_sha256"]
           for row, layout in zip(rows, protocol["layouts"], strict=True)):
        raise ValueError("A completed trial disagrees with its declared identity")
    if report["summary"] != summarize(rows):
        raise ValueError("Completed report summary disagrees with its trial rows")
    expected = protocol["source_sha256_before"]
    if report["source_sha256_before"] != expected or report["source_sha256_after"] != expected:
        raise ValueError("Completed report dependencies changed")
    if report.get("runtime_source_changes") or report.get("checkpoint_unchanged") is not True:
        raise ValueError("Completed report did not preserve frozen sources/checkpoint")


def aggregate(out):
    protocol = read_protocol(out)
    verify_sources(protocol)
    reports = {}
    for condition in CONDITIONS:
        reports[condition] = {}
        for robot in ROBOTS:
            path = out / condition / robot / "report.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["status"] == "complete":
                    verify_completed_report(out, protocol, report, robot, condition)
                reports[condition][robot] = {"status": report["status"], "report": str(path),
                    "summary": report.get("summary")}
                nominal_path = out / "nominal" / robot / "report.json"
                if condition != "nominal" and nominal_path.exists():
                    nominal = {r["seed"]: r for r in json.loads(nominal_path.read_text())["trials"]}
                    pairs = [(nominal[r["seed"]]["pick_place_success"], r["pick_place_success"]) for r in report["trials"]]
                    reports[condition][robot]["paired_success_transitions"] = {
                        "both_succeed": sum(a and b for a, b in pairs),
                        "nominal_only": sum(a and not b for a, b in pairs),
                        "variation_only": sum(not a and b for a, b in pairs),
                        "both_fail": sum(not a and not b for a, b in pairs)}
    complete = all(reports[c].get(r, {}).get("status") == "complete" for c in CONDITIONS for r in ROBOTS)
    result = {"schema": protocol["schema"], "status": "complete" if complete else "running",
        "protocol": str(out / "protocol.json"), "protocol_sha256": sha256_file(out / "protocol.json"),
        "checkpoint_sha256": protocol["checkpoint_sha256"], "conditions": reports,
        "planned_trials": 20 * len(ROBOTS) * len(CONDITIONS),
        "completed_trials": sum(item.get("summary", {}).get("n", 0) for arms in reports.values() for item in arms.values()),
        "physical_calibration_accepted": False, "real_hf_weights_used": False}
    save(out / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--robot", choices=ROBOTS)
    parser.add_argument("--condition", choices=list(CONDITIONS))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out, args.source)
    elif args.aggregate:
        aggregate(args.out)
    elif args.robot and args.condition:
        run_condition(args.out, args.robot, args.condition, video=args.video)
    else:
        parser.error("select --prepare, --aggregate, or both --robot and --condition")
