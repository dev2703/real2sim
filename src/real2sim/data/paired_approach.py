"""Verified stationary-ball approach renders; real states/actions/splits are untouched.

Simulator object positions are offline rendering setup, never extra policy inputs.
Out-of-limit recorded joints are rendered kinematically without stepping physics.
Unknown contact/transport images remain real. No simulated actions or episodes.
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from real2sim.artifacts import sha256_file as file_sha256
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.sim.camera_fit import backproject_to_plane, ball_centroid_px
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def stationary_prefix(detections, *, max_shift_px=3.0, initial_ticks=5):
    """Stop permanently at the first missing/moved/partially occluded ball detection."""
    if len(detections) < initial_ticks or any(d is None for d in detections[:initial_ticks]):
        return np.empty(0, dtype=int), None, "initial_detection_missing"
    first = np.asarray(detections[:initial_ticks], float)
    centre = np.median(first[:, :2], axis=0)
    area = float(np.median(first[:, 2]))
    if np.max(np.linalg.norm(first[:, :2] - centre, axis=1)) > max_shift_px:
        return np.empty(0, dtype=int), None, "initial_ball_not_stationary"
    good = []
    reason = "time_cap"
    for i, d in enumerate(detections):
        if d is None:
            reason = "ball_not_visible"
            break
        if np.linalg.norm(np.asarray(d[:2]) - centre) > max_shift_px:
            reason = "ball_moved_or_detection_changed"
            break
        if not .7 <= d[2] / area <= 1.3:
            reason = "ball_area_changed_possible_occlusion"
            break
        good.append(i)
    return np.asarray(good, dtype=int), centre, reason


def camera_parameters(env):
    cam = env.model.camera("iphone")
    R_mj = env.data.cam_xmat[cam.id].reshape(3, 3)
    R = np.diag([1, -1, -1]) @ R_mj.T
    pos = env.data.cam_xpos[cam.id] - env.offset
    rvec = cv2.Rodrigues(R)[0].reshape(3)
    tvec = -R @ pos
    camera = env.task["camera_iphone"]
    from real2sim.sim.cameras import camera_intrinsics_px

    K = camera_intrinsics_px(env.model, cam.id, camera["width"], camera["height"])
    return rvec, tvec, K, pos, R_mj


def pose_render(env, real_q, ball_xy):
    """Static render, deliberately no limit clamp or physics step; not a rollout."""
    q = sim_from_real(env.mapping, real_q)
    env.data.qpos[env.qadr] = q
    env.data.ctrl[env.actuator_ids] = q
    env.data.qvel[:] = 0
    env.data.qpos[env._ball_q:env._ball_q + 3] = [
        *(np.asarray(ball_xy) + env.offset[:2]), env.table_top + env.task["ball"]["radius"]]
    mujoco.mj_forward(env.model, env.data)
    return q


def audit_camera(env, out):
    """Reproject existing observations through the current camera; never refit/write config."""
    out.mkdir(parents=True, exist_ok=False)
    rvec, tvec, K, pos, axes = camera_parameters(env)
    prior = json.loads(Path("results/audit/camera_fit/report.json").read_text())
    rows = []
    for c in prior["correspondences"]:
        if c["kind"] == "held":
            pose_render(env, c["state_deg"], env.task["ball"]["start_region"]["center_task_xy"])
            xyz = env.ee_pose()[:3, 3]
        else:
            xy = (env.task["ball"]["start_region"]["center_task_xy"] if c["kind"] == "start"
                  else env.task["bowl"]["center_task_xy"])
            z = env.table_top - env.offset[2] + env.task["ball"]["radius"]
            if c["kind"] == "bowl":
                z += env.task["bowl"]["outer_height"] - env.task["bowl"]["inner_depth"]
            xyz = np.r_[xy, z]
        pixel = cv2.projectPoints(np.asarray(xyz, float).reshape(1, 3), rvec, tvec, K, None)[0].reshape(2)
        rows.append({**c, "point_base": xyz.tolist(), "projected_pixel_current": pixel.tolist(),
                     "error_px_current": float(np.linalg.norm(pixel - c["pixel"]))})
    report = {"method": "Fixed committed camera reprojection; existing observed pixels retained, no fitting",
              "opencv_pose": {"rvec": rvec.tolist(), "tvec": tvec.tolist()},
              "intrinsics_used": dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2])),
              "correspondences": rows,
              "rms_by_kind_px": {k: float(np.sqrt(np.mean([r["error_px_current"] ** 2 for r in rows if r["kind"] == k])))
                                 for k in ("start", "bowl", "held")},
              "note": "Held ball=tool site is an approximation; old detections do not establish current calibration accuracy."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return rvec, tvec, K, pos, axes


def prepare(config, out, audit_out, *, max_seconds=20.0, max_shift_px=3.0, contact_distance=.08,
            bowl_placements=None):
    if not np.isfinite(contact_distance) or contact_distance < 0:
        raise ValueError("contact distance must be finite and nonnegative")
    if out.exists() or audit_out.exists():
        raise FileExistsError("refusing to overwrite pair cache or camera audit")
    ds = DerivedTickDataset(".cache/derived_joint", list(range(40)), 20)
    placements = None
    if bowl_placements is not None:
        from real2sim.data.bowl_placement import validate_placements
        placements = validate_placements(json.loads(Path(bowl_placements).read_text()), ds.manifest)
    env = So101PickPlaceEnv(config, render_images=True)
    env.reset(seed=0)
    key = ds.camera_keys[0]
    report = {"kind": "STATIC_SIM_RENDERS_PAIRED_WITH_REAL_APPROACH", "episodes": [],
              "thresholds": {"max_seconds": max_seconds, "max_shift_policy_pixels": max_shift_px,
                             "min_ball_area_policy_pixels": 60, "area_ratio": [.7, 1.3],
                             "stop_before_tool_ball_distance_m": contact_distance},
              "timing": "Same derived tick state; recorded image lag is not silently compensated",
              "config_sha256": file_sha256(config),
              "task_config": env.task, "sim_config": env.config,
              "pose_note": "Recorded joints without clamping; forward kinematics only, no physics steps",
              "review_status": "pending_visual_review"}
    if placements is not None:
        report["bowl_placements"] = {"path": str(bowl_placements), "sha256": file_sha256(bowl_placements)}
    out.mkdir(parents=True)
    try:
        rvec, tvec, K, pos, axes = audit_camera(env, audit_out)
        all_pairs = {}
        for split in ("train", "val"):
            ticks_out, images_out, previews = [], [], []
            for ep in ds.manifest["split"][split]:
                ticks = np.flatnonzero(ds.arrays["episode_index"] == ep)[:int(max_seconds * 10)]
                dets = [ball_centroid_px(np.asarray(ds.images[key][t]), min_area=60) for t in ticks]
                prefix, centre, reason = stationary_prefix(dets, max_shift_px=max_shift_px)
                info = {"episode": ep, "split": split, "eligible_ticks": 0, "stop_reason": reason}
                if not len(prefix):
                    report["episodes"].append(info)
                    continue
                lb = ds.manifest["letterbox"]
                full_pixel = (centre - [lb["pad_left"], lb["pad_top"]]) / lb["scale"]
                z = env.table_top - env.offset[2] + env.task["ball"]["radius"]
                xyz = backproject_to_plane([full_pixel], rvec, tvec, K, z)[0]
                if not np.isfinite(xyz).all() or not (0.1 < xyz[0] < .6 and -.3 < xyz[1] < .4):
                    info["stop_reason"] = "backprojection_outside_work_area"
                    report["episodes"].append(info)
                    continue
                if placements is not None:
                    bowl_pixel = (np.asarray(placements[ep]["pixel_policy"]) - [lb["pad_left"], lb["pad_top"]]) / lb["scale"]
                    bowl_xy = backproject_to_plane([bowl_pixel], rvec, tvec, K,
                        env.table_top - env.offset[2] + env.task["bowl"]["outer_height"])[0, :2]
                    env.reset(seed=0, scenario={"ball_xy": xyz[:2], "bowl_xy": bowl_xy})
                    info["bowl_xy_estimated"] = bowl_xy.tolist()
                    # A kinematic render can otherwise conceal an impossible
                    # object setup, such as the ball intersecting the bowl wall.
                    bad = [c for c in env._contact_pairs() if "ball_geom" in (c["geom1"], c["geom2"])
                           and c["distance_m"] < -.002]
                    if bad:
                        info.update(stop_reason="invalid_initial_object_overlap", initial_contacts=bad)
                        report["episodes"].append(info)
                        continue
                ep_pairs = []
                outside = 0
                for local in prefix:
                    t = int(ticks[local])
                    q = pose_render(env, ds.arrays["state"][t], xyz[:2])
                    if np.linalg.norm(env.ee_pose()[:3, 3] - xyz) < contact_distance:
                        info["stop_reason"] = "near_contact_uncertain_ball_pose"
                        break
                    outside += int(np.any((q < env.limits[:, 0]) | (q > env.limits[:, 1])))
                    rgb = np.rint(env.observation()[key].transpose(1, 2, 0) * 255).astype(np.uint8)
                    ticks_out.append(t)
                    images_out.append(rgb)
                    ep_pairs.append((t, rgb))
                info.update(eligible_ticks=len(ep_pairs), ball_xy_estimated=xyz[:2].tolist(),
                            first_pixel_policy=centre.tolist(), outside_joint_limits_ticks=outside,
                            moving_ticks=int(sum(np.max(abs(ds.arrays["state"][t, :5] - ds.arrays["state"][ticks[0], :5])) > 5
                                                 for t, _ in ep_pairs)))
                if ep_pairs:
                    # First / halfway / final approach frame, annotated real and static sim.
                    for idx in sorted({0, len(ep_pairs) // 2, len(ep_pairs) - 1}):
                        t, sim = ep_pairs[idx]
                        real = np.asarray(ds.images[key][t]).copy()
                        for image, text in ((real, f"REAL ep{ep} f{ds.arrays['frame_index'][t]}"),
                                            (sim := sim.copy(), "STATIC SIM PAIR (no rollout)")):
                            cv2.putText(image, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
                        previews.append(np.concatenate([real, sim], axis=1))
                report["episodes"].append(info)
                print(f"{split} ep{ep}: {len(ep_pairs)} pairs; {info['stop_reason']}", flush=True)
            if not ticks_out:
                raise ValueError(f"no eligible {split} approach pairs")
            dest = out / split
            dest.mkdir()
            np.save(dest / "ticks.npy", np.asarray(ticks_out, dtype=np.int64))
            np.save(dest / "images.npy", np.asarray(images_out, dtype=np.uint8))
            manifest = {"split_name": split, "camera_key": key, "derived_manifest": ds.manifest,
                        "n_pairs": len(ticks_out), "sim_config_sha256": file_sha256(config),
                        "sha256": {n: file_sha256(dest / n) for n in ("ticks.npy", "images.npy")}}
            if placements is not None:
                manifest["bowl_placements_sha256"] = file_sha256(bowl_placements)
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            # Paginate previews so inspection remains legible (four episodes per page).
            for i in range(0, len(previews), 12):
                grid = np.concatenate(previews[i:i + 12], axis=0)
                cv2.imwrite(str(audit_out / f"pairs_{split}_{i // 12:02d}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            all_pairs[split] = len(ticks_out)
        report["counts"] = all_pairs
        if sum(r.get("moving_ticks", 0) >= 10 for r in report["episodes"] if r["split"] == "train") < 8:
            raise ValueError("insufficient moving approach coverage: need >=8 train episodes with >=10 moving ticks")
        # Backprojection audit uses current fixed camera; training never reads validation estimates.
        (audit_out / "backprojection.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("results/metrics/geometry_3988577/configs/sim_so101.yaml"))
    p.add_argument("--out", type=Path, default=Path(".cache/paired_approach_3988577"))
    p.add_argument("--audit-out", type=Path, default=Path("results/audit/paired_approach_3988577"))
    p.add_argument("--bowl-placements", type=Path)
    p.add_argument("--contact-distance", type=float, default=.08,
                   help="Conservative tool-ball cutoff in metres; 0 retains visible stationary-ball frames up to motion/occlusion")
    args = p.parse_args()
    prepare(args.config, args.out, args.audit_out, bowl_placements=args.bowl_placements,
            contact_distance=args.contact_distance)


if __name__ == "__main__":
    main()
