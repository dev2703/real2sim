"""Exploratory first-three joint zeros with the static-foot camera held fixed.

Only training horn-axis image residuals choose offsets. Native limits, contacts and
validation captures do not enter optimization. No physical calibration is adopted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from diag_v2_camera_joint_overlay import project
from diag_v2_joint_hypotheses import score
from scipy.optimize import least_squares

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping
from real2sim.sim.env import So101PickPlaceEnv

# Read native image coordinates before producing projections or fitting offsets.
CAPTURES = [
    {"episode": 0, "frame": 350, "split": "train", "horns": {
        "shoulder_lift": [338., 192.], "elbow_flex": [478., 33.], "wrist_flex": [703., 44.]}},
    {"episode": 6, "frame": 350, "split": "train", "horns": {
        "shoulder_lift": [280., 221.], "elbow_flex": [388., 99.], "wrist_flex": [642., 199.]}},
    {"episode": 23, "frame": 0, "split": "train", "horns": {
        "shoulder_lift": [316., 209.], "elbow_flex": [112., 213.], "wrist_flex": [335., 141.]}},
    {"episode": 23, "frame": 400, "split": "train", "horns": {
        "shoulder_lift": [273., 222.]}},
    {"episode": 31, "frame": 400, "split": "val", "horns": {
        "shoulder_lift": [298., 224.], "elbow_flex": [397., 98.], "wrist_flex": [653., 190.]}},
    {"episode": 36, "frame": 400, "split": "val", "horns": {
        "shoulder_lift": [292., 221.], "elbow_flex": [385., 93.], "wrist_flex": [625., 210.]}},
]


class ConditionalFit:
    def __init__(self, arrays, task, camera):
        self.mapping = JointMapping.from_task_config(task)
        self.model = mujoco.MjModel.from_xml_path("assets/menagerie/robotstudio_so101/so101.xml")
        self.data = mujoco.MjData(self.model)
        self.qadr = self.model.jnt_qposadr[[self.model.joint(n).id for n in (*ARM_JOINTS, "gripper")]]
        self.k = np.asarray(camera["base_foot"]["candidate_K_early"])
        self.transform = np.asarray(camera["base_foot"]["candidate_T_optical_base_early"])
        self.late = np.asarray(camera["image_similarity_models"]["without_fixed_base_foot"][
            "matrix_reference_to_late"])
        self.captures = []
        for capture in CAPTURES:
            index = np.flatnonzero((arrays["episode_index"] == capture["episode"]) &
                                   (arrays["frame_index"] == capture["frame"]))
            if len(index) != 1:
                raise ValueError("source state not unique")
            self.captures.append({**capture, "real": np.asarray(arrays["joint_state"][index[0]], float)})

    def rows(self, offsets):
        out = []
        for capture in self.captures:
            real = capture["real"]
            arm = self.mapping.arm_real_to_sim_rad(real) + np.deg2rad(np.r_[offsets, 0., 0.])
            self.data.qpos[self.qadr] = np.r_[arm, self.mapping.gripper_real_to_sim_rad(real[5])]
            mujoco.mj_kinematics(self.model, self.data)
            affine = self.late if capture["episode"] >= 7 else None
            for name, observed in capture["horns"].items():
                jid = self.model.joint(name).id
                anchor, axis = self.data.xanchor[jid], self.data.xaxis[jid]
                segment = project(np.array([anchor - .04 * axis, anchor + .04 * axis]),
                                  self.k, self.transform, affine)
                direction = segment[1] - segment[0]
                normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                residual = float(np.dot(np.asarray(observed) - segment[0], normal))
                fraction = float(np.dot(np.asarray(observed) - segment[0], direction) /
                                 np.dot(direction, direction))
                out.append({"episode": capture["episode"], "frame": capture["frame"],
                            "split": capture["split"], "joint": name, "observed_px": observed,
                            "residual_px": residual, "projected_axis_px": segment.tolist(),
                            "closest_projected_axis_fraction_minus40_to_plus40mm": fraction})
        return out

    def residuals(self, offsets):
        return np.array([r["residual_px"] / 8 for r in self.rows(offsets) if r["split"] == "train"])

    def point_rows(self, parameters):
        offsets, axial_mm = parameters[:3], parameters[3:]
        out = []
        for capture in self.captures:
            real = capture["real"]
            arm = self.mapping.arm_real_to_sim_rad(real) + np.deg2rad(np.r_[offsets, 0., 0.])
            self.data.qpos[self.qadr] = np.r_[arm, self.mapping.gripper_real_to_sim_rad(real[5])]
            mujoco.mj_kinematics(self.model, self.data)
            affine = self.late if capture["episode"] >= 7 else None
            for name, observed in capture["horns"].items():
                jid = self.model.joint(name).id
                which = ("shoulder_lift", "elbow_flex", "wrist_flex").index(name)
                point = self.data.xanchor[jid] + self.data.xaxis[jid] * axial_mm[which] / 1000
                predicted = project(point[None], self.k, self.transform, affine)[0]
                out.append({"episode": capture["episode"], "frame": capture["frame"],
                            "split": capture["split"], "joint": name, "observed_px": observed,
                            "predicted_px": predicted.tolist(),
                            "error_px": float(np.linalg.norm(predicted - observed)),
                            "residual_px": (predicted - observed).tolist()})
        return out

    def point_residuals(self, parameters):
        return np.array([r["residual_px"] for r in self.point_rows(parameters)
                         if r["split"] == "train"]).ravel() / 8


def summarize(rows):
    return {part: {"landmarks": len(values), "rms_axis_distance_px": float(np.sqrt(np.mean(values**2))),
                   "median_axis_distance_px": float(np.median(np.abs(values))),
                   "max_axis_distance_px": float(np.max(np.abs(values)))}
            for part in ("train", "val")
            if len(values := np.array([r["residual_px"] for r in rows if r["split"] == part]))}


def point_summary(rows):
    return {part: {"landmarks": len(values), "rms_point_error_px": float(np.sqrt(np.mean(values**2))),
                   "median_point_error_px": float(np.median(values)), "max_point_error_px": float(values.max())}
            for part in ("train", "val")
            if len(values := np.array([r["error_px"] for r in rows if r["split"] == part]))}


def identifiability(function, parameters):
    n = len(parameters)
    jacobian = np.column_stack([(function(parameters + np.eye(n)[j] * .001) -
                                 function(parameters - np.eye(n)[j] * .001)) / .002 for j in range(n)])
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    sigma = np.sqrt(np.diag(covariance))
    return {"rank": int(np.linalg.matrix_rank(jacobian, tol=singular[0] * 1e-7)),
            "parameters": n, "scalar_observations": len(function(parameters)),
            "singular_values_after_8px_weighting": singular.tolist(),
            "weakest_direction": vt[-1].tolist(), "condition_number": float(singular[0] / singular[-1]),
            "conditional_one_sigma_known_8px_independent_noise": sigma.tolist(),
            "parameter_correlation": (covariance / np.outer(sigma, sigma)).tolist(),
            "camera_base_CAD_uncertainty_included": False}


def overlay(fit, offsets, out):
    baseline, candidate = fit.rows(np.zeros(3)), fit.rows(offsets)
    paths = []
    for ep, frame in ((23, 0), (31, 400), (36, 400)):
        photo = cv2.imread(f"outputs/v2/geometry_alignment_20261021/ep{ep:02d}_f{frame:04d}.jpg")
        panels = []
        for title, rows in (("fixed camera; zero offsets", baseline), ("fixed camera; fitted offsets", candidate)):
            image = photo.copy()
            for row in rows:
                if row["episode"] != ep or row["frame"] != frame:
                    continue
                axis = np.asarray(row["projected_axis_px"]).astype(int)
                point = np.asarray(row["observed_px"]).astype(int)
                cv2.line(image, tuple(axis[0]), tuple(axis[1]), (255, 80, 40), 2, cv2.LINE_AA)
                cv2.drawMarker(image, tuple(point), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(image, f"{row['joint']} {abs(row['residual_px']):.1f}px", tuple(point + [8, 12]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4)
            cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1)
            panels.append(cv2.resize(image, (640, 360)))
        path = out / f"ep{ep:02d}_f{frame:04d}.jpg"
        cv2.imwrite(str(path), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data_path = Path(".cache/v2_exploratory_20261022/cartesian_iphone_top/arrays.npz")
    camera_path = Path("outputs/v2/static_base_camera_20261023/report.json")
    task_path = Path("configs/v2/task.exploratory_20261022.yaml")
    split_path = Path("results/v2/split.json")
    arrays, task = np.load(data_path), load_yaml(task_path)
    splits = json.loads(split_path.read_text())
    for capture in CAPTURES:
        if capture["episode"] not in splits[capture["split"]]:
            raise ValueError("capture split mismatch; reserved episodes must not enter diagnostic")
    fit = ConditionalFit(arrays, task, json.loads(camera_path.read_text()))
    starts = [np.zeros(3), np.full(3, 5.), np.full(3, -5.),
              np.array([5., -5., 5.]), np.array([-5., 5., -5.])]
    solutions = [least_squares(fit.residuals, start, bounds=(-10, 10), loss="soft_l1",
                               f_scale=1., diff_step=1e-4, max_nfev=2000) for start in starts]
    best = min(solutions, key=lambda s: s.cost)
    offsets = best.x
    jacobian = np.column_stack([(fit.residuals(offsets + np.eye(3)[j] * .001) -
                                 fit.residuals(offsets - np.eye(3)[j] * .001)) / .002 for j in range(3)])
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.linalg.matrix_rank(jacobian, tol=singular[0] * 1e-7))
    baseline, candidate = fit.rows(np.zeros(3)), fit.rows(offsets)
    # Predeclared sensitivity: widen the *diagnostic pan-offset* bound only. Native
    # joint limits remain fixed. No held-out scores or contacts select any fit.
    broad_solutions = [least_squares(fit.residuals, start,
                       bounds=([-20., -10., -10.], [20., 10., 10.]), loss="soft_l1",
                       f_scale=1., diff_step=1e-4, max_nfev=2000) for start in starts]
    broad_best = min(broad_solutions, key=lambda s: s.cost)
    broad_rows = fit.rows(broad_best.x)
    # Actual 3D point model: one unknown horn position along each hinge axis,
    # shared across ALL captures. Baseline fits these nuisance positions only.
    nuisance_fits = [least_squares(lambda axial: fit.point_residuals(np.r_[np.zeros(3), axial]),
                                  np.full(3, start), bounds=(-40., 40.), loss="soft_l1",
                                  f_scale=1., max_nfev=1000) for start in (0., 20., -20.)]
    nuisance = min(nuisance_fits, key=lambda s: s.cost)
    point_starts = [np.r_[start, nuisance.x] for start in (np.zeros(3), offsets, broad_best.x)]
    point_fits = [least_squares(fit.point_residuals, start,
                 bounds=([-20., -10., -10., -40., -40., -40.], [20., 10., 10., 40., 40., 40.]),
                 loss="soft_l1", f_scale=1., x_scale=[5., 5., 5., 10., 10., 10.],
                 max_nfev=2000) for start in point_starts]
    point_best = min(point_fits, key=lambda s: s.cost)
    point_baseline = fit.point_rows(np.r_[np.zeros(3), nuisance.x])
    point_candidate = fit.point_rows(point_best.x)
    args.out.mkdir(parents=True)
    report = {
        "schema": "v2_fixed_camera_conditional_joint_zeros_v1",
        "physical_calibration_accepted": False, "production_calibration_changed": False,
        "objective": "Training-only signed perpendicular distances from manually read horn centres"
                     " to projected hinge axes, divided by 8px; robust soft-L1. Fixed camera/base/signs.",
        "fitted_joint_names": list(ARM_JOINTS[:3]), "offsets_deg": offsets.tolist(),
        "fixed_wrist_flex_roll_offsets_deg": [0., 0.], "offset_bounds_deg": [-10., 10.],
        "active_parameter_bounds": best.active_mask.tolist(),
        "train_solution_costs": [float(s.cost) for s in solutions],
        "train_solution_offsets_deg": [s.x.tolist() for s in solutions],
        "baseline": {"summary": summarize(baseline), "rows": baseline},
        "candidate": {"summary": summarize(candidate), "rows": candidate},
        "pan_bound_sensitivity": {
            "fixed_before_heldout_scoring": True,
            "offset_bounds_deg": [[-20., 20.], [-10., 10.], [-10., 10.]],
            "offsets_deg": broad_best.x.tolist(), "active_parameter_bounds": broad_best.active_mask.tolist(),
            "train_solution_costs": [float(s.cost) for s in broad_solutions],
            "summary": summarize(broad_rows), "rows": broad_rows,
            "identifiability": identifiability(fit.residuals, broad_best.x)},
        "shared_3D_horn_point_model": {
            "parameter_names": [*ARM_JOINTS[:3], "shoulder_lift_horn_axial", "elbow_flex_horn_axial", "wrist_flex_horn_axial"],
            "parameter_units": ["degrees"] * 3 + ["mm"] * 3,
            "parameter_bounds": [[-20., 20.], [-10., 10.], [-10., 10.]] + [[-40., 40.]] * 3,
            "zero_offset_baseline_horn_axial_mm": nuisance.x.tolist(),
            "zero_offset_baseline_summary": point_summary(point_baseline),
            "candidate_parameters": point_best.x.tolist(),
            "active_parameter_bounds": point_best.active_mask.tolist(),
            "train_solution_costs": [float(s.cost) for s in point_fits],
            "candidate_summary": point_summary(point_candidate),
            "candidate_rows": point_candidate, "zero_offset_baseline_rows": point_baseline,
            "identifiability": identifiability(fit.point_residuals, point_best.x),
            "interpretation": "One rigid 3D horn point per hinge shared across all poses; camera, base, signs and remaining joints fixed."
                              " +/-40mm axial extent is a diagnostic bound, not measured horn geometry."},
        "local_identifiability": {
            "rank": rank, "parameters": 3, "scalar_observations": len(fit.residuals(offsets)),
            "singular_values_per_degree_after_8px_weighting": singular.tolist(),
            "weakest_direction_pan_lift_elbow": vt[-1].tolist(),
            "condition_number": float(singular[0] / singular[-1]),
            "conditional_one_sigma_deg_known_8px_independent_noise": np.sqrt(
                np.diag(np.linalg.pinv(jacobian.T @ jacobian))).tolist()},
        "sources": [{"episode": c["episode"], "frame": c["frame"], "split": c["split"],
                     "real_state_deg_percent": c["real"].tolist(), "horn_centres_px": c["horns"],
                     "image_sha256": sha256_file(Path(
                         f"outputs/v2/geometry_alignment_20261021/ep{c['episode']:02d}_f{c['frame']:04d}.jpg"))}
                    for c in fit.captures],
        "input_sha256": {str(p): sha256_file(p) for p in (data_path, camera_path, task_path, split_path,
            Path("assets/menagerie/robotstudio_so101/so101.xml"), Path(__file__),
            Path(__file__).with_name("diag_v2_camera_joint_overlay.py"),
            Path(__file__).with_name("diag_v2_joint_hypotheses.py"))},
        "limitations": [
            "Conditional on exploratory static-foot/mat camera and repeated-hole CAD registration, not independent physical calibration.",
            "Manual nominal horn identities and +/-8px uncertainty are approximate; all captures are exploratory.",
            "Validation capture31:400 was viewed in the preceding camera diagnostic. It does not provide untouched acceptance.",
            "Signed 2D hinge-axis distances do not identify horn depth along the joint axis or prove full mesh correspondence.",
            "Wrist flex/roll zeros are fixed because these landmark axes do not constrain their own downstream rotations.",
            "Local covariance assumes fixed, exact camera/base/CAD and independent 8px errors. Real systematic uncertainty is larger.",
            "Image/state timing, backlash, physical support height and camera distortion remain unverified.",
            "Axis-only candidates ignore shared horn depth along each hinge. They are NOT joint-zero calibration; the separate shared-point model tests this constraint.",
            "Covariance at an active bound is an unconstrained local approximation, not a calibrated confidence interval.",
            "No contacts, native-bound counts, held-out image errors, or task-success feedback enter fitting or start selection.",
        ],
    }
    report["overlays"] = overlay(fit, offsets, args.out)
    with So101PickPlaceEnv("configs/sim_so101.yaml", render_images=False) as env:
        env.model.body_pos[env.model.body("base").id, 2] += .0024 - env.task["robot_so101"]["base_z_offset"]
        report["post_fit_static_diagnostics"] = {}
        for label, candidate_offsets in (("zero_offsets", np.zeros(5)),
                                          ("fitted_offsets", np.r_[offsets, 0., 0.]),
                                          ("pan_bound_20_offsets", np.r_[broad_best.x, 0., 0.]),
                                          ("shared_point_offsets", np.r_[point_best.x[:3], 0., 0.])):
            report["post_fit_static_diagnostics"][label] = {}
            for part in ("train", "val"):
                mask = np.isin(arrays["episode_index"], splits[part])
                report["post_fit_static_diagnostics"][label][part] = score(
                    env, arrays["joint_state"][mask], arrays["episode_index"][mask],
                    arrays["frame_index"][mask], np.ones(5), candidate_offsets)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"offsets_deg": offsets.tolist(), "baseline": summarize(baseline),
                      "candidate": summarize(candidate), "rank": rank, "singular": singular.tolist(),
                      "pan_bound20_offsets_deg": broad_best.x.tolist(),
                      "shared_point_parameters_deg_mm": point_best.x.tolist(),
                      "shared_point_baseline": point_summary(point_baseline),
                      "shared_point_candidate": point_summary(point_candidate)}, indent=2))


if __name__ == "__main__":
    main()
