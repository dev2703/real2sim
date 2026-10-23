"""Conditional wrist-zero image fit with explicit downstream landmark ambiguity.

This is an offline diagnostic. It does not write task/camera/mapping configuration,
clip recorded positions, optimize contacts, or use reserved episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from diag_v2_camera_joint_overlay import project
from diag_v2_conditional_joint_zeros import ConditionalFit
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim.config import load_yaml
from real2sim.data.raw import sha256_file
from real2sim.geometry.so101_fk import ARM_JOINTS, JointMapping

# Native 1280x720 pixels read from source images before generating any downstream
# CAD projections. Horn identities and the lens's partly occluded front-cap centre
# remain manual nominal-CAD hypotheses, with approximate independent 8px noise.
CAPTURES = [
    {"episode": 0, "frame": 200, "split": "train", "lens": [696., 77.]},
    {"episode": 0, "frame": 350, "split": "train", "lens": [812., 135.]},
    {"episode": 6, "frame": 150, "split": "train", "lens": [592., 86.], "horn": [528., 86.]},
    {"episode": 6, "frame": 350, "split": "train", "lens": [862., 338.], "horn": [774., 331.]},
    {"episode": 23, "frame": 0, "split": "train", "lens": [530., 236.], "horn": [461., 231.]},
    {"episode": 23, "frame": 100, "split": "train", "lens": [559., 165.], "horn": [460., 186.]},
    {"episode": 23, "frame": 350, "split": "train", "lens": [739., 314.], "horn": [658., 330.]},
    {"episode": 31, "frame": 250, "split": "val", "lens": [763., 137.], "horn": [704., 150.]},
    {"episode": 31, "frame": 400, "split": "val", "lens": [893., 254.], "horn": [800., 258.]},
    {"episode": 36, "frame": 250, "split": "val", "lens": [771., 109.], "horn": [714., 125.]},
    {"episode": 36, "frame": 400, "split": "val", "lens": [844., 327.], "horn": [758., 324.]},
]


def summary(rows):
    result = {}
    for split in ("train", "val"):
        result[split] = {}
        for landmark in ("all", "lens", "horn"):
            values = np.array([r["error_px"] for r in rows if r["split"] == split
                               and (landmark == "all" or r["landmark"] == landmark)])
            if len(values):
                result[split][landmark] = {"landmarks": len(values),
                    "rms_px": float(np.sqrt(np.mean(values**2))),
                    "median_px": float(np.median(values)), "max_px": float(values.max())}
    return result


def numerical_jacobian(function, x, step=1e-3):
    return np.column_stack([(function(x + np.eye(len(x))[j] * step) -
                             function(x - np.eye(len(x))[j] * step)) / (2 * step)
                            for j in range(len(x))])


def identifiability(function, x):
    jac = numerical_jacobian(function, np.asarray(x, float))
    _, singular, vt = np.linalg.svd(jac, full_matrices=True)
    rank = int(np.linalg.matrix_rank(jac, tol=singular[0] * 1e-7))
    covariance = np.linalg.pinv(jac.T @ jac)
    return {"rank": rank, "parameters": len(x), "scalar_observations": len(function(x)),
        "singular_values_after_8px_weighting": singular.tolist(),
        "weakest_direction": vt[-1].tolist(),
        "conditional_one_sigma": np.sqrt(np.diag(covariance)).tolist() if rank == len(x) else None,
        "systematic_camera_base_CAD_uncertainty_included": False}


class WristFit:
    def __init__(self, arrays, task, camera, first_three, captures=CAPTURES):
        self.mapping = JointMapping.from_task_config(task)
        self.model = mujoco.MjModel.from_xml_path("assets/menagerie/robotstudio_so101/so101.xml")
        self.data = mujoco.MjData(self.model)
        self.qadr = self.model.jnt_qposadr[[self.model.joint(n).id for n in (*ARM_JOINTS, "gripper")]]
        self.k = np.asarray(camera["base_foot"]["candidate_K_early"], float)
        self.transform = np.asarray(camera["base_foot"]["candidate_T_optical_base_early"], float)
        self.late = np.asarray(camera["image_similarity_models"]["without_fixed_base_foot"][
            "matrix_reference_to_late"], float)
        self.first_three = np.asarray(first_three, float)
        self.gripper_body = self.model.body("gripper").id
        self.lens_geom = self.model.geom("wrist_camera_lens").id
        self.gripper_joint = self.model.joint("gripper").id
        self.captures = []
        for capture in captures:
            indices = np.flatnonzero((arrays["episode_index"] == capture["episode"]) &
                                    (arrays["frame_index"] == capture["frame"]))
            if len(indices) != 1:
                raise ValueError("capture must have exactly one unclipped recorded joint state")
            self.captures.append({**capture, "real": np.asarray(arrays["joint_state"][indices[0]], float)})

    def rows(self, parameters, *, first_three=None, lens_offset_mm=None, horn_axial_mm=0.):
        first = self.first_three if first_three is None else np.asarray(first_three, float)
        lens_offset = np.zeros(3) if lens_offset_mm is None else np.asarray(lens_offset_mm) / 1000
        rows = []
        for capture in self.captures:
            real = capture["real"]
            arm = self.mapping.arm_real_to_sim_rad(real) + np.deg2rad(np.r_[first, parameters[:2]])
            self.data.qpos[self.qadr] = np.r_[arm, self.mapping.gripper_real_to_sim_rad(real[5])]
            mujoco.mj_kinematics(self.model, self.data)
            # The visible optical front is the negative local cylinder-Z cap.
            lens_rotation = self.data.geom_xmat[self.lens_geom].reshape(3, 3)
            lens = self.data.geom_xpos[self.lens_geom] - lens_rotation[:, 2] * self.model.geom_size[self.lens_geom, 1]
            lens += self.data.xmat[self.gripper_body].reshape(3, 3) @ lens_offset
            jid = self.gripper_joint
            horn = self.data.xanchor[jid] + self.data.xaxis[jid] * horn_axial_mm / 1000
            affine = self.late if capture["episode"] >= 7 else None
            for name, point in (("lens", lens), ("horn", horn)):
                if name not in capture:
                    continue
                predicted = project(point[None], self.k, self.transform, affine)[0]
                residual = predicted - capture[name]
                rows.append({"episode": capture["episode"], "frame": capture["frame"],
                    "split": capture["split"], "landmark": name, "observed_px": capture[name],
                    "predicted_px": predicted.tolist(), "residual_px": residual.tolist(),
                    "error_px": float(np.linalg.norm(residual)), "depth_m": float(
                        (self.transform[:3, :3] @ point + self.transform[:3, 3])[2])})
        return rows

    def residuals(self, parameters, *, first_three=None, lens_offset_mm=None,
                  horn_axial_mm=0., landmarks=("lens", "horn")):
        return np.array([r["residual_px"] for r in self.rows(parameters, first_three=first_three,
            lens_offset_mm=lens_offset_mm, horn_axial_mm=horn_axial_mm)
            if r["split"] == "train" and r["landmark"] in landmarks]).ravel() / 8


def solve(function, x0, bounds):
    solution = least_squares(function, np.asarray(x0, float), bounds=bounds, loss="soft_l1",
                             f_scale=1., diff_step=1e-4, max_nfev=1000)
    if not solution.success or not np.all(np.isfinite(solution.fun)):
        raise ValueError("conditional wrist fit did not converge to finite residuals")
    return solution


def native_limit_check(fit, real_states, offsets_deg):
    """Inspect raw arm coordinates; never clip, rescale, or fit to these counts."""
    arm = fit.mapping.arm_real_to_sim_rad(real_states) + np.deg2rad(offsets_deg)
    limits = np.array([fit.model.jnt_range[fit.model.joint(j).id] for j in ARM_JOINTS])
    violations = (arm < limits[:, 0]) | (arm > limits[:, 1])
    return {"ticks": len(arm), "ticks_with_any_violation": int(violations.any(1).sum()),
        "per_joint_violating_ticks": dict(zip(ARM_JOINTS, violations.sum(0).tolist(), strict=True)),
        "mapped_range_deg": np.rad2deg(np.c_[arm.min(0), arm.max(0)]).tolist(),
        "native_limits_deg": np.rad2deg(limits).tolist(), "used_in_fitting": False}


def save_overlay(rows, out):
    panels = []
    for ep, frame in ((23, 100), (31, 400)):
        source = Path(f"outputs/v2/geometry_alignment_20261021/ep{ep:02d}_f{frame:04d}.jpg")
        image = cv2.imread(str(source))
        for label, values in rows.items():
            for row in values:
                if (row["episode"], row["frame"]) != (ep, frame):
                    continue
                p = tuple(np.rint(row["predicted_px"]).astype(int))
                q = tuple(np.rint(row["observed_px"]).astype(int))
                cv2.drawMarker(image, q, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.circle(image, p, 5, (0, 120, 255) if label == "baseline" else (255, 40, 0), 2)
        cv2.putText(image, "green observed | orange baseline | blue wrist fit", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3)
        cv2.putText(image, "green observed | orange baseline | blue wrist fit", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        panels.append(cv2.resize(image, (640, 360)))
    path = out / "wrist_landmarks.jpg"
    cv2.imwrite(str(path), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    arrays_path = Path(".cache/v2_exploratory_20261022/cartesian_iphone_top/arrays.npz")
    task_path = Path("configs/v2/task.exploratory_20261022.yaml")
    camera_path = Path("outputs/v2/static_base_camera_20261023/report.json")
    prior_path = Path("outputs/v2/conditional_joint_zeros_20261023/report.json")
    split_path = Path("results/v2/split.json")
    splits = json.loads(split_path.read_text())
    for capture in CAPTURES:
        if capture["episode"] not in splits[capture["split"]]:
            raise ValueError("capture split mismatch: reserved episodes cannot enter diagnostic")
    arrays = np.load(arrays_path)
    task, camera = load_yaml(task_path), json.loads(camera_path.read_text())
    first = np.array(json.loads(prior_path.read_text())["shared_3D_horn_point_model"]["candidate_parameters"][:3])
    fit = WristFit(arrays, task, camera, first)
    # Model family and diagnostic bounds are fixed before any validation scoring.
    baseline_rows = fit.rows(np.zeros(2))
    fixed_results = [solve(fit.residuals, start, (-30., 30.))
                     for start in ([0., 0.], [10., -10.], [-10., 10.])]
    best = min(fixed_results, key=lambda s: s.cost)
    candidate_rows = fit.rows(best.x)
    def horn_fun(x):
        return fit.residuals(x[:2], horn_axial_mm=x[2])

    horn_best = solve(horn_fun, np.r_[best.x, 0.], ([-30., -30., -30.], [30., 30., 30.]))
    horn_rows = fit.rows(horn_best.x[:2], horn_axial_mm=horn_best.x[2])
    # A boundary optimum is a rejected small-offset model, not a measurement.
    # Explicitly test the entire roll cycle with multistart, chosen by train only.
    def broad_fun(x):
        return fit.residuals(x[:2], horn_axial_mm=x[2])

    broad_results = [solve(broad_fun, [0., roll, 10.],
        ([-45., -180., -30.], [45., 180., 30.])) for roll in (-150., -90., -30., 30., 90., 150.)]
    broad_best = min(broad_results, key=lambda s: s.cost)
    broad_rows = fit.rows(broad_best.x[:2], horn_axial_mm=broad_best.x[2])
    # Known-CAD lens alone versus a freely relocated lens. Roll zero and an
    # arbitrary rigid point's azimuth have an exact algebraic gauge freedom.
    def lens_fun(x):
        return fit.residuals(x, landmarks=("lens",))

    lens_best = solve(lens_fun, best.x, (-30., 30.))
    def free_fun(x):
        return fit.residuals(x[:2], lens_offset_mm=x[2:], landmarks=("lens",))

    free_best = solve(free_fun, np.r_[lens_best.x, np.zeros(3)],
                      ([-30., -30., -20., -20., -20.], [30., 30., 20., 20., 20.]))
    # First-three uncertainty is an explicit sensitivity, not a camera/zero fit.
    sensitivities = []
    for j in range(3):
        for sign in (-1, 1):
            perturbed = first.copy()
            perturbed[j] += 3 * sign
            sol = solve(lambda x, first=perturbed: fit.residuals(x, first_three=first),
                        best.x, (-30., 30.))
            sensitivities.append({"first_three_deg": perturbed.tolist(), "wrist_deg": sol.x.tolist(),
                "train_cost": float(sol.cost), "summary": summary(fit.rows(sol.x, first_three=perturbed))})
    # Pose perturbations are illustrative systematic tests, not a confidence interval.
    nominal_transform = fit.transform.copy()
    camera_sensitivities = []
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.zeros(3)
            delta[axis] = np.deg2rad(sign)
            r = Rotation.from_rotvec(delta).as_matrix()
            fit.transform = nominal_transform.copy()
            fit.transform[:3, :] = r @ nominal_transform[:3, :]
            sol = solve(fit.residuals, best.x, (-30., 30.))
            camera_sensitivities.append({"optical_axis_rotvec_deg": np.rad2deg(delta).tolist(),
                "wrist_deg": sol.x.tolist(), "train_cost": float(sol.cost),
                "summary": summary(fit.rows(sol.x))})
    fit.transform = nominal_transform
    # Repeat systematic tests away from the rejected small-offset roll boundary.
    # The wider diagnostic bound does not change any model/native joint limits.
    broad_sensitivities = []
    broad_bounds = ([-45., -180., -30.], [45., 180., 30.])
    for kind in ("first_three_deg", "optical_rotation_deg"):
        for axis in range(3):
            for sign in (-1, 1):
                first_perturbed = first.copy()
                delta = np.zeros(3)
                delta[axis] = sign * (3 if kind == "first_three_deg" else 1)
                if kind == "first_three_deg":
                    first_perturbed += delta
                else:
                    fit.transform = nominal_transform.copy()
                    fit.transform[:3] = (Rotation.from_rotvec(np.deg2rad(delta)).as_matrix()
                                         @ nominal_transform[:3])
                sol = solve(lambda x, q=first_perturbed: fit.residuals(
                    x[:2], first_three=q, horn_axial_mm=x[2]), broad_best.x, broad_bounds)
                broad_sensitivities.append({"kind": kind, "delta_deg": delta.tolist(),
                    "parameters_flex_roll_deg_axial_mm": sol.x.tolist(),
                    "train_cost": float(sol.cost), "summary": summary(fit.rows(sol.x[:2],
                        first_three=first_perturbed, horn_axial_mm=sol.x[2]))})
                fit.transform = nominal_transform
    # Conditional joint fit including all five offsets, while the earlier fixed
    # physical horn coordinates remain nuisance parameters learned on train only.
    proximal = ConditionalFit(arrays, task, camera)
    previous = np.array(json.loads(prior_path.read_text())["shared_3D_horn_point_model"]["candidate_parameters"])
    def all_residuals(x):
        return np.r_[proximal.point_residuals(np.r_[x[:3], x[5:8]]),
            fit.residuals(x[3:5], first_three=x[:3], horn_axial_mm=x[8])]
    all_best = solve(all_residuals, np.r_[first, best.x, previous[3:], 0.],
        ([-20., -15., -15., -30., -30., -40., -40., -40., -30.],
         [20., 15., 15., 30., 30., 40., 40., 40., 30.]))
    all_rows = fit.rows(all_best.x[3:5], first_three=all_best.x[:3], horn_axial_mm=all_best.x[8])
    args.out.mkdir(parents=True)
    report = {"schema": "v2_conditional_downstream_wrist_zeros_v1",
        "physical_calibration_accepted": False, "production_calibration_changed": False,
        "reserved_episodes_used": False, "contacts_limits_or_task_success_used_for_fit": False,
        "input_sha256": {str(p): sha256_file(p) for p in (arrays_path, task_path, camera_path,
            prior_path, split_path, Path(__file__), Path("assets/menagerie/robotstudio_so101/so101.xml"))},
        "annotation_protocol": "Manual native source pixels read before downstream CAD projection;"
            " approximately 8px per-coordinate uncertainty; lens partly occluded and horn identity tentative.",
        "first_three_conditional_deg": first.tolist(), "zero_wrist_baseline": summary(baseline_rows),
        "known_CAD_lens_and_horn": {"offsets_wrist_flex_roll_deg": best.x.tolist(),
            "diagnostic_bounds_deg": [-30., 30.], "active_parameter_bounds": best.active_mask.tolist(),
            "start_costs": [float(s.cost) for s in fixed_results], "summary": summary(candidate_rows),
            "rows": candidate_rows, "identifiability": identifiability(fit.residuals, best.x)},
        "shared_gripper_horn_axial_nuisance": {"parameters_flex_roll_deg_axial_mm": horn_best.x.tolist(),
            "summary": summary(horn_rows), "identifiability": identifiability(horn_fun, horn_best.x)},
        "whole_roll_cycle_diagnostic": {"parameters_flex_roll_deg_axial_mm": broad_best.x.tolist(),
            "diagnostic_bounds": [[-45., 45.], [-180., 180.], [-30., 30.]],
            "native_joint_limits_changed": False,
            "active_parameter_bounds": broad_best.active_mask.tolist(),
            "start_costs": [float(s.cost) for s in broad_results],
            "start_parameters": [s.x.tolist() for s in broad_results],
            "summary": summary(broad_rows), "rows": broad_rows,
            "identifiability": identifiability(broad_fun, broad_best.x),
            "native_limit_check": {split: native_limit_check(fit, arrays["joint_state"][
                np.isin(arrays["episode_index"], splits[split])], np.r_[first, broad_best.x[:2]])
                for split in ("train", "val")}},
        "known_CAD_lens_only": {"offsets_deg": lens_best.x.tolist(),
            "summary": summary(fit.rows(lens_best.x)), "identifiability": identifiability(lens_fun, lens_best.x)},
        "free_lens_mount_point": {"parameters_flex_roll_deg_xyz_mm": free_best.x.tolist(),
            "summary": summary(fit.rows(free_best.x[:2], lens_offset_mm=free_best.x[2:])),
            "identifiability": identifiability(free_fun, free_best.x),
            "interpretation": "Roll zero and an arbitrary downstream point's azimuth are gauge-equivalent;"
                " a fixed verified CAD point/shape or independent roll reference is necessary."},
        "all_five_offsets_and_shared_horn_coordinates": {"parameters": all_best.x.tolist(),
            "parameter_names": [*ARM_JOINTS, "lift_horn_axial_mm", "elbow_horn_axial_mm",
                "wrist_horn_axial_mm", "gripper_horn_axial_mm"],
            "summary": summary(all_rows), "identifiability": identifiability(all_residuals, all_best.x)},
        "first_three_plus_minus_3deg_sensitivity": sensitivities,
        "illustrative_camera_optical_plus_minus_1deg_sensitivity": camera_sensitivities,
        "whole_roll_cycle_systematic_sensitivity": broad_sensitivities,
        "baseline_rows": baseline_rows,
        "sources": [{"episode": c["episode"], "frame": c["frame"], "split": c["split"],
            "real_state_deg_percent": c["real"].tolist(),
            "image_sha256": sha256_file(Path(f"outputs/v2/geometry_alignment_20261021/ep{c['episode']:02d}_f{c['frame']:04d}.jpg"))}
            for c in fit.captures],
        "limitations": ["Exploratory verification on separate validation captures, not untouched acceptance.",
            "Static-foot camera and physical CAD identities remain unverified systematic assumptions.",
            "Horn point and camera lens front centre are manual/partly occluded, not measured fiducials.",
            "Gripper opening does not rotate the lens or gripper hinge axis; these cannot identify its zero.",
            "Known-CAD lens is nominal manufacturer geometry, not a measured camera mount on this robot.",
            "Small-offset models hit the roll bound; their local covariance is not a bounded confidence interval.",
            "Finite-difference local covariance does not include camera/base/CAD/timing systematics.",
            "Sensitivity bounds are diagnostic hypotheses and do not widen native joint limits."]}
    report["overlay"] = save_overlay({"baseline": baseline_rows, "candidate": broad_rows}, args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("zero_wrist_baseline", "known_CAD_lens_and_horn",
        "shared_gripper_horn_axial_nuisance", "free_lens_mount_point",
        "all_five_offsets_and_shared_horn_coordinates")}, indent=2))


if __name__ == "__main__":
    main()
