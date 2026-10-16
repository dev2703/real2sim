"""Recorded-control ablation of independently image-fitted arm, roll and jaw mapping."""

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import yaml
from diag_geometry_contact_sweep import replay

from real2sim.geometry.so101_fk import JointMapping
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=Path("results/audit/geometry_control_20261016/g1_image_fit/report.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = Path("results/metrics/dinov2_act_pilot_20261015/configs/sim_so101.yaml")
    base = yaml.safe_load(config.read_text())
    original_task = yaml.safe_load(Path(base["task_config"]).read_text())
    fit = json.loads(args.fit.read_text())
    fitted = np.array(fit["models"][fit["train_selected_candidate"]]["parameters"])
    reference = json.loads(Path("results/audit/geometry_control_20261016/g0_complete/report.json").read_text())
    joint = dict(np.load(".cache/derived_joint/arrays.npz", allow_pickle=False))
    cart = dict(np.load(".cache/derived_cartesian/arrays.npz", allow_pickle=False))
    with So101PickPlaceEnv(config, render_images=False) as env:
        limits = env.limits.copy()
    report = {"kind": "IMAGE_FIT_PARAMETER_FAMILY_ABLATION_NOT_CALIBRATION", "runs": []}
    for arm, roll, grip in itertools.product((False, True), repeat=3):
        label = f"arm{int(arm)}_roll{int(roll)}_grip{int(grip)}"
        task = copy.deepcopy(original_task)
        offsets = np.zeros(5)
        if arm:
            offsets[:4] = fitted[:4]
        if roll:
            offsets[4] = fitted[4]
        task["robot_so101"]["real_to_sim"]["offset_deg"] = offsets.tolist()
        if grip:
            task["robot_so101"]["gripper"]["sim_range_rad"] = (
                np.array(original_task["robot_so101"]["gripper"]["sim_range_rad"]) * fitted[5] +
                np.deg2rad(fitted[6])).tolist()
        task_path = args.out / f"task_{label}.yaml"
        task_path.write_text(yaml.safe_dump(task, sort_keys=False))
        cfg = copy.deepcopy(base)
        cfg["task_config"] = str(task_path.resolve())
        mapping = JointMapping.from_task_config(task)
        cfg["initial_qpos"] = np.clip(sim_from_real(mapping, cfg["initial_qpos_real_deg"]), *limits.T).tolist()
        cfg_path = args.out / f"sim_{label}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for ep in (1, 2, 3):
            candidate = {"xy_offset_mm": [0., 0.], "gripper_scale": 1.,
                         "friction": .6, "damping": 1., "base_offset_mm": 12.,
                         "mapping_family": label}
            run = replay(cfg_path, joint, cart, ep, reference["episodes"][str(ep)],
                         reference["review"][str(ep)], candidate, "leader", capture=False)
            report["runs"].append(run)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"mapping": label, "ep": ep, "valid": run["valid"],
                              "lift": run.get("lifted"), "success": run.get("success"),
                              "bilateral_lift_s": run.get("contact", {}).get("longest_bilateral_lift_s")}), flush=True)


if __name__ == "__main__":
    main()
