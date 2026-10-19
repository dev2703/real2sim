"""R5 registered two-camera renders at recorded states, or object counterfactuals.

Requires an independently registered per-tick scene NPZ: ticks, ball_xyz_task,
bowl_xy_task. No ball/bowl coordinates enter policy observations. Counterfactual
renders are diagnostic-only and cannot be mixed with unchanged actions for training.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from real2sim.data.transforms import letterbox_rgb
from real2sim.policy.dataset import DerivedTickDataset
from real2sim.policy.paired import file_sha256
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument(
        "--registration",
        type=Path,
        required=True,
        help="JSON: scene/data/task hashes and independent verification",
    )
    p.add_argument("--config", type=Path, default=Path("configs/v2/sim_so101.yaml"))
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--move", choices=("ball", "bowl"))
    p.add_argument("--shift-xy-m", nargs=2, type=float)
    a = p.parse_args()
    if bool(a.move) != bool(a.shift_xy_m):
        raise ValueError("object and task-frame displacement must be supplied together")
    if a.shift_xy_m and not np.isfinite(a.shift_xy_m).all():
        raise ValueError("finite displacement required")
    if a.out.exists():
        raise FileExistsError(a.out)
    m = json.loads((a.data / "manifest.json").read_text())
    ds = DerivedTickDataset(a.data, m["split"][a.split], 25)
    source = np.load(a.scenes, allow_pickle=False)
    ticks = source["ticks"]
    ball = source["ball_xyz_task"]
    bowl = source["bowl_xy_task"]
    if (
        ticks.ndim != 1
        or ticks.dtype.kind not in "iu"
        or len(set(ticks)) != len(ticks)
        or not len(ticks)
        or not np.isin(ticks, ds.indices).all()
        or ball.shape != (len(ticks), 3)
        or bowl.shape != (len(ticks), 2)
        or not np.isfinite(ball).all()
        or not np.isfinite(bowl).all()
    ):
        raise ValueError("invalid registered scenes or cross-split ticks")
    registration = json.loads(a.registration.read_text())
    if (
        registration.get("scene_sha256") != file_sha256(a.scenes)
        or registration.get("data_manifest_sha256") != file_sha256(a.data / "manifest.json")
        or registration.get("independent_verification") is not True
    ):
        raise ValueError("unverified or mismatched registration")
    a.out.mkdir(parents=True)
    with So101PickPlaceEnv(a.config, render_images=True) as env:
        if registration.get("task_sha256") != file_sha256(Path(env.config["task_config"])):
            raise ValueError("scene calibration hash differs")
        if env.camera_keys != m["camera_keys"]:
            raise ValueError("camera order differs")
        env.reset(seed=0)
        files = {key: f"images_{key.split('.')[-1]}.npy" for key in env.camera_keys}
        images = {
            key: np.lib.format.open_memmap(
                a.out / name, mode="w+", dtype=np.uint8, shape=(len(ticks), *m["image_shape_hwc"])
            )
            for key, name in files.items()
        }
        for i, t in enumerate(ticks):
            q = sim_from_real(env.mapping, ds.arrays["joint_state"][t])
            if np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]):
                raise ValueError(
                    f"tick {t}: recorded pose outside limits; do not disguise it with clipping"
                )
            b = ball[i].copy()
            bowl_xy = bowl[i].copy()
            if a.move == "ball":
                b[:2] += a.shift_xy_m
            if a.move == "bowl":
                bowl_xy += a.shift_xy_m
            env.data.qpos[env.qadr] = q
            env.data.qpos[env._ball_q : env._ball_q + 3] = b + env.offset
            env.model.body_pos[env.model.body("bowl").id, :2] = bowl_xy
            mujoco.mj_forward(env.model, env.data)
            for key, name in env.camera_bindings.items():
                if key in images:
                    images[key][i] = letterbox_rgb(
                        env.render(name),
                        *m["image_shape_hwc"][:2],
                        pad_value=env.data_config["image"]["pad_value"],
                    )[0]
        for view in images.values():
            view.flush()
    np.save(a.out / "ticks.npy", ticks)
    paired = {
        "split_name": a.split,
        "derived_manifest": m,
        "image_files": files,
        "sha256": {name: file_sha256(a.out / name) for name in ("ticks.npy", *files.values())},
        "registration_sha256": file_sha256(a.registration),
        "sim_config_sha256": file_sha256(a.config),
        "usage": "diagnostic_only" if a.move else "registered_real_action_pairs",
        "counterfactual": {"object": a.move, "shift_xy_m": a.shift_xy_m} if a.move else None,
    }
    (a.out / "manifest.json").write_text(json.dumps(paired, indent=2) + "\n")


if __name__ == "__main__":
    main()
