"""Native-resolution jaw landmarks for independent real/sim registration review."""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from real2sim.data.paired_approach import camera_parameters
from real2sim.data.raw import RawDataset
from real2sim.sim.camera_fit import decode_frame
from real2sim.sim.env import So101PickPlaceEnv, sim_from_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "results/metrics/dinov2_act_pilot_20261015/configs/sim_so101.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=["1:261", "1:390", "1:435", "2:246", "2:390", "3:288", "5:300", "12:480"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(".cache/derived_joint/manifest.json").read_text())
    arrays = np.load(".cache/derived_joint/arrays.npz")
    raw = RawDataset(manifest["source"]["root"])
    rows = []
    with So101PickPlaceEnv(args.config, render_images=False) as env:
        env.reset(seed=0)
        rv, tv, K, _, _ = camera_parameters(env)
        for sample in args.samples:
            ep, frame = map(int, sample.split(":"))
            idx = np.flatnonzero((arrays["episode_index"] == ep) & (arrays["frame_index"] == frame))
            if len(idx) != 1:
                raise ValueError(f"No exact derived sample {sample}")
            state = arrays["state"][idx[0]]
            env.data.qpos[env.qadr] = sim_from_real(env.mapping, state)
            mujoco.mj_forward(env.model, env.data)
            image = decode_frame(raw.video_segment("observation.images.iphone", ep), frame, raw.fps)
            cv2.imwrite(str(args.out / f"ep{ep}_f{frame}_raw.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            projected = {}
            for name, color in (("fixed_jaw_sph_tip1", (255, 30, 30)),
                                ("moving_jaw_sph_tip1", (30, 200, 255))):
                position = env.data.geom_xpos[env.model.geom(name).id] - env.offset
                point = cv2.projectPoints(position[None], rv, tv, K, None)[0].reshape(2)
                projected[name] = point.tolist()
                cv2.drawMarker(image, tuple(np.rint(point).astype(int)), color, cv2.MARKER_CROSS, 20, 2)
                cv2.putText(image, name, tuple(np.rint(point + [5, 20]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
            cv2.putText(image, f"ep{ep} frame{frame}: follower gripper={state[5]:.2f}%", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 70, 30), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"ep{ep}_f{frame}_projected.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append({"episode": ep, "frame": frame, "state": state.tolist(), "projection_px": projected})
    (args.out / "report.json").write_text(json.dumps({"kind": "UNFITTED_JAW_PROJECTIONS_FOR_REVIEW", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
