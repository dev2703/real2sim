"""Seeded source-resolution geometry captures, selected before viewing model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import RawDataset, sha256_file


def sample_frames(raw_root: Path, split_path: Path, out: Path, seed: int = 20261024):
    if out.exists():
        raise FileExistsError(out)
    raw = RawDataset(raw_root)
    split = json.loads(split_path.read_text())
    rng = np.random.default_rng(seed)
    selections = []
    # Camera-regime boundary is inherited from the earlier fixed-camera motion audit.
    # Static fitting and whole-frame verification are assigned before any image inspection.
    for regime, episodes in (("early", [e for e in split["train"] if e < 7]),
                             ("late", [e for e in split["train"] if e >= 7])):
        chosen = rng.choice(sorted(episodes), size=3, replace=False).tolist()
        for j, episode in enumerate(chosen):
            selections.append({"episode": episode, "frame": int(rng.integers(raw.episode_length(episode))),
                               "camera_regime": regime, "role": "fit" if j < 2 else "verification",
                               "partition": "train"})
    for episode in rng.choice(sorted(split["val"]), size=2, replace=False).tolist():
        selections.append({"episode": episode, "frame": int(rng.integers(raw.episode_length(episode))),
                           "camera_regime": "early" if episode < 7 else "late",
                           "role": "verification", "partition": "val"})
    out.mkdir(parents=True)
    images = []
    for selection in selections:
        episode, frame = selection["episode"], selection["frame"]
        rows = raw.episode_rows(episode)
        row = rows.loc[rows["frame_index"] == frame].iloc[0]
        selection.update(observation_state=list(map(float, row["observation.state"])),
                         recorded_action=list(map(float, row["action"])),
                         row_index=int(row["index"]), recorded_timestamp_s=float(row["timestamp"]))
        for camera in ("observation.images.iphone", "observation.images.top"):
            for position, timestamp, decoded in raw.iter_frames(camera, episode):
                if position == frame:
                    rgb = decoded.to_ndarray(format="rgb24")
                    name = f"ep{episode:03d}_f{frame:04d}_{camera.rsplit('.', 1)[-1]}.png"
                    path = out / name
                    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    segment = raw.video_segment(camera, episode)
                    images.append({**selection, "camera_key": camera, "path": str(path),
                                   "sha256": sha256_file(path), "source_size_wh": list(rgb.shape[1::-1]),
                                   "video_file": str(segment.path), "video_sha256": sha256_file(segment.path),
                                   "video_pts_s": timestamp, "video_episode_from_s": segment.from_ts})
                    break
            else:
                raise ValueError(f"source frame not decoded: episode={episode} frame={frame} camera={camera}")
    report = {"schema": "random_source_geometry_frames_v1", "seed": seed,
              "raw_root": str(raw_root), "split_path": str(split_path), "split_sha256": sha256_file(split_path),
              "selection_protocol": "Uniform episodes without replacement within predeclared early/late training strata (3 each), then uniform frame within each selected episode; two distinct validation episodes similarly sampled. First two in each training stratum fit, third verifies. No image/label quality screening or result-conditioned retries.",
              "camera_regime_assumption": "Episode <7 early; >=7 late, inherited from prior camera-motion evidence",
              "reserved_episodes_used": [], "selections": selections, "images": images,
              "source_info_sha256": sha256_file(raw_root / "meta/info.json"),
              "script_sha256": sha256_file(Path(__file__))}
    (out / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Inspection montage only; all measurements retain the original source pixels.
    thumbs = []
    for item in images:
        thumb = cv2.resize(cv2.imread(item["path"]), (640, 360))
        text = f"ep{item['episode']} f{item['frame']} {item['camera_key'].rsplit('.',1)[-1]} {item['role']}"
        cv2.putText(thumb, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
        thumbs.append(thumb)
    montage = np.concatenate([np.concatenate(thumbs[j:j + 2], axis=1)
                              for j in range(0, len(thumbs), 2)], axis=0)
    cv2.imwrite(str(out / "montage.jpg"), montage)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=Path("~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2").expanduser())
    p.add_argument("--split", type=Path, default=Path("results/v2/split.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261024)
    args = p.parse_args()
    result = sample_frames(args.raw_root, args.split, args.out, args.seed)
    print(json.dumps(result["selections"], indent=2))
