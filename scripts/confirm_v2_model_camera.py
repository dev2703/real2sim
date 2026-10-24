"""Freeze, sample and score fresh camera confirmation frames without refitting."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim.data.raw import RawDataset, sha256_file

CAMERA_REPORT = Path("outputs/v2/model_multiframe_camera_20261024_v4/report.json")
EXTRACTOR = Path("outputs/v2/model_mat_edges_20261024_v4/executed_extractor.py")
ORIGINAL_SELECTION = Path("outputs/v2/random_vision_geometry_20261024/selection.json")
MEASURER = Path("scripts/measure_v2_vision_geometry.py")
SPLIT = Path("results/v2/split.json")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(out, seed=2026102402):
    if out.exists():
        raise FileExistsError(out)
    split, old = json.loads(SPLIT.read_text()), json.loads(ORIGINAL_SELECTION.read_text())
    raw_root = Path(old["raw_root"])
    raw = RawDataset(raw_root)
    used = {row["episode"] for row in old["selections"]}
    rng = np.random.default_rng(seed)
    selections = []
    for regime, candidates in (("early", [2, 3, 6]),
                               ("late", [e for e in split["train"] if e >= 7 and e not in used])):
        candidates = sorted(set(candidates) & set(split["train"]) - used)
        episode = int(rng.choice(candidates))
        selections.append({"episode": episode, "frame": int(rng.integers(raw.episode_length(episode))),
                           "camera_regime": regime, "role": "confirmation", "partition": "train"})
    if sha256_file(EXTRACTOR) != json.loads(Path("outputs/v2/model_mat_edges_20261024_v4/extraction.json").read_text())["script_sha256"]:
        raise ValueError("exact v4 extractor is not the previously executed implementation")
    frozen = {str(path): sha256_file(path) for path in (CAMERA_REPORT, EXTRACTOR, ORIGINAL_SELECTION, MEASURER, SPLIT, Path(__file__))}
    plan = {"schema": "v2_fresh_camera_confirmation_plan_v1", "seed": seed,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(), "frozen_files_sha256": frozen,
            "original_episodes_excluded": sorted(used), "selections": selections,
            "rule": "Both frames are confirmation only. No model prompt, edge rule, camera K/T, bounds, parameter or frame substitution may change after this plan. Missing/failed measurements remain explicit failures; reserved episodes are excluded.",
            "working_corner_rms_threshold_px": 3., "reserved_episodes_used": []}
    out.mkdir(parents=True)
    # This persisted plan precedes source-image decoding or model inference.
    write(out / "plan.json", plan)
    images = []
    for selection in selections:
        episode, position = selection["episode"], selection["frame"]
        rows = raw.episode_rows(episode)
        row = rows.loc[rows["frame_index"] == position].iloc[0]
        metadata = {**selection, "observation_state": list(map(float, row["observation.state"])),
                    "recorded_action": list(map(float, row["action"])), "row_index": int(row["index"]),
                    "recorded_timestamp_s": float(row["timestamp"])}
        key = "observation.images.iphone"
        for index, timestamp, decoded in raw.iter_frames(key, episode):
            if index != position:
                continue
            rgb = decoded.to_ndarray(format="rgb24")
            path = out / f"ep{episode:03d}_f{position:04d}_iphone.png"
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            segment = raw.video_segment(key, episode)
            images.append({**metadata, "camera_key": key, "path": str(path), "sha256": sha256_file(path),
                           "source_size_wh": list(rgb.shape[1::-1]), "video_file": str(segment.path),
                           "video_sha256": sha256_file(segment.path), "video_pts_s": timestamp,
                           "video_episode_from_s": segment.from_ts})
            break
        else:
            raise ValueError("planned source frame was not decoded; no replacement permitted")
    selection = {"schema": "random_source_geometry_frames_v1", "seed": seed, "raw_root": str(raw_root),
                 "split_path": str(SPLIT), "split_sha256": sha256_file(SPLIT),
                 "plan_path": str(out / "plan.json"), "plan_sha256": sha256_file(out / "plan.json"),
                 "selection_protocol": "One uniform unused training episode per early/late stratum and one uniform source frame, assigned confirmation before decoding. Exact v4 extractor and original K/T frozen.",
                 "reserved_episodes_used": [], "selections": selections, "images": images}
    write(out / "selection.json", selection)
    return selection


def score(root, model_path, annotations_path, out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((root / "plan.json").read_text())
    for path, digest in plan["frozen_files_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen confirmation input changed: {path}")
    selection = json.loads((root / "selection.json").read_text())
    annotations = json.loads(annotations_path.read_text())
    model = json.loads(model_path.read_text())
    if (annotations["selection_sha256"] != sha256_file(root / "selection.json")
            or model["selection_sha256"] != sha256_file(root / "selection.json")):
        raise ValueError("model/extractor did not consume the frozen confirmation selection")
    camera = json.loads(CAMERA_REPORT.read_text())
    expected = {(row["episode"], row["frame"]): row for row in selection["images"]}
    if {(row["episode"], row["frame"]) for row in annotations["frames"]} != set(expected):
        raise ValueError("confirmation frames were substituted or omitted")
    points = np.array([[0., 0., 0.], [.305, 0., 0.], [.305, .406, 0.], [0., .406, 0.]])
    rows = []
    for row in annotations["frames"]:
        original = expected[row["episode"], row["frame"]]
        if row["sha256"] != original["sha256"] or row["path"] != original["path"]:
            raise ValueError("source-image identity changed")
        result = {key: original[key] for key in ("episode", "frame", "camera_regime", "role", "path", "sha256")}
        if row["role"] == "excluded":
            result.update(status="extraction_failed", reason=row["reason"], passes_3px=False)
        else:
            if row["role"] != "confirmation":
                raise ValueError("confirmation image became fitting/verification data")
            lines = []
            for name in ("top", "right", "bottom", "left"):
                pixels = np.asarray(row["mat_edge_samples_source_px"][name])
                mean = pixels.mean(0)
                _, _, vt = np.linalg.svd(pixels - mean, full_matrices=False)
                lines.append(np.r_[vt[-1], -vt[-1] @ mean])
            corners = []
            for a, b in ((3, 0), (0, 1), (1, 2), (2, 3)):
                cross = np.cross(lines[a], lines[b])
                corners.append(cross[:2] / cross[2])
            baseline = camera["regimes"][row["camera_regime"]]["baseline"]
            K, T = np.asarray(baseline["K"]), np.asarray(baseline["T_optical_plane"])
            prediction = cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
            error = np.linalg.norm(prediction - corners, axis=1)
            rms = float(np.sqrt(np.mean(error**2)))
            result.update(status="scored_original_camera_without_refit", corners_source_px=np.asarray(corners).tolist(),
                          original_camera_prediction_px=prediction.tolist(), per_corner_error_px=error.tolist(),
                          rms_px=rms, p95_px=float(np.percentile(error, 95)), passes_3px=rms <= 3.)
        rows.append(result)
    report = {"schema": "v2_fresh_model_camera_confirmation_v1", "physical_calibration_accepted": False,
              "seed": plan["seed"], "frames": rows, "all_frames_pass_3px": all(row["passes_3px"] for row in rows),
              "parameters_refitted": False, "rules_adapted_after_confirmation": False, "reserved_episodes_used": [],
              "frozen_plan": plan, "frozen_plan_sha256": sha256_file(root / "plan.json"),
              "artifact_sha256": {str(path): sha256_file(path) for path in (root / "selection.json", model_path, annotations_path, CAMERA_REPORT, EXTRACTOR)},
              "interpretation": "Fresh measurement-pipeline confirmation independent of the eight development frames, under the same conditional mat dimensions/optical assumptions. Passing two images would not identify all intrinsics, verify CAD identities or establish physical calibration. Failure receives no fitting, detector changes or frame replacement."}
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("prepare")
    sample.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out)
        print(json.dumps(result["selections"], indent=2))
    else:
        result = score(args.root, args.model, args.annotations, args.out)
        print(json.dumps(result["frames"], indent=2))
