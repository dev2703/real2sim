"""Read-only collection audit; stored timestamps are not physical capture times."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from real2sim.data.raw import RawDataset, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    ds = RawDataset(args.data)
    rows = ds.episode_rows(args.episode)
    q, action = ds.stack(rows, "observation.state"), ds.stack(rows, "action")
    ts = rows.timestamp.to_numpy(dtype=float)
    assert np.isfinite(q).all() and np.isfinite(action).all()
    assert len(rows) == ds.episode_length(args.episode)
    assert np.array_equal(rows.frame_index, np.arange(len(rows)))
    window = int(round(ds.fps)) + 1
    joints = []
    for j, name in enumerate(ds.state_names):
        # A stable target for one nominal second; no motor outcome enters selection.
        holds = np.array([i for i in range(window - 1, len(q))
                          if np.ptp(action[i-window+1:i+1, j]) <= 1.0], dtype=int)
        residual = np.array([abs(q[i, j] - np.median(action[i-window+1:i+1, j])) for i in holds])
        # Same reference rows at every lag; descriptive only, not a causal delay fit.
        max_lag = min(int(ds.fps), len(q) // 4)
        reference = np.arange(1, len(q) - max_lag)
        moving = (abs(np.diff(q[:, j]))[reference - 1] > .2) | (abs(np.diff(action[:, j]))[reference - 1] > .2)
        reference = reference[moving]
        errors = [float(abs(action[reference, j] - q[reference+lag, j]).mean()) if len(reference) else None
                  for lag in range(max_lag + 1)]
        joint = {"name": name, "units": "degrees assumed from collector default/quantization; not declared in dataset" if j < 5 else "percent of gripper range",
                 "same_row_mae": float(abs(action[:, j] - q[:, j]).mean()),
                 "same_row_max_abs": float(abs(action[:, j] - q[:, j]).max()),
                 "next_row_mae": float(abs(action[:-1, j] - q[1:, j]).mean()),
                 "stable_target_window_end_count": len(holds),
                 "stable_target_end_frames": holds.tolist(),
                 "stable_target_residual_median": float(np.median(residual)) if len(residual) else None,
                 "stable_target_residual_p95": float(np.percentile(residual, 95)) if len(residual) else None,
                 "stable_target_residual_max": float(residual.max()) if len(residual) else None,
                 "descriptive_lag_frames": list(range(max_lag+1)), "moving_reference_count": len(reference),
                 "moving_mae_by_lag": errors,
                 "best_descriptive_lag_frames": int(np.argmin(errors)) if len(reference) else None}
        joints.append(joint)
    np.savetxt(args.out / "joint_traces.csv", np.c_[ts, action, q], delimiter=",",
               header=",".join(["nominal_time_s", *[f"target_{n}" for n in ds.state_names],
                                *[f"measured_{n}" for n in ds.state_names]]), comments="")

    videos = {}
    selected = np.linspace(0, len(rows)-1, 12, dtype=int)
    for key in ds.image_keys:
        segment = ds.video_segment(key, args.episode)
        pts, changes, identical, previous, snapshots = [], [], 0, None, {}
        with av.open(str(segment.path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("decoded frame lacks PTS")
                time = float(frame.pts * stream.time_base)
                if not segment.from_ts <= time < segment.to_ts - 1e-7:
                    continue
                index = len(pts)
                small = frame.reformat(width=160, height=90, format="gray").to_ndarray()
                if previous is not None:
                    identical += int(np.array_equal(small, previous))
                    changes.append(float(np.abs(small.astype(float) - previous).mean()))
                previous = small
                if index in selected:
                    snapshots[index] = frame.to_image().resize((320, 180))
                pts.append(time - segment.from_ts)
        video = {"decoded_frames": len(pts), "expected_rows": len(rows), "exact_equal_adjacent_downscaled_frames": identical,
                 "max_nominal_pts_row_error_s": float(np.max(abs(np.array(pts)-ts))) if len(pts) == len(ts) else None,
                 "mean_adjacent_gray_difference_0_255": float(np.mean(changes)),
                 "note": "Frame count/nominal PTS agreement does not establish capture-time alignment or exclude missing physical samples."}
        videos[key] = video
        sheet = Image.new("RGB", (4*320, 3*204), "white")
        draw = ImageDraw.Draw(sheet)
        for position, index in enumerate(selected):
            x, y = (position % 4)*320, (position // 4)*204
            if index in snapshots:
                sheet.paste(snapshots[index], (x, y))
            draw.text((x+5, y+183), f"frame {index} | nominal {ts[index]:.2f}s", fill="black")
        sheet.save(args.out / f"{key.rsplit('.', 1)[-1]}_contact_sheet.jpg")
    report = {"data": str(args.data.resolve()), "episode": args.episode, "rows": len(rows), "fps_nominal": ds.fps,
              "nominal_timestamp_span_s": float(ts[-1]-ts[0]), "columns": list(rows),
              "nominal_timestamp_dt_min_max_s": [float(np.diff(ts).min()), float(np.diff(ts).max())],
              "physical_timing_available": False, "separate_requested_and_sent_targets_available": False,
              "target_semantics": "Current collector writes returned sent target; dataset itself lacks collector commit/semantics version.",
              "hold_definition": "Overlapping windows spanning one nominal second with target peak-to-peak <=1 arm unit or gripper percentage point; endpoint error versus window median target. Not independent holds or a motor specification test.",
              "lag_note": "Positive lag compares recorded target to later state; nominal frame lag cannot establish wall-clock latency or separate limiter dynamics.",
              "joints": joints, "videos": videos,
              "source_sha256": {str(f.relative_to(args.data)): sha256_file(f) for f in args.data.rglob('*') if f.is_file()},
              "script_sha256": sha256_file(Path(__file__)), "training_data_modified": False,
              "physical_success_gate": "not evaluated; no simulation or hardware executed"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "videos": videos,
                      "joints": [{k:v for k,v in j.items() if k not in ("stable_target_end_frames", "descriptive_lag_frames", "moving_mae_by_lag")} for j in joints]}, indent=2))


if __name__ == "__main__":
    main()
