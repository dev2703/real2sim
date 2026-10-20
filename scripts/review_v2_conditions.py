"""Export reproducible visual evidence for manual episode-condition review.

This reads the recording without changing it. Contact sheets do not classify conditions,
measure metric shifts, establish camera calibration, or determine task phase boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from real2sim.data.raw import RawDataset, sha256_file

FRAMES = tuple(range(0, 176, 25))
CAMERA = "observation.images.iphone"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/so101_act_pickplace_v2",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to replace review evidence: {args.out}")
    ds = RawDataset(args.root.expanduser())
    if CAMERA not in ds.image_keys:
        raise ValueError(f"missing fixed camera: {CAMERA}")
    args.out.mkdir(parents=True)
    records, initial = [], []
    for episode in ds.episode_indices:
        sheet = Image.new("RGB", (1600, 500), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        frames = []
        segment = ds.video_segment(CAMERA, episode)
        for position, pts, frame in ds.iter_frames(CAMERA, episode, stride=25):
            if position > FRAMES[-1]:
                break
            full = frame.to_image()
            if position == 0:
                path = args.out / f"iphone_ep{episode:02d}_f0000.jpg"
                full.save(path, quality=90)
                initial.append((episode, path))
            thumb = full.copy()
            thumb.thumbnail((400, 225))
            index = len(frames)
            x, y = index % 4 * 400, index // 4 * 250
            sheet.paste(thumb, (x, y + 25))
            draw.text((x + 5, y + 5), f"Episode {episode}, source frame {position}", fill="white")
            frames.append({"source_frame": position, "video_pts_s": pts})
        if [f["source_frame"] for f in frames] != list(FRAMES):
            raise ValueError(f"episode {episode}: incomplete opening-frame evidence")
        path = args.out / f"temporal_ep{episode:02d}.jpg"
        sheet.save(path, quality=90)
        records.append({
            "episode_index": episode,
            "video": str(segment.path.relative_to(ds.root)),
            "segment_from_s": segment.from_ts,
            "segment_to_s": segment.to_ts,
            "frames": frames,
            "contact_sheet": path.name,
        })
    for offset in range(0, len(initial), 14):
        sheet = Image.new("RGB", (1280, 820), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for index, (episode, path) in enumerate(initial[offset:offset + 14]):
            with Image.open(path) as source:
                thumb = source.copy()
            thumb.thumbnail((320, 180))
            x, y = index % 4 * 320, index // 4 * 205
            sheet.paste(thumb, (x, y + 25))
            draw.text((x + 5, y + 5), f"Episode {episode}, source frame 0", fill="white")
        sheet.save(args.out / f"overview_{offset:02d}.jpg", quality=90)
    source_paths = {
        ds.root / "meta/info.json",
        *ds.root.glob("meta/episodes/**/*.parquet"),
        *(ds.video_segment(CAMERA, ep).path for ep in ds.episode_indices),
    }
    report = {
        "version": "v2_condition_visual_evidence_v1",
        "camera_key": CAMERA,
        "root": str(ds.root),
        "source_sha256": {
            str(p.relative_to(ds.root)): sha256_file(p) for p in sorted(source_paths)
        },
        "script_sha256": sha256_file(Path(__file__)),
        "nominal_fps": ds.fps,
        "episodes": records,
        "artifact_sha256": {p.name: sha256_file(p) for p in sorted(args.out.glob("*.jpg"))},
        "interpretation": "Manual review evidence only; no automatic labels or metric calibration.",
    }
    (args.out / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(records)} episodes to {args.out}")


if __name__ == "__main__":
    main()
