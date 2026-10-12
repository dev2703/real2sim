"""Build a derived, policy-ready dataset from the raw recording (never modifying it).

    uv run python -m real2sim.data.prepare --mode joint --config configs/data.yaml \
        --local-root <dir> --out-dir .cache/derived_joint

Output directory:

* ``images_<key>.npy`` — uint8 ``.npy`` memmap ``[N, H, W, 3]`` of letterboxed frames at the policy clock
* ``arrays.npz`` — per tick: ``episode_index, frame_index, row_index, timestamp, ep_start,
  ep_len, state[N,6], action[N,6]`` (joint mode: action = recorded leader target, D-003 C)
* ``stats.json`` — mean/std/min/max of state and action over **training episodes only**
* ``manifest.json`` — source revision, split, exclusions, transform geometry, label semantics,
  hashes of every output file

``--mode cartesian`` writes 13-float proprioception as ``state`` and 7-D actions as ``action``
(plus raw joints and label diagnostics) and requires ``labels.cartesian_gripper_source`` (D-013).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from real2sim.config import load_yaml, validate_camera_keys
from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES
from real2sim.data.raw import RawDataset, resolve_dataset_root
from real2sim.data.resample import episode_tick_rows, policy_stride
from real2sim.data.split import make_episode_split
from real2sim.data.transforms import letterbox_rgb

PREPARE_VERSION = "v2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def compute_stats(x: np.ndarray) -> dict:
    std = x.std(axis=0)
    return {
        "mean": x.mean(axis=0).tolist(),
        "std": np.maximum(std, 1e-3).tolist(),  # guard against constant dims
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "count": int(len(x)),
    }


def prepare_joint(root: Path, cfg: dict, out_dir: Path, repo_id: str | None, revision: str | None, episodes: list[int] | None = None) -> dict:
    return prepare(root, cfg, out_dir, repo_id, revision, episodes=episodes, mode="joint")


def prepare(
    root: Path,
    cfg: dict,
    out_dir: Path,
    repo_id: str | None,
    revision: str | None,
    episodes: list[int] | None = None,
    mode: str = "joint",
) -> dict:
    if mode not in ("joint", "cartesian"):
        raise ValueError(mode)
    gripper_source = None
    fk = None
    if mode == "cartesian":
        from real2sim.config import load_yaml as _load
        from real2sim.geometry.so101_fk import JointMapping, So101FK

        gripper_source = cfg["labels"].get("cartesian_gripper_source")
        if gripper_source in (None, "PENDING"):
            raise ValueError("labels.cartesian_gripper_source is not set: awaiting decision D-013")
        task_cfg = _load(cfg.get("task_config", "configs/task.yaml"))
        fk = So101FK(mapping=JointMapping.from_task_config(task_cfg))
    ds = RawDataset(root, repo_id=repo_id, revision=revision)
    keys = validate_camera_keys(cfg, ds.image_keys)
    excluded = sorted(int(e) for e in cfg.get("excluded_episodes", []))
    split_cfg = cfg["split"]
    n_val = min(int(split_cfg["n_val"]), max(len(ds.episode_indices) - 1, 0))
    split = make_episode_split(
        ds.episode_indices, n_val=n_val, seed=int(split_cfg["seed"]), repo_id=repo_id, revision=revision, exclude=excluded
    )
    keep = sorted(split.train + split.val)
    if episodes is not None:
        keep = [e for e in keep if e in set(episodes)]
    stride = policy_stride(ds.fps, float(cfg["policy_hz"]))
    H, W = int(cfg["image"]["height"]), int(cfg["image"]["width"])
    pad_value = int(cfg["image"].get("pad_value", 114))

    # ---- tick table
    ticks = []
    for ep in keep:
        g = ds.episode_rows(ep)
        rows = episode_tick_rows(len(g), stride)
        ticks.append((ep, g, rows))
    N = sum(len(r) for _, _, r in ticks)

    out_dir.mkdir(parents=True, exist_ok=True)
    keys_ = ["episode_index", "frame_index", "row_index", "timestamp", "ep_start", "ep_len", "state", "action"]
    if mode == "cartesian":
        keys_ += ["joint_state", "joint_action", "label_is_extrapolated", "leader_follower_pos_err", "jump_flag"]
    arrays = {k: [] for k in keys_}
    start = 0
    for ep, g, rows in ticks:
        n = len(rows)
        if mode == "cartesian":
            from real2sim.data.labels import episode_labels

            S = ds.stack(g, "observation.state")[rows]
            A = ds.stack(g, "action")[rows]
            lab = episode_labels(
                fk.fk_real(S), S[:, 5], A[:, 5], hz=float(cfg["policy_hz"]), gripper_source=gripper_source,
                lead=int(cfg["labels"].get("cartesian_lead_ticks", 1)), leader_poses=fk.fk_real(A),
            )
            arrays["joint_state"].append(S.astype(np.float32))
            arrays["joint_action"].append(A.astype(np.float32))
            arrays["label_is_extrapolated"].append(lab.label_is_extrapolated)
            arrays["leader_follower_pos_err"].append(lab.leader_follower_pos_err.astype(np.float32))
            arrays["jump_flag"].append(lab.jump_flags)
        arrays["episode_index"].append(np.full(n, ep, np.int64))
        arrays["frame_index"].append(g["frame_index"].to_numpy()[rows].astype(np.int64))
        arrays["row_index"].append(g["index"].to_numpy()[rows].astype(np.int64))
        arrays["timestamp"].append(g["timestamp"].to_numpy()[rows].astype(np.float64))
        arrays["ep_start"].append(np.full(n, start, np.int64))
        arrays["ep_len"].append(np.full(n, n, np.int64))
        if mode == "cartesian":
            arrays["state"].append(lab.proprio.astype(np.float32))
            arrays["action"].append(lab.action.astype(np.float32))
        else:
            arrays["state"].append(ds.stack(g, "observation.state")[rows].astype(np.float32))
            arrays["action"].append(ds.stack(g, "action")[rows].astype(np.float32))
        start += n
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}

    # ---- images
    geometry = None
    image_files = {}
    for key in keys:
        path = out_dir / f"images_{key.replace('observation.images.', '')}.npy"
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(N, H, W, 3))
        t0 = time.time()
        offset = 0
        for ep, _, rows in ticks:
            want = set(rows.tolist())
            got = 0
            for pos, _pts, frame in ds.iter_frames(key, ep, stride=stride):
                if pos not in want:
                    raise RuntimeError(f"unexpected frame position {pos} in episode {ep}")
                rgb = frame.to_ndarray(format="rgb24")
                img, geometry = letterbox_rgb(rgb, H, W, pad_value=pad_value)
                mm[offset + got] = img
                got += 1
            if got != len(rows):
                raise RuntimeError(f"{key} episode {ep}: decoded {got} frames for {len(rows)} ticks")
            offset += len(rows)
            print(f"  {key} ep{ep:03d}: {got} ticks", file=sys.stderr)
        mm.flush()
        del mm
        image_files[key] = path.name
        print(f"{key}: {N} frames in {time.time() - t0:.1f}s", file=sys.stderr)

    np.savez(out_dir / "arrays.npz", **arrays)

    train_mask = np.isin(arrays["episode_index"], split.train)
    stats = {
        "observation.state": compute_stats(arrays["state"][train_mask]),
        "action": compute_stats(arrays["action"][train_mask]),
        "image_normalization": {"mode": "imagenet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "computed_on": "train episodes only",
        "train_episodes": split.train,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    files = ["arrays.npz", "stats.json", *image_files.values()]
    manifest = {
        "prepare_version": PREPARE_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "source": {"repo_id": repo_id, "revision": revision, "root": str(root), "codebase_version": ds.codebase_version, "fps": ds.fps},
        "policy_hz": float(cfg["policy_hz"]),
        "stride": stride,
        "camera_keys": keys,
        "image_files": image_files,
        "image_shape_hwc": [H, W, 3],
        "letterbox": geometry.to_dict() if geometry else None,
        "state_names": ds.state_names if mode == "joint" else PROPRIO_NAMES,
        "action_names": ds.action_names if mode == "joint" else ACTION_NAMES,
        "label_semantics": (
            "joint: recorded leader target action[t] at each 10 Hz tick (D-003 option C)"
            if mode == "joint"
            else f"cartesian: achieved motion FK(q_obs[k+1]) relative to FK(q_obs[k]) in task frame (D-003 C); "
            f"gripper_target from {gripper_source} (D-013); proprio 13 floats (D-008 A); FK: Menagerie so101 "
            "gripperframe with provisional real->sim joint map (configs/task.yaml)"
        ),
        "label_diagnostics": None if mode == "joint" else {
            "jump_flags": int(arrays["jump_flag"].sum()),
            "extrapolated_ticks": int(arrays["label_is_extrapolated"].sum()),
            "leader_follower_pos_err_m_median": float(np.median(arrays["leader_follower_pos_err"])),
            "leader_follower_pos_err_m_p95": float(np.percentile(arrays["leader_follower_pos_err"], 95)),
        },
        "split": {"seed": split.seed, "train": split.train, "val": split.val, "excluded": split.excluded, "sha256": split.sha256},
        "n_ticks": int(N),
        "n_ticks_train": int(train_mask.sum()),
        "n_ticks_val": int((~train_mask).sum()),
        "files_sha256": {f: _sha256(out_dir / f) for f in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir}: {N} ticks ({manifest['n_ticks_train']} train / {manifest['n_ticks_val']} val)", file=sys.stderr)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m real2sim.data.prepare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["joint", "cartesian"], default="joint")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--local-root", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", default=None, help="comma-separated subset (debug only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    repo_id = args.repo_id or cfg["source"]["repo_id"]
    revision = args.revision or cfg["source"]["revision"]
    root = resolve_dataset_root(repo_id, revision, args.local_root)
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else None
    prepare(root, cfg, Path(args.out_dir), repo_id, revision, episodes=eps, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
