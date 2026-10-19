"""Explicit, immutable condition-stratified v2 splits; never infer labels from order."""

import hashlib
import json

import numpy as np

CONDITIONS = ("baseline", "ball_shift", "bowl_shift")
COUNTS = {"baseline": (23, 4, 4), "ball_shift": (3, 1, 1), "bowl_shift": (3, 1, 1)}


def manifest_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stratified_split(annotations: dict, episode_ids: list[int], seed: int = 0) -> dict:
    rows = annotations["episodes"]
    ids = [r["episode_index"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(episode_ids):
        raise ValueError("annotations must cover every source episode exactly once")
    groups = {c: [] for c in CONDITIONS}
    for row in rows:
        condition = row.get("condition")
        if condition not in groups or row.get("condition_status") != "confirmed":
            raise ValueError(f"episode {row['episode_index']}: condition is not confirmed")
        if condition != "baseline" and not row.get("shift_direction"):
            raise ValueError(f"episode {row['episode_index']}: shifted condition needs direction")
        groups[condition].append(int(row["episode_index"]))
    result = {"version": "v2_conditions_v1", "seed": seed, "train": [], "val": [], "reserved": [],
              "excluded": [], "condition_counts": COUNTS,
              "annotations_sha256": manifest_hash(annotations), "conditions": {str(r["episode_index"]): r["condition"] for r in rows}}
    rng = np.random.default_rng(seed)
    for condition, counts in COUNTS.items():
        group = sorted(groups[condition])
        if len(group) != sum(counts):
            raise ValueError(f"{condition}: expected {sum(counts)} episodes, got {len(group)}")
        shuffled = rng.permutation(group).tolist()
        offset = 0
        for name, count in zip(("train", "val", "reserved"), counts, strict=True):
            result[name].extend(shuffled[offset:offset + count])
            offset += count
    for name in ("train", "val", "reserved"):
        result[name].sort()
    result["sha256"] = manifest_hash(result)
    return result


def validate_fixed_split(split: dict, episode_ids: list[int]) -> dict:
    if split.get("sha256") != manifest_hash(split):
        raise ValueError("fixed split checksum mismatch")
    lists = [split.get(k, []) for k in ("train", "val", "reserved", "excluded")]
    flat = [int(e) for group in lists for e in group]
    if not lists[0] or not lists[1] or len(flat) != len(set(flat)) or set(flat) != set(episode_ids):
        raise ValueError("fixed split must partition all episodes without overlap")
    if split.get("version") == "v2_conditions_v1":
        if lists[3]:
            raise ValueError("the v2 split retains all 41 episodes, including episode 9")
        for condition, counts in COUNTS.items():
            observed = tuple(sum(split["conditions"].get(str(e)) == condition for e in group)
                             for group in lists[:3])
            if observed != counts:
                raise ValueError(f"v2 stratification differs for {condition}: {observed}")
    return split
