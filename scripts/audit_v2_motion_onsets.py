"""Describe hold-to-motion supervision without editing labels or policy inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.data.raw import sha256_file

EPISODES = (11, 8, 23)
RULES = {"translation_motion_mm": .4, "stationary_rotation_rad": .01,
         "causal_history_ticks": 4, "causal_history_speed_m_s": .005,
         "causal_history_arm_span_deg": .5, "near_arm_max_difference_deg": .25,
         "near_gripper_difference_percentage_points": .5,
         "near_velocity_difference_m_s": .005, "similar_image_mae_uint8": 1.,
         "executed_prefix_steps": 2, "full_chunk_steps": 25}


def valid_mask(arrays):
    result = np.ones(len(arrays["episode_index"]), bool)
    for name in ("label_is_extrapolated", "action_is_invalid", "observation_is_invalid"):
        if name in arrays:
            mask = np.asarray(arrays[name])
            if mask.dtype != np.bool_ or mask.shape != result.shape:
                raise ValueError(f"{name} must be a boolean per-tick mask")
            result &= ~mask
    return result


def chunk_ticks(arrays, tick, horizon, valid):
    episode = arrays["episode_index"][tick]
    ticks = np.arange(tick, min(tick + horizon, len(valid)))
    ticks = ticks[arrays["episode_index"][ticks] == episode]
    return ticks[valid[ticks]]


def pair_metrics(arrays, images, a, b, valid, rules):
    # Compare the same offsets and never shift across an invalid target.
    limit = rules["full_chunk_steps"]
    offsets = np.arange(limit)
    eligible = ((a + offsets < len(valid)) & (b + offsets < len(valid)))
    offsets = offsets[eligible]
    offsets = offsets[(arrays["episode_index"][a + offsets] == arrays["episode_index"][a])
                      & (arrays["episode_index"][b + offsets] == arrays["episode_index"][b])
                      & valid[a + offsets] & valid[b + offsets]]
    difference = np.linalg.norm(arrays["action"][a + offsets, :3]
                                - arrays["action"][b + offsets, :3], axis=1) * 1000
    prefix = difference[offsets < rules["executed_prefix_steps"]]
    return {"ticks": [int(a), int(b)],
            "frames": [int(arrays["frame_index"][a]), int(arrays["frame_index"][b])],
            "first_action_translation_difference_mm": float(np.linalg.norm(
                arrays["action"][a, :3] - arrays["action"][b, :3]) * 1000),
            "prefix_mean_translation_difference_mm": float(prefix.mean()) if len(prefix) else None,
            "chunk_mean_translation_difference_mm": float(difference.mean()) if len(difference) else None,
            "prefix_valid_offsets": int(len(prefix)), "chunk_valid_offsets": len(offsets),
            "current_image_mae_uint8": float(np.abs(images[a].astype(np.float32)
                                                     - images[b].astype(np.float32)).mean()),
            "exact_proprioception": bool(np.array_equal(arrays["state"][a], arrays["state"][b]))}


def audit_episode(arrays, images, episode, rules=RULES):
    valid = valid_mask(arrays)
    episode_ticks = np.flatnonzero(arrays["episode_index"] == episode)
    if not len(episode_ticks):
        raise ValueError(f"missing episode {episode}")
    ticks = episode_ticks[valid[episode_ticks]]
    translation, rotation = (np.full(len(valid), np.nan) for _ in range(2))
    translation[episode_ticks] = np.linalg.norm(arrays["action"][episode_ticks, :3], axis=1) * 1000
    rotation[episode_ticks] = np.linalg.norm(arrays["action"][episode_ticks, 3:6], axis=1)
    stationary = (translation <= rules["translation_motion_mm"]) & (rotation <= rules["stationary_rotation_rad"])
    onsets, causal_hold_labels = [], []
    prefix_stationary_tail_motion, comparable_prefixes = 0, 0
    for tick in ticks:
        horizon = chunk_ticks(arrays, tick, rules["full_chunk_steps"], valid)
        prefix = horizon[horizon < tick + rules["executed_prefix_steps"]]
        if len(prefix) == rules["executed_prefix_steps"]:
            comparable_prefixes += 1
            tail = horizon[horizon >= tick + rules["executed_prefix_steps"]]
            if np.all(stationary[prefix]) and np.any(translation[tail] > rules["translation_motion_mm"]):
                prefix_stationary_tail_motion += 1
        start = int(tick) - rules["causal_history_ticks"] + 1
        if start < episode_ticks[0]:
            continue
        history = np.arange(start, tick + 1)
        # Observation validity matters here; next-label validity is not an input.
        observation_valid = arrays.get("observation_is_invalid", np.zeros(len(valid), bool))
        if np.any(observation_valid[history]):
            continue
        slow = np.all(np.linalg.norm(arrays["state"][history, 9:12], axis=1)
                      <= rules["causal_history_speed_m_s"])
        steady = np.max(np.ptp(arrays["joint_state"][history, :5], axis=0)) <= rules["causal_history_arm_span_deg"]
        if slow and steady:
            causal_hold_labels.append(int(tick))
            if translation[tick] > rules["translation_motion_mm"]:
                onsets.append({"tick": int(tick), "frame": int(arrays["frame_index"][tick]),
                               "next_translation_mm": float(translation[tick]),
                               "causal_history_ticks": history.astype(int).tolist()})
    states = arrays["state"][ticks]
    _, inverse, counts = np.unique(states, axis=0, return_inverse=True, return_counts=True)
    repeated_pairs = []
    for group in np.flatnonzero(counts > 1):
        group_ticks = ticks[inverse == group]
        final = group_ticks[-1]
        earlier = [int(t) for t in group_ticks[:-1] if np.linalg.norm(
            arrays["action"][t, :3] - arrays["action"][final, :3]) * 1000 > rules["translation_motion_mm"]]
        if earlier:
            repeated_pairs.append(pair_metrics(arrays, images, earlier[-1], int(final), valid, rules))
    adjacent_pairs = []
    for b in ticks:
        a = b - 1
        if a < episode_ticks[0] or not valid[a]:
            continue
        arm = float(np.max(np.abs(arrays["joint_state"][a, :5] - arrays["joint_state"][b, :5])))
        gripper = float(abs(arrays["joint_state"][a, 5] - arrays["joint_state"][b, 5]))
        velocity = float(np.linalg.norm(arrays["state"][a, 9:12] - arrays["state"][b, 9:12]))
        if (arm <= rules["near_arm_max_difference_deg"]
                and gripper <= rules["near_gripper_difference_percentage_points"]
                and velocity <= rules["near_velocity_difference_m_s"]
                and np.linalg.norm(arrays["action"][a, :3] - arrays["action"][b, :3]) * 1000 > rules["translation_motion_mm"]):
            row = pair_metrics(arrays, images, int(a), int(b), valid, rules)
            row.update(arm_max_difference_deg=arm, gripper_difference_percentage_points=gripper,
                       velocity_difference_m_s=velocity,
                       similar_current_image=row["current_image_mae_uint8"] <= rules["similar_image_mae_uint8"])
            adjacent_pairs.append(row)
    return {"episode": episode, "valid_ticks": len(ticks), "unique_proprioceptive_states": len(counts),
            "repeated_state_groups": int(np.sum(counts > 1)),
            "repeated_state_groups_with_first_action_conflict": len(repeated_pairs),
            "two_step_prefixes_with_two_real_labels": comparable_prefixes,
            "stationary_two_step_prefixes_with_motion_in_chunk_tail": prefix_stationary_tail_motion,
            "causal_low_motion_history_ticks": len(causal_hold_labels),
            "causal_low_motion_history_followed_by_moving_label": len(onsets),
            "onsets": onsets, "exact_state_conflict_pairs": repeated_pairs,
            "adjacent_near_state_conflict_pairs": adjacent_pairs,
            "adjacent_near_state_and_image_conflict_pairs": sum(p["similar_current_image"] for p in adjacent_pairs)}


def run(data, out):
    if out.exists():
        raise FileExistsError(out)
    manifest_path, arrays_path = data / "manifest.json", data / "arrays.npz"
    manifest = json.loads(manifest_path.read_text())
    if not set(EPISODES) <= set(manifest["split"]["train"]):
        raise ValueError("all audited episodes must be training episodes")
    if set(EPISODES) & (set(manifest["split"]["val"]) | set(manifest["split"]["reserved"])):
        raise ValueError("audit episodes overlap development or reserved partitions")
    if manifest["action_representation"] != "cartesian" or manifest["camera_keys"] != ["observation.images.iphone"]:
        raise ValueError("requires the original single-iPhone Cartesian data")
    image_path = data / manifest["image_files"][manifest["camera_keys"][0]]
    inputs = [manifest_path, arrays_path, image_path, data / "stats.json", Path(__file__)]
    # Freeze the descriptive analysis and input hashes before loading selected images.
    out.mkdir(parents=True)
    plan = {"schema": "three_episode_onset_plan_v1", "episodes": list(EPISODES), "rules": RULES,
            "reserved_episodes_used": [], "policy_inference": False,
            "selection": "All valid ticks in the three declared training episodes. Thresholds are inherited motion scales or explicit descriptive bounds. No label/image/policy outcome removes examples.",
            "source_sha256": {str(p.resolve()): sha256_file(p) for p in inputs}}
    for filename, digest in manifest["files_sha256"].items():
        if sha256_file(data / filename) != digest:
            raise ValueError(f"derived payload differs: {filename}")
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    arrays = dict(np.load(arrays_path, allow_pickle=False))
    images = np.load(image_path, mmap_mode="r")
    episodes = [audit_episode(arrays, images, episode) for episode in EPISODES]
    report = {"schema": "three_episode_onset_audit_v1", "plan_sha256": sha256_file(out / "plan.json"),
              "data": str(data.resolve()), "episodes": episodes,
              "reserved_episodes_used": [], "labels_changed": False, "normalization_changed": False,
              "causal_inputs_changed": False, "policy_inference": False,
              "interpretation": ["Current state/history are causal descriptors; next labels and future chunk tails are inspected only as supervision outcomes and never inserted into policy inputs.",
                  "A stationary executed prefix with a moving future tail is a valid demonstration, not automatically a corrupt label. It shows the longer training objective includes future onset timing beyond immediate execution.",
                  "Exact proprioception plus similar images does not mean identical complete visual observations or prove an irreducible error bound.",
                  "No episodes/ticks are filtered, reweighted, relabeled or dropped by this audit. All measurements depend on provisional FK and original sampling/timestamp contracts.",
                  "Counts/pairs within each episode are correlated. Near-state/image thresholds are descriptive, not calibrated observation equivalence."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/v2_exploratory_20261022/cartesian_iphone"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.out)
    print(json.dumps([{k: v for k, v in ep.items() if not isinstance(v, list)} for ep in result["episodes"]], indent=2))
