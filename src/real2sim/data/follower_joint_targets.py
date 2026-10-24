"""Matched absolute-joint targets with native-feasible observations and explicit timing.

Raw sampled rows and episode identities never move. Invalid gaps split action chunks,
so a mask cannot accidentally train from an impossible current observation or bridge
an unobserved period. This checks native ranges, not physical calibration or collision.
"""

from __future__ import annotations

import numpy as np

from real2sim.geometry.so101_fk import JointMapping

VERSION = "matched_next_follower_joint_v1"
VARIANTS = ("follower_next_arm_leader_gripper", "leader_command_matched_mask")
JOINT_ACTION_NAMES = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def native_joint_rows(real, mapping: JointMapping):
    real = np.asarray(real, dtype=float)
    if real.ndim != 2 or real.shape[1] != 6 or not np.isfinite(real).all():
        raise ValueError("finite sampled joints[N,6] required")
    return np.c_[mapping.arm_real_to_sim_rad(real), mapping.gripper_real_to_sim_rad(real[:, 5])]


def segment_metadata(episodes, valid):
    """Give each contiguous valid run an independent chunk clock, without resampling."""
    episodes, valid = np.asarray(episodes), np.asarray(valid, bool)
    if episodes.ndim != 1 or valid.shape != episodes.shape:
        raise ValueError("one boolean validity value per episode row required")
    n = len(episodes)
    starts, lengths, segment_ids = np.arange(n), np.ones(n, dtype=np.int64), np.full(n, -1, np.int64)
    segment = 0
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        end = i + 1
        while end < n and valid[end] and episodes[end] == episodes[i]:
            end += 1
        starts[i:end], lengths[i:end], segment_ids[i:end] = i, end - i, segment
        segment += 1
        i = end
    return starts.astype(np.int64), lengths, segment_ids


def make_matched_targets(arrays: dict, mapping: JointMapping, limits: np.ndarray,
                         allowed_episodes: list[int]) -> tuple[dict[str, dict], dict]:
    s, leader = np.asarray(arrays["joint_state"]), np.asarray(arrays["joint_action"])
    episode = np.asarray(arrays["episode_index"])
    n = len(episode)
    if s.shape != (n, 6) or leader.shape != (n, 6):
        raise ValueError("aligned observation and leader command joints[N,6] required")
    if np.asarray(arrays["state"]).shape != (n, 13):
        raise ValueError("untransformed canonical13 source required")
    limits = np.asarray(limits, float)
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("six native lower/upper limits required")
    successor = np.arange(n)
    has_successor = np.zeros(n, bool)
    if n > 1:
        has_successor[:-1] = episode[:-1] == episode[1:]
        successor[has_successor] += 1
    follower = s.copy()
    follower[:, :5] = s[successor, :5]
    # Preserve current command semantics; next measured aperture is a different target.
    follower[:, 5] = leader[:, 5]
    native = {"current": native_joint_rows(s, mapping),
              "follower": native_joint_rows(follower, mapping),
              "leader": native_joint_rows(leader, mapping)}
    in_range = {key: ((values >= limits[:, 0]) & (values <= limits[:, 1])).all(axis=1)
                for key, values in native.items()}
    terminal = np.asarray(arrays.get("label_is_extrapolated", ~has_successor), bool)
    allowed = np.isin(episode, allowed_episodes)
    shared = allowed & has_successor & ~terminal
    for valid in in_range.values():
        shared &= valid
    starts, lengths, segments = segment_metadata(episode, shared)
    results = {}
    for name, actions in zip(VARIANTS, (follower, leader), strict=True):
        result = {key: value.copy() for key, value in arrays.items()}
        result.update(action=actions.astype(np.float32),
                      source_ep_start=arrays["ep_start"].copy(), source_ep_len=arrays["ep_len"].copy(),
                      ep_start=starts.copy(), ep_len=lengths.copy(), segment_index=segments.copy(),
                      action_is_invalid=~shared, observation_is_invalid=~shared)
        results[name] = result
    summary = {
        "version": VERSION, "source_rows": n, "allowed_rows": int(allowed.sum()),
        "shared_valid_rows": int(shared.sum()), "valid_segments": int(segments.max() + 1),
        "valid_segment_length_median": float(np.median(lengths[shared & (starts == np.arange(n))])) if shared.any() else None,
        "exclusion_counts_overlap": {
            "no_successor": int((allowed & ~has_successor).sum()),
            "source_terminal_extrapolated": int((allowed & terminal).sum()),
            **{f"{key}_native_out_of_range": int((allowed & ~valid).sum()) for key, valid in in_range.items()},
        },
        "reference_clipping": False, "native_limits_rad": limits.tolist(),
        "mask_contract": "Both targets share exactly the same current/target native-range mask; excluded rows retain raw values and timestamps. Invalid observations never anchor a sample. Chunks stop at contiguous-valid-run boundaries.",
        "gripper_contract": "Current recorded leader gripper command in percent for both variants; never next observed gripper aperture.",
        "collision_scope": "Native bounds only. Collision, workspace and dynamic feasibility are not certified by this data mask; guarded inference validates independently.",
    }
    return results, summary
