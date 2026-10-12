# Data contract — observed source dataset

**Status:** AUDITED 2026-10-12 22:26 AEDT with `python -m real2sim.data.audit` (full decode of
every frame of both cameras; artifacts in `results/audit/`). Items marked VERIFY still need the
operator. The episode list for training is fixed by decision D-010 (pending).

Reproduce:

```
uv run python -m real2sim.data.audit \
  --repo-id bottowork/so101_act_pickplace_v1 \
  --revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 \
  --local-root ~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1 \
  --out results/audit --n-val 10 --seed 0
```

## 1. Source

| Field | Value |
|---|---|
| Repo | `bottowork/so101_act_pickplace_v1` (Hugging Face, public, not gated) |
| Pinned revision | `fd65a6600b0cd111a434ece1abfa3d44dc7b4d57` |
| Last modified | 2026-10-11T23:15:41Z |
| License | apache-2.0 (dataset card) |
| Format | LeRobot `codebase_version` `v3.0` |
| `robot_type` | `so_follower` |
| `fps` | 30 |
| Episodes / frames | 40 / 47,393 (26.3 min) |
| Tasks | 1: `"Pick up the object and place it in the target area"` (task_index 0 for all frames) |
| `splits` | `{"train": "0:40"}` (no validation split supplied) |
| Local byte-identical copy | `~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1` |
| Recording software | LeRobot v3.0 format; the operator's clone is `lerobot` commit `6156e82a` (0.6.2-dev). Exact version at record time: VERIFY with operator. |

The dataset card (`dataset.yaml` in the repo) states: iphone is the `primary_training_view`,
`top` is "mounted at a poor angle and should be excluded from ACT training", and "Successful
episodes were retained; follower tracking lag is expected from the configured PID controller."
These are operator claims; the per-episode outcome manifest is still required (see §6).

## 2. Non-image features

| Key | dtype | shape | names |
|---|---|---|---|
| `observation.state` | float32 | [6] | `shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos` |
| `action` | float32 | [6] | same six names (leader-arm goal positions sent to the follower) |
| `timestamp` | float32 | [1] | seconds from episode start |
| `frame_index`, `episode_index`, `index`, `task_index` | int64 | [1] | |

**Units:** joints are degrees, gripper is a 0–100 opening percentage. Basis: observed ranges
exceed ±100 (shoulder_lift reaches −105), which rules out LeRobot's `RANGE_M100_100` mode, and
`SOFollowerConfig.use_degrees` defaults to `True` in the operator's LeRobot clone. VERIFY with
operator's recording command.

**Joint order** matches the Menagerie `robotstudio_so101` MJCF joint names exactly
(`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`). Sign and zero
conventions between the real calibration and the MJCF are NOT yet verified (M2 FK test).

### Observed ranges (whole dataset, degrees / percent)

| Joint | state min | state max | action min | action max |
|---|---|---|---|---|
| shoulder_pan | −29.6 | 32.3 | −30.3 | 33.0 |
| shoulder_lift | −105.0 | 57.8 | −106.2 | 57.2 |
| elbow_flex | −44.9 | 96.8 | −72.9 | 96.7 |
| wrist_flex | −98.8 | 80.5 | −101.0 | 88.0 |
| wrist_roll | −27.3 | 23.0 | −27.9 | 24.4 |
| gripper | 1.15 | 83.6 | 0.0 | 84.0 |

### Action-versus-state audit (computed from all 47,393 rows)

| Joint | median \|a−s\| | p95 \|a−s\| | max \|a−s\| | best lag (frames @30 Hz) | residual at best lag |
|---|---|---|---|---|---|
| shoulder_pan | 0.48 | 2.77 | 12.1 | 5 | 0.39 |
| shoulder_lift | 1.85 | 64.4 | 139.0 | 16 | 12.95 |
| elbow_flex | 2.37 | 17.9 | 75.3 | 8 | 3.16 |
| wrist_flex | 1.49 | 15.3 | 73.0 | 7 | 2.15 |
| wrist_roll | 0.35 | 1.49 | 13.9 | 4 | 0.37 |
| gripper | 0.78 | 35.6 | 59.5 | 8 | 6.51 |

Interpretation (inferred, not asserted):
- Follower lags leader by roughly 0.15–0.5 s depending on joint, consistent with the P=12, I=0,
  D=2 setting used for recording.
- `shoulder_lift` has a systematic offset: mean action −31.8° vs mean state −18.7°, and a 13°
  residual that lag alone does not remove. The follower does not reach the leader's shoulder
  target under gravity load, or the two calibrations differ. Either way, **raw `action` is not
  achieved motion for this joint.**
- `elbow_flex` action minimum (−72.9°) is 28° beyond the state minimum (−44.9°): leader targets
  were commanded outside what the follower reached (clipping or limit).
- Gripper: 34 % of frames command < 5 (closed). While closed, the follower state is bimodal:
  ≈1.2 (closed on nothing) or ≈30–40 (closed on the ball). This is expected physical behaviour
  and is the observable "ball grasped" signal.
- No action values sit on their global min/max for more than 0.4 % of frames, so there is no
  evidence of hard software clamping in the saved `action`.
- Episode starts can have large leader/follower mismatch (episode 0 frame 0: wrist_flex action
  74° vs state 15.5°).

### Timing

`timestamp` steps are uniformly 1/30 s (min 0.033333, max 0.033337) in all 40 episodes;
`frame_index` starts at 0 and is contiguous in every episode. No duplicate or missing rows.
Within-episode per-frame state steps never exceed 4.9° (p99 ≤ 1.4°) on any joint, so there are
no encoder glitches or wrap-arounds in the recorded state.

### Episodes

| Statistic | Frames | Seconds |
|---|---|---|
| min | 841 | 28.0 |
| median | 1154.5 | 38.5 |
| mean | 1184.8 | 39.5 |
| max | 1794 | 59.8 |

Episodes 0 and 1 have exactly 1794 frames (59.8 s) and episode 4 has 1782; a 60 s recording cap
was probably hit. VERIFY with operator whether those episodes completed the task before the cap.

## 3. Image features

| Key | shape (H,W,C) | codec | fps | files | total size | mean bitrate | video frames | data rows |
|---|---|---|---|---|---|---|---|---|
| `observation.images.top` | 720x1280x3 | AV1 yuv420p, g=2, crf=30 | 30 | 4 | 101 MB | ≈0.5 Mbit/s | 47,393 | 47,393 |
| `observation.images.iphone` | 720x1280x3 | AV1 yuv420p, g=2, crf=30 | 30 | 9 | 1.29 GB | ≈6.8 Mbit/s | 47,376 | 47,393 |

There is **no** `observation.images.wrist` or any third image key.

### Per-camera quality (full decode, every frame, `results/audit/camera_quality.json`)

Metrics are computed on 320x180 downscaled frames. Thresholds are in the JSON next to the
verdicts.

| Camera | Decoded / expected | Failures | Stream verdict | Episodes USABLE / PARTIAL / UNUSABLE | Median sharpness | Ball detected (fraction of frames) | Image-after-joint lag (median) |
|---|---|---|---|---|---|---|---|
| `top` | 47,393 / 47,393 | 0 | **UNUSABLE** | 0 / 3 / 37 | 2.3 | 0.16 (false positives from the warm ceiling light) | +2 frames |
| `iphone` | 47,376 / 47,393 | 0 | **PARTIALLY_USABLE** (see below) | 37 / 3 / 0 | 604 | 1.00 | +2 frames (67 ms) |

`top` decodes perfectly but is aimed at the ceiling: 37 episodes fail the sharpness rule
(median Laplacian variance below 5) and the three that pass marginally (episodes 1, 25, 26 at
5.3–8.6) still show no workspace. It is a **misaimed camera, not a corrupt stream**, and is
excluded from training as a documented data issue. Evidence: `results/audit/camera_montage.html`.

`iphone` is sharp, never black/blank, has the ball visible in every frame of every episode and
a consistent +2-frame lag between joint motion and image motion (the image trails the state by
about 67 ms, which is normal for a phone stream and is the same in all healthy episodes). The
stream-level PARTIALLY_USABLE verdict comes **only** from three episodes whose video window
contains fewer frames than the data rows:

| Episode | Decoded / rows | Deficit | Motion lag (healthy median +2) | Split |
|---|---|---|---|---|
| 25 | 1,135 / 1,137 | −2 | 0 | train |
| 33 | 1,079 / 1,093 | −14 | 0 | train |
| 34 | 926 / 927 | −1 | −1 | validation |

The lag shift toward 0 is what a drop near the start of the episode produces, so the rest of
each affected episode is offset by up to the deficit (0.47 s for episode 33). LeRobot's
timestamp lookup does not raise on this; it returns neighbouring frames silently. Handling is
decision D-010 (recommended: exclude episode 33, keep 25 and 34 with the ≤ 2-frame shift noted).

Idle periods ("frozen" runs of ≥ 15 near-identical frames) cover 0–11 % of each iphone episode
and are expected for teleoperation pauses, not a defect.

## 4. Preliminary observation contract (not frozen)

- `camera_keys = ["observation.images.iphone"]` — audit-driven single-stream rule per plan §3.2:
  `top` is UNUSABLE, `iphone` is USABLE in 37/40 episodes. Final once D-010 fixes the handling
  of episodes 25/33/34 (the audit tool itself reports "requires user decision" for that reason).
- Model image tensor: candidate `[3, 192, 256]` from aspect-preserving resize of 16:9 source with
  letterbox to 4:3 (decision D-004 covers the alternative sizes). Padding geometry and scale
  factors are recorded alongside the checkpoint.
- Proprioception: raw 6-joint state (degrees, gripper %) for the joint baseline; canonical 13-float
  EEF state for the Cartesian policy (decision D-008).
- Action: raw 6-joint target for the joint baseline; canonical 7-D task-frame delta for the
  Cartesian policy; label semantics are decision D-003.
- Policy clock: 10 Hz resampled from the 30 Hz recording, episode boundaries preserved
  (decision D-007).

## 5. Split (generated, `results/audit/split.json`)

Deterministic whole-episode split, seed 0, 30 train / 10 validation (D-006 approved).
Validation episodes: 1, 2, 3, 4, 11, 18, 23, 24, 27, 34. Manifest SHA-256
`bb8bfb57499ce13346b5a5927f87dfaa21f4f70e6f7718f6a69e617486803c4a`. Will be regenerated with
the exclusion list once D-010 is approved (the manifest records the source revision). Validation
episodes never enter normalisation statistics or training.

## 6. Episode outcomes

No per-episode success labels exist in the dataset. `annotations/episode_outcomes.csv` with
`episode_id,success,reason,start_region,reviewer` must be filled by the operator; until then
every episode is `unknown`. The dataset card's claim that only successful episodes were retained
is recorded as an operator statement, not as a label.

## 7. Known gaps

- D-010 (episodes 25/33/34) pending; episode list and split not final until then.
- Dataset license is recorded from the Hugging Face card (apache-2.0); the local copy has no
  README, so `dataset_report.json` shows `license: null`.
- The ball detector is a plain HSV threshold; it is a visibility signal for the audit, not a
  perception module, and produces false positives on warm light sources (see `top`).
- Real/sim joint sign and zero conventions unverified.
- Physical measurements (ball, bowl, mat, phone pose) unknown; see `docs/calibration.md`.
- Operator confirmations outstanding: joint units (`use_degrees`), per-episode outcomes, the
  three ≈60 s episodes (0, 1, 4).
