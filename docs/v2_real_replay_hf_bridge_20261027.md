# Real-data replay and frozen ACT retest — 27 October 2026

The user's real-demonstration ACT weights are used unchanged throughout this
experiment. Both published root and `latest/` checkpoints are tested in evaluation
mode, with gradients disabled and their saved native processors applied once.
The [curated evidence](../results/v2/real_replay_hf_bridge_20261027.json) links
the frozen plan, full traces, paired images, videos and source hashes.

The experiment exposes two problems before a reliable real-to-simulation transfer
claim is possible: recorded real motions cannot be reproduced cleanly under the
current registration, and rendered images materially change the frozen policy's
commands at identical observed joints. No independently accepted camera/base/joint
correction was available to adopt. These results retain the existing provisional
scene rather than selecting a calibration by task success.

## Recorded demonstration reconstruction

All six checkpoint validation episodes—4, 7, 18, 32, 38 and 40—start outside the
native model's joint bounds. Their first native-limit and strict table/support
compatible poses occur at frames 110, 127, 145, 110, 137 and 142 respectively
(4.40–5.80 seconds). No source state is clipped and no native limit is expanded.
Consequently **0/6 full original starts are replayable** under this mapping.

Two explicit suffix diagnostics are run from each compatible pose. The recorded
leader action is the first reference; the next recorded follower readback is a
separate, noncausal tracking reference. Each uses unchanged 25 Hz control,
500 Hz physics, joint rate limits and sampled native/path/collision gates.

| Reference, six suffixes each | Commands accepted | Rejected | Aborted runs |
| --- | ---: | ---: | ---: |
| Recorded leader commands | 2,133/3,025 (70.5%) | 892 | 0/6 |
| Next recorded follower readbacks | 1,815/3,025 (60.0%) | 1,210 | 0/6 |

Leader-reference rejections include 417 native-limit, 466 target-collision and
nine sampled path-collision requests. The weighted shoulder/wrist source-readback
errors are 16.25°/21.07°. These are not pure servo-tracking errors: incompatible
references and their rejection-induced holds contribute. All raw commands,
accepted/applied targets and achieved joints remain in the traces.

These suffixes begin with zero velocity and actuator targets equal to the observed
pose because the original physical servo command/history is unavailable. Simulated
objects use the provisional seed-0 layout, not the real episode's reconstructed
layout. Thus these twelve runs are controller/registration diagnostics and are
**not scored task-success reproductions**.

## Same joints, recorded versus rendered images

Three early compatible states per episode are fixed before inference: the strict
suffix start plus 0, 25 and 50 source frames. All 18 candidate states are usable.
For each checkpoint and state, only the two RGB inputs change; the raw six-joint
input is identical. The predicted first ten actions are compared with the recorded
leader commands. This gives 36 checkpoint/state pairs, 72 image-condition
predictions, plus 36 identical-input repetitions.

| Frozen checkpoint | Mean absolute error across five arm joints, recorded RGB | Rendered RGB | Mean shoulder action change when images are swapped |
| --- | ---: | ---: | ---: |
| Root | 2.94° | 6.12° | 14.14° |
| `latest/` | 2.86° | 6.13° | 15.30° |

All repeated identical-input predictions are exactly equal (maximum difference
0.0). No model randomness or weight update explains the paired difference.
The [first paired view](../outputs/v2/real_replay_hf_bridge_20261027/paired_images/episode_004_frame_0110_real_left_sim_right.jpg)
shows recorded images on the left, simulated images on the right, wrist camera
above and global camera below.

The image swap changes optics, rendering and object layout/appearance jointly;
it does not identify a causal physical camera correction. Recorded 1280×720 RGB
frames are directly bilinear-resized to 640×360 with OpenCV `INTER_LINEAR`, then
passed through the saved native preprocessing once. The historical training resize
kernel is not pinned in `experiment.json`. These early-state diagnostics therefore
do not reproduce the original whole-episode validation metric or prove grasping.

## Fresh closed-loop feedback screen

Each condition runs seed 0 for the full 750 ticks/30 simulated seconds with the
configured valid start. Saved-prefix execution uses ten actions per policy call;
the new comparison re-infers every action. No task goals or corrective IK enter
the frozen policy's commands.

| Frozen weights | Execution prefix | Held lift / placement | Closest tool–ball distance | Commands accepted |
| --- | ---: | ---: | ---: | ---: |
| Root | 10 (saved) | 0/1; 0/1 | 239.56 mm | 750/750 |
| Root | 1 | 0/1; 0/1 | 255.15 mm | 750/750 |
| `latest/` | 10 (saved) | 0/1; 0/1 | 183.34 mm | 750/750 |
| `latest/` | 1 | 0/1; 0/1 | 169.86 mm | 750/750 |

All four fail approach without a rejection stall, saturation or abort. More
frequent feedback does not repair the task in this one-seed screen; it establishes
no broader failure or success rate. Simulation pauses during host inference,
whose mean is approximately 110–113 ms. These are not hardware real-time results.
Both weight files and every recorded runtime input hash are unchanged.

## Reproduction and verification

```sh
.venv/bin/python -m scripts.diag_v2_real_replay_hf_bridge_20261027 \
  --out outputs/v2/real_replay_hf_bridge_NEW
.venv/bin/pytest -q tests/integration/test_real_replay_hf_bridge_20261027.py
```

Two focused integration checks pass: incompatible original references remain
unmodified and rejected, and the real episode's invalid full start remains distinct
from its explicit strict-compatible suffix. Ruff passes. All sixteen replay/live
MP4 files pass full FFmpeg decoding.

The executed runner finished every experiment, then a final hash-bookkeeping line
passed a string to a Path-only helper. Completion was recovered using
`sha256_file(Path(name))` before editing the runner, with the original runtime
hashes preserved. The authoritative [executed snapshot](../outputs/v2/real_replay_hf_bridge_20261027/executed_script.py)
and [post-run edit ledger](../outputs/v2/real_replay_hf_bridge_20261027/post_run_bookkeeping_fix.json)
record old/new hashes. The current runner fixes that finalization line, asserts
the checkpoint validation split, and includes episode metadata hashes in future
plans. Current-run metadata associations were separately hashed before paired
decoding and verified unchanged after all trials. No inference, motion or scoring
code changed during the run.
