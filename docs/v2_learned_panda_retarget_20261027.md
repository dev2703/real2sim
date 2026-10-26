# Frozen learned motions on Panda — 27 October 2026

The frozen **simulation-teacher ACT** supplies joint-command trajectories that
complete held lift, release over the bowl and settled placement on Panda in
**2/5 original development layouts**. This uses an explicit robot adapter and a
recorded learned-source trajectory. It does **not** establish Panda-image
closed-loop policy transfer: that separate test fails approach in **0/2** trials.
This checkpoint is the new matched-scene ACT, not the user's longer-trained
`bottowork/so101-act-h256-b16` checkpoint.

The [curated evidence](../results/v2/learned_panda_retarget_20261027.json) links
every full report, video, accepted-command trace, source evaluation and hash.
The checkpoint remains
`01d13f93aebcbe68779b482cde0ecaa26e971f65f2c9efc39160f66aa0f037e5`.

| Fixed source checkpoint / Panda execution | Held lift | Released resting placement | Successful seeds |
| --- | ---: | ---: | --- |
| Recorded learned trajectory, 8 mm pad seating | 3/5 | 2/5 | 300, 302 |
| Same trajectories, 12 mm seating comparison | 3/5 | 2/5 | 301, 302 |
| Same trajectories, 8 mm seating and fixed 1.25× slower playback | 3/5 | 2/5 | 300, 302 |
| Actual Panda RGB plus virtual SO-101 joint observations | 0/2 | 0/2 | None; seeds 100, 300 |

All five source development seeds 300–304 remain in each trajectory denominator,
including the failed SO-101 source trajectories. No best seating value is chosen
independently for each seed. The seating/retiming comparisons use development
outcomes and do not improve the aggregate. All trials retain the original
30-second horizon and unchanged success scorer. Robot/table penetration is zero
in these Panda runs. Rejected workspace requests remain in the traces.

## What the adapter transfers

The source ACT predicts five SO-101 arm targets plus a gripper command. Panda
cannot execute that vector directly. The adapter uses SO-101 FK to convert each
accepted learned command into an equivalent sphere grasp point:

`p_grasp = p_SO101_TCP + R_SO101_TCP * [-10, 0, 30] mm`.

It then places Panda's between-pad TCP under that point, using its declared
8 mm seating offset and fixed top-down orientation. Panda IK converts the pose
to seven native arm targets, under the original 3 mm position, 0.05 rad rotation,
joint/workspace and 2 mm penetration gates. Original actuator forces, gravity,
friction and contact physics are retained. SO-101 rotational motion is not copied.

Panda's original mounting pose cannot represent the folded SO-101 start. A
declared robot-only search over base translations chooses **−150 mm along task x**,
the first candidate that solves the source start. Object goals and task outcomes
do not choose that placement. The starting joints are replaced by this mapped
source start and this is recorded explicitly. This is a placement/kinematic
engineering adapter, not an unchanged-reset zero-shot test.

Source jaw CAD separation can exceed Panda's 80 mm maximum. Clipping that metric
width delays Panda closure until the source has already begun lifting: the first
metric-width test loses the ball during transport. Mapping command intent
**0% closed / 90% open → 0 / 80 mm** gives Panda time to close during the source
close segment. This change uses only the learned gripper command, with no held
flag, object pose or task-phase timing. Measured gripper readbacks remain a
separate, explicitly lossy CAD-gap projection in the live-feedback test.

The trajectory test performs no ACT inference from Panda observations. It reads
accepted commands from a prior unassisted, live SO-101 ACT rollout; the source
report must identify the same checkpoint and no teacher/IK corrections. Both
trace and source report are hashed. No scripted waypoint actions or recorded
human demonstration actions supply Panda's motion. Panda follows those learned
motions with its own joint feedback and IK adapter.

## What remains unestablished

In the actual Panda-image test, Panda's pose/opening are projected to virtual
SO-101 joint observations and its own RGB image is fed to frozen ACT. All
375 Panda IK requests per trial are accepted, but neither trial approaches the
ball successfully. The input projection is lossy: multiple SO-101 configurations
share one grasp point, and jaw readback percentages are not common physical
coordinates across these grippers. This experiment does not isolate appearance
from proprioception mismatch. It establishes neither frozen visual transfer nor
an embodiment-independent learned state/action interface.

The 8/12 mm seating tradeoff and transport slip leave loaded grasp robustness
unresolved. The next learned interface should use an explicit common grasp point,
object-relative goals and causal gripper intent rather than native joint vectors.
Physical camera/base/jaw calibration remains unaccepted.

## Reproduce

Use fresh output paths:

```sh
.venv/bin/python scripts/diag_v2_learned_panda_retarget_20261027.py \
  --mode trajectory --seeds 300 \
  --trace outputs/v2/bounded_gripper_act_20261027/development/seed_300/ticks.jsonl \
  --out outputs/v2/panda_learned_motion_NEW
.venv/bin/python scripts/diag_v2_learned_panda_retarget_20261027.py \
  --mode live --seeds 100 300 --out outputs/v2/panda_live_RGB_NEW
.venv/bin/pytest -q tests/integration/test_learned_panda_retarget_20261027.py
```

Three focused tests pass for distinct tool-origin compensation, gripper endpoint
meaning and rejection of teacher/mismatched-checkpoint trajectory provenance.
Ruff passes. Every scored report has unchanged source/checkpoint hashes; executed
script snapshots preserve the compared adapter revisions.

Successful source-300 Panda video:
[learned-motion retarget](../outputs/v2/learned_panda_trajectory_20261027_verified/seed_source_300/seed_300/rollout.mp4).
