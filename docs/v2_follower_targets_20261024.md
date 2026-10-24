# Next-achieved joint targets — 24 October 2026

Next-follower joint supervision improves the common raw FK command error in this
short pilot, from **32.114 to 24.579 mm**, but both models remain worse than the
recorded-current hold reference of **5.434 mm**. Both fail all three bounded
approach trials. This does not repair the visual approach failure or establish a
checkpoint suitable for reserved evaluation.

The [preparation evidence](../results/v2/follower_joint_targets_20261024.json) and
[pilot/rollout report](../results/v2/follower_joint_pilot_20261024.json) retain source,
task, checkpoint and script hashes. Full rows and per-tick traces remain under
`outputs/v2/follower_joint_*_20261024`.

## Targets, invalid observations and time

Both variants consume the same recorded images and unchanged canonical 13-field
proprioception. They output absolute arm joint angles in degrees and gripper
commands in percent. The experimental variant targets the **next observed follower
arm state** at the 12.5 Hz policy clock, with the **current recorded leader gripper
command**. The matched control retains the current recorded leader arm command.
Next measured gripper aperture is never substituted for the gripper command.

Both variants use the same mask: current observed native joints and both target
definitions must lie inside the exact pinned six-hinge ranges. Original episode
terminal/extrapolation flags remain distinct. References are never clipped.
Native-invalid observations do not anchor training samples. Invalid gaps split
chunks into contiguous valid runs, with original episode identities, row indices,
frame indices and timestamps unchanged. Original chunk metadata is retained as
`source_ep_start` and `source_ep_len`. This avoids both impossible starting contexts
and a false inference timeline compressed across invalid gaps.

| Partition | Original ticks | Shared retained ticks | Retained |
|---|---:|---:|---:|
| Training | 9,617 | 6,362 | 66.15% |
| Validation | 2,077 | 1,427 | 68.70% |

There are 38 valid runs with median length 215 ticks. Validation retention is
68.16% for baseline, 72.43% for ball-shift and 67.72% for bowl-shift episodes.
These exclusions change the represented distribution; this is not the original
full-task validation score. Reserved rows remain in the arrays but enter neither
normalization nor policy evaluation. State statistics use shared-valid training
rows. Both action normalizers are identical, fitted to the pooled paired training
targets only. Raw joint observations, images and ticks are equal between variants.

The checkpoint saves explicit `joint_target_semantics`, including timing, units,
gripper meaning and the guarded inference interface. Initial checkpoint transfer
and resume reject changed semantics. The existing generic joint controller remains
unchanged for older contracts.

## Static feasibility exposes a separate geometry bottleneck

On the same 1,427 native-compatible validation rows, original recorded leader
commands differ from next achieved follower FK by **17.623 mm** before rate bounds.
Next-follower targets have zero discrepancy by construction under the same FK;
this is a label definition, not independent evidence that the FK is physically right.

A separate diagnostic checks the original provisional scene's native ranges,
workspace and collision rule. It tests requested endpoints and 17 static samples
along the rate-bounded joint-path proxies. The native-only training mask is not
changed after these outcomes.

| Recorded target definition | Accepted | Endpoint collision | Workspace | Path collision | Mean next-follower FK error after rejection/rate bounds |
|---|---:|---:|---:|---:|---:|
| Next follower arm/current leader gripper | 1,153 | 206 | 60 | 8 | 0.580 mm |
| Current leader command | 1,159 | 142 | 28 | 98 | 15.676 mm |

Thus **19.20%** of the recorded next-follower targets fail a configured scene gate:
**15.00%** have collision reasons and **4.20%** fall outside the configured workspace.
Recorded-achieved motion being incompatible with this provisional scene is a
geometry/control consistency problem. It does not prove that the real recording
collided, and it does not justify relaxing the gates or silently training only on
the scene's preferred targets. Native compatibility alone does not establish
collision or dynamic feasibility.

These are teacher-forced static checks: command ramps start at observed current
joints because executed servo targets were not recorded. Collision sampling is
not continuous collision certification. The full row-level evidence is
`outputs/v2/follower_joint_label_feasibility_20261024.json`.

## Matched training and common physical-space proxies

Each model receives 256 updates at effective batch 16, **4,096 training examples**,
seed 0, the same frozen pretrained ResNet18 ACT architecture and schedule, identical
normalization and sampled contexts. Checkpoints are shortlisted using the shared
executed-prefix protocol, then compared on the same achieved-follower reference.
Own-target normalized losses are not treated as a physical controller ranking.

| Common next-follower FK proxy, 1,427 actions | Leader-command supervision | Next-follower supervision |
|---|---:|---:|
| Mean translation error | 32.114 mm | 24.579 mm |
| Translation p95 | 62.559 mm | 48.450 mm |
| Mean orientation error | 0.12138 rad | 0.08676 rad |
| Positive movement direction | 78.94% | 78.32% |
| Median command-motion/label magnitude | 6.118 | 4.674 |
| Stationary target displacement | 31.769 mm | 23.578 mm |
| Raw predicted native-invalid actions | 191 | 164 |
| Current-leader gripper error | 4.463 percentage points | 4.232 percentage points |

These raw commands are scored before guarded rejection, rate limits or physics;
none is clipped for reporting. The large errors and overshoot show substantial
short-budget underfitting. The new target meaning alone is insufficient. All
Cartesian quantities remain conditional on the provisional FK mapping.
Next-achieved joint positions are desired future setpoints, not the actuator
commands that caused the recording. Passing them to the unchanged native servo
does not establish matched physical tracking or compensate an unidentified lag;
that requires measured response and closed-loop comparison.

## Guarded bounded rollouts

Only the explicit new contract uses `GuardedJointActionAdapter`. It preserves
native joint ranges, workspace, penetration tolerance and the existing collision
exceptions. It checks the requested endpoint and both observed-joint-to-applied
and previous-command-to-applied paths. The rate-bounded endpoint is computed about
the previous native actuator command, matching the unchanged simulator servo.
Rejected requests hold observed arm and gripper joints. No IK pose residual is
reinterpreted: these are absolute joint targets with their exact coupled FK pose.

Both candidates then run on three common seeds for 125 ticks/10 seconds, using
the existing strict recorded episode-9/frame-100 reset and original scene.
Their initial arm, tool and ball states match exactly. Joint policies use the
explicit `current_pose` evaluation setting; no Cartesian target integration is
introduced. Ground-truth objects are used only for scoring.

| Seed | Closest distance, leader-command model | Next-follower model |
|---|---:|---:|
| 0 | 274.74 mm | 278.98 mm |
| 1 | 265.17 mm | 269.01 mm |
| 2 | 266.37 mm | 270.48 mm |

Both are **0/3**, fail approach, and show no lift or release. There are zero guard
rejections, numerical aborts or robot/table penetrations during these six short
trials. NoSlip iterations remain zero. Source hashes stay unchanged throughout
the 750 executed ticks. Neither checkpoint advances to reserved evaluation.

## Verification and reproduction

Nine new unit regressions cover target timing, shared masks, exact segment clocks,
missing-successor handling, finite evidence and unchanged native/path gates. A
real MuJoCo integration check accepts a hold, rejects a 10,000-degree request,
and verifies that joint ranges and collision geometry remain unchanged. The
combined focused unit suite passes 68 tests; the new physics integration passes.
Ruff and whitespace checks pass.

Use fresh paths:

```sh
.venv/bin/python scripts/prepare_v2_follower_joint_targets.py \
  --out .cache/v2_follower_joint_NEXT
.venv/bin/python scripts/diag_v2_follower_joint_labels.py \
  --data .cache/v2_follower_joint_NEXT \
  --out outputs/v2/follower_joint_labels_NEXT.json
.venv/bin/python scripts/run_v2_follower_joint_pilot.py \
  --data .cache/v2_follower_joint_NEXT --steps 256 \
  --out outputs/v2/follower_joint_pilot_NEXT
.venv/bin/python scripts/diag_v2_follower_joint_rollouts.py \
  --pilot outputs/v2/follower_joint_pilot_NEXT \
  --out outputs/v2/follower_joint_rollouts_NEXT
```
