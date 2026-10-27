# Successful-motion reconstruction — 27 October 2026

All five declared replay variants complete held lift, release over the bowl and
settled placement on **5/5 known-successful SO-101 source demonstrations**.
The existing shared XYZ plus binary-gripper interface can express these successful
motions. Its earlier learned-policy failures therefore need a separate investigation
of predicted commands, feedback drift and execution timing. Discarded orientation
alone is not established as the cause of those failures.

These are **simulation-teacher command replays, with no ACT inference**. No model
was trained or changed. They are positive controls for an action interface, not
additional learned-policy successes, unseen-layout evaluation or physical validation.

The [curated evidence](../results/v2/motion_interface_20261027.json) links all
25 videos, per-tick requested/applied/achieved commands, source hashes and separate
tracking measurements. The complete [execution report](../outputs/v2/motion_interface_20261027/report.json)
and [tracking analysis](../outputs/v2/motion_interface_20261027/tracking_analysis.json)
remain available.

| Replay variant, same five source episodes | Held lift | Complete placement | Guard rejections | Peak robot/table penetration |
| --- | ---: | ---: | ---: | ---: |
| Original accepted native joint commands | 5/5 | 5/5 | 73 | 0.428 mm |
| Current shared XYZ, fixed jaw orientation, binary intent | 5/5 | 5/5 | 0 | 0.601 mm |
| Same XYZ, source-orientation reference, binary intent | 5/5 | 5/5 | 1 | 0.338 mm |
| Same XYZ, fixed orientation, continuous source gripper | 5/5 | 5/5 | 0 | 0.601 mm |
| Same XYZ, source-orientation reference, continuous gripper | 5/5 | 5/5 | 2 | 0.133 mm |

Source seeds **100–104** are the first five successful training episodes in the
retained source manifest, selected before execution. Each guarded native positive control
completes at its recorded source length: 101, 102, 104, 102 and 105 ticks. The native
guard rejections occur during late wrist-flex readback overshoot beyond the 95°
native bound: at most **0.001925°**, with 14/14/16/12/17 rejected ticks by seed.
They are retained; no bound, velocity cap or guard was relaxed. The original
waypoint teacher checked native FK endpoints and planned paths, but did not run
the learned-command adapter's strict current-native-limit check at every tick.
Thus this is a guarded source-command replay, rather than a bit-exact reproduction
of the teacher controller. Native replay's achieved grasp point differs from the
recorded source successor by at most 2.89 mm.

## Tracking fidelity and what the orientation reference changes

Every orientation reference is the source command's jaw rotation, passed into the
same **soft 0.002 m/rad least-squares term**. Cartesian solves start from the live
measured robot, never from the source joint solution. Source joint commands supply
only the declared Cartesian, orientation and jaw references. The current interface
uses the original fixed orientation and receives no source rotation.

| Pooled tracking measurement | Current XYZ + binary | Source orientation + binary |
| --- | ---: | ---: |
| Static accepted full-command grasp error against bounded XYZ, mean / p95 | 0.00165 / 0.00675 mm | 0.000241 / 0.000641 mm |
| Achieved grasp error against recorded source successor, mean / p95 | 18.32 / 59.57 mm | 0.621 / 4.116 mm |
| Achieved jaw rotation error against recorded source successor, mean / p95 | 0.136 / 0.495 rad | 0.00633 / 0.0178 rad |

Static accepted-command IK residual and achieved tracking error measure different
things. The latter includes the existing 24 mm Cartesian step bound, joint servo
rate limiting, contact dynamics and changed jaw orientation. It must not be called
an IK error. Rejected IK candidates are unavailable in the original adapter API;
static IK residual statistics include accepted commands only. Source-orientation
binary replay's maximum full-command rotation error is 0.02594 rad, within the
existing 0.35 rad SO-101 orientation tolerance. Rotation was still a soft reference,
not an enforced exact constraint.

Orientation references improve trajectory fidelity, but all variants have the same
five-trial placement count. Continuous source gripper commands also show no placement
benefit in this small reconstruction test. Neither result establishes an improvement
for frozen ACT execution; the current semantic policy does not predict jaw rotation.

## Fixed execution contract

The scene and task are the original teacher configuration, checked against the
manifest's task hash. Reset joints, seeded ball/bowl layouts, 12.5 Hz commands,
500 Hz physics, actuator forces, friction, native limits, workspace, 17 static
joint-path samples and the 2 mm penetration gate remain unchanged. No settling
reset, per-seed parameter selection or new task-directed waypoint is used. The
source's accepted servo command is distinct from its achieved successor readback.
Binary intent comes from the original requested gripper command (45% threshold),
then maps to the existing 90%/0% endpoints. Continuous replay retains the source's
accepted hinge setpoint. Failures would run for the original 30-second horizon;
the final source request is held after the successful source recording terminates.

All runtime dependency hashes agree before and after, and both frozen ACT weight
files remain unchanged. Three focused integration regressions pass: explicit grasp
offset/source-command semantics, numerical parity with the original binary adapter
at a binary endpoint, and retention/rejection of continuous jaw requests. Ruff passes.
All 25 MP4s decode without errors; the five-variant seed-100 preview was inspected.

Reproduce using a fresh output path:

```sh
.venv/bin/python scripts/diag_v2_motion_interface_20261027.py \
  --out outputs/v2/motion_interface_NEW --seeds 100 101 102 103 104
.venv/bin/pytest -q tests/integration/test_motion_interface_20261027.py
```

Representative videos:
[current shared interface](../outputs/v2/motion_interface_20261027/xyz_binary/seed_100/rollout.mp4),
[source orientation reference](../outputs/v2/motion_interface_20261027/orientation_binary/seed_100/rollout.mp4),
[native positive control](../outputs/v2/motion_interface_20261027/native/seed_100/rollout.mp4).
