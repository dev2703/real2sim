# Executable RGB pick-and-place baseline — 26 October 2026

The repaired RGB waypoint controller completes **4/5 SO-101 and 4/5 Panda
pick-and-place trials** on seeds 0–4. These are geometric controller results in
MuJoCo, not learned ACT success or frozen ACT transfer. The final
[machine-readable report](../results/v2/rgb_waypoint_results_20261026.json)
links every full report, source hash, requested/applied/readback trace and video.

| Controller / robot | Held lift → released resting placement | Remaining failure | Maximum robot/table penetration |
| --- | --- | --- | --- |
| RGB waypoints / SO-101 | 4/5 | Seed 1: no grasp branch passes sampled native feasibility checks | 0.558 mm |
| RGB waypoints / Panda, 8 mm seated grasp | 4/5 | Seed 4: RGB targets unavailable in the initial view | 0 mm |
| Earlier Panda, ball at pad centre | 0/5 | Four lifted balls slip during transport; one missing detection | 0 mm |

The scorer is the existing `TrialTracker`: a held lift exceeding 30 mm, a release
over the bowl, then an unheld ball resting in the bowl for one second. All final
successes satisfy this sequence. Both runs retain their configured initial joint
states and the full episode horizon. Neither run settles, clips or replaces the
start pose. There are no aborts, command saturations or final-run IK rejections.
The failed seeds remain in the denominator and their traces run to timeout.

## Concrete fixes and what they establish

`scripts/run_visual_waypoint.py` referenced a missing module, so it could not run.
The new `real2sim.policy.visual_waypoint` implements its intended strategy: obtain
ball/bowl goals from RGB and supplied camera geometry, select a feasible grasp
branch, then execute bounded joint segments with joint-readback feedback. IK is
used to plan grasp branches and phase endpoints. FK checks sampled paths and each
motion endpoint against the supplied scene map. Successful SO-101 trials use
41–49 IK calls over 102–108 control ticks; successful Panda trials use six calls
over 76–80 ticks. Most SO-101 solves occur in the initial branch search.

The ball frontend originally included the warm wood background in its hue mask.
For the actual initial scene this joined the ball to an image-border contour and
discarded it as an invalid silhouette. Excluding wood hues restores the ball
measurement. The shallow bowl's visible sidewall required a small, explicit
metric contour allowance; position still comes from the observed contour, with
no nominal object-centre fallback. Missing/ambiguous targets fail closed.

Panda's pads are short and its between-pad TCP is a different physical point from
SO-101's fixed-jaw-tip site. With the sphere centred at the Panda pad centre,
gravity progressively moves it below the pads during lift/transport. The retained
seed-0 baseline trace loses contact at tick 51, with the sphere about 30 mm below
the TCP. Placing the pad centre **8 mm below the RGB ball centre** seats the
sphere before lifting; the same four visible seeds then place successfully.
This changes a declared grasp parameter, without changing contacts, friction,
gripper force, actuator gains, gravity or the success definition. It is a
simulator grasp-depth result and needs physical jaw/TCP verification.

## Geometry and scope

Both final configs use `task.scalar_measured_20261026.yaml`, including the
operator's measured ball radius of 32.34 mm. The new base-to-ball and
ball-to-bowl distances contradict the old scene placement, and their task-frame
directions remain unresolved. The camera/base registration therefore remains
provisional. These successes do not validate physical calibration or explain
away that discrepancy.

Object **goals** come from RGB. The controller also receives camera projection,
object dimensions/support height and the simulator's static collision map,
including exact bowl/table collision geometry. It does not use dynamic object
poses, measured object contacts, held flags or success metrics to select goals or
advance phases. Collision checks use virtual contacts in that supplied map; this
is not map-free control. The sphere/tool offsets are explicit grasp assumptions.
SO-101 carry orientation is intentionally flexible because its five arm joints
cannot realize every six-dimensional pose. Ball-waypoint planning still requires
the original 3 mm position tolerance. Native joint, velocity, workspace and 2 mm
penetration gates remain in force. Sampled geometry checks do not certify every
continuous swept path or physical loaded response.

The runner logs requested native action, accepted/applied commands and achieved
joint readbacks. Panda action commands contain seven arm targets plus total
opening; Panda readbacks contain seven arm joints plus **two individual** finger
slide positions. Mean per-tick times including planning/rendering are about
22 ms (SO-101) and 17 ms (Panda). These are local simulator times, not measured
camera-exposure-to-motor latency.

## Reproduction and validation

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 \
  --seeds 0 1 2 3 4 --video --out outputs/v2/rgb_so101_NEW
.venv/bin/python scripts/run_visual_waypoint.py --robot panda \
  --seeds 0 1 2 3 4 --video --out outputs/v2/rgb_panda_NEW
.venv/bin/pytest -q tests/unit/test_visual_targets.py \
  tests/integration/test_visual_waypoint.py
```

The checked-in default configs are
`configs/v2/sim_so101.waypoint_20261026.yaml` and
`configs/v2/sim_panda.waypoint_20261026.yaml`. They do not require ignored output
configs or the nonexistent `task.calibrated.yaml`. All eight focused tests pass,
including forbidden privileged-metric access, missing-target behavior, sparse IK
execution, synthetic shifted targets and the wood-background regression. Ruff
passes for the modified implementation, runner and tests. Final reports have no
runtime source changes.

Successful seed-0 videos:
[SO-101](../outputs/v2/rgb_waypoint_so101_20261026_verified/seed_000/rollout.mp4),
[Panda](../outputs/v2/rgb_waypoint_panda_20261026_verified/seed_000/rollout.mp4).
This establishes an executable image-guided control baseline for the scene.
ACT fitting, recovery and transfer of frozen learned weights remain separate work.

## Actual final ACT execution comparison

The final step-2,000 four-state-history ACT checkpoint was also executed in
three complete 30-second, seed-0 development rollouts. These use its matching
**original provisional scene and 33.5 mm ball radius**, separately from the
32.34 mm measured-radius geometric baseline above. No training gate was treated
as a substitute for running the policy. The
[ACT execution report](../results/v2/history4_execution_20261026.json) preserves
the configs, videos, full requested/applied/achieved traces and checkpoint hashes.

| Execution of the same frozen ACT | Outcome | Closest tool/ball distance | Collision-gate rejections / 375 ticks | Mean full control step |
| --- | --- | --- | --- | --- |
| Two actions per inference | Approach failure, no held lift | 250.39 mm | 326 | 20.97 ms |
| Fresh inference every tick | Approach failure, no held lift | 238.99 mm | 319 | 29.27 ms |
| Every tick plus aligned temporal ensemble, coefficient 0.01 | Approach failure, no held lift | 241.03 mm | 329 | 28.94 ms |

All modes retain persistent accepted-target integration and the same native IK,
joint, velocity, workspace and contact gates. Initial joints, tool poses and RGB
image hashes match exactly. Each trial uses all 375 control ticks. Ensembling
uses only the **two supervised action positions**; the other 23 predictions in
the chunk never enter the average. The checkpoint hash remains
`c6ee5facfcd8d4240af56cca395074848cf89d33777fd680c0eea637fe77fd3b`.
No source or checkpoint changed during execution, and no checkpoint is promoted.

The policy initially moves near the base and then repeatedly requests targets
that fail the supplied scene's collision gate. First rejection occurs at tick
50, 56 and 47 respectively. Maximum achieved robot/table penetration remains
0.60–0.66 mm, below the unchanged 2 mm gate. These provisional-scene rejections
do not establish real hardware collisions. More frequent feedback and averaging
alone do not repair this checkpoint's approach behavior; successful geometric
control does not establish learned object-directed behavior.

Steady policy inference means are 16.27, 16.17 and 15.92 ms. Full control-step
p95 values, including action execution and rendering the next observation, are
31.32, 31.32 and 31.05 ms. These local simulated host timings fit within an
80 ms control interval; physical acquisition-to-motor delay remains unmeasured.
This is one shared development seed, with no reserved episodes, rather than
generalization evidence or a frozen-Panda-transfer result.

Reproduce this execution comparison with:

```sh
.venv/bin/python scripts/diag_v2_history_execution.py \
  --checkpoint outputs/v2/three_episode_history4_20261026/training/checkpoints/002000 \
  --out outputs/v2/history4_execution_NEW
```
