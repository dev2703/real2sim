# Calibration, execution and training status — 26 October 2026

**Measurements and task recordings are available.** The earlier missing-input
conclusion is superseded. `physical_measurements.md` is now ingested with its
inch values, ±5% bounds and source hash. The local task dataset is
`/Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2`:
41 episodes, 27,736 frames, 25 Hz. A second `so101_act_pickplace_v2_25fps`
folder contains one episode/467 frames. The DAgger rollout folder contains zero
episodes; the collector's `traces` folder is empty. Existing task recordings
are usable, even though independent optical and loaded timing captures are absent.

## Actual control result

The repaired RGB waypoint runner completes **4/5 SO-101 and 4/5 Panda
pick-and-place trials**, including held lift, release over the bowl and one second
of unheld resting placement. It uses the measured ball radius, supplied scene
calibration/collision geometry, sparse IK planning and bounded joint feedback.
These are geometric-controller results, not learned ACT or frozen-weight transfer.
The [rollout report](v2_waypoint_results_20261026.md) links videos, every failed
seed, command/readback traces and reproduction commands.

Concrete fixes include the missing controller module, a wood-background contour
that hid the ball, feasible five-joint carry paths and Panda's grasp depth. Seating
the Panda pads 8 mm below the ball centre changes its result from 0/5 to 4/5
without contact-physics or force changes. Successful final runs stay below the
original 2 mm robot/table penetration threshold.

## Root causes supported by local evidence

* **Scene placement disagrees with the new measurements.** Existing ball–bowl
  spacing is 242.65 mm versus reported 304.8 ± 15.24 mm; base–ball planar distance
  is 410.40 mm versus 334.08 ± 16.70 mm. These comparisons require the same
  measurement references and recording layout. Signed axes remain unresolved.
  Changing a network's learning rate cannot repair this registration discrepancy.
* **Recorded joint zero and native limits differ.** 32.94% of training/development
  readbacks violate at least one native bound; shoulder lift is involved in 91.04%
  of those violations. Its recorded minimum is −105.143° against a −100° native
  limit. Degree units are supported by encoder arithmetic; LeRobot degree zero
  comes from calibration range midpoints, which do not independently define CAD
  neutral. No limits were expanded or arbitrary offsets fitted to remove violations.
* **Grasp frames differ between embodiments.** Panda pad seating affects loaded
  retention; equal seven-element action vectors do not establish equal physical
  TCP/grasp meaning. The successful geometric adapter provides a concrete
  reference for testing learned execution.
* **The temporal-ensemble setting was bypassed.** `ChunkExecutor` called
  `predict_action_chunk` and queued outputs, bypassing upstream `select_action`
  ensembling. It now ensembles predictions for the same absolute control tick,
  honours the saved supervised prefix and clears history at reset. Prefix-two
  checkpoints never average their untrained future tail. Execution overrides are
  reported as ablations; original checkpoint weights/contracts remain unchanged.

The [measurement report](v2_measurements_20261026.md) records conversions,
contradictions, normalization evidence and unchanged FK scope. The
[primary-source research](v2_external_research_20261026.md) compares relevant
SO-101, ACT, real-time execution and task-space transfer implementations.

The operator's subsequent reply confirms camera height **406.4 mm above the
tabletop**. A separate height candidate preserves that fact and its uncertainty;
the previous provisional camera was 533.4 mm above the table. Changing height
alone with the old orientation breaks both RGB detections in the five tested
views, so it is not substituted into the successful baseline or called a complete
camera calibration. Horizontal signs/orientation and the task/base datum remain
unresolved. The verbatim clarification and revised measurement lineage are linked
from the measurement report.

## Training and timing

A bounded history-four ACT run completed at
`outputs/v2/three_episode_history4_20261026`: **2,000 updates/32,000 examples**,
the same three training episodes, original labels and two-action supervision.
The [comparison](../results/v2/history4_capacity_20261026.json) records a small
training-fit translation improvement, **1.895 → 1.837 mm**, and moving-vector
error **2.316 → 2.245 mm**. Mean stationary-run error worsens **1.272 → 1.299 mm**;
the unchanged per-condition capacity gate still fails. This is not a successful
learned pick-and-place policy or a generalization result.
It tests causal temporal context, not a ball-size-induced FK change. Ball radius
does not alter arm FK labels, so relabelling identical kinematics would not be
calibrated retraining. Original training dependencies are retained in
`training_source_snapshot` before the separate runtime executor fix.

Labelled development rollouts diagnose reach/grasp/transport/release even if the
offline capacity criterion fails; that criterion is not a reason to prohibit
testing. Reserved episodes remain separate. Policy-call and full simulation-step
timings are reported separately. Full-step timing includes next-observation
rendering; it is not physical exposure-to-motor timing.

The final checkpoint was actually tested for 375 feedback ticks/30 simulated
seconds per execution setting on the same development seed and identical initial
images/readbacks. Queue-two, queue-one and temporal ensemble 0.01 all fail during
approach, with closest ball distances of **250, 239 and 241 mm** respectively.
They incur **326, 319 and 329** configured collision-gate rejections, with no held
lift. These provisional-scene rejections do not establish real-world collisions.
The [execution comparison](../results/v2/history4_execution_20261026.json) retains
all traces, unchanged checkpoint/source hashes, and full-step timings. Faster
feedback/ensembling alone does not repair this learned trajectory. Geometric
controller success is retained separately and is not attributed to ACT.

Validation: 195 focused unit/integration checks passed across measurement/frame
conversion, RGB geometry, history, execution/reset guards, ensembling, timing-trace
software and both Cartesian adapters. Ruff and whitespace checks pass. This test
count excludes the actual rollout trials, which are scored above.

## Remaining physical information

The most useful next inputs are the signed length/width directions and exact
base reference used for `(2,13)`, centre/edge references for the ball–bowl spacing,
inside versus outside bowl diameter and bowl depth, jaw gaps at known gripper
readbacks, and a timestamped loaded hold/release. Camera identity/settings and
any movement since recording establish whether new measurements apply to old
footage. Those inputs resolve physical alignment and response; they do not block
the execution and training work above.

The combined [session log](../configs/v2/calibration_capture_20261026.template.json)
remains available for timing, loaded response and optical/base/joint observations.
Full physical calibration, recovery and frozen ACT transfer are not established.
