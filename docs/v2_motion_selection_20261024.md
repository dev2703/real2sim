# Executed-prefix metrics and checkpoint selection — 24 October 2026

New v2 ACT and frozen-feature head training now share `executed_prefix_motion_v1` in
`real2sim.policy.validation`. The full-chunk ACT L1 remains a separate diagnostic.
The selection report scores episode-local anchors 0, 2, 4, … and the first two
predicted actions. It checks that every nonpadding, nonextrapolated action occurs
exactly once. Reserved episodes are rejected before image access or inference.

The report includes normalized executed-prefix L1, physical translation error in
millimetres, SO(3) rotation error in radians, gripper error as a fraction of its
calibrated range, and joint errors in degrees where applicable. It reports motion
direction and amplitude, stationary drift, per-episode errors, and causal hold and
persistence baselines. Cartesian metrics inherit the prepared FK calibration;
degrees cannot establish Cartesian accuracy for a joint policy.

Movement means translation norm above 0.4 mm per tick, or joint-target displacement
above 0.5 degrees in joint-vector norm. Stationary Cartesian actions have translation
at most 0.4 mm and rotation at most 0.01 rad per tick. Absolute targets are converted
to displacement from the recorded current pose for movement metrics. Tool vectors
are rotated into the base frame before integrating errors. Stationary sums use
contiguous runs within individual episodes; they are teacher-forced drift proxies.

Both baseline actions use the gripper observed at the inference anchor. Hold uses
zero deltas or the anchor's observed absolute pose/joints. Persistence repeats the
previous achieved delta or target, transporting tool deltas into the anchor frame.
Neither baseline uses observations after the inference anchor.
For canonical joint targets, persistence specifically repeats the prior recorded
leader command target; it does not measure achieved follower joint motion.

The offline shortlist score is the equal mean of available fixed components:

1. Executed-prefix normalized L1.
2. Moving-vector error divided by hold's moving-vector error, with a denominator
   floor at the declared movement threshold.
3. One minus the fraction of moving predictions with a positive direction dot
   product. Zero predictions count as direction failures.
4. Stationary predicted step magnitude divided by the movement threshold.

The component definitions and weights are fixed before new training. Scores can be
compared only within the same split, labels, normalization, action representation
and available strata. They are a transparent heuristic for retaining offline
candidates, not an estimate of pick-and-place success. Existing promotion still
requires 30 common development rollouts, ten per condition, measured latency below
120 ms, and complete-task outcomes. Only frozen finalists proceed to the separate
90-case reserved suite. No reserved rollout or dataset image was consumed here.

ACT writes each full report to `validation/<step>.json`, links the report and score
from `training_history.csv`, and embeds the report in checkpoint `last_val` and the
training result. Frozen-feature head checkpoints contain the same common report.
`best_step`/`best_examples` now identify the offline-score checkpoint;
`best_val_l1`/`best_validation_l1` remain the lowest legacy L1 observed and can refer
to a different checkpoint. Resume rejects a changed selection protocol.

## Existing iPhone checkpoint audit

The existing `L1_resnet_iphone/attempt-001/checkpoints/004000` checkpoint was audited
on all six validation episodes: 2,071 valid actions, six extrapolated terminal
placeholders excluded. The 29 training and six reserved episodes were untouched.
Evidence: [selection_metrics_20261024.json](../results/v2/selection_metrics_20261024.json).

| Metric | Learned ACT | Causal persistence | Hold |
| --- | ---: | ---: | ---: |
| Executed-prefix normalized L1 | 0.34894 | 0.32632 | 0.50194 |
| Mean translation error | 2.547 mm | 2.466 mm | 4.347 mm |
| Moving translation error, 1,585 actions | 3.145 mm | 3.083 mm | 5.672 mm |
| Positive movement direction | 92.18% | 89.97% | 0% |
| Predicted stationary step, 471 actions | 0.569 mm | 0.431 mm | 0 mm |
| Maximum stationary-run summed error | 20.164 mm | 15.381 mm | 0.398 mm |
| Mean rotation error | 0.01286 rad | 0.01220 rad | 0.01864 rad |
| Gripper MAE | 0.03178 fraction | 0.02157 fraction | 0.02157 fraction |

The ACT moves mostly in the recorded direction but has lower median amplitude
(0.792 times the moving label), remains worse than causal persistence in mean
translation error, and introduces stationary drift. The new shortlist score is
0.60120. This audit neither changes the historical checkpoint selection nor
promotes it for reserved evaluation. All Cartesian conclusions remain conditional
on the provisional geometry used to produce its labels.

Reproduce into a new output path:

```sh
.venv/bin/python scripts/audit_v2_policy.py \
  --checkpoint outputs/v2/learning_screen_20261022/runs/L1_resnet_iphone/attempt-001/checkpoints/004000 \
  --data .cache/v2_exploratory_20261022/cartesian_iphone \
  --device cpu --out results/v2/<new-audit>.json
```
