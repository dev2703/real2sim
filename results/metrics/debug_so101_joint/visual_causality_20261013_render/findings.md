# Visual-causality audit — 13 October 2026

Diagnostic only. Six fixed episode starts (train 0/5/6/7; validation 1/3), original
joint checkpoint 020000, CPU inference, no training or benchmark changes. Exact
checkpoint/input SHA-256 values, copied source/configs, normalized image tensors,
and all 20 predicted actions were saved beside this note locally. The Git commit
includes the reports; tensor/source snapshots remain local diagnostic artifacts. No input file changed
during the successful run. This uses the **current appearance snapshot**, not
necessarily the yellow-robot render used by the initial investigation.

## New interventions

Mean predicted wrist targets (degrees), holding the recorded start state fixed
except where explicitly marked:

| Image / state | Action 1 | Action 2 | Action 20 |
|---|---:|---:|---:|
| Real start / start | 80.67 | 80.32 | 71.10 |
| Sim, ball present / start | 41.96 | 37.64 | -16.20 |
| Sim, ball moved to corner / start | 41.11 | 36.20 | -27.84 |
| Sim, ball truly outside scene / start | 46.50 | 41.72 | -12.04 |
| Real end / start | -1.89 | -6.93 | -27.10 |
| Real start / end | 8.35 | 8.60 | 5.46 |
| Real end / end | -82.40 | -82.42 | -82.13 |
| Real start / state after five hold ticks | 82.01 | 81.58 | 72.15 |

The controller executes only actions 1 and 2 before replanning. The visual effect
is already substantial in those actions, independently of the unexecuted tail.
Real/end image-state crossovers are deliberately inconsistent inputs; they do not
identify an internal categorical phase or prove a particular learned mechanism.

The old "no ball" intervention leaves **932 native-render pixels of ball visible**
in every tested episode. Present-ball renders contain 1,800–2,038 ball pixels.
Moving the ball to world [100,100,100] without stepping gives exactly zero visible
ball pixels, verified using MuJoCo segmentation. This truth is used for diagnostic
verification only, never as a policy input. True removal changes executed-prefix
wrist targets by mean absolute **4.31 degrees**, maximum **6.84 degrees**. It fails
to restore real-image behaviour, but the nonzero response does not justify saying
the policy literally cannot see the ball. Initial-frame sensitivity also cannot
establish whether the policy localizes it during approach.

Dataset and runtime image normalization agree exactly for the six real inputs
(maximum absolute tensor difference **0.0**). This checks the implemented
normalization paths, not all possible camera timing or calibration errors.

## Data and actuator compatibility

`action_limit_audit.json` compares every derived state and leader label to the
current combined MuJoCo joint/actuator limits under the existing mapping. Training
shoulder targets exceed those limits on **29.32%** of ticks; wrist targets on
**12.50%**. Absolute shoulder leader–follower disagreement has p95 **60.48 degrees**
in training and **70.07 degrees** in validation. These are valid recordings of
commands, but commands are not identical to achieved poses or simulator execution.
This is the pre-existing D-003 concern, now quantified on the derived splits.

At 10 Hz, flag a shoulder interval if its observed increment is below 0.2 degrees
and its contemporaneous command/state gap exceeds 10 degrees. Such intervals are
**18.51%** of train transitions and **29.45%** of validation transitions. Longest
contiguous flagged stretch: **20.4 s**, training episode 30. These descriptive
thresholds do not prove a faulty servo; logs alone cannot separate mechanical
stall, limits, loading, mappings or other recording/control causes.

`split_end_pose_audit.json`: **16/28** training episodes end with observed wrist
below -90 degrees, versus **0/9** validation episodes. Mean final wrist is -83.91
versus -66.46 degrees. This is a split distribution difference worth measuring
with phase-specific errors before interpreting all validation loss as overfitting.
Do not change the frozen split based on this result. Most demos exceed 30 s, but
their length alone does not establish when task success occurred.

## Existing artifacts reviewed, not rerun here

`outputs/tmp_appearance/grid_results.json`: three training starts, mean final
predicted wrist -19.69 degrees with legacy appearance versus -19.91 with the
current variant; tested variants range approximately -22.60 to -15.29 degrees.
Thus these appearance edits have not restored the real-image response. They lack
the full provenance snapshots of the new audit and should not be mistaken for
controlled closed-loop success evidence.

Settled-start trials remain 0/10. Their five settling ticks consume the same
300-tick episode budget, leaving 295 policy ticks versus 300 in the control.
This small budget difference does not explain the large image-induced prediction
change, but a future matched comparison should give both 300 policy ticks.

## Interpretation and next discriminating test

The visual input is a demonstrated cause of inappropriate **initial commands** in
this frozen joint policy. An exclusive root cause of all 0/10 outcomes is not yet
identified; correcting vision may expose command/dynamics or grasp failures.
Prioritize matching camera projection and observed scene structure, then test
executed-prefix behaviour at approach/grasp frames and run the unchanged E1
scenarios. Fit appearance against training images and physical measurements;
do not optimize against held-out action predictions or equate a matching rest
chunk with successful task transfer. Retain the original failed benchmark.

## Commands

```sh
.venv/bin/python scripts/diag_visual_causality.py --output results/metrics/debug_so101_joint/visual_causality_20261013_render
.venv/bin/ruff check scripts/diag_visual_causality.py
UV_CACHE_DIR=/tmp/real2sim-uv-cache uv run --offline pytest -q tests/unit/test_resample_and_policy_data.py tests/unit/test_policy_guards.py
```

Diagnostic completed, Ruff passed, and the targeted data/policy suites passed
**54 tests**. No full-suite or closed-loop rerun was required for this isolated audit.

First sandboxed attempt could not create a macOS CoreGraphics context. The
successful run used approved execution outside the sandbox for offscreen rendering.
No training, hardware, primary evaluation results or shared simulation files were
changed by this audit.
