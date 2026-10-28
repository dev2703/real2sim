# Retention feedback and source representation — 29 October 2026

The closed-jaw Panda failures are progressive slip, rather than commanded
release. New diagnostics preserve this distinction, and ACT-dependent commands
now pass a source-representation check before execution. The frozen phase head
and the successful geometric fallback remain separate controllers.

## Retention evidence

The [retrospective audit](../results/v2/recovery_retention_audit_20261029.json)
rechecks all eight Panda extension trials, including successes and failures.
It finds five sustained losses after a genuine lift while the selected jaw
command stays at most 10% open throughout the preceding 0.4 seconds.

| Development seed | Loss tick | Ball descent relative to TCP in preceding 0.4 s | Peak measured TCP speed | Peak measured TCP acceleration |
| --- | ---: | ---: | ---: | ---: |
| 12003 | 229 | 17.6 mm | 0.026 m/s | 0.423 m/s² |
| 12005 | 173 | 21.2 mm | 0.046 m/s | 1.029 m/s² |
| 12006 | 235 | 15.9 mm | 0.116 m/s | 1.424 m/s² |
| 12007 | 233 | 16.0 mm | 0.036 m/s | 0.812 m/s² |
| 12009 | 204 | 20.8 mm | 0.031 m/s | 0.651 m/s² |

Measured jaw gaps narrow from approximately 59–64 mm to 44–51 mm across those
windows. That signature is compatible with a ball slipping downward through
closed fingers. It does not isolate grasp seating, friction or motion as the
cause. Acceleration is a finite difference at the saved 25 Hz control rate;
substep peaks are unavailable. Simulator ball poses and held/lift flags are
used only in this retrospective diagnostic.

## Observable runtime feedback

`ObservableGraspMonitor` uses issued jaw intent, measured jaw opening, TCP
motion, and fresh confident on-table RGB detections. It distinguishes a grasp
candidate, narrowing-gap warning, and suspected loss. Candidate means that the
jaw resisted closure while the tool rose; obstruction can produce the same
evidence. Every record explicitly reports `held_verified: false`.

The learned controller logs this feedback and its complete phase probabilities.
The existing head's 99 inputs remain unchanged. A nearby recent RGB ball
reference is anchored in the measured tool frame when an actual close command
begins. Later fresh on-table evidence far from that hypothesized carried pose
can reveal a missed pickup or loss. Occlusion alone does not establish holding
or loss, and the hypothesis is not a measurement of the carried ball.

`--phase-guard` enables additional, disclosed execution assistance: temporal
phase confirmation, closed intent during carry, an observable-candidate gate
before transport, bounded command translation speed/acceleration, and bounded
open/hold retries on suspected loss or missing evidence. It uses neither teacher
actions nor simulator object state. These rules are experimental; neither a
phase prediction nor the guard proves that the ball is securely held. Baseline
monitoring is the default, and assisted outcomes must remain separately labelled.

DAgger's teacher now measures continuous close/release dwell against the actual
issued native jaw command. An opposing issued command resets dwell. A selected
close request which is rejected into an open-jaw hold therefore cannot advance
the teacher through CLOSE. Selected-action history remains unchanged for the
existing head's observation contract.

## Panda source representation

Panda-native feasibility does not guarantee that a pose can be represented by
the virtual SO101 ACT input. A pure `project_pose` query now exposes position
residual, acceptance and representation semantics without modifying source
state or command history. Additional deterministic solver seeds do not make
the two historical failing poses valid: residuals remain 14.604 and 7.638 mm.

ACT-dependent common execution checks native endpoints and sampled joint paths
for source representability. Command endpoints have a 1 mm margin inside the
unchanged 3 mm measured-state tolerance. Backtracking changes magnitude only;
it supplies no task goal. The check applies to collection, ACT heads and phase
heads. No-ACT execution retains its original native workspace.
The diagnostic `--expert-execution native` collection option bypasses the common
executor for teacher commands and reports that bypass explicitly.

This is a kinematic constraint, not a dynamics guarantee or a complete Panda
state representation. Actual out-of-tolerance measured feedback still raises
a structured error; stale or invented source states and decoder features are
never supplied. Rotation coordinates of the position-only inverse retain the
causal source-command prior, rather than representing measured Panda rotation.

## Camera evidence and remaining work

The [18-anchor camera factorial audit](v2_camera_factorial_20261029.md) preserves
exact input images and ACT chunks and confirms substantial interaction between
the two camera views. It adopts no camera correction. Independent intrinsic and
camera-to-base captures are still needed; physical measurements, lighting,
contact parameters and frozen ACT weights have not been changed.

New corrective collection must use the fixed missed-pickup teacher and keep
old packs immutable. Training a head to consume grasp feedback requires an
explicit new observation contract and fresh learner-visited examples. Current
head validation errors and phase accuracy remain distinct from task success.

## Verification and development smoke

The [smoke audit](../results/v2/retention_representation_smoke_20261029.json)
retains all six engineering trials, including the initial guard version's
failures. After review corrections, the optional guard completed SO101 seed
12000 in 497 ticks with zero IK rejections. Panda seed 12003 still failed to
grasp, also with zero rejections. Both were previously inspected failures;
these two cases do not establish an improved success rate. The optional guard
remains experimental and the geometric fallback remains the working default.

Fresh expert-only contract checks on training seed 15012 and validation seed
16004 completed 2/2 tasks per arm. Their separate packs retain 372/353 training/
validation rows across the arms with issued-command phase labels. They verify
that the changed teacher and mapping guards can execute a complete task; they
provide neither learned-only results nor learner-visited failure coverage. No
head was retrained from these small smoke packs.

All runs preserved source snapshots, unchanged frozen weights and unchanged
lighting. The unit suite plus relevant integration tests passes 1,037 checks;
two CUDA-only checks skip. Ruff and `git diff --check` pass.

Reproduce the retention diagnostic:

```sh
.venv/bin/python scripts/diagnose_recovery_retention.py \
  --reports outputs/v2/recovery_mlp_tracked1_phase_panda_dev8_20261029/report.json \
  --out results/v2/recovery_retention_NEW.json
```
