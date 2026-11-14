# Observable retention sequencing — 15 November 2026

Implemented an optional execution sequencer for the existing stage-balanced
DAgger head. It remains **off by default**: the first matched smoke check did
not improve completion and regressed Panda lift. This is additional assistance,
not a new trained head. Entry gating, SiLU, frozen ACT weights and native
acceptance thresholds are unchanged.

## Implementation

`RetentionGuard` consumes only proposed actions, causal jaw feedback, measured
TCP motion, RGB reference ages, an RGB close-edge carry hypothesis and native
execution feedback. It never receives simulator object state or teacher phases.

- ACQUIRE requires recent RGB before moving or initiating closure.
- CLOSE/LIFT latch closed intent and suppress lateral/rotational transport until
  the existing resisted-jaw plus achieved-lift monitor supplies a grasp candidate.
  A candidate is explicitly not proof of holding an object.
- TRANSPORT/LOWER retain closure. Opening requires the carry hypothesis to be
  within 25 mm horizontally of a recent RGB bowl rim reference, 0–50 mm above
  it, with measured TCP speed at most 30 mm/s for 200 ms. Retention warnings and
  rejected native motions reset that dwell. These are provisional thresholds.
- RELEASE holds the arm; RETREAT waits for measured jaw opening. It keeps the
  jaws open so a subsequent learner close proposal cannot immediately regrasp.
- Suspected loss or candidate timeout opens while holding the arm, clears the
  close anchor/monitor, forces RGB refresh and waits for a detection timestamped
  after loss before another acquisition. Reacquisition has a timeout and retries
  are bounded. Eight consecutive native rejections enter an absorbing arm hold
  that preserves issued jaw intent, rather than dropping a possible payload.
- Translation limits use achieved motion. Stage transitions invalidate queued
  frozen ACT actions and aligned decoder features together.

Enable with `--retention-guard` in `run_recovery_dagger.py evaluate`, or benchmark
controller options `{"head": "...", "retention_guard": true}`. Requires tracked
RGB and the native recovery frame; cannot combine with `--phase-guard`.
Per-tick traces disclose stage, intervention reasons, retry count, dwell,
rejection streak and proposed versus filtered actions.

Corrective collection now excludes rejected native teacher commands as well as
terminal holds. Common-action labels are additionally executed in the shadow
from the actual learner state through the common executor. Rejected projections
are excluded; accepted labels use the executable backtracking fraction. This
checks executability, not whether the teacher will ultimately succeed. Existing
packs and fitted artifacts are not rewritten.

## Validation and development evidence

Original experiment unit suite: **1,574 passed, 3 skipped**. Ruff and `git diff --check` pass.
New tests cover delivery dwell/reset, premature opening, stale acquisition,
loss/reacquisition, bounded retries, rejection hold, candidate timeout, controller
wiring, ACT invalidation and rejected/terminal teacher label masking.

Artifacts: `outputs/v2/retention_guard_20261115/`.

`smoke_frozen/` runs the unchanged stage-balanced head and frozen prefix on seed
12221 for each arm, using the same configured jaw and provisional scene as the
earlier experiment. Both initial-state hashes match the earlier unguarded
stage-balanced records. Both trials completed without abort, and
`smoke_frozen/evidence_audit.json` verifies both trial/trace records.

| Arm | Unguarded | Retention assistance | IK rejections, before → after |
| --- | --- | --- | --- |
| SO101 | Lift; transport failure | Lift; transport failure | 278 → 8 |
| Panda | Lift; transport failure | Grasp failure | 0 → 0 |

SO101 reached TRANSPORT, detected empty-jaw loss, reacquired a fresh RGB target,
then held after repeated native rejection. Its reduction in rejection count
partly reflects stopping; it does not establish better transport. Panda timed
out twice with only about 9–10 mm measured lift, below the candidate monitor's
20 mm threshold. It never entered TRANSPORT. Neither trial exercised a successful
release; release behavior is covered by synthetic observable-stream tests only.
Two development rollouts cannot establish a success-rate change.

The PR branches are based on newer main commit `6d666d1`, including the later
CLI cleanup and Panda retarget fixes. The smoke rollouts above predate that
publication base; their saved source hashes describe the experimental runtime.
They have not been rerun on the publication branches. Unit checks are repeated
on those branches separately.

The earlier `smoke/` attempt was invalidated by the benchmark's source-change
check while implementation was still being edited. Exclude it from comparisons.

`collector_smoke/` exercised shadow common-action validation on SO101 training
seed 10330 and validation seed 11330. Both teacher-only attempts completed with
zero execution rejections/aborts, producing 163 train and 165 validation rows.
These packs have not been aggregated or used to fit a new head. Panda collection
with this new label-validation path has not been smoke-tested.

## Remaining work

Do not promote this guard as a reliable controller. The matched smoke result
shows that closed-jaw sequencing alone does not prevent mechanical loss, and
restricting pre-candidate motion can disrupt the learned Panda pickup. A next
controlled ablation should separate jaw latching, lateral gating and motion
limits, then collect executable transport/drop corrections using fixed data
roles. Use validation for fitting and fresh development layouts for comparison;
do not tune thresholds solely to make seed 12221 pass. A learned recurrent stage
model and improved teacher coverage remain future work, not claims of this patch.
