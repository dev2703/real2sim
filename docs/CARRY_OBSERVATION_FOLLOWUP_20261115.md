# Carry observation and seating follow-up — 15 November 2026

The earlier complete-silhouette detector supplied 0/243 carried-object
measurements. This follow-up addresses partial jaw occlusion and an inaccurate
close-edge carry estimate. Visual assistance remains experimental and disabled
by default. No ACT weights, recovery weights, scene physics, camera mounts or
native execution acceptance thresholds are changed.

This PR is stacked on `codex/recovery-data-phase-followup` (`2108b54`, PR #64),
which depends on #63 and #62. The publication runtime includes the newer main's
Panda retarget changes. Results from the original dirty checkout in
[the earlier feedback record](VISUAL_RETENTION_FEEDBACK_20261115.md) are historical
evidence, not measurements of this branch. Large artifacts remain local.

## Implementation

`locate_ball_3d(..., fit_mode="arcs")` robustly fits the known-radius sphere's
tangent cone using RGB contour arcs. Fixed RANSAC hypotheses exclude occluding
edges, followed by a half-pixel boundary correction and geometric refinement.
Acceptance requires at least 70% sampled boundary support within 0.8 pixels, tangent
directions spanning at least 240 degrees, a projected radius of at least eight
pixels and no sampled foreground boundary more than 1.5 pixels outside the fit. Missing,
clipped, small, ambiguous and insufficiently constrained shapes return no
measurement. These are explicit diagnostic assumptions, not calibrated error
bounds. The original `complete` fit remains the default.

This uses the existing global RGB camera. The exploratory replay first showed
unoccluded Panda balls because it omitted the benchmark's 150mm robot-base
translation; it is not evidence of live sensing coverage. A live, instrumented
rerun reproduced the saved Panda trajectory exactly and exposed the actual jaw
occlusion. Evidence below comes from live controller observations.

Passive feedback retains the original supported-plane goal detector, refresh
cadence, reference ages, observation dimensions and learned-head inputs. A
measurement absent on the current tick is never passed as fresh. Fresh 3D RGB
can seed a close-edge carry estimate when the supported-plane reference is old.
Both are hypotheses, never simulator ball poses or proof of contact.

The close-edge estimate can become wrong as the jaws seat: in the first SO101
pilot, measured ball/tool separation from that estimate reached 30–40mm despite
scored holding and accurate RGB. The final implementation can establish one
new estimate after a current resisted-jaw candidate and at least 20mm independently
observed ball lift. The ball must be within 60mm of measured TCP. This cannot
follow a stationary ball, use a missing current measurement, override suspected
loss, or repeatedly follow slip. Fresh matches must accumulate against the new
estimate before support is declared. Opening/reset clears that one-update latch.
The 25mm visual match threshold is unchanged; `held_verified` remains false.

For visual assistance only, a current 3D observation within 15mm of the existing
pickup reference corroborates that goal's freshness without changing the head's
reference or age. After loss, a current 3D measurement at the declared support
plane can supply a fresh pickup reference if the strict supported-plane detector
is occluded. Elevated measurements cannot become reacquisition goals.

The PR also publishes the earlier visual-mode correctness work: observed lift
latching, freshness expiry, conservative payload timeout/rejection holds,
measured delivery/release dwell, post-loss reference/history invalidation and
ACT action/feature invalidation. The guard still requires tracked RGB, native
recovery and a separate explicit enable flag.

## Audit and tests

`python -m real2sim.evaluation.visual_feedback_audit` reports measurement coverage,
error, missing intervals, visual support and scored contact-loss transitions.
Feedback at tick t is compared with scoring at t−1, because controller feedback
precedes execution. Simulator state is scoring-only and never enters sensing,
anchor construction or guard decisions. Loss flags over the bowl are reported
separately from outside-bowl contact loss; contact flicker is not a sustained
transport-drop label.

The passive comparison checks all saved head inputs, proposals, issued/achieved
actions, native controls, reference metadata and the complete scored state.
Only RGB grasp diagnostics, sensing-mode metadata and wall-clock latency are
excluded. Different trajectory lengths or any control/scoring difference fail
the comparison. Trial integrity is checked separately with
`scripts/audit_frozen_act_results.py`.

Focused tests cover shallow occlusion without false lift on reveal, multiple
native resolutions, ambiguous/weak/square/elongated shapes, stale measurements,
post-loss supported reacquisition, pickup-reference corroboration, single seating
correction followed by independently detected separation, release sequencing,
and passive-action/causal-audit integrity.

Final validation: **162 focused tests passed**. Full publication-branch unit
suite with graphics/Metal access: **1,291 passed, 2 skipped, 2 failed, 3 errors**.
All five broken cases are in `test_v2_wrist_camera_diagnostics.py`, which imports
the deleted `diag_v2_wrist_zeros.py` / `diag_v2_camera_uncertainty.py` scripts.
Those same five failures reproduce on the unchanged publication parent
`2108b54`; they are inherited cleanup failures. The full-suite log is
`/private/tmp/carry_observation_final_unit.log`; parent reproduction is
`/private/tmp/carry_observation_parent_failures.log`. Ruff and diff checks pass.

## Development protocol

Artifacts are under `outputs/v2/carry_observation_20261115/`. All variants use
the configured jaw reset, 25Hz control, the existing global camera,
`configs/v2/weight_space_20261111/sim_so101.yaml`, and the same shared phase head:

- ACT revision: `e696c8a078935432e3501e2c28c1fb2598fbff7e/latest`;
  weights SHA256 `5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
- Head: `outputs/v2/retention_data_20261115/head_phase_augmented`;
  SHA256 `5473fe93c10f52d716832de882d5f5d0aeb91b8358d4647a2f59aafe6e9eded7`.
- Seeds 12660–12661 per arm were new at the first pilot, then reused to debug
  seating. They are development layouts, not fresh confirmation or training data.

`baseline_new`, `complete_new`, `passive_new` and `guard_new` are the first pilot
before the seating correction. All 16 records pass trial/trace integrity audits;
both passive variants exactly preserve baseline control and scored trajectories.
Baseline and both passive variants complete SO101 1/2 and Panda 1/2. The guard
completes SO101 0/2 and Panda 1/2. These results must not be presented as the
final implementation's controller performance.

In that pilot, strict sensing measures 1/242 held ticks, versus 220/242 with arc
fitting. All measured held positions are within 12.1mm of pre-action scoring.
The arc fit therefore addresses much of the measured visibility gap on these
layouts, while the rigid close-edge hypothesis still prevents SO101 support.
The failed Panda transport has an unflagged outside-bowl contact-loss transition;
good image measurements alone do not validate timely drop detection.

## Final reviewed-source results

`baseline_reviewed`, `complete_reviewed`, `passive_reviewed` and `guard_reviewed`
rerun all four variants after the seating fix. All **16 trial/trace records**
pass integrity audits, without aborts. Their runtime source maps are identical,
every saved source/artifact hash matches, and initial-state hashes match for each
of the four layouts. Exact source text is preserved in each variant's
`executed_sources/`. The checked-in [compact evidence record](evidence/CARRY_OBSERVATION_20261115.json)
contains protocol, trace, trial-report and snapshot-manifest hashes, per-trial
measurements and guard stage counts. It references the larger local artifacts.

| Final variant | SO101 scored completion | Panda scored completion |
| --- | ---: | ---: |
| Baseline, sensing off | 1/2 | 1/2 |
| Passive complete-silhouette fit | 1/2 | 1/2 |
| Passive tangent-arc fit | 1/2 | 1/2 |
| Visual assistance with tangent arcs | 1/2 | 1/2 |

Both passive variants preserve the baseline's complete control/scoring traces
**exactly on all four trials**, including failed trajectories. Their matched
carried-object measurement comparison is:

| Passive trial | Complete fit | Arc fit | Arc held-position error in sample |
| --- | ---: | ---: | ---: |
| SO101 12660 | 1/12 | 12/12 | ≤11.1mm |
| SO101 12661 | 0/92 | 92/92 | ≤12.1mm |
| Panda 12660 | 0/84 | 84/84 | ≤11.4mm |
| Panda 12661 | 0/54 | 32/54 | ≤4.2mm |
| Total | **1/242** | **220/242** | **≤12.1mm** |

These denominators count scored held states immediately before controller
feedback. They are not independent trials or proof of secure transport. The
worst missing held interval remains **0.68s** on Panda 12661. Across all measured
states, including free balls, the largest position error is 21.6mm. On these
four passive trials, all 147 support ticks coincide with scored contact, but
contact itself is not held-object verification. The seating fix restores 71
support ticks in SO101 12661, versus zero in the first arc pilot.

The guard recovers the pilot's lost SO101 scored completion but does **not**
improve either arm over baseline. No guarded trial enters RELEASE or RETREAT,
and **0/4 validate release**. Scored completion after eventual ball separation
is not evidence that the guard executed its intended release procedure.

- SO101 12660 still fails acquisition after loss/retry; reliable RGB alone does
  not make the learned acquisition proposals succeed.
- SO101 12661 now establishes support and reaches LOWER, then stops after eight
  native rejections. It eventually receives a scored completion while stopped.
  The lowering/release proposals still need executable corrections.
- Panda 12660 reaches LOWER, then loses fresh visual support and stops with
  `visual_hold_timeout`. It also eventually receives a scored completion without
  guard release validation.
- Panda 12661 still loses the object outside the bowl and fails. In passive
  sensing, scored contact ends at tick 152 and a loss flag appears at 154. In
  guarded control, those ticks are 145 and 149; a jaw warning appears at 146.
  The 3D separation flag triggers a fresh-target retry, but does not prevent
  the mechanical drop or recover completion. These are observed timings of
  individual development events, not estimated detection sensitivity/latency.

Keep the guard disabled. The next bottlenecks are feasible lowering/release
corrections, responses to narrowing-jaw warnings before a drop, and the remaining
Panda occlusion interval. Native acceptance thresholds should not be loosened
to turn a stopped trial into a claimed controller success. Broader independent
layouts, deliberate loss events and physical calibration remain necessary;
there is no new selected controller or reliable-transport claim from this PR.

## Reproduce

Set `SNAPSHOT` to the pinned ACT directory. For passive arc sensing:

```sh
.venv/bin/python scripts/benchmark_frozen_act.py --snapshot "$SNAPSHOT" \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --robots so101 panda --conditions hf_learned --seeds 12660 12661 \
  --configured-gripper \
  --controller-factory scripts.run_recovery_dagger:controller_factory \
  --controller-options '{"head":"outputs/v2/retention_data_20261115/head_phase_augmented","visual_feedback":true,"visual_ball_fit":"arcs"}' \
  --out outputs/v2/carry_passive_NEW

.venv/bin/python scripts/audit_frozen_act_results.py \
  outputs/v2/carry_passive_NEW/report.json --out outputs/v2/carry_passive_NEW/integrity.json
.venv/bin/python -m real2sim.evaluation.visual_feedback_audit \
  outputs/v2/carry_passive_NEW/report.json \
  --baseline outputs/v2/carry_baseline_NEW/report.json \
  --out outputs/v2/carry_passive_NEW/sensing.json
```

For baseline, omit `visual_feedback` and `visual_ball_fit`. For strict passive
sensing, retain `visual_feedback` and omit `visual_ball_fit`. Experimental visual
assistance uses `"retention_guard":true,"retention_mode":"visual","visual_ball_fit":"arcs"`.
Use new output paths and retain separate per-arm outcomes. These commands require
the local pinned artifacts; the PR does not distribute trained weights.
