# RGB carry feedback and loss recovery — 15 November 2026

This records the historical experiment in the original checkout based on
`38e2fb4`, with dirty sources preserved in the referenced artifacts. Its 1,605
passing tests and 3/4 versus 0/4 rollouts do not evaluate the newer publication
runtime. The [carry-observation follow-up](CARRY_OBSERVATION_FOLLOWUP_20261115.md)
records subsequent implementation and matched publication-branch experiments.

The new RGB feedback is an experimental diagnostic, not reliable held-object
sensing. Keep visual retention assistance **disabled**. On the matched
development probe, passive sensing preserves the selected controller's actions
and outcomes, but supplies no 3D ball measurements during carried-object ticks.
The guarded variant regresses completion. No new weights or controller are
selected from this work.

## Implementation

`locate_ball_3d` fits the known-radius sphere's RGB silhouette without assuming
table support or proximity to the tool. It compensates for the half-pixel inward
bias of raster contour centres and uses a stricter confidence gate than the
existing supported-plane detector. Missing, clipped, ambiguous and insufficiently
complete silhouettes return no measurement. Camera calibration, object radius,
colour and resolution remain explicit assumptions. A monocular shape fit is not
contact evidence, and `held_verified` remains false.

`--visual-feedback`, or benchmark option `visual_feedback: true`, observes this
additional signal without filtering actions. It preserves the existing supported
ball references, their refresh cadence, the trained observation dimensions and
the head's inputs. No simulator ball pose, contact flag, teacher phase or scored
success enters runtime feedback.

`--retention-guard --retention-mode visual` enables separate, disclosed
experimental assistance:

- Jaw intent stays closed before validated delivery. Initial pickup retains
  learned motion within a bounded observation window. After visual support was
  established, expired support holds the arm closed; missing vision alone never
  establishes a drop or authorizes opening.
- Visual support requires repeated fresh matches to the close-edge carry
  hypothesis and an observed lift of both tool and ball. Lift is latched as an
  event so lowering does not erase it. Expired observations cannot establish a
  new lift event. The carry hypothesis remains an estimate, not object truth.
- Opening requires a current measured 3D ball at the fresh bowl reference,
  achieved low tool speed, a delivery dwell, and no jaw warning or native motion
  rejection. Retreat additionally waits for measured jaw opening and settled
  ball observations in the bowl region. Timeout records `release_unconfirmed`;
  jaw opening alone never marks release validated.
- Suspected loss clears the old pickup reference, carry anchor, grasp monitor,
  observation history and queued ACT actions/features. Missing-target ticks skip
  head inference. Reacquisition requires a supported RGB detection after loss;
  arm motion waits during the pause and acquisition. Retries and waits are bounded.
- Repeated unreachable acquisition proposals trigger fresh-target retry when
  there is no payload evidence. Historical visual support or a jaw candidate
  instead preserves closure and stops. A jaw-candidate timeout likewise cannot
  open a possibly carried object.

The older `full` and `jaw` modes remain available. Native execution acceptance,
scene physics, selected shared head and frozen ACT weights are unchanged.

## Independent review and validation

Three Astra reviewers checked controller sequencing, perception and evidence.
Their reproducible findings became regression tests: raster depth bias,
partial-occlusion false lift, stale-evidence lift confirmation, lowering after
lift, carry-stage transition memory, and opening on timeout/rejection despite
possible payload evidence. The stricter silhouette gate rejects the demonstrated
occlusion-reveal false lift; this does not establish general perceptual robustness.

Full unit suite: **1,605 passed, 3 skipped** with macOS graphics/Metal access.
The sandbox-only run's three camera failures were CoreGraphics access errors;
the camera subset passes with that access. Repository Ruff and `git diff --check`
pass. Synthetic streams exercise successful release validation and post-loss
fresh-target retry; they are not successful manipulation rollouts.

## Matched development probe

Artifacts are local under `outputs/v2/visual_retention_20261115/`. The checkpoint
is pinned to native ACT revision
`e696c8a078935432e3501e2c28c1fb2598fbff7e/latest`, SHA256
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
The selected shared head remains `retention_data_20261115/head_phase_augmented`,
SHA256 `5473fe93c10f52d716832de882d5f5d0aeb91b8358d4647a2f59aafe6e9eded7`.
Use the configured-jaw reset and
`configs/v2/weight_space_20261111/sim_so101.yaml` on both arms. Seeds 12640–12641
are development layouts reused during debugging, not fresh confirmation.

The `baseline_final`, `passive_final` and `visual_final` directories are an
intermediate pilot before the last stale-lift/payload-timeout correctness fixes.
All 12 trial/trace records pass integrity audits and share matching initial
states and runtime source maps. Exact hashed text is retained under
`audited_pilot_sources/executed_sources/`; directory names containing `final`
do not identify a selected or promoted controller.

| Pilot variant | SO101 completed | Panda completed |
| --- | ---: | ---: |
| Selected head, unchanged | 1/2 | 2/2 |
| Same head, passive RGB feedback | 1/2 | 2/2 |
| Same head, visual assistance | 0/2 | 0/2 |

`independent_passive_audit.json` verifies exact equality between baseline and
passive traces for references, ages, observations, proposals, issued/achieved
actions, controls and entire scored trajectories. Feedback at tick *t* is
compared with scoring at *t−1*, the pre-action state, solely for offline auditing.

| Passive trial | Held ticks with a 3D observation | Longest missing held interval |
| --- | ---: | ---: |
| SO101 12640 | 0/82 | 3.28 s |
| SO101 12641 | No holding | — |
| Panda 12640 | 0/98 | 3.84 s |
| Panda 12641 | 0/63 | 2.32 s |

Thus the passive pilot has **0/243 carried-object measurements** and no visual
support events. Low false-support counts in the absence of support establish
neither sensitivity nor useful feedback. Successful transports and an approach
failure do not validate failed-transport loss detection. Jaw/3D separation loss
indications after bowl placement must not be counted as detected transport drops.
The remaining single-view visibility gap is a measured blocker; improved carry
observation is needed before this guard can assure retention. Marker-equipped
tracking and mechanical retention changes were not evaluated here.

The final reviewed-source rerun is recorded separately in `baseline_reviewed`,
`passive_reviewed` and `visual_reviewed`, with exact source snapshots under
`reviewed_sources/executed_sources/`. It reproduces the per-arm completion counts
above: baseline and passive 3/4 overall, visual assistance 0/4. All 12 final
trial/trace records pass evidence audits, with no aborts. These repeated
development layouts do not add independent confirmation evidence. The final
passive audit is `independent_reviewed_passive_audit.json`.
It confirms exact baseline/passive trajectories and the same 0/243 carried
measurement coverage. No guarded trial enters RELEASE or RETREAT, so release
validation remains exercised only by synthetic tests. All three runtime source
maps, the four matched initial states and all saved source/artifact hashes agree.

Earlier `baseline`/`visual` attempts used the wrong checkpoint and aborted before
motion; exclude them. The first `baseline_matched`/`visual_matched` pilot mixed
3D estimates into the pickup references and gated pickup too early. It is
retained as rejected development evidence, not an isolated retention comparison.

## Reproduce

Set `SNAPSHOT` to the pinned ACT directory above and choose a new output path.
For passive feedback:

```sh
.venv/bin/python scripts/benchmark_frozen_act.py --snapshot "$SNAPSHOT" \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --robots so101 panda --conditions hf_learned --seeds 12640 12641 \
  --configured-gripper \
  --controller-factory scripts.run_recovery_dagger:controller_factory \
  --controller-options '{"head":"outputs/v2/retention_data_20261115/head_phase_augmented","visual_feedback":true}' \
  --out outputs/v2/visual_feedback_NEW
```

Omit `visual_feedback` for baseline. For experimental assistance, replace it with
`"retention_guard":true,"retention_mode":"visual"`. Keep comparisons per arm;
do not select a shared controller that improves Panda by regressing SO101.
