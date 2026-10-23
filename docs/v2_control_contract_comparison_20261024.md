# Development control contract comparison — 24 October 2026

Changing joint targets or softening orientation costs does not yet establish an
approach improvement. A bounded common-input comparison now tests both target
representations in physical FK units, retains every native acceptance gate, and
reports rejected commands alongside motion direction and stationary motion.
No checkpoint is promoted from this diagnostic.

[The complete report](../results/v2/control_contract_comparison_20261024.json)
uses the completed matched two-view ResNet Cartesian and canonical-joint
checkpoints, both at step 4,000. They receive identical recorded validation images
and identical canonical proprioception; image bytes are checked directly. The
six development episodes are 9, 13, 22, 31, 36 and 37. Reserved episodes are not used.

Twelve evenly spaced two-action anchors per episode produce 141 nonpadding
actions. Fifty-one have native-incompatible observed joint states and are
explicitly excluded without clipping; the following results cover the remaining
90 common actions. They are a limited development sample, not a complete
validation score. The second action uses the recorded successor observation,
and each sampled prefix starts with an explicitly rebased target. Persistent
accepted-target integration remains active within each two-action prefix.

| Candidate | Original endpoint gates accepted | Mean bounded FK command error to next follower pose | Mean stationary command movement |
| --- | ---: | ---: | ---: |
| Original Cartesian IK | 80/90 | 4.632 mm | 0.656 mm |
| Cartesian optional position-priority fallback | 80/90 | 4.632 mm | 0.656 mm |
| Cartesian bounded joint step, orientation weight 0.025 m/rad | 80/90 | 4.666 mm | 0.656 mm |
| Cartesian bounded joint step, orientation weight 0.0015625 m/rad | 80/90 | 4.661 mm | 0.656 mm |
| Learned canonical joint commands | 76/90 | 12.991 mm | 6.583 mm |
| Hold baseline | 80/90 | 5.400 mm | 0 mm |

The fallback does not change any sampled request because this sample has no
numerical residual failures: Cartesian rejections are four workspace failures
and six collisions. Both soft orientation projections retain the original hard
3 mm position, 0.35 rad orientation, 2 mm penetration and native joint gates.
Their small difference does not support changing the default action interface.

On 76 active actions (recorded translation greater than 0.4 mm), the Cartesian
candidate has positive direction agreement on 69.7%, versus 60.5% for learned
joint commands. The median command/label magnitude ratio is 0.693 versus 2.422.
These are FK command proxies after rejection and common-progress rate bounds.
They are not measured movement, complete policy rollouts, or the full validation
prefix audit. At prefix offsets zero and one, original Cartesian error is
3.355 and 5.908 mm respectively; the report retains this distinction.

The raw learned joint commands fail native limits on eight actions. Before
rate bounding or rejection, their mean target error is 28.526 mm. Even the
recorded leader commands have mean FK mismatch of 18.636 mm to the recorded
next follower pose. Joint supervision therefore has a different target meaning
from Cartesian achieved-motion labels. Normalized losses across these targets
cannot identify the better controller.

The recorded follower's next joint state is also included as a label-feasibility
reference. It specifies the same provisional FK target as the Cartesian label
before acceptance/rate handling: raw FK target mismatch is zero, 79/90 targets
pass the original gates, and mean error after rejection/rate handling is 0.298 mm.
This is a useful future joint-space training
target, but no checkpoint has been trained on it in this comparison. It must
retain explicit invalid-label handling and a matching next-follower-joint
inference contract; silently clipping original leader commands would create a
different, unverified label meaning.

Six scored source poses already have endpoint collisions. Two learned joint
targets with accepted full endpoints acquire colliding endpoints after the
common-progress rate bound. The report exposes those flags. Endpoint feasibility
and joint limits do not certify a swept path or actuator tracking. This runner
performs zero physics steps, changes no simulator parameters, and installs no
controller. Joint and collision limits remain unchanged.

Reproduce with:

```sh
.venv/bin/python scripts/diag_v2_control_contract_comparison.py \
  --out results/v2/control_contract_comparison_NEW.json
```

The new gate checks and existing projection checks pass: 16 tests, with three
accelerator-only tests skipped on this CPU run. The complete input/checkpoint
hashes, per-episode results, per-prefix results and every scored/excluded row
remain in the report.

## Loaded holds and releases

[The measurement evidence review](../results/v2/loaded_hold_release_evidence_20261024.json)
finds no matching physical loaded-hold/release trace in the reviewed inventory or
measurement template. Table/mat dimensions, supplied bowl dimensions, nominal
CAD and an unchanged setup statement do not measure loaded force, ball/tool
slip, or release response. The two loaded trace entries remain unfilled.

Existing exact-state simulator tests retain zero of four 30-second holds with
the original contact regularization, three of four with `impratio=100`, and
four of four with explicit `noslip_iterations=3`. Existing privileged transport
tests achieve released, resting placements on four of four cases under both
the original and NoSlip options. These establish a simulator sensitivity and
a small release regression check; no corresponding physical 30-second hold or
release measurement validates their realism.

NoSlip remains an explicit simulator option. Defaults stay unchanged, and its
physical relevance remains unestablished. A measured comparison needs matched
loaded/unloaded poses, requested/sent/readback commands and timing provenance,
ball/tool displacement with registration uncertainty, loaded jaw response, and
release onset/separation/settling at declared ordinary openings. Geometry and
contact fitting must remain separate from verification trials.

## Separate learned ball-input approach screen

The [bounded ball-input comparison](../results/v2/ball_grounding_rollouts_20261024.json)
executes the matched 4,096-example pilot checkpoints on three common development
seeds for 125 ticks each (10 seconds at 12.5 Hz). Each model uses its saved
training/inference detector contract through the normal policy adapter.
Initial native, tool and ball states match exactly across each seed pair.
The unchanged provisional scene uses the strict recorded episode-9 frame-100
start. NoSlip remains disabled; native joint and collision gates stay unchanged.

| Development seed | Disabled features closest distance | Ball-input closest distance |
| --- | ---: | ---: |
| 0 | 178.98 mm | 178.12 mm |
| 1 | 213.61 mm | 197.18 mm |
| 2 | 165.28 mm | 157.69 mm |

Both candidates are **0/3**, with approach failure in every case, no lifts,
releases or aborts. The disabled control has 55 collision IK rejections on seed
1; the ball-input candidate has no IK rejections on these three trajectories.
Its detector supplies a visible ball feature on all 125 ticks in each run.
Maximum table penetration is 0.553 mm for the control and zero for ball input,
under the unchanged 2 mm gate. All source hashes stay unchanged during execution.

These three small improvements remain over 150 mm from the ball. They do not
establish grasping, complete task success or a checkpoint-promotion decision.
Reserved evaluation remains unused. The full report retains per-tick observed
features, action requests, IK results, native states and privileged scoring.
A sandbox graphics failure before any control tick is retained separately;
the same workload completed with native CoreGraphics access.

Reproduce with the completed pilot checkpoints:

```sh
.venv/bin/python scripts/diag_v2_ball_grounding_rollouts.py \
  --out outputs/v2/ball_grounding_rollouts_NEW
```
