# Latest HF ACT on both arms — 28 October 2026

The latest frozen Hugging Face ACT **with explicit RGB geometric recovery**
completed **10/10 SO101 and 10/10 Panda** fresh simulation tasks. Each task
required a held lift, delivery, release and one second of resting placement under
the existing scorer. HF-only transfer still fails approach. The working system
retains the HF policy and hands control to a geometric controller when its
initial approach stalls; these results do not establish successful unassisted
ACT transfer or a trained recovery MLP.

The [curated report](../results/v2/hf_both_arms_20261028.json) links complete
reports, traces, videos, source hashes and the matched-start raw comparisons.
Lighting, HF weights, actuator/contact parameters, native joint limits and IK
acceptance tolerances remain unchanged. Panda's task workspace box was corrected
to include the measured task footprint; its native limits remain enforced.

| Controller and layouts | SO101 complete placements | Panda complete placements |
| --- | ---: | ---: |
| HF + RGB recovery, development seeds 300–304 | 5/5 | 5/5 |
| Same implementation, fresh seeds 500–509 | 10/10 | 10/10 |
| Latest HF alone, matched initial gripper, seeds 300–304 | 0/5 | 0/5 |
| Separate RGB geometric baseline, seeds 300–304 | 5/5 | 5/5 |

Every fresh assisted trial executes 50 native HF actions at 25 Hz, then detects
less than 3 mm of approach progress over two seconds and transfers control.
SO101 then needs 198–219 recovery ticks; Panda needs 264–304. Thus the geometric
controller supplies most of the successful execution. All 20 fresh trials remain
in the denominator, with zero aborts and zero IK rejections. All 20 videos fully
decode. No trial changes its source or checkpoint hashes during execution.

## Measurements are present and applied

`physical_measurements.md` supplies ball circumference, bowl diameter, mat size,
base-to-ball components, ball-to-bowl distance and camera-to-ball distances with
±5% uncertainty. The [measurement audit](../results/v2/measurement_application_reaudit_20261028.json)
verifies these scalars in both scenes. Referring to the measurement file as
missing was incorrect. Remaining registration questions concern the transform
between the operator's length/width axes and the robot CAD axes, joint zero
offsets, and both cameras' optical orientation/intrinsics. Those transforms are
not determined by the supplied distances alone.

The executable scene preserves its explicit assumptions about bowl bearing,
outer diameter and camera orientation. Its geometric controller knows the supplied
camera model and static collision map. It obtains movable-ball goals from RGB;
simulator ball poses, held flags and success flags never drive its control.
An initially observed static bowl can be remembered if an arm shadow later
prevents its detection. A fresh ball detection is always required at recovery.

## Fixes

- Active evaluation now selects HF `latest/`, rather than the different root
  weights. The live Hub revision was verified as
  `e696c8a078935432e3501e2c28c1fb2598fbff7e`; latest weights SHA-256 is
  `5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
  Both cameras, saved processors, absolute degree/percent actions, 50-action
  chunks and 10-action execution prefixes are retained.
- RGB bowl detection scans fixed image brightness levels and deduplicates
  repeated observations, preserving existing geometric and ambiguity gates.
  It changes image processing, not scene lighting.
- Panda's old x workspace minimum of 50 mm excluded several valid ball draws
  around the measured 50.8 mm centre. The workspace now includes the measured
  start disc, radius uncertainty and bowl footprint. Deterministic grasp-branch
  search retains collision/path/pose checks.
- Panda's virtual SO101 observation preserves commanded unobservable wrist
  motion while correcting observable position from actual Panda feedback.
  Previously even perfect execution could alter wrist feedback by 22 degrees.
- The recovery runner records its intervention, control source, fresh ball
  acquisition source, initial RGB targets, lighting and source hashes. Setup failures
  become aborted rows rather than disappearing from the denominator.
- Local residual-head training rejects cached decoder features from different
  HF weights. The existing root-trained local head is not a latest-checkpoint
  head and is not used by this working system. Its single diagnostic run did
  not complete the task; low offline validation error alone was insufficient.
- Measurement generation consumes current reported dimensions/uncertainty and
  preserves source provenance instead of retaining hardcoded camera values.

## Remaining bottlenecks

1. **Frame alignment.** Latest ACT tends toward native forward x, while the
   literal measured scene places the ball predominantly along task y. A
   training-only FK-derived axis hypothesis reduces closest approach from
   approximately 362/376 mm to 124/152 mm in two trials, but still completes
   zero tasks. This hypothesis is retained separately and has not replaced the
   measured scene by tuning to task success.
2. **Visual transfer.** At identical joints, swapping recorded and simulated
   cameras changes latest shoulder commands by about 15 degrees in the prior
   paired diagnostic. Panda's nominal wrist view initially omits the ball.
   Current optics/material/background and robot appearance differ from the
   physical inputs on which the user observed 4/5 successes.
3. **Proprioception and embodiment.** Physical shoulder starts around −104.8°
   remain incompatible with the native CAD −100° stop under zero-offset
   mapping. Panda has seven arm joints, and its position-only projection into
   five SO101 arm coordinates is lossy even after correcting command history.
4. **Unassisted recovery learning.** Current success depends on a geometric
   recovery controller. A learned recovery head would need on-policy recovery
   supervision and closed-loop evaluation under the corrected scene; a small
   teacher-feature fit did not establish that capability.

Ten fresh draws per arm establish working examples within the configured 3 cm
ball start disc and fixed bowl/camera arrangement. They do not establish physical
Panda performance or broad layout generalization. No hardware was commanded.

Validation: **924 unit and relevant integration tests passed**, with two CUDA
tests skipped; Ruff and `git diff --check` passed. All 20 fresh evaluation videos
fully decode. Maximum observed robot/table penetration was 0.576 mm for SO101
and zero for Panda, below the existing 2 mm acceptance tolerance.

## Reproduce

```sh
.venv/bin/python scripts/evaluate_hf_recovery.py --robot so101 \
  --seeds 500 501 502 503 504 505 506 507 508 509 --video \
  --out outputs/v2/hf_recovery_so101_NEW
.venv/bin/python scripts/evaluate_hf_recovery.py --robot panda \
  --seeds 500 501 502 503 504 505 506 507 508 509 --video \
  --out outputs/v2/hf_recovery_panda_NEW
```

Use `scripts/evaluate_hf_act.py --robot so101|panda --out <new-directory>` for the
separate HF-only evaluation. MuJoCo rendering on this Mac requires access to
CoreGraphics; sandboxed render attempts are retained as aborted diagnostics.
