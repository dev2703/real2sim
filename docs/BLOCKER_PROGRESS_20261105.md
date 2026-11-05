# Blocker progress — 5 November 2026

Two execution defects are fixed: the native loader now accepts the v3 export's
default dtype metadata, and learned SO-101 recovery can issue a separately
guarded jaw command when its arm request is rejected. Fresh checkpoint-specific
rollouts still fail raw ACT approach on both arms. Physical registration and
reliable learned/source-only recovery remain unresolved.

The work starts from `cf98ffd`. All new artifacts below are local, ignored
outputs under [`outputs/v2/blockers_20261105/`](../outputs/v2/blockers_20261105/).
No physical robot was commanded.

## Export identity

Authenticated Hub metadata and accompanying saved training states were inspected
at immutable revisions. Requested training duration is distinct from the scalar
step stored beside exported weights.

| Export | Saved training step | Basis |
| --- | ---: | --- |
| `so101-act-h256-b16`, root | 15,000 | Downloaded `training_state.pt`; verified Hub SHA-256 |
| `so101-act-h256-b16`, `latest/` | 30,000 | Retained `training_state.pt` |
| `so101-act-h512-b32`, root and `latest/` | 5,000 | Identical state hashes; downloaded state verified |
| `so101-act-h512-b16`, root / `latest/` | 4,800 / 5,000 | Both downloaded states verified |
| `act_so101_pickplace_v3_varied_kaggle` | Unknown | Provenance requests 10,000; no exported training state |

No verified 40k export was found in those repositories or the inspected caches.
The h256 experiment requests 40k updates, but its accompanying saved states do
not establish that duration. A different 40k artifact still needs an exact path
or repository/revision before its performance can be assessed.

The v2 revision is `e696c8a078935432e3501e2c28c1fb2598fbff7e`; v3 is
`076452e6024c778a2bffabe99ecf33776de141e1`. Exact additional revisions and hashes
are retained in [saved training steps](../outputs/v2/blockers_20261105/checkpoint/other_candidates/saved_training_steps.json),
[export validation](../outputs/v2/blockers_20261105/checkpoint/validation.json)
and the pinned metadata/provenance beside them. Training state was read with
`torch.load(weights_only=True)`.

## V3 loading fix and matched task evaluation

LeRobot 0.6.2 saves `dtype: null`; the pinned 0.6.1 configuration decoder rejects
that field. `load_native_act_config` omits only this known default in memory when
the installed loader lacks it. Explicit dtype requests and other unknown or
invalid fields still fail. Strict weight loading, saved processors and original
export bytes are preserved. All three locally available ACT exports passed
finite `(50, 6)` inference and saved-normalization checks.

Both v2 `latest/` and v3 were then evaluated with five fixed development seeds
12040–12044 per arm, raw/RGB/ACT→RGB conditions, matching reset fingerprints,
25 Hz control, saved 50-action chunks/ten executed actions, a 30-second horizon
and the existing measured-scalar scene. Each checkpoint has its own protocol.
The legacy matched jaw reset remains 1.2%; optical/joint/jaw registration remains
provisional. Source geometry, lighting, native limits and weights were retained.

| Checkpoint | Controller | SO-101 complete tasks | Panda complete tasks |
| --- | --- | ---: | ---: |
| V2 `latest/` (saved 30k) | Raw ACT | 0/5 | 0/5 |
| V2 `latest/` (saved 30k) | RGB geometric | 5/5 | 5/5 |
| V2 `latest/` (saved 30k) | ACT→RGB | 5/5 | 5/5 |
| V3 varied | Raw ACT | 0/5 | 0/5 |
| V3 varied | RGB geometric | 5/5 | 5/5 |
| V3 varied | ACT→RGB | 5/5 | 5/5 |

All 60 trial/trace pairs pass the independent evidence audit, with no aborts.
All 20 raw ACT trials failed approach, with no held lift or guard/IK rejections;
their closest tool-to-ball distances range from 0.343 to 0.378 m. Every ACT→RGB
trial switched at tick 50. Ten robot/seed reset fingerprints also match across
checkpoints. These are development simulation results; neither reserved
performance nor hardware reliability is established. Shared-host latency is not
compared.

Evidence: [cross-checkpoint comparison](../outputs/v2/blockers_20261105/checkpoint_comparison.json),
[v2 audit](../outputs/v2/blockers_20261105/matched_v2_latest/evidence.json),
[v3 audit](../outputs/v2/blockers_20261105/matched_v3/evidence.json). Each run
retains its protocol, reports, full traces and 54 verified source/config files.
The [benchmark driver](../outputs/v2/blockers_20261105/run_checkpoint_benchmarks.py)
and [comparison checker](../outputs/v2/blockers_20261105/summarize_checkpoint_results.py)
are retained beside the outputs.

## Guarded SO-101 jaw fallback

The previous SO-101 executor held its measured jaw whenever Cartesian arm IK
and all backtracking candidates failed. That suppressed close/open intent even
when a separate jaw-only command was feasible. Panda already retained jaw intent
on arm rejection. Retained failed SO-101 traces reproduced the suppression.

After arm rejection, the SO-101 executor now holds the measured arm and checks
the jaw-only request through the same actual-pose, workspace, native-limit,
static-collision and 17-sample command/rate-path gates. This includes the path
from the previous actuator target when control lags readback. Unsafe jaw requests
remain rejected. Arm rejection and zero arm feasibility stay recorded separately;
history receives achieved TCP motion and issued jaw intent. Traces declare
`native_common_cartesian_v2_guarded_so101_jaw_fallback` semantics.

A frozen joint-arm no-ACT head was compared using the exact old/new executor
sources on matched SO-101 development seeds 12040/12041, with no runtime teacher
or phase guard. Both versions completed **0/2** tasks with no lifts or aborts.
All four 750-tick trial/trace pairs pass the independent audit. On seed 12040,
requested-close/issued-open ticks fell from 33 to zero. Across the two new
trials, 1,149 jaw fallbacks passed their independent guards and 93 were rejected.
One failure advanced from approach to grasp, but reliable pickup is unresolved.

Evidence: [matched executor comparison](../outputs/v2/blockers_20261105/recovery/comparison_result_graphics.json),
[old audit](../outputs/v2/blockers_20261105/recovery/evidence_old_graphics.json),
[new audit](../outputs/v2/blockers_20261105/recovery/evidence_new_graphics.json).
The earlier zero-tick CoreGraphics sandbox aborts remain separately retained
and are excluded from this declared graphics-capable comparison.

## One corrective-training iteration

After the executor correction, a predeclared SO-101 DAgger collection used four
training seeds 10080–10083 and two validation seeds 11080/11081, distinct from
the original fitted roles and development seeds. The frozen original no-ACT
head visited states with expert execution probability 0.5. Five of six mixed
collection tasks completed, with 730 training/690 validation correction rows;
these mixture successes are collection evidence.

The new pack was deliberately aggregated with the original four immutable
SO-101/Panda packs. A no-ACT head was refit for the same 4,000-step budget,
training seed and saved phase-head hyperparameters: 6,445 training and 2,445
validation rows, with validation selecting step 2,700. Observations remained
tracked RGB, achieved motion/issued jaw history and grasp-relative geometry.

Original/refit heads then used the fixed executor on matched development seeds
12042/12043 per arm, without a runtime geometric teacher or phase guard:

| Head | SO-101 complete tasks | Panda complete tasks |
| --- | ---: | ---: |
| Original frozen no-ACT head | 0/2 | 1/2 |
| Refit after new corrections | 0/2 | 0/2 |

The refit is retained as experimental and is
[not promoted](../outputs/v2/blockers_20261105/recovery/corrective_iteration/promotion_decision.json).
One refit SO-101
trial achieved a held lift but aborted after 389 completed ticks when measured
`wrist_flex` exceeded its native upper limit by 10.355 mrad, beyond the existing
1 mrad soft-stop seed tolerance. Its last issued wrist command was in bounds;
the held lift had already been lost by tick 159. The retained trace does not
establish a physical cause. That abort remains a failure; limits and physics
were preserved. All original packs, head/checkpoint weights
and declared runtime inputs remained unchanged, and resets match. The original
packs include Panda, so this experiment makes no source-only transfer claim.

Evidence: [predeclared plan](../outputs/v2/blockers_20261105/recovery/corrective_iteration/plan.json),
[collection, fit and matched results](../outputs/v2/blockers_20261105/recovery/corrective_iteration/result.json),
[abort diagnostic](../outputs/v2/blockers_20261105/recovery/corrective_iteration/abort_diagnostic.json).
All eight evaluation trial/trace pairs pass the independent
[original](../outputs/v2/blockers_20261105/recovery/corrective_iteration/evidence_original.json)
and [refit](../outputs/v2/blockers_20261105/recovery/corrective_iteration/evidence_refit.json) audits.
Full collection/evaluation traces, source snapshots, head artifacts and fit
metrics are retained alongside these records. The small development comparison
does not establish recovery reliability; further corrective training did not
resolve the task blocker in this iteration.

## Physical registration dependency

The fresh [readiness audit](../outputs/v2/blockers_20261105/registration/readiness.json)
finds no completed independent optical/joint correspondences, measured jaw-gap
rows or loaded physical command traces. Encoder lattice checks support degree
units, while physical signs, zeros, camera/base transforms and contact geometry
remain unverified.

Prepared [camera inputs and capture checklist](../outputs/v2/blockers_20261105/registration/CAPTURE_CHECKLIST.md)
declare 12 individually measured target points, six fitting/two independent
verification views per camera and separate verification scoring points. Base/link
and jaw-gap measurements are also specified. The input inspections find missing
physical measurements with no structural input errors. Actual captures and
measurements are required before physical calibration can be accepted.

## Verification

- Final full unit suite: **1,351 passed, three skipped** (two CUDA-only and one optional
  bounded experiment).
- Focused native ACT/controller/source-frame integrations: **nine passed**.
- Native-config compatibility and retained ACT checks: **33 passed**.
- Recovery executor/feasibility checks, including lagged actuator control:
  **25 passed**; independent related review checks also passed.
- Calibration readiness/capture/optics checks: **19 passed**.
- Repository Ruff and whitespace checks pass. Focused counts overlap the full
  suite and are reported separately.

The final [unit log](../outputs/v2/blockers_20261105/unit_final.log) and
[validation manifest](../outputs/v2/blockers_20261105/validation.json) are retained.
