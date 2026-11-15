# Recovery retention and data follow-up — 15 November 2026

The selected phase head improves fresh confirmation completion from 4/10 to
7/10: SO101 stays at 3/5, while Panda improves from 1/5 to 4/5. All ten candidate
trials reach lift, but three still fail transport. These are small simulation
samples; robust both-arm recovery and physical calibration remain unresolved.

This experiment follows the stage-balanced DAgger head and retention-guard smoke
test. It keeps native ACT v2 frozen, the provisional scene, lighting, approach
handoff thresholds, and native execution acceptance thresholds unchanged.
Both embodiments provide supervision; this is not zero-shot transfer.

## Runtime provenance

The 22 collections and 80 benchmark trials below ran in the original local
checkout based on `38e2fb4`, with the exact dirty sources preserved in each
artifact's executed-source snapshot. That checkout predates the cleanup and
Panda retarget commits now included in publication base `6d666d1`.

The follow-up PR is stacked on the retention/training branches based on that
newer main. The historical 7/10 confirmation result must not be represented as
an evaluation of the publication runtime. Publication-branch checks and any
new smoke evidence are reported separately below. Saved data and model weights
remain local; the PR contains code, tests and this evidence record.

## Changes

- Collection accepts `--configured-gripper`, passing the scene's configured jaw
  reset to both live and shadow environments. This matches the configured-reset
  benchmark rather than the historical 1.2% collection override. The legacy
  collection default remains available and is recorded in manifests.
- Every attempt reports accepted, terminal, native-rejected and common-projection
  rejected teacher label counts. Rejected and terminal labels stay excluded.
- Retention assistance accepts `--retention-mode jaw` (benchmark option
  `retention_mode: "jaw"`). This retains freshness, loss/reacquisition, delivery
  and rejection handling while preserving the learner's pre-candidate arm motion
  and omitting speed/acceleration smoothing. `full` retains its prior behavior.
  Both modes remain optional execution assistance.
- The existing predicted-phase trainer accepts `--balance-phases`, using equal
  robot, episode and issued-command phase exposure. Teacher phases supervise
  training and never enter the runtime inputs.

## New data

Root: `outputs/v2/retention_data_20261115/`. Checkpoint:
`bottowork/so101-act-h256-b16`, revision
`e696c8a078935432e3501e2c28c1fb2598fbff7e/latest`, weights SHA256
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
Scene: `configs/v2/weight_space_20261111/sim_so101.yaml`.

Six training layouts (10340–10345) and two validation layouts (11340–11341)
per arm. Each executes frozen ACT until the existing observable stall/timeout
gate, then a teacher through the guarded common executor. Teacher-only
collection succeeded 7/8 on SO101 and 8/8 on Panda, without execution rejections
or aborts. The failed SO101 training attempt remains recorded. These are
teacher collection outcomes, not learned-policy performance.

The packs contain 1,695 training and 578 validation rows. New training data
includes 179 SO101 / 239 Panda TRANSPORT and 105 / 126 RELEASE labels.
`coverage_audit.json` verifies saved packs, source snapshots, row counts, phase
coverage and split integrity against the previous head's input contract.

Residual fits use MLX, stage balancing and 6,000 steps; phase fits use the existing
PyTorch trainer on CPU. Validation chooses the
checkpoint. The augmented fit combines the previous six packs with both new
packs (9,601 train / 3,615 validation rows). A separate residual fit uses only
the new packs. The existing predicted-phase model is also fitted, using hard
predicted component routing. All exports retain existing checksummed interfaces.

## Evaluation protocol

Jaw-only assistance uses the prior five seeds 12220–12224 to isolate the previous
head's execution changes. New head comparisons use fresh development layouts
12240–12244 per arm with identical configured-jaw resets. Candidate selection
maximizes the worse-arm completion count, then total completions. An unassisted
residual is preferred over added assistance on a tie. Confirmation layouts
12340–12344 remain separate from fitting and development selection.

The clean-data-only predicted-phase fit is retained as a fitting pilot; the
phase model intended for the rollout comparison uses the aggregate data to
retain learner-visited coverage. Neither action validation error nor phase
classification accuracy establishes completed manipulation.

A further Panda DAgger round uses the aggregate phase head, 80% teacher action
probability, four new training seeds 10350–10353 and validation 11350–11351.
All six assisted collection attempts succeeded, without rejections or aborts,
adding 555 train / 286 validation rows. Each attempt executes 23–39 learner
actions. Teacher commands are queried at those actual learner-visited states.
This is assisted collection success, not the learner's success rate.
`dagger_coverage_audit.json` checks this additional pack. A phase head is refitted
on all nine packs with the same architecture, routing and training budget.

The first completed development comparison is:

| Head | SO101 complete | Panda complete |
| --- | ---: | ---: |
| Previous stage-balanced residual | 1/5 | 1/5 |
| New teacher packs only, residual | 0/5 | 1/5 |
| Previous + new teacher packs, residual | 3/5 | 0/5 |
| Same aggregate, predicted-phase head | 4/5 | 3/5 |
| Phase head + targeted Panda DAgger round | 1/5 | 4/5 |

The phase head reaches lift on 4/5 SO101 and 5/5 Panda layouts. Its remaining
failures are one SO101 grasp and two Panda transports. Native rejections drop
from 631 to 141 on SO101 and 1,130 to 862 on Panda; these totals include time
spent in failed states and do not establish improved reachability everywhere.

On the separate older five-seed ablation, jaw-only retention completes 1/5 on
each arm, versus the earlier unguarded 2/5 SO101 and 0/5 Panda. Total completions
do not improve, and both modes still have losses under closed jaw commands.
That assistance is not enabled in the new-head comparisons.

The targeted Panda round improves Panda's development score but regresses
SO101. It is retained as evidence and **not selected as the shared controller**.
Across both new collection rounds there are 22 attempted episodes, 2,250 new
training rows and 864 validation rows. The selected shared head uses the original
eight-pack aggregate, not the final Panda-only pack.

`selected_candidate.json` freezes `head_phase_augmented` using development
reports only and records their hashes. No confirmation report was read during
selection. The chosen model predicts nine nonterminal control components from
causal observation history, frozen ACT proposals and decoder features. It uses
SiLU and hard predicted component routing, with no additional retention or phase
guard. Expert phases remain training labels. It reuses the existing phase model
with new coverage and balanced sampling.

## Fresh confirmation and remaining failures

After candidate selection, the frozen chosen head is evaluated on layouts
12340–12344 per arm against the previous head, with identical initial-state
fingerprints. The baseline was executed earlier for scheduling efficiency; its
outcomes were not inspected until candidate selection was frozen.

| Head | SO101 complete | Panda complete | Total |
| --- | ---: | ---: | ---: |
| Previous head | 3/5 | 1/5 | 4/10 |
| Selected phase head | 3/5 | 4/5 | 7/10 |

All ten selected-head trials reach a scored lift. SO101 fails transport on
12342/12344, both losing sustained holding while issuing an open command outside
the bowl. Panda fails transport on 12340 with a closed-command loss. These
retrospective signatures do not establish the physical cause of each drop.
Failed SO101/Panda trials retain ball-reference ages up to 18.44/20.32 seconds.
The controller still lacks direct continuous observation of the carried ball.

SO101 execution rejections increase from 36 to 445 on this confirmation set;
Panda decreases from 1,116 to 446. SO101 seed 12344 and Panda seed 12340 account
for 360 and 446 rejections respectively. Thus reachability/loss recovery is
**not solved**, despite the higher task completion. Do not relax native guards
or interpret aggregate rejection counts as a universal kinematic improvement.

The selected head's SHA256 is
`5473fe93c10f52d716832de882d5f5d0aeb91b8358d4647a2f59aafe6e9eded7`.
It is an experimental candidate for further simulation, not a reliable controller
or a real-robot validation. The previous default routing and ACT weights remain
unchanged. Development scores selected the design, so they must not be pooled
with confirmation as an unbiased reliability estimate.

All eight benchmark reports (80 trial/trace records) pass evidence audits.
`summary.json` records cross-report initial-state agreement and retrospective
loss diagnostics; each run has `evidence_audit.json`. All three new collection
packs complete without source drift. The two coverage audits verify data roles
and retained source snapshots. No model was refitted after reading confirmation.

## Reuse the selected artifact

The selected weights and checksummed metadata are local under
`outputs/v2/retention_data_20261115/head_phase_augmented/`. Exact fitting commands
are retained in `training_commands.json`; scene/reset/data roles and candidate
selection are in `plan.json`. Preserve these artifacts rather than overwriting
an experiment directory. Set `SNAPSHOT` to the pinned v2 `latest/` directory.

```sh
.venv/bin/python scripts/benchmark_frozen_act.py --snapshot "$SNAPSHOT" \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --robots so101 panda --conditions hf_learned --seeds 12440 12441 \
  --configured-gripper \
  --controller-factory scripts.run_recovery_dagger:controller_factory \
  --controller-options '{"head":"outputs/v2/retention_data_20261115/head_phase_augmented"}' \
  --out outputs/v2/retention_candidate_NEW
```

Overlapping jobs are not an isolated latency benchmark. The current tracker
still estimates supported on-table objects. Physical camera calibration, live
carried-object tracking and robust post-loss reacquisition remain future work.

## Validation

Original experiment checkout: 1,583 unit tests passed, three skipped. Focused
recovery/scene/retention/phase suite: 90 passed. Ruff and `git diff --check` pass.

Publication branch, stacked on `codex/recovery-retention-sequencing` (`3c8fd3e`):

- Focused suite: 90 passed. Repository-wide Ruff and `git diff --check` pass.
- Full unit suite: 1,232 passed, two skipped, two failures and three errors.
  All five broken cases are in `test_v2_wrist_camera_diagnostics.py`, which
  references deleted `diag_v2_wrist_zeros.py` and `diag_v2_camera_uncertainty.py`.
  The exact same five failures/errors reproduce in unchanged main `6d666d1`.
  They are inherited cleanup failures and are not masked by this PR.
- Fresh matched publication-runtime smoke, seed 12540 per arm: the previous head
  fails transport on both arms; the selected candidate fails SO101 transport
  and completes Panda. No trial aborts. All four trial/trace records pass
  evidence audits, and per-arm initial-state hashes match across policies.
  Artifacts: `outputs/v2/retention_pr_20261115/previous/` and `candidate/`.
  One layout per arm is a smoke check, not a revalidation of the historical
  confirmation rate on the newer retarget runtime.

No model or threshold was tuned on the publication smoke seed. Broader validation
and, if needed, fresh corrective collection on the newer runtime are still
required before promoting the head. The PR leaves retention assistance disabled
by default and keeps model weights and collection packs local.
