# DAgger recovery stages — 15 November 2026

The controlled comparison improved SO101 recovery but did not establish reliable
Panda recovery. The dominant remaining failures are retention and transport,
not failure to activate the entry gate.

## Results on identical fresh resets

Five new development seeds per arm, 12220–12224; these never entered fitting.
Every policy first executes the same frozen HF ACT v2 checkpoint, then the same
observable latched stall gate. All 30 trial/trace records passed evidence audits;
initial-state hashes match across the three policy reports for each arm/seed.

| Recovery head | SO101 complete tasks | Panda complete tasks |
| --- | ---: | ---: |
| Previous teacher-only MLP | 0/5 | 0/5 |
| Aggregated DAgger, ordinary sampling | 1/5 | 0/5 |
| Aggregated DAgger, stage-balanced sampling | 2/5 | 0/5 |

The earlier Panda 1/2 result used different seeds; it is not a matched estimate
of regression. Five trials per arm remain development evidence, not a reliable
success-rate estimate. No trial outcomes selected training checkpoints.

Stage-balanced SO101 reached a scored lift on all five trials: two completed,
two failed transport, one failed release. Panda reached a scored lift on four
trials; four failed transport and one failed grasp. Ordinary DAgger had two
SO101 approach failures, two grasp failures, and one completion; Panda failed
grasp twice and transport three times.

## What was changed

1. Added `--act-prefix-until-stall` to corrective collection. It uses the same
   `ApproachMonitor` and shared RGB-to-grasp distance calculation as runtime,
   rather than always starting recovery after 50 ACT ticks. Actual handoffs
   include 50, 80 and 125 ticks. Reference ages are retained through handoff.
2. Collected two rounds of expert labels at actual learner-visited physical
   states, using shadow simulation to query the expert. The executed-action
   mixtures use 50% teacher probability in round one and 20% in round two.
   The second round uses the selected first-round head. Labels include issued
   command stage IDs and ticks; teacher phase is never a runtime observation.
3. Aggregated original examples plus both DAgger rounds and trained residual
   MLPs with MLX/Metal. Added optional `--balance-phases`: equal robot exposure,
   then equal episode exposure, then equal issued-command stage exposure within
   that episode. This changes sampling, not runtime architecture or action gates.
4. Fitted an ordinary-sampling control on exactly the same aggregate data and
   training budget. All heads retain SiLU, 256/128 hidden widths, the frozen ACT
   prior/features, achieved-motion history, grasp-relative RGB observations,
   and task-event jaw clocks. Existing native guards remain enabled.

Checkpoint: `bottowork/so101-act-h256-b16`, revision
`e696c8a078935432e3501e2c28c1fb2598fbff7e`, latest export, SHA256
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.
This is the v2/dataset-v2 checkpoint from the preceding retry. ACT stays frozen;
only the correction head is fitted. Runtime export remains PyTorch-compatible;
MLX performs training.

## Data and training

- 32 new corrective attempts: 2 rounds × 2 arms × (6 train + 2 validation).
- New training seeds 10220–10225; validation 11220–11221. Roles remain fixed
  across rounds. Original packs retain their original train/validation roles.
- Aggregate: **7,906 training rows and 3,037 validation rows**, up from 1,090 and
  537 respectively. These counts are labeled states, not independent episodes.
- Original and new packs include failed attempts. Terminal failed-expert idle
  holds are excluded from labels, so not every learner step supplies supervision.
- Round-one fit selected step 1,000/6,000. Final stage-balanced fit selected
  5,000/6,000; ordinary fit selected 2,600/6,000.
- Final held-out robot/episode-balanced normalized action MAE: stage-balanced
  0.086729, ordinary 0.078722. The lower-MAE ordinary head completed fewer SO101
  tasks; action loss remains insufficient as a task-success ranking.

The original four training layouts per arm are supplemented by six new layouts,
revisited across DAgger rounds. This is not 32 independent new layout seeds.
Collection retains the collector's 1.2%-jaw reset override; all evaluation
policies share the configured-jaw reset. Thus gate semantics match but the
collection and evaluation initial jaw distributions are not identical. Scene
geometry remains the explicitly selected provisional
`configs/v2/weight_space_20261111/sim_so101.yaml`.

## Current blockers and interpretation

**1. Stage transitions and retention remain the main learned-policy failure.**
The previous head issued 26–66 semantic open/closed threshold switches per new
trial. Ordinary DAgger reduced that to 0–8; stage balancing to 4–31. That is a
large reduction in command oscillation, not proof of stable grasp. The
stage-balanced head reaches later stages, but sometimes opens before successful
delivery. For example, SO101 seed 12221 loses its held interval while commanding
about 0.88 jaw opening and ultimately fails transport. Some other losses occur
while commanding closed, so a jaw threshold alone cannot explain or fix them.

**2. Unsafe/unreachable proposals and stale target memory persist after failure.**
Stage-balanced Panda transport failures include 147, 431 and 234 IK rejections
(on seeds 12220, 12222, 12224). The execution guards correctly block those
proposals. Terminal ball-reference ages are 7–24 seconds in several failed
rollouts. The controller can continue acting after loss or escape using aged
RGB memory. These are logged observations, not proof that every rejection or
stale frame caused the original loss. Privileged held/object state was used only
for post-hoc diagnosis and scoring.

**3. The correction teacher is not a universal recovery oracle.**
SO101 mixed collection completed 2/8 in each DAgger round; Panda completed 5/8
in round one and 0/8 in learner-heavy round two. These are assisted collection
outcomes, not standalone model scores. Some states become terminal or the object
escapes reach. The expert cannot supply meaningful recovery commands indefinitely.
More steps spent in those states are not automatically useful training data.

**4. Neither a new entry activation nor blindly longer training is established
as the fix.** The entry gate fired in these rollouts; failures occur afterward.
All candidate runtime activations and entry thresholds were held fixed to
isolate data/sampling effects. Earlier fitting overfit beyond step 500; with the
larger DAgger dataset, the stage-balanced model benefited up to step 5,000, while
the ordinary model selected 2,600. Training length should follow validation and
closed-loop outcomes rather than a fixed assumption that more is better.

The next focused policy change should explicitly model grasp/retain/transport/
release transitions using causal measured jaw, achieved motion and fresh RGB
feedback. Test a hysteretic release decision and loss/reacquisition handling;
train on transport/drop corrections and accepted executable actions. The
existing grasp monitor/phase-policy code provides a starting point, but this
experiment did not activate an additional heuristic guard or fit a recurrent
stage model. Do not substitute simulator truth into policy inputs or relax IK
acceptance to make the scores improve. A broader manipulation dataset or larger
backbone is not yet shown to be the necessary remedy.

## Evidence and reproduction

Root: `outputs/v2/dagger_stages_20261115/`.

- `plan.json`: exact collection, MLX training and evaluation commands.
- `round1_so101/`, `round1_panda/`, `round2_so101/`, `round2_panda/`: immutable
  NPZ packs, attempted-episode outcomes, handoff traces and executed sources.
- `head_round1/`, `head_round2/`, `head_unbalanced/`: weights and training reports.
- `eval_previous/`, `eval_dagger_stage/`, `eval_dagger_plain/`: 30 complete trials
  and per-report `evidence_audit.json` files.
- `summary.json`: aggregate results, stage counts, held-out stage errors, jaw
  switch diagnostics and cross-policy reset verification.
- `transport_diagnostic.json`: held intervals, jaw commands and reference ages.

A direct-script import issue in the new stage-loading path was corrected before
fitting; the failed attempt log is retained and did not produce a head. The
interrupted evaluation supervisor was resumed using hash-validated existing
trials, and all final reports are complete. The final two evaluations overlapped
in wall time; latency is not an isolated performance benchmark.

Validation: **127 focused tests passed**, including MLX optimization/export,
phase sampling, collection option validation, frozen handoff and workflow
regressions. Ruff and `git diff --check` pass. Neither candidate is promoted as
a reliable both-arm controller.
