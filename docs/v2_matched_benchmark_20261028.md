# Matched frozen-ACT benchmark

`scripts/benchmark_frozen_act.py` compares raw frozen ACT, the RGB geometric
controller alone, and frozen ACT with RGB takeover under one reset protocol.
It also accepts learned-recovery and no-ACT controller factories.

This distinction matters for Panda: the earlier standalone RGB runner used its
configured base/start, while the HF runner translated the base by −150 mm and
mapped the SO101 start. Comparing those reports directly does not isolate the
controller. This benchmark gives every condition the same adapted Panda start,
base translation, closed 1.2% source-gripper command, two cameras, 25 Hz control,
static collision map and task timeout. It records initial native state, object
layout, lighting and a reset fingerprint, and rejects differences across
conditions for the same robot/seed. Ball/bowl positions are recorded for scoring
and reset verification; no privileged object metrics enter controller inputs.

The development smoke uses seed 12000 for both arms. The reserved benchmark uses
seeds 13000–13099, separate from train 10000–10023, validation 11000–11005 and
development 12000–12009. Training and model selection must not consume reserved
outcomes. Initial lighting remains fixed in all conditions.

Each report includes complete-task success, aborted trials, failure phase,
intervention frequency, fraction of ticks using ACT, assistance fraction,
simulated completion/timeout time, wall time and measured control/inference
latency. ACT participation and assistance can overlap for a residual controller;
these quantities describe actual control use, not causal model contribution.
The no-ACT ablation tests whether frozen ACT inputs add value to a learned head.

The geometric controller is an executable teacher/fallback. A recovery MLP is
a learned model. DAgger is an iterative collection/training procedure which can
teach that model from corrective labels at learner-visited states; it is not
the controller which currently takes over.

## Completed 100-seed result (audited 29 October 2026)

The reserved nominal simulation benchmark completed 600 trials: 100 seeds ×
three conditions × two arms. Frozen weights and each arm's existing lighting
remained unchanged. Every condition used the same reset fingerprint for its
robot/seed; all 600 saved row hashes, trace hashes and trace lengths passed an
independent audit. All 54 declared source/configuration/checkpoint inputs still
matched their recorded hashes at audit time. The executed benchmark source is
preserved beside each original report.

| Controller | SO101 success | Panda success | SO101 mean simulated time | Panda mean simulated time |
| --- | ---: | ---: | ---: | ---: |
| Frozen ACT alone | 0/100 | 0/100 | 30.000 s (timeout) | 30.000 s (timeout) |
| RGB geometric alone | 100/100 | 100/100 | 9.058 s | 10.868 s |
| Frozen ACT, then RGB takeover | 100/100 | 100/100 | 10.302 s | 13.415 s |

There were no aborted trials or IK rejections. All raw-ACT failures were during
approach. Every ACT-plus-RGB trial triggered takeover at tick 50 (2.0 seconds).
ACT supplied an average 19.44% of SO101 ticks and 14.97% of Panda ticks;
geometric recovery supplied the remainder. With paired seeds, ACT-plus-RGB
completion took 1.244 seconds longer on SO101 and 2.547 seconds longer on Panda
than RGB alone. These comparisons provide no evidence that ACT improves success
in this nominal simulation; the geometric controller completes the task after
takeover.

Mean measured control latency for raw / RGB / ACT-plus-RGB was
25.51 / 2.61 / 9.51 ms on SO101 and 25.28 / 1.24 / 3.51 ms on Panda.
These are descriptive shared-host measurements across resumed batches;
inference spikes and rendering costs are included, and they are not a hardware
real-time guarantee. The curated result also records inference-call latency,
mean per-trial 95th-percentile latency, wall time and assistance counts.

For 100/100 successes, the two-sided exact 95% binomial interval is
96.38–100%; for 0/100 it is 0–3.62%. These intervals apply only to the tested
simulation distribution. They do not establish physical robot performance or
cover camera/domain mismatch. The user's reported 4/5 real-checkpoint result
remains separate evidence; this simulation failure does not show that the real
ACT checkpoint is poor.

The [curated audited result](../results/v2/matched_benchmark_100_20261029.json)
links original reports, protocol IDs, hashes and detailed counts. The
[alignment and fixed-lighting robustness report](v2_alignment_and_fixed_lighting_robustness_20261028.md)
examines the remaining camera mismatch and wider geometric-controller pilot.
Learned recovery is evaluated separately in the
[recovery MLP/DAgger report](v2_learned_recovery_dagger_20261028.md).

## Run and resume

```sh
.venv/bin/python scripts/benchmark_frozen_act.py \
  --out outputs/v2/matched_benchmark_PLAN --plan-only
.venv/bin/python scripts/benchmark_frozen_act.py --robots so101 \
  --out outputs/v2/matched_benchmark_so101_NEW
.venv/bin/python scripts/benchmark_frozen_act.py --robots panda \
  --out outputs/v2/matched_benchmark_panda_NEW
```

Repeating the exact command resumes completed attempts. A protocol hashes its
checkpoint, saved processors, source/configuration/static assets and chosen
seeds. Changed inputs require a new output directory. Partial attempts remain
on disk and are retried in a new attempt directory; completed failed/aborted
trials stay in the denominator. Resumption checks protocol/seed identity, row
and trace hashes, trace length, and unchanged runtime/lighting/checkpoint flags.
Source or lighting changes during a run terminate pooling with a clear error.

`--scenario-plan <json> --scenario-condition <name>` consumes a frozen layout
plan with explicit shared ball/bowl placements. Radius/friction conditions must
be declared as separate conditions and retain their unchanged-lighting evidence.

## Learned-controller integration

`SharedEpisode(robot, setup_directory, source_scene)` exposes `env`, `source`,
`guard`, `robot`, `reset(seed, scenario=None)` and `close()` for training and
evaluation. The setup directory must be new. `FrozenDriver(shared, native_act)`
executes the same saved HF prefix and native motion gates.

Pass `--controller-factory module:function` and JSON `--controller-options` for
`learned`/`no_act` conditions. The factory receives `env`, `source`, `robot`,
`snapshot`, `condition`, `options`; it returns a controller exposing `reset()`
and `step(obs, env) -> ApplyResult`. The returned detail can declare `uses_act`,
`intervention` and `inference_ms`. The factory never receives scoring metrics.
Model artifacts belong in an options `head`, `head_dir`, `checkpoint` or `model`
directory or file so they are included in protocol hashes. A custom controller
may expose `close()` to release feature hooks; the benchmark calls it after
completion or abort. Factory and no-ACT trial rows retain the reference snapshot
checkpoint hash even when the factory owns its policy. Direct-file and module
invocation accept the same `scripts.module:function` factory name.

After the completed 600-trial run, instrumentation was extended to hash these
model artifacts and the common/MLP/phase/tracker runtime modules, close custom
controllers, and support both CLI forms. These changes produce a **new protocol
ID**: use a new output directory for new runs. Original reports, protocol IDs and
executed source copies remain untouched. The curated result records historical
executed and patched script hashes, plus new plan-only protocol IDs. Fourteen
focused tests cover artifact-change rejection, provenance, cleanup on normal and
aborted trials, base-checkpoint binding, and equivalent CLI imports.
