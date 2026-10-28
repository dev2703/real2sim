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
Model artifacts belong in an options `head_dir`, `checkpoint` or `model`
directory so they are included in protocol hashes.
