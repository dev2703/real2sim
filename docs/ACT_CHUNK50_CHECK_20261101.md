# Frozen ACT at 25 Hz with 50-action chunks

The v2 checkpoint already predicts 50 actions. Its saved execution setting uses
the first ten, then observes and predicts again. This check also executes all
50 actions before refreshing the observation: two simulated seconds per chunk
at 25 Hz, compared with 0.4 seconds for ten actions.

Both runs use the unchanged `latest/` v2 checkpoint, five matched seeds
14000–14004 per arm, native starts, 30-second horizons and no RGB recovery.
Weights SHA-256:
`5dab91076d1e3b274674a888f522ce5fdddaf69ddd05c1adf3bce1742b99f65c`.

| Executed actions per prediction | SO-101 pick-and-place | Panda pick-and-place | Aborted trials |
| --- | ---: | ---: | --- |
| 10, saved setting | 0/5 | 0/5 | None |
| 50, requested full chunk | 0/5 | 0/5 | Two Panda trials |

Neither setting produced a held lift. All non-aborted trials failed approach.
With 50 executed actions, closest tool-to-ball distance was 0.329–0.363 m on
SO-101 and 0.336–0.370 m on Panda. Panda seeds 14003 and 14004 stopped after
698 and 148 successful ticks, respectively: the next source arm command exceeded
the unchanged SO-101 limits. Both failures occurred on the 49th queued action
(offset 48), before `env.step`. The failing action itself is absent from the
trace, so the exact offending joint is not established. Limits were preserved.

## Timing

Every complete trial has 750 simulated control ticks. Executing ten actions
requires 75 predictions; executing 50 requires 15, an 80% reduction in calls.
Actual prepared SO-101/source/Panda configurations and trace clocks all verify
25 Hz, with exactly the expected prediction cadence.

| Observation | Execute 10: SO-101 / Panda | Execute 50: SO-101 / Panda |
| --- | --- | --- |
| Mean policy-call time including rendering/projection | 149 / 141 ms | 118 / 116 ms |
| Mean control-tick computation | 16.80 / 15.19 ms | 3.84 / 3.23 ms |

Timings are exploratory local observations; the ten-action run overlapped
regression execution. Panda's 50-action mean also includes early-aborted trials.
The exact reduction in prediction count is stronger evidence than timing ratios.
Synchronous inference pauses simulated physics. Each prediction still exceeds
the 40 ms period of a 25 Hz real-time loop; reducing call frequency does not
establish uninterrupted 25 Hz wall-clock operation. At a one-in-50 call rate,
the 95th-percentile control-tick time can hide inference stalls. The comparison
artifact therefore also records call-specific latency, p99/max tick latency and
counts of ticks taking longer than 40 ms.

## Reproduction and validation

The benchmark now accepts `--execute-steps` without modifying saved model files.
Its protocol records chunk size, saved execution setting, requested execution
setting and timing model; changing the prefix invalidates resume into an old run.

```sh
.venv/bin/python scripts/benchmark_frozen_act.py \
  --robots so101 panda --conditions raw --execute-steps 50 \
  --seeds 14000 14001 14002 14003 14004 \
  --out outputs/v2/act_chunk50_NEW
```

Use `--execute-steps 10` in a separate directory for the reference condition.
macOS offscreen graphics access is required for these simulations.

All 20 trial/trace pairs passed the independent evidence audit. A separate cadence
check verified matching resets across settings, unchanged weights/configuration
and runtime hashes, 25 Hz prepared scenes, raw ACT-only controller accounting,
and observation refreshes every ten or 50 ticks. Thirteen new prefix tests and
25 existing benchmark/evidence tests passed; Ruff and whitespace checks pass for
the changed benchmark and tests.

Evidence is retained under ignored `outputs/v2/act_chunk50_20261101/`:

- `execute_10/` and `execute_50/`: protocols, all trial reports/traces and copied
  executed Python sources.
- `evidence_execute_10.json`, `evidence_execute_50.json`: each verifies ten trials.
- `compare_cadence.py`, `cadence_comparison.json`: reproducible independent
  cadence, scene, reset, hash and timing comparison.

Executing the complete chunk reduces inference frequency but did not resolve
pick-and-place on these seeds. It also exposes source-limit violations in two
Panda chunk tails that were not executed by the ten-action setting.
