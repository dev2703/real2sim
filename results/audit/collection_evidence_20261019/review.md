# Collection evidence, 19 October 2026

The local evidence has not gained a traced recording. [report.json](report.json) inventories
the two local source datasets and searches their roots plus `results/` for `SESSION.json`.
It found **zero trace sessions**. This is a statement about the listed local paths; a session
saved elsewhere would need its path before it can be inspected.

| Source | Episodes | Rows | Nominal FPS | Integer stride to 10 Hz |
|---|---:|---:|---:|---:|
| `so101_act_pickplace_v1` | 40 | 47,393 | 30 | 3 |
| `so101_act_pickplace_v2_25fps` | 1 | 467 | 25 | None |

Both sources have uniform stored timestamps, one saved `action` field, and no separate
requested/sent targets. Their real acquisition cadence, cumulative timing drift, and
per-tick limiter intervention cannot be recovered from those fields. Source hashes are in
the report. This inventory does not repeat the earlier video/geometry screens or reclassify
the successful real v2 demonstration. The older 37 usable training/validation episodes and
the new one-episode source remain separate; a larger nominal row count would not resolve
these missing measurements.

## Implemented diagnostic changes

[audit_recording_trace.py](../../../scripts/audit_recording_trace.py) now reports cumulative
host-observation-start drift against nominal row time, the interval distortion inherited by
nominal 10 Hz row selection, separate leader-to-processed and processed-to-sent differences,
send-side command-change fractions, prior-command age, and tracking residuals with long gaps
shown separately. It does not infer exact camera or motor acquisition times from host API
call starts. The velocity-scale diagnostic is the host observation interval divided by its
nominal interval, not an automatic correction to labels.

Dataset matching now checks stored follower observations as well as sent targets and row
timestamps, and explicitly lists any dataset episodes without a matching saved trace. Raw
observation parity assumes the collector's default identity position processor; a different
processor needs an explicit transform/provenance contract. Session markers are visible so a
saved episode in an interrupted session is not confused with a finalized collection.

The new [audit_collection_evidence.py](../../../scripts/audit_collection_evidence.py) makes this
inventory repeatable. Nominal timestamps and even a same-name/FPS completed sidecar do not
become proof of physical cadence: candidate sessions still require row-level trace checks.
The resampling module documentation now makes its nominal-clock assumption explicit.

Twenty focused tests pass, including a 10% host interval dilation, a long stall, delayed
feedback, noninteger 25-to-10 Hz conversion, mismatched observations, untraced dataset
episodes, discarded/interrupted attempts, and absent timing evidence. Ruff passes for the
changed audit scripts and tests. These are software tests, not physical collection results.

## Reproduce and collect the missing evidence

```sh
.venv/bin/python scripts/audit_collection_evidence.py \
  --dataset /Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v1 \
  --dataset /Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2_25fps \
  --trace-root results/collection_traces --trace-root results \
  --output results/audit/collection_evidence_next.json

.venv/bin/python -m pytest tests/unit/test_collection_evidence.py \
  tests/unit/test_recording_trace_audit.py tests/unit/test_recording_trace_integration.py -q
```

The next useful collection remains **five short pilots at nominal 20 FPS**, with the installed
`--recording_trace_dir` option enabled, the current hardware encoding/cameras/calibration,
and exact controller/limiter settings recorded. Include comparable ordinary unloaded and
ball-loaded holds, approach, closure, lift/transport, and release. Preserve the raw request,
processed request, returned sent target, and measured positions separately. Changing gains
or relaxing the limiter during this comparison would confound interpretation.

The working-command fragment and collector metadata details are in the
[collection findings](../../../docs/collection_pipeline_findings_20261016.md). After capture:

```sh
.venv/bin/python scripts/audit_recording_trace.py \
  results/collection_traces/session_UUID \
  --dataset /Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v3_20fps_trace \
  --policy-hz 10 --output results/audit/collection_v3_20fps_trace.json
```

Review actual cadence tails, drift, sent-label/observation parity, subsequent tracking and
loaded-versus-unloaded holds, alongside video integrity and freshness. Camera exposure times
and acquisition IDs remain unavailable through this interface. Once those pilots establish
reproducible recording/task behavior, scale to the proposed 50 episodes with a predefined
episode split and controlled object/approach variation. This work did not start hardware
collection, edit recorded data, or establish that dataset size alone causes the failures.
