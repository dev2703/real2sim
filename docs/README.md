# Project documentation

Start with the [learning screen](v2_learning_screen_20261022.md) for the current
experiment budget, commands, data provenance and artifact locations. Read live
progress from `outputs/v2/learning_screen_20261022/STATUS.json`; dated documents
are not a substitute for the runner status.

| Topic | Reference |
|---|---|
| V2 preparation, evaluation and geometry gates | [V2 runbook](v2_implementation_20261020.md) |
| Physical joint zeros and camera alignment | [Geometry findings](v2_geometry_alignment_20261021.md) |
| Calibration conventions | [Calibration](calibration.md) |
| Data and label contracts | [Data contract](data_contract.md) |
| Evaluation rules | [Experiment protocol](experiment_protocol.md) |
| Implementation structure | [Architecture](architecture.md) |
| Recording and operator inputs | [Recording checklist](recording_checklist.md), [operator session](v2_operator_session.md) |
| Collection timing and provenance | [Collection findings](collection_pipeline_findings_20261016.md) |
| Approved historical choices | [Decision register](decisions.md) |
| Known limitations | [Limitations](limitations.md) |

The architecture and decision register describe the original 10 Hz pipeline.
For the current 12.5 Hz two-camera screen, use the v2 runbooks and frozen configs.
Old system-design and implementation-plan documents duplicated these references
and have been removed. Historical experiment reports remain evidence, not instructions
to restart obsolete queues.

Cleanup on 22 October directly deleted stale optimizer states, superseded cached
features and a failed preparation cache. No new archive was created. See
`results/cleanup_20261022.json` for paths and sizes. Completed historical checkpoints
remain available for inference; their deleted optimizer states cannot be used for
exact continuation. Active runs, prepared data, pretrained weights, source,
tests, configs, geometry evidence and required robot assets were retained.
