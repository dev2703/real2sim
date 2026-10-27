# Project documentation

The [27 October frozen generalization test](v2_frozen_generalization_20261027.md)
evaluates one unchanged semantic ACT on 20 new paired layouts and six separate
scene variations: 280 trials, with every failure retained. Nominal placement is
15/20 on Panda and 0/20 on SO-101; brighter lighting and smaller balls expose
initial-perception failures on both arms.

The [27 October frozen ACT follow-up](v2_frozen_act_followup_20261027.md)
distinguishes the successful simulation-trained models from the uploaded
real-demonstration ACT, and tracks the requested motion-interface reconstruction
and real-data replay/registration investigations. Frozen ACT is already used;
a shared checkpoint completing on both arms and real-data ACT transfer remain
unestablished. See that report before interpreting the historical status below.

The [26 October status](v2_calibration_status_20261026.md) supersedes the earlier
missing-measurement conclusion. The [operator measurements](v2_measurements_20261026.md)
are ingested, and the [RGB waypoint controller](v2_waypoint_results_20261026.md)
completes 4/5 placements on each arm with measured ball size and sparse IK.
These are geometric-control results; ACT success and frozen-policy transfer remain
separate outcomes. [External primary sources](v2_external_research_20261026.md)
inform the execution, timing and grasp-frame corrections.

The latest [fixed-base tracking](v2_fixed_landmark_tracking_20261025.md) separates local
capture-coordinate changes from raw mat-boundary changes across 115 frames. Two fixed
regions support 112 frames; late mat-edge measurements include explicit search-range
limits. Physical camera/base registration remains unresolved.
The [matched-object action probes](v2_matched_object_actions_20261025.md) hold arm inputs
fixed during ball/bowl image interventions and quantify weak action dependence.
The [three-episode capacity test](v2_three_episode_motion_20261025.md) applies movement
and stationary-run error gates against both hold and persistence. Labelled development
rollouts can still diagnose failures when that offline gate fails; it does not
certify or prohibit an actual grasp test.
The [executed-prefix comparison](v2_prefix_capacity_20261025.md) tests supervision on
the two executed actions with the same backbone, seed and three training episodes.
Training errors improve, but the capacity gate still fails and the
[matched object probes](v2_prefix_object_actions_20261025.md) show mixed action dependence.
The [motion-onset audit](v2_motion_onset_audit_20261025.md) describes conflicting
near-identical observations; the [capture sheet](v2_calibration_capture_20261025.md)
prepares independent camera/base and loaded hold/release evidence.

The earlier [vision alignment and bottleneck experiments](v2_vision_alignment_and_bottlenecks_20261024.md)
use local Grounding DINO/SAM2 on random source frames, fresh frozen-method confirmation,
an auxiliary keypoint bottleneck and matched next-follower joint targets. Camera
candidates fail confirmation and all new policy approach screens fail; no physical
mapping or checkpoint is promoted.

Start with the [24 October corrective experiments](v2_corrective_experiments_20261024.md)
for the ball-image input pilot, executed-prefix selection metrics, nominal CAD wrist
camera binding, calibration search, and control-target comparisons. The nominal
wrist camera exists in the pinned robot model; independent optical registration
and physical joint/jaw verification remain separate gates.

All 22 learning runs have completed. Start with the [completed results](v2_learning_results_20261024.md)
for the experiment comparisons and the [geometry/control investigation](v2_geometry_control_20261023.md)
for full learned-policy failures and corrective tests. The [learning screen](v2_learning_screen_20261022.md)
records the experiment budget, commands, data provenance and artifact locations. Read live
progress from `outputs/v2/learning_screen_20261022/STATUS.json`; dated documents
are not a substitute for the runner status.

| Topic | Reference |
|---|---|
| Two executed actions versus full-chunk supervision | [Prefix capacity comparison](v2_prefix_capacity_20261025.md) |
| Fixed object swaps after prefix supervision | [Prefix object/action probes](v2_prefix_object_actions_20261025.md) |
| Causal hold-to-motion label patterns | [Motion-onset audit](v2_motion_onset_audit_20261025.md) |
| Calibration sequence, session log and printable board | [Capture sheet](v2_calibration_capture_20261025.md) |
| Fixed-base tracks, raw mat edges and range limits | [25 October tracking](v2_fixed_landmark_tracking_20261025.md) |
| Matched arm states and ball/bowl action interventions | [Matched-object probes](v2_matched_object_actions_20261025.md) |
| Three training episodes versus hold/persistence | [Capacity and motion gate](v2_three_episode_motion_20261025.md) |
| Multi-height CAD identity evidence and capture plan | [Fixed-base landmark review](v2_fixed_base_landmark_review_20261024.md) |
| Random-frame vision measurements and new learning experiments | [Vision alignment and bottlenecks](v2_vision_alignment_and_bottlenecks_20261024.md) |
| Model camera estimates and base-contour ambiguity | [Model camera geometry](v2_model_camera_geometry_20261024.md) |
| Auxiliary localization versus action dependence | [Keypoint bottleneck](v2_aux_keypoint_bottleneck_20261024.md) |
| Feasible follower targets and guarded joint execution | [Follower targets](v2_follower_targets_20261024.md) |
| Corrective experiments and remaining evidence | [24 October follow-up](v2_corrective_experiments_20261024.md) |
| Existing wrist camera, measurement search and FK regeneration | [Calibration follow-up](v2_calibration_followup_20261024.md) |
| Executed-prefix and physical-unit selection metrics | [Motion selection](v2_motion_selection_20261024.md) |
| Feasible joint targets and soft orientation comparison | [Control contracts](v2_control_contract_comparison_20261024.md) |
| V2 preparation, evaluation and geometry gates | [V2 runbook](v2_implementation_20261020.md) |
| Results from all eight experiment groups | [Completed learning results](v2_learning_results_20261024.md) |
| Physical joint zeros and camera alignment | [Geometry findings](v2_geometry_alignment_20261021.md) |
| New camera, joint and feasible-carry comparisons | [23 October investigation](v2_geometry_control_20261023.md) |
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
