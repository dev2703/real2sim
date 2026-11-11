# Native ACT weight-space investigation — 11 November 2026

Raw task completion remains the acceptance criterion. Lower teacher action error,
RGB teacher completion, brief contact, and a pushed-in ball are separate results.
All experiments use the confirmed v3 checkpoint, revision
`076452e6024c778a2bffabe99ecf33776de141e1`, and preserve the original export.

## Causes established by matched experiments

The historical scene places the ball 2 inches forward and 13 inches sideways.
The operator correction retained in `plan.md` says 13 inches forward and 2 inches
sideways. On development seed 12160, correcting these axes, rotating the previous
relative layout, and changing the provisional global FOV from 22.79° to 43°
reduced raw ACT closest approach from 353 to 99 mm on SO101 and 357 to 111 mm on
Panda. Neither raw trial completed the task. RGB control completed both arms in
both scenes. This supports a major scene/input mismatch; it does not isolate the
individual effects of axes and camera changes or establish physical calibration.

`configs/v2/weight_space_20261111/` retains the candidate separately. The measured
scene used by historical benchmarks is unchanged. Bowl/camera bearings and wrist
optics remain hypotheses, and Panda still uses the existing robot geometry
adapter.

Scenery increases the SO101 model extent enough that the default near plane is
84.85 mm. The new `render.near_clip_m` setting converts metres after model
compilation. A further integration bug in native `prepare_scene` discarded this
setting while replacing image dimensions; this is fixed. Actual benchmark setup
tests now assert a 5 mm near plane on both arms. Feature reports record the
effective value rather than relying only on the requested YAML value.

The previous collection supported only SO101. Panda has seven arm joints and a
metric jaw width, while native ACT emits five source angles and gripper percent.
The new collector labels Panda **applied** actuator targets through FK and the
same virtual-source projection used at execution, including the inverse 0–90%
jaw endpoint mapping. Unrepresentable or rejected targets are masked. Projection
does not uniquely recover five joints from a 3D point: its two unobserved
coordinates follow causal source command history. This is target-arm supervision,
not source-only transfer.

## Direct native weight edits

Both fitters balance arms, use separate training/validation episodes, include the
unchanged checkpoint in selection, and reject candidates that worsen either
arm's baseline validation MAE. Cache aggregation checks scene YAML, referenced
task/data YAMLs and scene preparation/render code, so pre-fix and post-fix camera
features cannot silently mix. ACT-prefix and rejected teacher commands never
become supervision, and unsuccessful teacher attempts remain in the evidence.
Float32 labels that round beyond a strict native joint stop are masked without
aborting the episode or loosening the execution guard.

The linear editor changes only `model.action_head.weight` and `.bias`.
The decoder editor additionally changes the final decoder layer's `linear1` and
`linear2` weights/biases: six existing tensors total. All other tensors, saved
processors, architecture, limits, and guards remain intact. Cached inputs to
the final feed-forward block exactly replay its native post-norm computation;
attention and upstream visual features remain frozen. Tests compare this replay
bit-for-bit with native two-layer ACT decoders for ReLU and GELU.

Initial both-arm head fits reduced validation error but did not complete raw
tasks. After a 50-tick corrective pass, Panda seed 12165 reached within 2.03 mm and
briefly held the ball, then reopened its jaw at a chunk boundary before a held
lift. Those grasp ticks were accepted by the native guard. Other failures
included below-table targets that the guards correctly rejected. These are
learned geometry and sequencing errors; relaxing guards would conceal them.

Training seeds are 10160–10162 and validation seeds 11160–11161, with roles
preserved across corrective rounds. Development seeds are separate. Early head
experiments and decoder caches named `ffn_features_round*` predate the camera
handoff fix. Caches named `ffn_near_features*` use verified 5 mm clipping.

## Final recorded results

The last fit aggregates teacher-only, 25-tick and 50-tick corrective collections:
26/30 teacher attempts succeeded, with 2,770 training and 1,995 validation prefix
actions. L1 fitting selected step 1,575. Robot-balanced validation MAE fell from
1.31934 to 0.14817 (88.8%). SO101 MAE fell from 1.75822 to 0.13611; Panda from
0.88046 to 0.16023. These values are normalized action errors.

Fresh matched development seeds 12168/12169 use verified 5 mm clipping, 25 Hz
control and the saved 10-action execution prefix:

| Arm | Original raw tasks | Final edited raw tasks | RGB teacher tasks |
| --- | ---: | ---: | ---: |
| SO101 | 0/2 | 0/2 | 2/2 |
| Panda | 0/2 | 0/2 | 2/2 |

The final edit improves closest SO101 approach on these seeds from
152/134 mm to 44/129 mm and Panda from 155/139 mm to 109/103 mm. All four
edited trials still fail approach. The earlier MSE decoder edit reached 12 mm
on a different Panda development seed but also failed the task. Validation
accuracy does not rank these policies reliably by closed-loop task outcome.
A separate timing ablation executes one action per inference on seed 12170;
both arms still fail. No export is promoted as a working raw policy.

Final artifacts are `final_baseline/`, `final_edited/`, `feedback1/`, and
`ffn_near_edited1/` under the local evidence directory. The final checkpoint hash
is `fd9d3e1d3293088dccd77b8c4a50f235e838a273e989b75460b25ac1a33298ab`.
Native strict loading, complete tensor comparison and processor byte comparison
pass: exactly six tensors (527,110 parameters) change, with relative delta L2
0.28368. Final evidence audits verify the retained trial/trace pairs and matched
reset accounting. Ruff passes; 139 focused unit/integration checks passed, and
the subsequent collector/stop-rounding regression run passed all 40 checks.
The full unit run's only three failures were sandbox-denied CoreGraphics
renders; all three passed with local graphics access.

The remaining blocker is learned closed-loop geometry and grasp/lift sequencing
on the policy's own states. The scene and interface fixes plus these limited
weight edits do not establish raw pick-and-place on either arm.

## Reproduce

Set `SNAPSHOT` to the original v3 native export. Use fresh output directories.

```sh
uv run python -m real2sim.policy.weight_edit_collection \
  --snapshot "$SNAPSHOT" --robots so101 panda \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --feature-layer decoder_ffn_input --out outputs/v2/ffn_features_NEW \
  --train-seeds 10160 10161 10162 --val-seeds 11160 11161

uv run python -m real2sim.policy.decoder_weight_edit \
  --snapshot "$SNAPSHOT" --data outputs/v2/ffn_features_NEW \
  --out outputs/v2/ffn_edited_NEW --steps 1500

uv run python scripts/benchmark_frozen_act.py \
  --snapshot outputs/v2/ffn_edited_NEW --robots so101 panda --conditions raw \
  --source-scene configs/v2/weight_space_20261111/sim_so101.yaml \
  --out outputs/v2/ffn_eval_NEW --seeds 12166 12167

uv run python scripts/audit_frozen_act_results.py outputs/v2/ffn_eval_NEW/report.json
```

For corrective collection, add `--rollin-snapshot EDITED --act-prefix-steps 50`
while keeping `--snapshot` pinned to the original feature base. Aggregate the
original and corrective caches with `--data BASE_CACHE CORRECTIVE_CACHE`.
The final recorded fit used all three near-clipping caches, `--loss l1`,
`--steps 2500`, and `--patience 16`; validation selected the earlier checkpoint.
Raw inference uses only the exported native model, live RGB/state, and existing
robot adapters/guards. Teacher object detections and scores never enter it.

Local evidence is under `outputs/v2/weight_space_20261111/`, including individual
trials, traces, teacher attempts, features, fit reports, and native exports.
