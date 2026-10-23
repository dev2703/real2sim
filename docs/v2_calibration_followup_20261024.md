# Camera mount, physical evidence and FK label lineage — 24 October 2026

The pinned SO-101 already contains the wrist mount **and nominal optical camera**.
`camera_mount` includes the mesh, PCB/lens and collision boxes; `wrist_cam` defines a
fixed optical pose and 61° vertical FOV. An explicit `observation.images.top` binding is
now available for provisional two-camera experiments. Physical camera correspondence
and calibration remain separate from the presence of these model assets.

The [binding diagnostic](../scripts/diag_v2_nominal_wrist_mount.py) writes isolated
task/data/simulator configurations with a new contract version. It copies the compiled
native camera's local position and normalized quaternion axes, preserving the existing
mount collisions. The [generated report](../outputs/v2/nominal_wrist_binding_20261024_v2/report.json)
records actual rendering of both inputs at `[3,192,256]`. Across five joint poses the
native and bound cameras have zero position difference and a maximum axis difference
of 1.25×10⁻¹⁶. Collision geometry/exclusions and joint/actuator limits are identical to
the reference scene. [iPhone](../outputs/v2/nominal_wrist_binding_20261024_v2/iphone.png)
and [nominal wrist](../outputs/v2/nominal_wrist_binding_20261024_v2/top_nominal_cad.png)
renders are retained. This verifies software binding, not a physical optical transform.
No active production task was replaced, and existing checkpoint/task hashes remain enforced.

## Local measurement search

The user's statement that physical measurements exist prompted a broader local search,
including hidden and ignored files. The [search record](../results/v2/calibration_measurement_search_20261024.json)
lists exact locations and hashes/classifies 13 plausible sources. Searches covered
`~/Downloads/real2sim`, the sibling `~/Downloads/lerobot`, Downloads sibling projects,
Desktop/Documents, the LeRobot Hugging Face cache, `~/data/ros2_ws`, `ros2_ws_hw2`, and
ROS/Gazebo configuration folders. `~/.cache/calibration` was not present. Git internals,
virtual environments, dependency/weight caches and unrelated private document contents
were excluded. Filename/content searches used `rg --files -uuu` and
`rg --no-ignore --hidden`; plausible JSON/YAML/CSV sources were inspected directly.

Actual scalar measurements already exist and remain in use: table height **0.737 m**,
mat dimensions **0.305 × 0.406 m**, camera distance/height **24/21 inches** with stated
±3% uncertainty, and supplied bowl dimensions. The [existing inventory](../results/v2/config_measurements_inventory_20261022.json)
separates these operator measurements from image-fitted scene coordinates.

| Located evidence | What it establishes | Remaining applicability check |
|---|---|---|
| Current follower/leader motor calibration JSON and older backups | Encoder homing offsets and endpoint ranges | Physical arm sign/zero alignment, source-recording provenance and jaw clearance |
| `camera_measured.json` in the historical camera audit | Camera fit constrained by operator scalars, provisional FK and mat pixels; joint/mat RMS 77.57/40.78 px | Independent source-resolution intrinsics/distortion and task/base pose |
| Historical joint-horn and jaw-tip pixel annotations | Approximate real image observations, with 10–25 px declared uncertainty | Metric jaw gap, verified landmark identities and independently fixed camera; these are v1 annotations |
| Static CAD landmark inventory | Nominal model coordinates | Entries explicitly retain `physical_identity_verified=false` |
| Root-cause hold annotation CSV | Intended loaded/unloaded conditions | Hold frame intervals are blank; it is an annotation plan |
| ROS camera xacro and `so101_new_calib.urdf` | Configurable camera defaults and generated Onshape geometry | No independent optical or measured-joint observation provenance |

Additional independent intrinsics/distortion, registered static/joint metric
correspondences, jaw-gap measurements and physical loaded hold/release traces did not
surface in those searched locations. This scoped search does not establish absence
elsewhere. The existing independent-geometry and measurement templates are still unfilled;
their nulls are not replaced with nominal defaults. The [readiness audit](../results/v2/calibration_contract_audit_20261024.json)
records the inspected inputs and hashes. The user-reported measurements should be linked
to these contracts and checked for capture/setup applicability before promotion.

The cached dataset is `~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2`:
41 episodes, 27,736 source frames, nominal 25 Hz and top/iPhone video. Its local files
are content-addressed in the audit; equivalence with a specific revision of the requested
`bottowork/so101_act_pickplace_v2` upstream repository was not established. Only the
35 train/validation episodes enter the numerical checks. Follower degree-to-encoder
inverse residual p95 is at most **0.00004292 tick**, versus approximately 0.40–0.49 tick
for ±100 normalization. Recorded actions are commands and can include limiter-generated
fractional tick targets; they are not follower readbacks. Neither check establishes
physical arm signs/zeros or historical command timing.

## Regenerated labels and matching inference

The new [FK regeneration tool](../scripts/regenerate_v2_fk_labels.py) rebuilds
Cartesian deltas, tool-frame deltas, absolute poses or canonical-joint proprioception
from the saved sampled joints. Canonical-joint actions remain recorded joint commands.
It requires a fresh output directory and contract version, verifies all source file
checksums, preserves the complete split/tick/image/gripper/terminal-mask contract, and
recomputes training-only normalization. It records exact task/config snapshots, arm
mapping, native limits and FK/model/label dependency hashes. References are never clipped.
Previously transformed visual-grounding/observation-variant datasets are rejected;
regenerate their canonical source and then rebuild the transformations.

Regenerating the two-view source under the new nominal-camera task produced
**exactly unchanged states and actions** on all 9,617 training and 2,077 validation
ticks. The [label report](../results/v2/fk_regeneration_nominal_wrist_20261024.json)
retains **3,219 training** and **642 validation** ticks outside native arm limits.
The 2,184 reserved ticks are preserved but not scored. The resulting dataset is
`.cache/v2_nominal_wrist_fk_20261024_checked`; its task hash and contract are new.
Changing camera appearance or registering a scene-only base translation cannot correct
labels expressed in the native robot base frame.

A justified sign/zero or TCP correction requires regeneration of every affected state,
Cartesian target, diagnostic and normalizer, followed by fresh training with matching
task/mapping/frame/camera contracts at inference. The new tool records lineage and never
grants physical acceptance. No repaired physical mapping or retraining on repaired labels
is claimed by the nominal-optics regeneration. The independent camera/static-landmark,
joint and gripper acceptance sequence remains in the [geometry investigation](v2_geometry_alignment_20261021.md).
Loaded holds/releases still require comparison to applicable physical observations;
the [separate evidence review](../results/v2/loaded_hold_release_evidence_20261024.json)
keeps simulator NoSlip sensitivity explicit and physically unvalidated.

Reproduce with new output paths:

```sh
.venv/bin/python scripts/audit_v2_calibration_readiness.py \
  --out results/v2/calibration_contract_audit_NEXT.json
.venv/bin/python scripts/diag_v2_nominal_wrist_mount.py \
  --sim-config outputs/v2/inference_sim_smoke_20261023/sim.yaml \
  --out outputs/v2/nominal_wrist_binding_NEXT
.venv/bin/python scripts/regenerate_v2_fk_labels.py \
  --source .cache/v2_exploratory_20261022/cartesian_iphone_top \
  --config outputs/v2/nominal_wrist_binding_NEXT/data.yaml \
  --out .cache/v2_nominal_wrist_fk_NEXT
```

Fifty-five focused unit checks pass, covering calibration-lattice arithmetic, all four
relabel representations, split/statistics isolation, transformed-source rejection,
overwrite protection, nominal camera projection and existing registration behavior.
Ruff passes. The separate macOS renderer diagnostic produced both saved views; two
existing pixel-render tests were excluded from the sandbox-only unit invocation because
CoreGraphics requires a connection outside that sandbox.
