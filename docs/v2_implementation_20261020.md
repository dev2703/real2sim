# V2 implementation and experiment runbook — 20 October 2026

**Execution update, 22 October:** the operator requested the learning comparisons run before
physical geometry acceptance. The [exploratory learning queue](v2_learning_screen_20261022.md)
has been launched locally on MPS: 24 matrix entries, 22 unique runs, 64,000 anchors per run.
Its [live status](../outputs/v2/learning_screen_20261022/STATUS.json) records actual progress.
The earlier geometry-first status below is historical; physical calibration and task-success
claims remain unresolved.

The two-camera pipeline and bounded experiment infrastructure are implemented. **No v2
learned task-success result or SO-101 60% milestone is claimed.** Episode conditions and the 29/6/6 split are now reviewed and saved.
Measured camera/jaw registration, phase annotations, and reference-task acceptance remain
prerequisites for ranking learned rollouts. No cloud jobs, private uploads or robot movements
were performed.

The current work is [geometry alignment](v2_geometry_alignment_20261021.md). The operator
confirmed unchanged motor calibration/base/cameras. Arm units are strongly supported as
degrees; physical zeros remain unverified. The gripper scale and visual CAD/table audit now
identify concrete calibration checks before resuming the learning matrix.

The [22 October follow-up](v2_geometry_alignment_20261021.md#follow-up--22-october-2026)
documents the apparent iPhone view change across recordings, stricter source-frame
separation and joint observability/depth diagnostics. Independent measured inputs remain
missing. K3 pose composition now uses at least float32 to prevent mixed-precision NaNs;
this implementation fix does not advance the physical or learned-success gates.

## Experiment status — 21 October 2026

All eight requested experiment groups remain in [screen matrix v4](../results/v2/screen_matrix_v4.json).
Its 24 entries define 22 distinct training configurations and two reused controls; every
entry is still `requires_evidence`. No v2 learning screen has produced a task-success result.
Geometry registration and reference replay acceptance come first, followed by the tiny-set
fitting check and the bounded learning comparison.

| Experiment | Matrix arms | Current status |
|---|---|---|
| iPhone vs wrist vs both cameras | L1 | Configured; awaiting geometry acceptance and training. |
| ResNet18 vs frozen DINOv2 | L1 dual-camera control, L2 | Configured; awaiting geometry acceptance and training. |
| Larger batch sizes and learning-rate sweep | L3, reusing the identical L1 control | Effective batch 16/64 and head LR 3e-5/1e-4/3e-4 configured with matched example exposure; awaiting training. |
| DINOv2 fine-tuning | L4 | Last two blocks plus output normalization implemented at backbone LR 3e-6/1e-5, with a frozen control; awaiting training. |
| CLIP ViT frozen and fine-tuned | L7 | Implemented; pinned pretrained loading verified by the [initialization smoke](../results/v2/clip_initialization_smoke.json); awaiting training. |
| DINO/CLIP flow matching vs matched regression heads | L6, L8 | Implemented; training/reload smoke checks cover the pipeline; dataset learning comparisons remain pending. |
| Joint targets, persistent end-effector deltas, absolute poses | L5 | Implemented; provisional controller diagnostics completed, with geometry-induced rejections unresolved; learned comparisons remain pending. |
| Tool-frame deltas and trajectory-consistency loss | K1, K3 | Implemented and covered by coordinate-roundtrip, terminal-mask, gradient and resume checks; awaiting training. K3 adds trajectory weight 0.1 to the matched K1 control. |

The [kinematic action checks](../tests/unit/test_kinematic_actions.py) cover K1/K3 semantics;
the [training-resume checks](../tests/integration/test_v2_resume.py) include trajectory loss.
These checks establish implementation behavior, not improved task completion. K2 bounded
retargeting remains a separate execution diagnostic and is not promoted after worsening all
three provisional motion windows. The 64,000-anchor screen budget and strongest-ACT plus
strongest-challenger finalist cap remain unchanged.

## Measured diagnosis

The local v2 dataset has 41 episodes and 27,736 source rows at nominal 25 FPS. The wrist is
stored as `observation.images.top`; iPhone is the other required view. Visual inspection
indicates a much more useful gripper/ball view than the old ceiling-facing stream. Its rigid
optical transform and coverage throughout delivery still require independent calibration.
Nominal row rate does not establish acquisition latency or command provenance.

[R0 mapping audit](../results/v2/r0_provisional_mapping_v2.json) finds every original start
outside at least one simulated joint limit under the provisional mapping: 40 shoulder
violations and one elbow violation. The old configured reset penetrates the table by about
11.4 mm. Later in-range poses are candidates, not automatically approved task starts.

[R3 unloaded holds](../results/v2/r3_provisional_holds_summary.json) compares three recorded
poses, four controllers, and 500/1,000 Hz physics at 12.5 Hz control. At 500 Hz, persistent
targets, fixed joints and absolute feedback have approximately 0.126–0.129 mm maximum
20-second drift. Current-pose delta integration drifts 26.6–27.1 mm. Doubling physics rate
does not resolve this difference. This supports testing controller integration independently
of architecture. It does not establish loaded retention or calibrated placement.

The subsequent [R1/K2 motion audit](../results/v2/kinematic_control_diagnosis_v2.json)
tests six controllers on the same three holds and contiguous in-limit recorded windows
(36 comparisons). These windows end before the next joint-limit violation and are **not**
complete-task demonstrations or confirmed representatives of the three recording conditions.

Recorded-motion tool error, p95 in millimetres:

| Controller | Episode 0, start 156 | Episode 21, start 126 | Episode 36, start 142 |
|---|---:|---:|---:|
| Joint reference | 32.93 | 4.07 | 5.34 |
| Current-pose delta | 20.57 | 40.67 | 30.32 |
| Persistent task delta | 17.98 | 21.15 | 5.23 |
| Absolute pose feedback | 43.97 | 10.32 | 5.75 |
| Persistent tool delta | 25.86 | 28.06 | 5.39 |
| Persistent delta + bounded projection | 22.75 | 25.09 | 9.57 |

Persistent tracking is initially accurate: p95 0.30–0.33 mm before the first IK rejection.
The mapped reference later penetrates the table by up to 18.12 mm and, in later poses, creates
shoulder/lower-arm overlap. Collision checks reject those targets. Subsequent incremental
commands continue from the last accepted target, so discarded translations leave a persistent
reference offset. Its magnitude matches the cumulative discarded translations to within
6.1e-14 mm in these windows, with zero action/feedback clipping. This identifies a concrete
geometry → rejection → accumulated-reference-error chain; changing the encoder cannot make
the invalid reference path physically executable.

K2 preserves the stable holds (maximum about 0.13 mm) but worsens motion p95 in every tested
window. **Do not promote K2 to the baseline.** Absolute targets avoid that particular
incremental bookkeeping issue but still encounter invalid geometry; their mixed replay
results are not a reason to relax collision checks. K1/K3 remain controlled learning
experiments, not demonstrated execution improvements.

Reproduce the diagnostic without rendered observations:

```sh
.venv/bin/python scripts/diag_v2_kinematic_control.py \
  --poses 0:156 21:126 36:142 --out outputs/v2/new_kinematic_control_version
.venv/bin/python scripts/diag_v2_mapping.py --full-contact-scan \
  --out results/v2/new_geometry_scan_version.json
```

The control command saves per-tick traces, compact summaries, source snapshots, data/config
hashes and a completion checksum. Moving ball/bowl away isolates unloaded tracking; these
runs cannot satisfy placement or retention gates.

The [full-recording geometry scan](../results/v2/r0_full_geometry_scan_v1.json) checks all
13,878 stride-two poses. Under the provisional mapping, 4,486 are outside joint limits and
are excluded from contact checking. Among the remaining 9,392 poses:

- 948 exceed the existing table-penetration tolerance, across 36 episodes.
- 712 exceed the self-penetration tolerance, across 40 episodes.
- Maximum reference penetration is 31.33 mm into the table and 24.80 mm between robot parts.

These counts include unannotated approach, task, and return motions. They do not establish
that every collision occurs before task completion. The report lists source frames for
review and mathematically feasible constant-offset intervals, explicitly **not calibrated
offset recommendations**. Do not apply interval midpoints or enlarge limits. Register the
table/base and joint zeros independently, then repeat the scan over reviewed task phases.

Keep the earlier [contact-retention findings](../results/audit/so101_contact_retention_20261019/README.md)
and [experiment status](experiments_status_20261016.md) separate from these provisional results.
Some bulky historical artifacts now have `.compact.json` summaries; exact originals remain
recoverable using the [cleanup manifest](cleanup_manifest_20261020.json).

## Implemented interfaces

- Exact every-second-row 25→12.5 Hz preparation, chunk 25, execute two actions, 500 Hz physics.
  Whole-episode train/validation/reserved splitting and normalization fitted only on training.
- Ordered multi-camera observations, calibrated link-mounted cameras on both embodiments,
  missing-view errors, calibration/task hashes, portable two-view checkpoint loading.
- Canonical 13-value observed tool state; separate joint, persistent task-delta, absolute
  position/rot6d, and tool-frame delta action contracts. Legacy single-camera/v1 paths remain.
- Strict reset checks, persistent-target reset/rejection handling, complete pick-and-place
  scoring separate from historical ball-in-bowl scoring.
- Exact training continuation with optimizer, sampler and RNG state; fixed-exposure checkpoints;
  matched flow/regression heads and frozen multi-camera feature caches.
- Reference replay/retention, real/rendered image replacement and observation-delay diagnostics,
  unique executed-prefix errors, phase groups, and complete inference timing.
- Frozen scenario/evaluation manifests, condition-stratified scenario-cluster uncertainty,
  and Panda checks for the same policy artifacts, scenario suite and execution settings.

## Additional experiments requested on 20 October

The current definition is [screen matrix v4](../results/v2/screen_matrix_v4.json).
V1–v3 matrices remain historical definitions; no completed v2 screens are being replaced.
Existing L1–L6 comparisons remain included.

| Arm | Controlled change | Root-cause question |
|---|---|---|
| L7 frozen CLIP | ViT-B/16 spatial patches, both cameras | Does a different pretrained representation improve object-conditioned actions? |
| L7 last-two-block CLIP | Last two blocks plus output normalization, backbone LR 3e-6 or 1e-5 | Does limited task adaptation improve grasp/placement beyond frozen features? |
| L8 CLIP regression / flow | Same frozen CLIP features, same head parameter schema and exposure | Does flow matching help beyond an equally sized regression head? |
| K1 tool deltas | Current observed tool axes for translation/rotation; same canonical state and baseline backbone | Does reducing task-frame dependence improve transfer? |
| K2 bounded retargeting | Same frozen delta policy; joint-margin/velocity-bounded Jacobian step followed by FK and the existing IK controller | Are unreachable outputs or embodiment-specific joint limits causing failures? |
| K3 composed trajectory loss | K1 plus weight 0.1 on differentiable rigid-motion trajectory consistency | Does supervising accumulated motion reduce chunk drift despite low per-step error? |

CLIP uses pinned `openai/clip-vit-base-patch16` revision
`57e6b9622bd450f20321a1e587a7778158ea0610`. The public safetensors snapshot is cached locally.
[Initialization evidence](../results/v2/clip_initialization_smoke.json) records its SHA-256.
Input normalization converts ImageNet-normalized tensors back to RGB and into CLIP's
normalization. Spatial patches retain the full 192×256 view with interpolated positional
embeddings; there is no center crop or text input. Fine-tuned encoders cannot use frozen
feature caches. Old DINO caches cannot be substituted for CLIP caches.

The short CPU smoke measured 118.7 ms p95 over ten timed synthetic-frame calls, including
normalization, input/output transfer and full ACT prediction. The action head was freshly
initialized. **This is not a deployment latency qualification**: repeat with real observations,
more samples, intended device and execution overhead before applying the 120 ms criterion.

K1 rotates both increments into the observed tool axes in training and reverses this
conversion on each execution tick, including queued actions. Persistent integration still
uses the last accepted target. K2 is an execution ablation, not a learned joint model or a
collision-avoidance guarantee. It logs projection residuals/failures, retains existing IK
acceptance, and never silently clips the measured reference to call it valid. Use
`eval_so101_kinematic.yaml` and `eval_panda_kinematic.yaml` in separate frozen bundles.
Do not charge another training run for K2.

K3 unnormalizes actions using training statistics and composes complete predicted/target
chunks. Its masked auxiliary loss combines translation scaled by 0.024 m and chordal
rotation error scaled by 0.24 rad. It is a training-only objective; robot joints and object
ground truth are not new policy inputs. Padded terminals contribute no loss. Compare K3
against K1 with all other choices fixed. This is our trajectory-consistency experiment,
not a reproduction of a published invariant loss.

## What the literature changes

[LEGATO](https://arxiv.org/abs/2411.03682) combines tool-frame motion, an invariant training
loss, and constrained robot-specific retargeting. It also uses common gripper hardware and
egocentric observations. K1/K3 and the explicit tool/camera contract test related ideas;
our different grippers remain a physical transfer gap.

[EmbodiSteer](https://arxiv.org/abs/2606.12965) guides diffusion sampling through target-robot
joint space using FK/Jacobian updates and whole-body constraints. K2 tests a cheaper
post-prediction retargeting baseline. We have not implemented its in-sampling guidance or
whole-body collision method, and its reported gains are not predictions for this project.

[CLIP-RT](https://arxiv.org/abs/2411.00508) studies language-conditioned policies with
contrastive imitation and substantial robot pretraining. Our CLIP visual backbone with
ACT or a flow head is a different experiment. [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
and [CrossFormer](https://github.com/rail-berkeley/crossformer) support studying shared
interfaces across robots; their large heterogeneous training corpora do not establish
zero-shot Panda success from our 41 SO-101 recordings.

## Run order and budgets

1. Use [reviewed annotations](../configs/v2/annotations.reviewed_v2.json) and the frozen
   [seed-0 split](../results/v2/split.json). Visual inspection found ball shifts in **6, 8, 9, 10**
   and bowl shifts in **21–25**; all other episodes are baseline. The operator acknowledged
   that only four ball-shift recordings may have been saved. Both shifts are reported as
   toward the robot, with no lateral shift; the task-coordinate vectors remain unmeasured.
   This explicitly revises the originally intended 31/5/5 condition counts to **32/4/5**.

   | Condition | Training | Validation | Reserved |
   |---|---:|---:|---:|
   | Baseline | 24 | 4 | 4 |
   | Ball shifted | 2 | 1 | 1 |
   | Bowl shifted | 3 | 1 | 1 |
   | Total | 29 | 6 | 6 |

   Episode 9 remains in validation with its bowl-tipping annotation. No episodes were
   excluded. All episodes have been visually inspected; reserved does not mean untouched.
   Protocol `v2_conditions_v2` preserves the old allocation under `v2_conditions_v1`.
   The initial-layout review is not a task-outcome or phase-boundary annotation.

   Reproduce the split to a new output path:

   ```sh
   .venv/bin/python scripts/run_v2_protocol.py split \
     --annotations configs/v2/annotations.reviewed_v2.json --out /tmp/v2_split_reproduced.json
   ```

2. Complete the [operator measurement session](v2_operator_session.md), including phase
   annotations and measured `configs/v2/task.calibrated.yaml`. Confirm a complete-task start.
   The split is ready; calibrated training labels and reference-task acceptance are pending.

3. Prepare each camera/action arm from its own versioned data config. Use `accepted_target`
   for task/tool deltas and `current_pose` for absolute/joint controls. Keep training episodes
   and image preprocessing identical when comparing action representations.

   ```sh
   .venv/bin/python -m real2sim.data.prepare --mode cartesian \
     --config configs/v2/data.yaml \
     --local-root ~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2 \
     --out-dir .cache/v2/cartesian_iphone_top
   .venv/bin/python -m real2sim.data.prepare --mode cartesian_tool \
     --config configs/v2/data.yaml \
     --local-root ~/.cache/huggingface/lerobot/local/so101_act_pickplace_v2 \
     --out-dir .cache/v2/cartesian_tool_iphone_top
   ```

4. Run R0–R6 prerequisites with measured inputs. Commands are in
   `diag_v2_mapping.py`, `diag_v2_numerics.py`, `replay_v2_references.py`,
   `render_v2_pairs.py`, `diagnose_v2_views.py`, and `audit_v2_policy.py` (`--help`).
   Generate evidence from those artifacts; do not fill a gate with invented passing values.
5. Tiny-fit three training episodes, one per confirmed condition, unregularized and
   deterministic, at most 32,000 anchors. Run the evidence check before broader comparisons.
6. Freeze data, task, sim, eval, development scenarios, evidence and matrix with
   `run_v2_protocol.py freeze --file name=path ... --out bundle.json`.
   Training requires roles `data_manifest`, `task`, `sim`, `eval`, `scenarios`, `evidence`,
   `matrix`; flow additionally requires `feature_cache_manifest`.
   Use `run_v2_protocol.py train --matrix ... --arm ... --bundle ... --out ...`
   with an explicit `--runtime-seconds`; only one training process runs at a time.

   To build CLIP features after preparation:

   ```sh
   .venv/bin/python scripts/train_flow_head.py cache \
     --data .cache/v2/cartesian_iphone_top --vision-config configs/v2/clip_backbone.yaml \
     --chunk-size 25 --out .cache/v2/clip_features
   ```

7. Screen seed 0 for 64,000 anchors, validating/saving every 8,000. **Declare a 320,000-anchor
   scheduler horizon from the first update**, including 16,000-anchor warmup. Thus seed 0 can
   resume exactly from 64,000 to 320,000; finalist seeds 1/2 train to 320,000 from scratch.
   This differs from the unused v1 matrix's 64,000-anchor schedule; do not mix those runs.
8. Select by 30-case development completion, closure error, then latency. Promote only the
   strongest ACT and strongest distinct challenger. New arms do not relax this finalist cap.
   Freeze final configuration before the 90 reserved scenarios × three seeds. Evaluation
   bundles additionally pin `checkpoint`, `checkpoint_config`, `checkpoint_stats`,
   `checkpoint_contract`, and `checkpoint_run_info`. Aggregate 270 rows with `score-final`.
9. Panda follows measured ≥60% SO-101 completion and scripted target-robot verification.
   Keep policy weights, normalizers, inference settings and scenario distribution frozen.
   SO-101's 70% remains a stretch target. Failed confirmation results require a new suite
   version before subsequent tuning.

Local preparation and screens remain first. Existing credit eligibility, cloud pricing and
finalist cost remain unverified; no new spending is authorized beyond the stated total
$10 Vast.ai cap. A concrete private upload bundle/destination/cost is required before upload.

## Repository cleanup and verification

The cleanup removed 423 committed generated media/trace files, creating compact evidence
summaries where needed: 328,970,853 net bytes removed from the 408,903,018-byte baseline
(80.45%, before small new implementation/docs). This measures tracked working-tree contents,
not Python LOC, Git history, `.venv`, local model caches, or the real dataset. Source and tests
were retained because deleting 80% of functioning code would contradict the functionality
requirement. Direct source/config artifact references, calibration source images, contact
retention evidence and trace NPZs were preserved.

Every removed path, original SHA-256, Git blob and base commit is in
`cleanup_manifest_20261020.json`. Recover a needed original without resetting other work:

```sh
git show 4a04a6d3bfaaf397caffff601caddfe94f9771ce:results/path/to/original > /tmp/original
```

Use the actual manifest path in place of the example. Compact summaries are explicitly
labelled and are not full substitutes for the original per-tick evidence.
Historical scripts that consume those archived traces/videos need their original inputs
restored before rerunning. The recovery command lists matching paths by default:

```sh
.venv/bin/python scripts/restore_archived_results.py \
  --path results/audit/recorded_grasp_control_20261015/verified
```

Add `--out-dir .` to restore those exact missing paths in place, or choose an output directory
and point the historical script there. Recovery checks hashes, refuses overwrites and never
resets source changes. New v2 training/evaluation does not depend on these archived artifacts.

Verification covers the unit suite; both-camera rendering on SO-101/Panda; CLIP pretrained
initialization, fine-tuning boundaries and offline reload; flow/regression cache paths;
tool-frame invariance; bounded joint projection on both native MuJoCo robots; terminal-mask
and trajectory-loss gradients; exact ACT training continuation. Rendering requires access to
macOS CoreGraphics outside the restricted process sandbox.
