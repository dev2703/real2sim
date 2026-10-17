# Targeted SO-101 root-cause experiment

The initial five tracks have completed their bounded offline screens except for the unresolved geometry gate. None establishes learned grasp success. The next experiment isolates control from geometry with known recorded motion, while a separate five-episode collection measures the missing real timing and tracking evidence. No further architecture search or large data collection is needed to perform these diagnostics.

## Status of the original five tracks

All prediction errors below are inner-validation active translation errors, in millimetres. These are one-seed screens with matched32,000-anchor exposure, not task success rates.

| Original track | Completed work and finding | Status / next decision |
|---|---|---|
| 1. Architecture | ResNet3.505 versus DINO3.744; ResNet selected. Effective batch64 gave3.429 but worsened stationary and closure-gripper errors. Encoder4/decoder1 and unchanged chunk20/prefix2 tested. Width-matched12 heads gave no consistent gain. SDPA ran on Mac; no FlashAttention kernel was observed. | Screen complete; no model promoted as a physical fix. |
| 2. Geometry | Jaw/arm mapping fits, registration and contact sweeps. One common exploratory mapping/5mm offset lifts3/3, places2/3; independent calibration unresolved. Episode1 loses contact during transport despite0.259mm local tool tracking error at that tick. | Still open; prioritize independent jaw alignment and matched control replay. |
| 3. Cleaning/proprioception | Raw-joint noise audit plus no velocity, two causal EMA filters, raw joints and canonical+joints comparisons. None improves active translation over3.505 in both splits. | Screen complete; preserve raw states and contact-related gripper opening. No blanket smoothing. |
| 4. Flow head | Small cached-DINO head is trainable on Mac. Flow6.483 versus matched regression5.253 and zero motion6.203;8 sampling steps did not help. It still predicts an action chunk. | Screen complete; no evidence to replace ACT or spend more on this head now. |
| 5. Additional hypothesis | Three-state causal history scored3.851 versus canonical3.505; input/label/normalizer audits passed. The earlier isolated IK convergence fix removed accepted no-motion solves without relaxing acceptance. | History screen complete; small-set memorization and same-embodiment joint-action versus Cartesian learning remain conditional follow-ups. |

The detailed prior reports are linked from [experiment status](experiments_status_20261016.md). Panda remains gated on successful reference replay, control tracking and learned SO-101 execution. A completed screen is not completion of these gates.

## R1: matched known-motion control experiment — executable now

Configuration: `configs/so101_root_cause.yaml`. Runner: `scripts/run_so101_root_cause.py`. Output: `outputs/so101_root_cause_20261017/`. The runner acquires the existing experiment lock and its own lock, writes progress, and produces `COMPLETE.json` or `FAILED.json`. A completed output is reused rather than launched again. No GPU training or hardware access is involved.

Replay all recorded ticks of episodes1,2,3 through each route, for nine runs:

| Route | Arm target | Question answered |
|---|---|---|
| Direct joints | Next measured follower joint state | Can the simulated mechanism follow the recorded arm path and retain the ball? |
| Cartesian deltas | Recorded achieved delta, applied to current simulated pose with polished IK | Does incremental execution accumulate error even with the correct motion sequence? |
| Absolute reference feedback | Delta from current simulated pose to next recorded FK pose, with the same bounded IK | Does correcting accumulated pose error restore tracking? This reference is privileged and cannot be used by the learned policy. |

Every route uses the same original recorded gripper command, initial objects, dynamic ball, mapping, physics, velocity bounds, collision checks and success rules. Original gripper command provenance is uncertain and explicitly retained as a limitation. Measured follower opening is recorded separately. The candidate geometry and common+5mm X offset were selected during earlier exploratory diagnostics; this experiment does not validate them independently.

The candidate mapping changes FK, so every Cartesian reference is recomputed from the recorded follower states using that scene's mapping. The old Cartesian labels are compared for audit but never blindly replayed under the changed geometry. Training arrays, policy inputs, normalization and weights are not changed. IK convergence is10µm/0.0001rad; hard acceptance stays3mm/0.35rad, penetration2mm, and existing workspace/joint/action/velocity limits remain intact. Direct joint targets bypass Cartesian IK and its Cartesian step cap by design; native joint/velocity limits still apply to every route.

Score approach, closure, lift/transport, release and after-release separately, plus active/stationary partitions. Save requested, bounded and achieved movements; accumulated reference-pose error; servo error; jaw-gap difference; gripper command/opening; limit events; bilateral contacts; and every loss-of-hold event. Zero-motion movement error is included as a sanity baseline. A good local movement score can coexist with large accumulated path error, so both are required.

Interpret results as follows:

| Observation | Next branch |
|---|---|
| Direct joints track; Cartesian deltas drift; absolute feedback improves | Control representation and accumulated execution error. Investigate trajectory-reference tracking or joint-action baseline, retaining all limits. |
| Direct joints track; both Cartesian routes fail | Inspect IK rejection,5-DOF reachability, workspace and limit coverage before any policy retraining. |
| All routes track the reference but jaws miss or drop the ball | Geometry, gripper-command provenance or contact support. Review real jaw-tip overlays and contact location; do not tune each episode its own successful offset. |
| Reference path itself violates limits or starts in invalid contact | Calibration/model operating-range mismatch remains a confound. Do not count the trial as a physical pass or widen limits to make it pass. |
| Valid reference replay and tracking pass, learned policy still fails | Stronger policy/observation/data-coverage evidence. Verify runtime observations, then run a small-set fit and matched new-data ACT baseline. |

### Completed R1 result

All nine runs completed with valid initial overlap under the candidate geometry. Direct replay reproduces the earlier three lifts/two placements. The completed-run flag does not open a physical or learned-policy gate.

| Route | Lifts | Placements | Closure pose error in episodes1/2/3 |
|---|---:|---:|---|
| Direct joints |3/3|2/3|0.30 /0.33 /1.14mm|
| Cartesian deltas |0/3|0/3|37.81 /43.23 /50.91mm|
| Absolute reference feedback |2/3|1/3|0.30 /0.33 /220.83mm|

Cartesian deltas have only1.26–2.12mm mean active per-tick movement error, yet tens of millimetres of accumulated reference error. Looking only at local movement error would conceal this drift. The feedback route rejects all456 episode3 targets; feedback is therefore diagnostic evidence, not a deployable solution. Recorded references also exceed native limits on168/124/136 ticks respectively, largely around rest/approach/return, and those limits remain unchanged.

Episode1 loses the held ball at29.2s in both the direct and feedback routes, with0.259/0.239mm reference error. That contact failure persists after good tracking and remains a separate geometry/contact investigation.

The first launch exposed a NumPy-integer JSON serialization bug in the new reporting code. It wrote a failure flag, was retained under `outputs/so101_root_cause_20261017_failed_initial/`, and was corrected before the successful nine-run retry. Regression tests now serialize the actual summaries. No safety or physics setting changed to resolve that software error.

### R1b: stationary hold isolates a control mechanism

`scripts/diag_so101_stationary_hold.py` then tested three in-limit transport poses for20s each. All routes start from the exact same measured simulated pose, with the ball away from the robot. This removes future trajectory labels, initial reference mismatch, images, policy inference and ball contact. The fixed-pose feedback target is the initial measured pose, so it needs no privileged future target.

| Control | End drift at the three poses |
|---|---|
| Fixed native joint target |0.346 /0.249 /0.249mm|
| Repeated zero Cartesian delta relative to current pose |58.635 /40.705 /41.132mm|
| Feedback to fixed initial Cartesian pose |0.346 /0.249 /0.249mm|

All nine probes have zero IK rejections, zero non-table ball contacts and zero robot/table penetration. The current-pose delta route repeatedly moves the commanded target with the displaced measured pose; small servo residuals can therefore accumulate. A fixed target retains the correction error and arrests the drift. This demonstrates a simulated control failure mode independently of ACT fitting, irregular recording cadence and contact. It does not identify the real follower's load-dependent residual or prove that this is the sole cause of learned-policy failure.

The first probe report incorrectly classified table contact as robot contact by comparing a geometry name with a variable name. The report was preserved under `stationary_hold_initial_contact_label_bug/`; the corrected probe uses numeric geometry IDs, passes a dedicated test and reproduces the drift results. Final outputs are `stationary_hold/report.json` and `stationary_hold/COMPLETE.json`.

The next control change to test is a persistent accepted pose target with bounded feedback, explicit reset/rebase behavior and rejection handling, compared with the current-pose delta interface on the same frozen sequences. Preserve all bounds. Do not substitute the privileged future-reference controller into ACT or silently change its action contract. First require the zero-motion hold test to remain stable, then the full recorded grasps, before rerunning a frozen policy. R2 below can proceed independently to measure the real mechanism.

## R2: five-episode real timing and load pilot — awaiting collection

The installed collector trace is ready. No traced20FPS dataset exists locally as of setup. Keep calibration, camera placement, hardware encoding, limiter and motor settings fixed. Use ordinary known reachable task poses; do not add extreme poses or increase limits.

| Episode | Condition | Procedure |
|---|---|---|
| 0 | Empty rehearsal A | Follow the ordinary transport route without the ball; hold about2s at each of two repeatable transport poses. |
| 1 | Loaded A | Grasp the ball, follow the same route and intended arm targets; repeat the two holds, then place. |
| 2 | Loaded B | Repeat the loaded run first to balance condition order. |
| 3 | Empty rehearsal B | Match episode2's route and intended targets without the ball. |
| 4 | Ordinary complete grasp | Normal approach, closure, lift, transport and release without diagnostic pauses. |

This is a short paired diagnostic, not a training dataset or statistically powered estimate. Do not mix empty rehearsals into pick-and-place training. Compare steady holds only where logged sent arm targets actually match; operator differences invalidate an assumed matched-pose comparison. Annotate visible load/phase and frame boundaries using `configs/so101_root_cause_hold_annotations.csv`; blank bounds mean unannotated, never an assumed interval. Frame IDs are dataset rows, not camera acquisition IDs.

In your existing full recording command, replace the corresponding options with:

```sh
--dataset.repo_id=local/so101_root_cause_20fps \
--dataset.fps=20 \
--dataset.num_episodes=5 \
--dataset.push_to_hub=false \
--recording_trace_dir=/Users/deveshvyas/Downloads/real2sim/results/collection_traces
```

If your command sets a dataset root, use a fresh matching root; do not resume25FPS recordings. Keep the existing robot/teleoperator/camera/encoder arguments. The fragment intentionally does not guess ports or start hardware.

Audit the emitted session with:

```sh
cd /Users/deveshvyas/Downloads/real2sim
.venv/bin/python scripts/audit_recording_trace.py \
  results/collection_traces/session_UUID \
  --dataset /Users/deveshvyas/.cache/huggingface/lerobot/local/so101_root_cause_20fps \
  --output results/audit/so101_root_cause_collection.json
```

Read median/p95/p99/max actual intervals and stall counts, then distinguish processed-request→sent clipping from prior-sent→later-measured residuals. At holds compare signed residuals, their time evolution and loaded-versus-empty differences; do not infer motor failure from a same-row or transient error. Gripper contact-induced opening is separate from arm tracking. Verify every saved action against its trace sent target and review both videos for freshness and matching rows. Host timing does not establish camera exposure timing, which remains unavailable in this interface.

If the20FPS pilot is coherent, collect the larger successful-task dataset with a fixed episode split. Before training, derive a separate10Hz dataset using supported real timing, reject long unobserved gaps, and verify geometry. The existing policy cadence remains10Hz/chunk20/prefix2. Compare its ACT baseline against the old-data baseline before combining architecture, timing and cleaning changes.
