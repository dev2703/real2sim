# Pick-and-place: why it fails, how to unblock it, and what to cut

This note is the working record of the 5–6 November audit. Raw ACT still does not approach, grasp, or lift. RGB geometric control still completes the same seeds. The native ACT path itself is not the failure: wrist-then-iPhone order, 640×360 RGB, normalization applied once, degree/percent joint mapping, and the 25 Hz rate limits all check out. With real recorded frames, the confirmed v3 checkpoint reproduces recorded actions to about 5° over a chunk. With sim renders at the same joints, that error rises to about 27°.

The confirmed v3 export is `bottowork/act_so101_pickplace_v3_varied_kaggle` at revision `076452e6024c778a2bffabe99ecf33776de141e1`. Do not overwrite `configs/v2/measured_registration_20261028/task_positive_all_scalars.yaml`. A year of RGB benchmark numbers was scored on that scene. Any axis correction belongs in a new measured scene, and only after the confirmation at the end of this note.

## 1. Why there is no grasp, lift, or pick-and-place

In all 20 raw v2/v3 trials (seeds 12040–12044, both arms) the closest tool-to-ball distance is the tick-0 distance, 0.343–0.378 m. The arm moves away from the sim ball, toward where the real ball was. The RGB geometric controller still scores 5/5 on the same resets because it looks at the sim ball, not at the training distribution. Those runs are in `outputs/v2/blockers_20261105/` (`checkpoint_comparison.json`, `matched_v2_latest/evidence.json`, `matched_v3/evidence.json`) and are summarized in `docs/BLOCKER_PROGRESS_20261105.md`.

These are not the cause of the 0/5 raw runs: start-state normalization (every joint was within 1.35σ of the checkpoint stats), action clipping (every raw command was accepted), or the inference wrapper.

The causes, in the order they matter:

**The table axes put the sim ball where the demonstrations never grasped.** `physical_measurements.md` records ball-from-base as `(2, 13)` inches, with the first number "length wise" (x) and the second "width wise" (y). `configs/v2/measured_registration_20261028/task_positive_all_scalars.yaml` sets `operator_axes_assumed_task_axes: true` and places `ball.start_region.center_task_xy` at `(0.0508, 0.3302)` m. That is 2 in forward and 13 in to the side (correction here if you interpret it like that that number is 2 to the side and 13 forward): about 81° to the robot's left. Real v2 grasp forward kinematics is about `(0.312, 0.091)` m, 16°±2° left of straight ahead. Real v3 grasps span −54° to +36°. Reaching the sim ball needs `shoulder_pan` around −84° to −92°, outside both checkpoints' training range. ACT drives toward the real ball and never approaches the sim ball.

The same yaml's bowl is at `(0.2563, 0.1051)` m, inherited from an older image layout yaw-aligned onto the new base/ball bearing (`center_provenance`, `inferred_layout_yaw_rad`). The older layout that the demonstrations actually used sat near ball `(0.356, 0.205)` and bowl `(0.318, −0.035)`. A layout consistent with the grasp kinematics is ball near `(0.312, 0.091)` and bowl near `(0.332, −0.164)`.
Take physical measurements in conjunction with the image findings that you have from the dataset. Don't take physical measurements for axis or directions take them as precise measurements and extract axis, direction from the dataset frames and try to overlay your findigns from extracted measurements with the physical_measurenets.md

**The iPhone camera is misplaced and about twice too zoomed.** `camera_iphone` is `fovy_deg: 22.79` with `fx`/`fy` 1786 px on a 1280×720 sensor. The earlier fit was near 924 px, about 43°. Swapping cameras offline: real wrist with real iPhone is 5.2°; sim wrist with real iPhone is 6.0°; real wrist with sim iPhone is 20.4°; sim with sim is 27.4°. The global view is most of the open-loop error. `extrinsics_provenance` already says the look-at and the old intrinsics are hypotheses, not a measured calibration. `independently_verified_registration` is false.

**The wrist camera clips the grasp, and the image looks rolled.** Scenery boxes in `configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml` (floor, wall, furniture; built in `src/real2sim/sim/model.py` around the `background_boxes` loop) inflate MuJoCo's scene extent, so the default near plane is about 8.5 cm. Jaw tips and the ball at grasp are closer than that, so the wrist image never shows the jaws. Nothing in `src/real2sim/sim/model.py` or `src/real2sim/sim/cameras.py` sets `znear`. Separately, the sim wrist image appears rolled about 180° relative to the real frames. That roll is a camera-frame bug of its own; fixing the near plane does not fix it.

**Learned SO-101 recovery freezes at the wrist stop.** `VisualWaypointController` in `src/real2sim/policy/visual_waypoint.py` selects a geometric grasp whose `wrist_flex` sits on the 95° joint stop. Human grasps in the demonstrations are 13–23°. `soft_stop_seed` in `src/real2sim/policy/common_recovery.py` then rejects any measured pose more than 1 mrad (`tolerance=.001`) past a limit, so after the wrist pins only the jaw still moves. The 5 November abort was a 10.355 mrad dynamic overshoot of `wrist_flex` against that 1 mrad seed tolerance; the last issued wrist command was inside the limit (`docs/BLOCKER_PROGRESS_20261105.md`, `outputs/v2/blockers_20261105/recovery/corrective_iteration/abort_diagnostic.json`). A smaller case of the same check is a 0.23 mrad elbow overshoot that rejects later in-range commands. `check_joint_request` in `src/real2sim/policy/follower_joint.py` (the `current_native_limits` return at lines 43–44) treats any measured joint outside `kinematics.limits` as fatal for the whole request.

**Panda slip is grip and friction, not an ACT bug.** The Panda keeps stock MuJoCo friction and about 1.6 N per finger on a 65 mm ball. SO-101 uses the stricter contact settings and about 37 N. That matches the earlier 16–21 mm drop in 0.4 s after a real-looking lift. Panda appearance is a second confound: `src/real2sim/panda/model.py` hard-codes headlight, a directional light, and table/mat/ball colours (`rgba` at the scene geoms) instead of the SO-101 appearance block, so an SO-101 versus Panda comparison mixes the arm with the scene.

## 2. How to unblock it

Each step is checkable on its own. None of them retrains ACT. v3-specific recovery training stays blocked until the near plane and the scene registration have landed; the existing learned heads are tied to the v2 checkpoint hash.

1. **Show the jaws.** Set an explicit near plane (`znear` about 5 mm) when building the scene, taken from the camera rather than from the scenery extent. Add a render test that the gripper is in frame at the grasp pose. Re-run the offline camera swap. Wrist-only error should stay near 6°.

2. **Re-register only after the axis confirmation below.** Add a new measured scene. Do not overwrite `task_positive_all_scalars.yaml`. Put the ball near `(0.312, 0.091)` m and the bowl near `(0.332, −0.164)` m, and re-aim the iPhone with `fovy` about 43°. Run:

   ```sh
   uv run python scripts/benchmark_frozen_act.py --robots so101 --conditions raw rgb \
     --snapshot ~/.cache/huggingface/hub/models--bottowork--act_so101_pickplace_v3_varied_kaggle/snapshots/076452e6024c778a2bffabe99ecf33776de141e1 \
     --seeds 12040 12041 12042 12043 12044 --out outputs/v2/scene_rereg_NEW
   ```

   Pass condition: raw ACT enters grasp (minimum tool-to-ball well under 0.15 m) and RGB still completes.

3. **Stop treating a 1 mrad soft-limit overshoot as fatal.** Clip the measured joints before the seed and limit check in `src/real2sim/policy/follower_joint.py` (lines 43–44) and in `src/real2sim/policy/visual_waypoint.py`. Record `measured_beyond_soft_stop` instead of aborting. `src/real2sim/evaluation/oracle.py` and `src/real2sim/policy/adapters.py` already clip the measured seed; recovery should do the same. Keep collision checks on the true pose. Add a regression that reproduces the 0.23 mrad elbow overshoot: a measured joint slightly past its soft limit must not reject every later in-range command.

4. **Keep geometric grasps off the wrist stop.** Constrain grasp selection so `wrist_flex` stays below about 1.4 rad, in the range of the demonstrated 13–23° grasps, then re-evaluate the learned head on seeds 12042 and 12043.

5. **Separate Panda slip from the policy.** Try the SO-101 contact solver (`cone: elliptic`, higher `impratio` or `noslip_iterations`; see `src/real2sim/sim/contact_solver.py`) on seeds 12040–12041 and log ball drop while the ball is held. Expect under 3 mm if friction is the cause. Grip force may still need a hardware measurement before any force change is accepted. Appearance and lighting in `src/real2sim/panda/model.py` should follow the SO-101 scene so the comparison is the arm, not the room.

6. **Hold a bad ACT command instead of blocking recovery.** Recovery and the Panda ACT runners should reject-and-hold out-of-limit ACT commands the way `scripts/benchmark_frozen_act.py` already does, so an ACT fault cannot block RGB recovery. `scripts/evaluate_hf_act_panda_20261028.py` should stop replacing `real_to_sim.offset_deg` with zeros unless an offset was actually passed. Supported CLIs should default to the confirmed v3 revision `076452e6024c778a2bffabe99ecf33776de141e1`, or require `--snapshot`, and the path should stay repo-relative. CLI defaults and retained learned heads still point at historical v2 (`README.md`).

## 3. Where an 80% cut is real, and where it is not

Tracked Python is about 64,000 lines: `scripts/` 25,948 lines in 114 files, `tests/` 20,142, `src/` 17,691. Strict imports from the six supported CLIs in `README.md` reach 11 scripts (3,600 lines) and 56 `src` modules (9,269 lines). Those six commands are:

- `scripts/evaluate_hf_recovery.py`
- `scripts/evaluate_hf_act.py`
- `scripts/benchmark_frozen_act.py`
- `scripts/audit_frozen_act_results.py`
- `scripts/run_recovery_dagger.py`
- `scripts/run_cross_embodiment_recovery.py`

**The real cut is scripts.** About 103 of those 114 files, roughly 22,000 lines (about 86%), are dated `diag_*` and one-off experiment drivers outside that closure. Deleting them is safe only together with the tests that exist solely to cover those drivers (about 2,400 lines that import nothing reachable from the six CLIs). A guard test should fail if a deleted script is still named by `tests/` or by `README.md`. Do this only after accepting the six supported CLIs as the functionality boundary, and do it as its own change, not mixed into the bugfixes. The lifecycle note is `docs/script_lifecycle_policy.md`.

**Do not delete these even though they sit outside the six-CLI closure.** `src/real2sim/policy/train.py`, `src/real2sim/data/audit.py`, `src/real2sim/evaluation/run.py`, and `recovery/` are still documented commands (`python -m real2sim.data.audit`, `python -m real2sim.policy.train`). Removing them breaks the README, not just history.

**The `src/` trim that is real is under 80%.** The six-CLI closure still pulls dead weight through imports: `src/real2sim/evaluation/oracle.py` (457 lines) because `visual_waypoint` imports grasp orientation, `src/real2sim/evaluation/replay.py` (379 lines) via retarget, and `policy/flow_matching.py` plus the CLIP/DINOv2/attention stack via `policy/act.py`. Moving `grasp_orientation` into `geometry/` and splitting the training-only ACT backends out of the inference import path drops on the order of 2,000–3,000 lines from the runtime closure without deleting behavior.

**Already gone.** The 1 November cleanup removed 80% of `results/` and 30% of `configs/`. Another 80% of `src/` cannot be removed without dropping training, audit, and the historical v1/v2 contracts.

## 4. Pull requests and follow-up issues

Follow-up issues, no code in the same change as the scene yaml:

- Scene re-registration, gated on the axis confirmation below. New scene only. Pass condition is the benchmark in section 2.
- Wrist image rolled about 180° versus real frames.
- Panda appearance and lighting hard-coded in `src/real2sim/panda/model.py` (headlight, directional light, and geom `rgba` around lines 112–159), so SO-101 versus Panda comparisons confound the arm with the scene.
- Panda grasp force and friction versus the SO-101 contact settings (stock friction, about 1.6 N per finger versus about 37 N; 16–21 mm slip).
- Geometric grasp poses that pin `wrist_flex` at the 95° joint stop.
- Benchmark resume hash omits `src/real2sim/perception/ball_hsv.py`, `src/real2sim/evaluation/oracle.py`, and `src/real2sim/policy/bounded_joint.py`. `runtime_hashes` in `scripts/benchmark_frozen_act.py` hashes `sim/`, `panda/`, `geometry/`, a fixed policy list, and `evaluation/metrics.py` plus `evaluation/video.py`. It does not hash those three modules, so a resume can ignore an edit that changes perception, the oracle, or joint bounding.
- Per-tick step cap does not scale with control rate. `VisualWaypointController.max_step_m` is 0.03 m except 0.024 m at 12.5 Hz (`src/real2sim/policy/visual_waypoint.py`). At 25 Hz that is 0.75 m/s; the cap is a distance per tick, not a speed.
- `TrialTracker.success` counts a pushed-in ball. `pick_place_success` is the complete-task count (released over the bowl after a held lift). `phase_failed` returns `"none"` whenever `success` is true (`src/real2sim/evaluation/metrics.py`), and CLIs that print `phase_failed` (`src/real2sim/evaluation/run.py`, the recovery and ACT evaluation scripts) then read a clean phase on a trial that was only pushed in. `pick_place_phase_failed` is the field that follows the real task.
- Script deletion of the unreferenced dated drivers, with the guard test, once the six supported CLIs are accepted as the functionality boundary.

## Evidence

| What | Where |
| --- | --- |
| 5 Nov matched raw/RGB runs, traces, audits | `outputs/v2/blockers_20261105/` |
| Written summary of those runs and the wrist abort | `docs/BLOCKER_PROGRESS_20261105.md` |
| Operator inches and axis words | `physical_measurements.md` |
| Canonical scene (do not overwrite) | `configs/v2/measured_registration_20261028/task_positive_all_scalars.yaml` |
| SO-101 scenery that inflates the near plane | `configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml`, `src/real2sim/sim/model.py` |
| Cameras, no `znear` | `src/real2sim/sim/cameras.py` |
| Measured-joint limit abort | `src/real2sim/policy/follower_joint.py` lines 43–44 |
| 1 mrad soft-stop seed | `src/real2sim/policy/common_recovery.py` (`soft_stop_seed`) |
| Grasp selection and 0.03 m step | `src/real2sim/policy/visual_waypoint.py` |
| Panda hard-coded look and contacts | `src/real2sim/panda/model.py` |
| Success versus pick-and-place | `src/real2sim/evaluation/metrics.py` (`TrialTracker`) |
| Resume hash | `scripts/benchmark_frozen_act.py` (`runtime_hashes`) |
| Confirmed checkpoint | v3 revision `076452e6024c778a2bffabe99ecf33776de141e1` |
| Supported command surface | `README.md` |

## Confirm the ball axes before any scene change

The axis correction contradicts the scene that the existing RGB numbers were scored on. Please confirm, with one photo or a yes/no, that the ball sits mostly in front of the robot: about 13 in forward and 2–3 in sideways, not 2 in forward and 13 in to the side. that's correct ball sits 2 to the side and 13 forward 

Today the yaml puts the ball at `(0.0508, 0.3302)` m. Real v2 grasp forward kinematics is about `(0.312, 0.091)` m. Until that yes/no is in, leave `task_positive_all_scalars.yaml` as it is and do not open a scene pull request that replaces it.
