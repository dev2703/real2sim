# Architecture — Real2Sim-RaC

**Status: APPROVED by the user 2026-10-12 (decision D-011).** Written 2026-10-12 at the user's
request after the M0 inspection, then updated the same evening with the user's approved choices
for D-002, D-003, D-004, D-005, D-006, D-007 and D-009 (see `docs/decisions.md`). D-008
(proprioception vector), D-010 and D-011 were approved later the same evening. Provenance: plan.md §2–§7 plus the observed
dataset facts in `docs/data_contract.md`.

## 1. One-paragraph summary

Real SO-101 teleoperation demonstrations (40 episodes, one usable camera: the iPhone side view)
are audited, resampled to 10 Hz and converted into two training sets: raw 6-joint targets for a
robot-specific joint-space ACT baseline, and canonical 7-D task-frame end-effector deltas for a
cross-embodiment Cartesian ACT. Both ACT policies are trained only on real data. A MuJoCo scene
built from Menagerie SO-101 and Panda models, with a tennis ball and a concave bowl, renders a
camera that mimics the iPhone view and executes policy actions through embodiment-specific
kinematic adapters. Closed-loop evaluation measures ball-in-bowl success with every failure
counted. A separately trained recovery MLP, gated by a finite-state machine, can take over from
the frozen ACT under bounded disturbances. Simulator ground truth is used for scoring and for
explicitly labelled diagnostics only.

## 2. Component diagram

```
                 Hugging Face / local cache (pinned revision fd65a66…)
                                   │
                        ┌──────────▼──────────┐
                        │  data.raw  (loader) │  LeRobot v3.0, --local-root / --repo-id
                        └──────────┬──────────┘
            ┌──────────────────────┼───────────────────────┐
   ┌────────▼────────┐    ┌────────▼────────┐     ┌────────▼────────┐
   │   data.audit    │    │  data.prepare   │     │ annotations/    │
   │ reports+montage │    │ 10 Hz resample, │     │ episode_outcomes│
   │ split manifest  │    │ letterbox, labels│    │ (operator)      │
   └─────────────────┘    └───┬─────────┬───┘     └─────────────────┘
                              │         │
                 .cache/derived_joint   .cache/derived_cartesian   (LeRobot-format, train-only stats)
                              │         │
                        ┌─────▼─────────▼─────┐
                        │   policy.train      │  LeRobot ACTConfig/ACTPolicy, resolved YAML, CSV history
                        └─────┬─────────┬─────┘
                  act_joint_so101    act_cartesian (frozen after M4)
                              │         │
   ┌──────────────┐     ┌─────▼─────────▼─────┐     ┌──────────────────┐
   │ sim.scene    │────▶│  policy.inference   │◀────│ geometry (FK/IK, │
   │ SO-101/Panda │     │  obs adapter + chunk│     │ SO(3), frames)   │
   │ ball, bowl,  │     └─────────┬───────────┘     └──────────────────┘
   │ iphone-cam   │               │ 7-D or 6-D action
   └──────▲───────┘     ┌─────────▼───────────┐
          │             │ recovery.gate (FSM) │  ACT → DEVIATION → SPECIALIST → HANDOFF → ACT / STOP
          │             │ detector + MLP      │
          │             └─────────┬───────────┘
          │             ┌─────────▼───────────┐
          └─────────────│ sim.controller      │  embodiment adapter: clamp → IK → rate-limited joint targets
                        └─────────┬───────────┘
                        ┌─────────▼───────────┐
                        │ evaluation.run      │  seeded trials, trials.csv, labelled video, metrics
                        └─────────────────────┘
```

## 3. Package layout (target)

| Package | Responsibility | Owner |
|---|---|---|
| `real2sim.data` | `raw` (LeRobot v3.0 access), `audit` (reports), `transforms` (letterbox, intrinsics), `resample` (10 Hz by timestamp), `labels` (joint and 7-D), `prepare` (derived datasets, manifests, stats), `split` | integrator |
| `real2sim.geometry` | `so3`, `se3`, `rot6d`, `frames` (named transforms), `so101_fk` (Placo URDF + MuJoCo check), `so101_ik`, `panda_ik`, `gripper` (semantic ↔ robot mapping) | integrator (IK adapters shared contract) |
| `real2sim.perception` | colour/feature ball and bowl estimator usable on real iphone frames and sim renders; outputs task-frame estimates + confidence; never GT | integrator |
| `real2sim.sim` | MJCF scene assembly, `So101PickPlaceEnv`, `PandaPickPlaceEnv`, camera rig, physics stepping, controllers, smoke CLI | simulation worktree |
| `real2sim.policy` | ACT config resolution, training loop wrapper, checkpoint manifest, `ObservationAdapter`, `ChunkExecutor` | integrator |
| `real2sim.recovery` | `detector`, `fsm`, `mlp`, correction-data collection, comparator controllers | integrator |
| `real2sim.evaluation` | trial runner, metrics, video capture/captioning, reproducibility checks | integrator |

Configs live in `configs/*.yaml`; resolved copies are written to `results/configs/` with every
run. Large artifacts (`.cache/`, `outputs/`) are git-ignored; manifests with hashes are tracked.

## 4. Frozen interfaces (contracts)

### 4.1 Observation contract

| Field | Joint baseline (`act_joint_so101`) | Canonical Cartesian (`act_cartesian`) |
|---|---|---|
| `observation.images.iphone` | `float32 [3,192,256]`, letterboxed from 1280x720, ImageNet-normalised by LeRobot | same (D-004, approved by delegation) |
| `observation.state` | `float32 [6]` degrees + gripper %, train-stat normalised | `float32 [13]` = `ee_xyz(3) + ee_rot6d(6) + ee_lin_vel(3) + gripper_semantic(1)` in task frame (D-008, approved) |
| Clock | 10 Hz (D-007, approved) | 10 Hz |
| Split | 28 train / 9 validation episodes, seed 0; episodes 25, 33, 34 excluded (D-006, D-010 approved) | same |

The same adapter builds this dict from the real dataset and from the simulator. Camera count,
key name, tensor shape, normalisation and meaning are identical in both; a checkpoint manifest
records them and inference refuses mismatched inputs.

### 4.2 Action contract

| | Joint baseline | Canonical |
|---|---|---|
| Vector | `[6]` joint targets, degrees + gripper %, semantics = recorded leader targets (D-003 option C) | `[7] = [dx, dy, dz, d_rx, d_ry, d_rz, gripper_target]` |
| Units | degrees, percent | metres, radians (rotation vector of `R_next · R_currentᵀ` in task frame), gripper in [0,1] |
| Reference | absolute | relative to the **observed** EE pose at the tick |
| Label source | raw `action[t]` at the 10 Hz sample | achieved motion `FK(q[t+1]) ⊖ FK(q[t])` (D-003) |
| Chunk | 20 steps, 2 executed per inference (D-004/D-007, approved) | same |
| Training | ACT dim 512, batch 16 (D-004, approved); joint on MPS, Cartesian on rented GPU (D-009) | same |
| Bounds | joint ranges from calibration | per-tick clamp (e.g. ‖dx‖ ≤ 3 cm, ‖d_r‖ ≤ 0.3 rad, provisional), workspace box, joint rate, IK residual |

### 4.3 Coordinate frames

| Frame | Definition | Status |
|---|---|---|
| `world` | MuJoCo world, z up, metres | defined by scene |
| `task` | tabletop origin at the robot base-plate front-centre projection on the table, x away from the robot, y left, z up | PROVISIONAL until measured |
| `so101_base`, `panda_base` | each robot's base link | from model |
| `ee` | SO-101: Menagerie `gripperframe` site; Panda: flange + TCP offset to fingertip centre | from model, TCP offsets documented |
| `cam_iphone` | OpenCV optical convention (z forward, x right, y down) | extrinsics estimated at M2 (`docs/calibration.md`) |
| `ball`, `bowl` | object centres | sim GT for scoring; perception estimates for policy |

All transforms are stored as `T_parent_child` (4x4, metres, radians, right-handed).

### 4.4 Embodiment adapters

```
class CartesianActionAdapter(Protocol):
    def observe(self, sim_state) -> dict        # 13-float canonical proprioception + images
    def apply(self, action7, obs) -> ControlResult   # clamp → target pose → IK → rate-limited joints
                                                     # returns residual, rejected flag, joint cmd
```

SO-101 adapter: position-prioritised IK with weighted (soft) orientation, 5 arm joints,
unreachable targets rejected and logged. Panda adapter: full-pose 7-joint IK plus gripper width
mapping. Both consume the identical 7-D contract; neither changes the policy.

### 4.5 Recovery gate

States `ACT → DEVIATION_CONFIRMED → SPECIALIST → HANDOFF_CHECK → ACT`, plus `STOP`.
Takeover flushes the ACT chunk; handoff forces a fresh observation and chunk. Specialist input is
the 29-float observable vector (D-005); no simulator ground truth.

### 4.6 Evaluation record

One row per attempt in `results/metrics/trials.csv`: `experiment_id, robot, seed, scenario,
controller_source (LEARNED_ACT | SCRIPTED_REPLAY | ORACLE_DIAG | ACT_DET_REC | ACT_MLP_REC),
checkpoint_sha, success, lift, delivered, released, rolled_out, phase_failed, collisions, stops,
ik_rejections, latency_ms, video_path`.

## 5. Experiment labelling (from plan §7)

E0 `raw_replay`, E1 `joint_ACT`, E2 `cart_ACT_so101`, E3 `cart_ACT_panda`, E4b
`oracle_geometry_diagnostic` (E4a is NOT APPLICABLE unless the trained schema contains estimated
geometry slots), E5 `ACT+det_rec`, E6 `ACT+MLP_rec`. Videos carry the matching caption.

## 6. Key trade-offs recorded

- **Single camera.** Forced by the audit. It removes multi-view ambiguity but makes depth along
  the camera axis weakly observable; proprioception and the bowl's fixed position compensate.
- **Achieved-motion labels for the canonical policy.** Matches the stiff simulated actuators,
  at the cost of learning a lag-smoothed version of operator intent.
- **Joint baseline keeps raw leader targets.** Keeps M1 a pure pipeline test but its simulated
  rollouts will over-reach relative to the real arm; it is labelled SO-101-specific.
- **Placo via LeRobot for IK.** Reuses a pinned, documented interface; macOS wheel risk is
  mitigated by a MuJoCo-based FK fallback.
- **Provisional physical parameters.** Ball/bowl geometry are labelled provisional until the
  operator measures them; results are reported with that caveat.
