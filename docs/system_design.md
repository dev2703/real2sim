# System design — runtime pipelines, configuration and verification

**Status: DRAFT — PENDING USER APPROVAL (decision D-011).** Companion to
`docs/architecture.md` (static structure). This file describes how data moves at runtime, the
proposed hyperparameters, compute budget and the test strategy. Open choices reference
`docs/decisions.md`.

## 1. Data pipeline

### 1.1 Access

`RawDataset(root | repo_id, revision)` reads `meta/info.json`, the episode table, the parquet
rows and decodes video frames by timestamp through PyAV (AV1 via libdav1d). It records the
pinned revision, file hashes and `codebase_version` into every downstream manifest. The local
byte-identical copy is the default source; the remote is used only if the hash check fails.

### 1.2 Audit (M0)

For each image key: decode every frame once per episode at full rate (≈47 k frames per camera,
AV1 720p decodes at roughly 200–400 fps on M3, so minutes not hours), compute per-frame grey
mean/std, Laplacian variance, difference to previous frame, decode failures, timestamp of the
decoded frame versus the requested timestamp. Flags: BLACK (mean < 10), BLANK (std < 5), FROZEN
(≥ 15 consecutive frames with mean absolute difference < 0.5), DUPLICATE, MISSING. Episode-level
roll-up decides USABLE / PARTIALLY_USABLE / UNUSABLE with thresholds written into the JSON.
The HTML montage shows first/middle/last frames per episode per camera plus the ten worst frames.

Action/state audit: the lag and offset table already computed in `docs/data_contract.md`, per
episode. Split manifest: seeded whole-episode split, 30 train / 10 validation (D-006), with SHA-256 of
the episode-ID list.

### 1.3 Resampling and labels (M1, M3)

- Select frames at `t_k = k / 10` s within each episode by nearest timestamp (exact, since the
  recording clock is uniform). Episode boundaries are preserved; the last partial chunk is masked.
- Joint labels: `action[t_k]` (raw). Cartesian labels: `FK(q[t_{k+1}])` relative to `FK(q[t_k])`
  (translation difference in task frame; rotation `log(R_{k+1} R_kᵀ)`); gripper semantic target
  from gripper state with verified sign (M3 check: "0 % = closed" is a hypothesis).
- Images: aspect-preserving resize of 1280x720 to 256x144, letterboxed to 256x192 with
  grey padding (24 rows top and bottom); scale 0.2 and offsets recorded for intrinsics.
  Approved by delegation in D-004; a square centre crop was rejected because it would remove
  the robot base and part of the ball start region from this view.
- Normalisation statistics computed on the training split only and stored with the derived
  dataset and the checkpoint manifest.

### 1.4 Derived datasets

`.cache/derived_joint` and `.cache/derived_cartesian` are LeRobot v3.0 datasets (written with
the pinned LeRobot writer) so `ACTPolicy` training uses the stock dataloader. Each carries
`manifest.json`: source revision, split hash, transform version, label semantics, stats hash.

## 2. Training (M1, M4)

### 2.1 ACT configuration (LeRobot 0.6.1 field names verified)

| Field | Joint baseline | Cartesian | Note |
|---|---|---|---|
| `vision_backbone` | resnet18 | resnet18 | ImageNet weights |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 | same | D-004; fallback 256 / 8 / 2048 |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | 4 / 1 | plan |
| `use_vae` / `latent_dim` / `kl_weight` | true / 32 / 10 | same | defaults |
| `chunk_size` / `n_action_steps` | 20 / 2 | 20 / 2 | D-004 |
| `n_obs_steps` | 1 | 1 | |
| `optimizer_lr` / `optimizer_lr_backbone` | 1e-5 / 1e-5 | same | defaults |
| batch size | 16 (2×8 accumulation if MPS memory forces it) | 16 | D-004 approved |
| steps | 20 k, ckpt every 2 k, val every 1 k | 20 k | D-004 |
| device | mps (smoke, baseline) | rented cuda (provider/budget pending) | D-009 approved in principle |

### 2.2 Loop

`real2sim.policy.train --config configs/act_*.yaml --data <derived> [--smoke-steps N]`:
builds `ACTConfig` from YAML, validates feature shapes against the derived dataset manifest,
runs the LeRobot training step, writes `training_history.csv` (step, train L1, train KL, val L1,
lr, seconds/step, memory), saves checkpoints with the manifest, supports `--resume`. The 2-step
train/save/resume test must pass on MPS before any rented GPU is used.

### 2.3 Expected cost (to be measured at M1, not asserted)

ResNet-18 at 192x256, batch 8, dim 512: estimated 0.5–2 s per step on M3 MPS. The timing table
goes into `docs/progress.md` and drives D-009.

## 3. Simulation (M2, M5)

### 3.1 Scene

- Menagerie `robotstudio_so101` and `franka_emika_panda` vendored at commit
  `0059d4335f8156206f63a35662313385f7ad6d74` with licences.
- Table top at provisional height; black mat as a visual plane; ball = sphere geom, radius
  0.0335 m, mass 0.058 kg (provisional, ITF standard), friction/restitution swept; bowl = compound
  collision (16–24 tilted boxes forming the inner wall + base cylinder) plus optional visual mesh.
- Camera `iphone` placed to match the real view (M2 calibration), rendering 1280x720 then the
  identical letterbox transform. Other cameras are diagnostic only.
- Physics at 500 Hz (`timestep 0.002`), control at 10 Hz with linear interpolation of joint
  targets and per-joint rate limits.

### 3.2 Environment API

```
obs = env.reset(seed, scenario)   # scenario: ball start pose distribution, bowl pose, perturbations
obs, info = env.step(action)      # action = 6 joint targets (joint policy) or 7-D canonical delta
# obs keys: observation.images.iphone (uint8 HxWx3), observation.state (6 or 13 per policy mode)
# info: GT ball/bowl poses, contacts, IK residuals, clamps, phase flags — for scoring only
```

### 3.3 Controllers

Joint mode: targets clamped to joint ranges and rate limits. Cartesian mode: delta → target pose
→ workspace clamp → IK (SO-101 position-prioritised, Panda full-pose) → joint targets →
rate limit. Every clamp and rejection is logged to `info`.

## 4. Inference loop (M4, M5)

```
reset → observe (render + proprio adapter) → policy.select_action (chunk of 20, execute 2)
     → controller.apply ×2 at 10 Hz → observe → ...
     until success (ball resting inside bowl for 1 s), timeout (30 s), STOP, or escape.
```

Latency per tick is measured and logged. The recovery gate, when enabled, wraps the policy call:
it evaluates the detector each tick, flushes the chunk on takeover and forces a new chunk on
handoff.

## 5. Perception (observable estimator)

Tennis-ball yellow and bowl-rim detection by HSV thresholding with connected components on the
letterboxed frame, back-projected to the table plane using the calibrated camera. Produces
`(x, y)` in task frame plus confidence. Used by the recovery detector and specialist; not part of
the ACT input unless a future decision adds it to the training schema.

## 6. Recovery (M6)

- Detector: ball displacement from pre-grasp corridor, approach miss (EE within reach but no
  gripper contact after N ticks), EE drift; thresholds measured on validation replays.
- FSM per architecture §4.5 with persistence counters, maximum recovery duration and retries.
- Data: on-policy corrections in MuJoCo from a scripted oracle (labelled), stored with
  `controller_source, intervention_start/end, expert_action, base_proposed_action, state,
  reason, success, provenance`.
- MLP per D-005; trained with ACT frozen; evaluated in matched seeds against ACT alone and
  ACT + deterministic correction.

## 7. Evaluation (M7)

Metrics per plan §4 M7: N, stable ball-in-bowl success, lift, delivery, release, roll-out
failure, FK/IK error, inference latency, action saturation, detector precision/handoff counts.
Seeds and scenarios fixed in `configs/task.yaml` before final runs. Videos captioned by
`controller_source`.

## 8. Verification strategy

| Layer | Tests (from plan §5) | Runs on |
|---|---|---|
| Data | `test_raw_dataset`, `test_images`, `test_camera_quality`, `test_labels`, `test_so3` | CI (CPU, fixtures) |
| Sim | `test_so101_scene`, `test_kinematics` | CI headless MuJoCo |
| Policy | `test_act_smoke` (real minibatch), `test_oracle_schema`, `test_closed_loop` | local MPS / opt-in CI |
| Recovery | `test_recovery_fsm` | CI |
| Repro | `test_reproducibility` | local |

Fixtures: a 2-episode, 60-frame synthetic LeRobot v3.0 dataset generated in `tests/fixtures`
(including a deliberately black and a frozen video) so CI never downloads the real data.

## 9. Compute and time budget (proposal)

| Activity | Where | Estimate |
|---|---|---|
| Full-frame audit both cameras | M3 CPU | 10–20 min |
| Derived dataset build | M3 CPU | 5 min per mode |
| ACT smoke + timing | MPS | 15 min |
| Joint ACT 20 k steps | MPS | 3–11 h (measure first) |
| Cartesian ACT 20 k steps | rented GPU if approved | 1–2 h |
| Sim trials, 20 × 30 s | CPU | 10–20 min per experiment |

## 10. Risks tracked

1. Placo wheel import failure on macOS → MuJoCo FK fallback, IK on Linux.
2. MPS step time too slow → dim 256 fallback or GPU rental (D-009).
3. iphone frame deficit misaligns episodes 25/33/34 → D-010.
4. Sim gripper cannot hold a 6.7 cm ball with the Menagerie pads → contact tuning, documented;
   no object teleportation.
5. Camera-view mismatch dominates the domain gap → calibrated extrinsics, lighting match,
   bounded randomisation; `0/N` reported honestly if unresolved.
