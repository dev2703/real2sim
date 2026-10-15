# Decision register

Rules (plan §0.1): the human decides research design. Each material decision lists the evidence,
two or three options, a recommendation, and its approval status. Silence is never approval.
Conflict order: latest explicit user approval → plan contracts → approved entries here → code.

| ID | Decision | Status | Date |
|---|---|---|---|
| D-001 | Camera feature set | AUDIT-DRIVEN RULE (preliminary), awaiting full-decode confirmation | 2026-10-12 |
| D-002 | Python environment and dependency pinning | **APPROVED: option A** (uv, Python 3.12, lerobot 0.6.1 PyPI) | 2026-10-12 |
| D-003 | Training-label semantics (achieved motion vs leader targets) | **APPROVED: option C** (joint = raw leader targets, Cartesian = achieved motion) | 2026-10-12 |
| D-004 | ACT size, batch size, image input size, chunking | **APPROVED: dim 512, batch 16; chunk 20 / execute 2.** Image tensor delegated to agent → `[3,192,256]` letterbox (see entry) | 2026-10-12 |
| D-005 | Recovery MLP architecture and inputs | **APPROVED: option A** (29-float observable input → 256 → 256 → 128 → 7) | 2026-10-12 |
| D-006 | Episode outcomes manifest and train/validation split | **APPROVED: 30 train / 10 validation** (user's own ratio); outcomes manifest still to be filled by operator | 2026-10-12 |
| D-007 | Policy tick rate | **APPROVED: option A** (10 Hz, chunk 20, execute 2) | 2026-10-12 |
| D-008 | Canonical proprioception vector | **APPROVED: option A** (13 floats) | 2026-10-12 |
| D-009 | Compute budget (MPS only vs rented GPU) | **APPROVED: option B** — Modal (USD 30 credit) first; vast.ai / Kaggle as additional sources | 2026-10-12 |
| D-010 | Handling of the 17 dropped iphone frames (episodes 25, 33, 34) | **APPROVED: option B** — exclude all three (train 28 / val 9) | 2026-10-12 |
| D-011 | Architecture / system design documents | **APPROVED** (both documents) | 2026-10-12 |
| D-012 | Calibration values before measurement | **APPROVED by user: use standard values**, labelled PROVISIONAL in `docs/calibration.md` | 2026-10-12 |
| D-013 | Gripper channel of the 7-D Cartesian action label | **APPROVED: option A** (leader command / 100) | 2026-10-12 |
| D-014 | Real-to-sim joint mapping and base height offset | **APPROVED: option A** (signs kept; `base_z_offset` 0.012 m PROVISIONAL) | 2026-10-13 |
| D-015 | Panda embodiment: gravity compensation, TCP frame, start pose | **IMPLEMENTED as engineering defaults; flagged for review** | 2026-10-13 |

Approvals above were given by the user in chat on 2026-10-12 (evening, AEDT) in answer to the
option sets below.

---

## D-001 Camera feature set

**Evidence.** The dataset has exactly two image keys, `observation.images.top` and
`observation.images.iphone`; no wrist key exists. Six sampled `top` frames across two video files
all show the ceiling (sharpness 0.2–2.6, no robot/ball/bowl). Six sampled `iphone` frames show
the full workspace (sharpness 312–441). The dataset card independently excludes `top`.

**Rule applied (plan §3.2, "exactly one usable stream").** `camera_keys =
["observation.images.iphone"]`. Single-camera ACT is the P0 baseline. The simulated camera must
approximate the iphone viewpoint. `top` is retained in the audit report as a labelled data issue
and never enters training.

**Open assumption.** Full-decode audit may still find blank/frozen iphone segments; if any
episode is affected, D-010 governs.

**Status:** audit-driven rule, not a user choice, but the user may veto. Becomes FINAL when
`results/audit/camera_quality.json` is generated.

**Full-decode audit result (2026-10-12 22:26, `results/audit/camera_quality.json`):**
`top` = **UNUSABLE** (37/40 episodes fail the sharpness rule, median Laplacian variance 2.3;
3 episodes scrape past at 5.3–8.6 but show no ball). `iphone` = **PARTIALLY_USABLE at stream
level only because of 3 episodes with frame deficits** (25: −2, 33: −14, 34: −1); the other
37 episodes are USABLE with median sharpness ≈ 600, ball detected in 100 % of frames, zero
decode failures and a consistent +2-frame image-after-joint lag. The tool therefore reports
"only partially usable streams: requires user decision" and D-010 resolves it. Once D-010 is
approved, `camera_keys = [observation.images.iphone]` is FINAL.

---

## D-002 Python environment and pinning

**Evidence.** Repo `.venv` is Python 3.13 and empty. The conda `lerobot` env has a broken
editable install. LeRobot 0.6.1 (PyPI) reads dataset format v3.0, requires Python ≥3.12, torch
<2.12, placo `>=0.9.6,<0.9.16`. uv resolves lerobot 0.6.1 + mujoco 3.15.0 + placo for both 3.12
and 3.13. The operator's recordings were made with Python 3.12 and a 0.6.2-dev LeRobot clone.

| Option | Benefits | Drawbacks | Cost |
|---|---|---|---|
| **A (recommended)** uv project, Python 3.12, `lerobot==0.6.1` from PyPI with `[dataset]` extra, `mujoco==3.15.0`, `placo<0.9.16`, torch 2.11 | Locked, reproducible by anyone with `uv sync --locked`; same dataset format; wheels for all deps on darwin-arm64 | Not byte-identical to the operator's dev clone; placo on macOS arm64 must be smoke-tested | low |
| B uv project with `lerobot` as an editable path dependency on `~/Downloads/lerobot@6156e82a` | Exact code that recorded the data | Not reproducible for reviewers without the clone at that path; lockfile ties to a local path | low now, high later |
| C reuse conda `lerobot` env | Nothing to install | Editable install broken; no mujoco/placo; not reproducible; conflicts with plan's `uv` requirement | low now, blocks M7 |

**Recommendation:** A. Changes: `pyproject.toml`, `.python-version` → 3.12, new `uv.lock`.
Fallback if placo fails to import on macOS: implement SO-101 FK directly from the MJCF with
MuJoCo and defer Placo IK to a Linux GPU box.

**Approved (user, 2026-10-12): A.**

---

## D-003 Training-label semantics

**Evidence.** Follower lags leader by 4–16 frames depending on joint. `shoulder_lift` has a 13°
systematic offset between commanded and achieved positions that persists after lag correction;
`elbow_flex` targets go 28° beyond any achieved state. The Menagerie SO-101 uses stiff `position`
actuators (kp 50–998), so a simulated arm fed raw leader targets will reach poses the real arm
never reached, which could drive the simulated gripper into the table or past the ball.

| Option | What the label is | Benefits | Drawbacks |
|---|---|---|---|
| **A** achieved motion: joint baseline target = `observation.state[t+1 tick]`; Cartesian label = `FK(q_obs[t+1 tick]) ⊖ FK(q_obs[t])` | Matches simulator execution semantics; immune to the shoulder offset and clipping; what the arm physically did | Diverges from LeRobot's default pipeline; lag means the "intent" is smoothed; needs a tick-shift test |
| **B** recorded leader targets: raw `action[t]` (LeRobot default); Cartesian label = `FK(action[t]) ⊖ FK(q_obs[t])` | Standard, zero preprocessing, captures operator intent earlier | Carries the 13° offset and out-of-reach elbow targets into sim; sim motion will not match real motion |
| **C (recommended)** split: joint baseline (`act_joint_so101`, M1) uses **B** exactly as recorded, labelled "leader targets, SO-101-specific"; canonical Cartesian policy (M3/M4) uses **A** with a one-tick (100 ms) lead at 10 Hz | M1 stays a pure pipeline proof with no derived semantics; the research policy gets physically achieved labels that match sim execution | Two label conventions to document and test; the joint baseline's sim rollouts will need a clamp/offset note |

**Recommendation:** C. Changes: `configs/data.yaml` (`label_source`), `src/real2sim/data/labels.py`,
`tests/unit/test_labels.py`, `docs/data_contract.md` §4.

**Approved (user, 2026-10-12): C.**

---

## D-004 ACT model size, batch size, image input, chunking

**Evidence.** Apple M3, 16 GB unified memory, no CUDA until D-009 is decided. LeRobot 0.6.1 ACT
defaults: `dim_model 512, n_heads 8, dim_feedforward 3200, n_encoder_layers 4, n_decoder_layers 1,
use_vae true, latent_dim 32, chunk_size 100, n_action_steps 100, resnet18 ImageNet weights,
lr 1e-5, kl_weight 10`. Data: 40 episodes, ≈15.8 k frames at 10 Hz. Memory for ResNet-18 at
192x256 with batch 8 is modest; actual MPS speed is unmeasured until the M1 smoke test.

Sub-decisions (one choice each):

1. **Transformer width** — 512 (plan preferred; recommended, fallback 256 if step time > 1.5 s on
   MPS) / 256 from the start (faster, lower capacity) / 384 (compromise, non-standard).
2. **Batch size** — 8 (recommended; gradient accumulation to effective 16 if loss is noisy) / 4
   (plan minimum) / 16 (only on rented GPU).
3. **Image tensor** — `[3,192,256]` letterboxed 16:9→4:3 (recommended; plan candidate, cheap on
   MPS) / `[3,240,320]` (sharper ball, ≈1.6× compute) / `[3,180,320]` (native aspect, no letterbox,
   non-standard for ResNet stride but valid).
4. **Chunking at 10 Hz** — `chunk_size 20, n_action_steps 2` for training and evaluation
   (recommended, plan) / `chunk 30, n_action_steps 5` (smoother, slower reaction) / `chunk 20`
   with temporal ensembling (`n_action_steps 1`, `temporal_ensemble_coeff 0.01`).
5. **Training length** — first useful run 20 k steps with checkpoints every 2 k and validation
   every 1 k (recommended) / 50 k / until validation L1 plateaus (open-ended, risky for deadline).
6. **Backbone learning rate** — keep 1e-5 (default, recommended) / freeze backbone (faster on MPS,
   less adaptation).

**Recommendation:** 512 / 8 / `[3,192,256]` / chunk 20 + 2 / 20 k steps / backbone 1e-5, with 256
as the explicit smaller-model fallback. Changes: `configs/act_joint.yaml`,
`configs/act_cartesian.yaml`, `configs/data.yaml`.

**Approved (user, 2026-10-12):** `dim_model 512`, **batch 16**, chunk 20 / `n_action_steps` 2
(via D-007). If batch 16 does not fit on MPS, gradient accumulation 2×8 keeps the effective batch
at 16 and is reported as such. Training length and backbone LR stay at the recommendation.

**Image tensor — delegated by the user** ("resize 256, center crop 224 or whatever you feel is
good"). Chosen: **`[3,192,256]` aspect-preserving letterbox**. Reason: in the iPhone view the
robot base sits at the left edge and the ball start marker near the right edge; a square 224
centre crop after a shorter-side resize to 256 keeps only the middle 49 % of the width and would
remove the base and part of the start region. The letterbox keeps the whole field of view at
the cheapest plan-candidate size. Padding (24 rows top and bottom) and the 0.2 scale factor are
recorded for intrinsics. Revisit to `[3,240,320]` if the ball is too small for the policy.

---

## D-005 Recovery MLP architecture and inputs

**Evidence.** Plan §4 M6 specifies `input -> 256 -> 256 -> 128 -> 7` trained with Huber/L1 on
expert corrections. The amount of correction data is unknown and will be small (collected in
MuJoCo at M6). No privileged object pose may enter the main-benchmark specialist.

Input vector (observable only), proposed: canonical proprioception (13) + estimated ball
position relative to the end-effector in the task frame (3) + estimated bowl centre relative to
EE (3) + perception confidence (2) + last ACT-proposed action (7) + gripper-contact flag from
gripper state (1) = **29 floats**. All estimated geometry comes from the same colour-tracking
pipeline available in real and sim.

| Option | Hidden layers | Parameters | Benefits | Drawbacks |
|---|---|---|---|---|
| **A (recommended)** | 256 → 256 → 128 | ≈ 108 k | Plan default; enough capacity; trains in minutes on CPU | Can overfit on < 2 k samples; needs dropout 0.1 and weight decay |
| B | 128 → 128 | ≈ 21 k | Hard to overfit with tiny data; fastest | May underfit if corrections are multimodal |
| C | 512 → 512 → 256 | ≈ 410 k | Headroom if correction data is larger than expected | Overfits; no evidence we will have the data |

Other fixed choices unless overridden: ReLU, LayerNorm on input, Huber loss (δ=0.05 in
normalised units), AdamW lr 3e-4, weight decay 1e-4, batch 256, early stop on held-out
correction episodes, outputs clamped to the same 7-D action bounds as ACT.

**Recommendation:** A, with the 29-float observable input above. Changes: `configs/recovery.yaml`,
`src/real2sim/recovery/mlp.py`, `tests/unit/test_recovery_fsm.py`.

**Approved (user, 2026-10-12): A.**

---

## D-006 Episode outcomes manifest and split

**Evidence.** No success labels exist. Operator states all retained episodes succeeded. Episodes
0, 1, 4 hit (or nearly hit) a 60 s cap. 40 episodes total.

Outcomes: the operator must confirm each episode in `annotations/episode_outcomes.csv`; the
audit's per-episode montage (first/middle/last iphone frame) will be generated to make that review
a 10-minute task. Until then all rows read `unknown` and training proceeds on `unknown` episodes
only if the user explicitly approves (recommended: approve, because the card says they succeeded,
and revisit after review).

Split options (whole episodes, seed 0, hashed manifest):

| Option | Train / Val | Benefits | Drawbacks |
|---|---|---|---|
| **A (recommended)** | 32 / 8 | Plan default; validation loss meaningful | 8 fewer training episodes |
| B | 34 / 6 | More training data | Noisier validation |
| C | 36 / 4 | Max training data | Validation nearly meaningless |

No separate test split: closed-loop simulation trials are the test. Validation episodes are
also the source of the "replay" diagnostics so they stay out of training and fit statistics.

**Approved (user, 2026-10-12): 30 train / 10 validation** (user-specified ratio, 75/25).
Implemented as a seeded whole-episode split with hashed manifest. Outcome manifest: still
required from the operator; training on `unknown`-outcome episodes proceeds on the strength of
the dataset card's statement and is revisited once the review is done.

---

## D-007 Policy tick rate

| Option | Benefits | Drawbacks |
|---|---|---|
| **A (recommended)** 10 Hz policy tick, 30 Hz source, MuJoCo physics at 500 Hz with interpolated rate-limited targets | Plan contract; chunk 20 = 2 s horizon; cheaper inference on MPS | Coarser than recording; fast ball rolls between ticks |
| B 15 Hz | Finer control | Non-integer relationship to 30 Hz needs careful resampling; chunk must grow |
| C 30 Hz native | No resampling | 3× inference cost, chunk 60 for the same horizon, slower training |

**Recommendation:** A.

**Approved (user, 2026-10-12): A.**

---

## D-008 Canonical proprioception

| Option | Vector | Benefits | Drawbacks |
|---|---|---|---|
| **A (recommended)** | `ee_xyz[3] + ee_rot6d[6] + ee_lin_vel[3] + gripper_semantic[1]` = 13 | Plan contract; velocity gives lag context; rot6d is continuous | velocity must be computed identically in sim (finite difference at 10 Hz) |
| B | drop velocity → 10 | simpler, no finite-difference noise | policy cannot infer motion phase from a single frame |
| C | add gripper state (actual opening) → 14 | encodes "ball held" signal seen in the data | leaks robot-specific gripper semantics into the cross-embodiment vector; Panda mapping must be defined |

**Approved (user, 2026-10-12): A.**

**Recommendation:** A for the cross-embodiment policy. Note that C's "ball held" signal is
genuinely informative on SO-101; if A underperforms it is the first ablation.

---

## D-009 Compute budget

**Evidence.** MPS-only M3 16 GB. ACT 512 at batch 8 is likely 0.5–2 s/step on MPS (to be
measured at M1); 20 k steps could be 3–11 hours per run. Two policies (joint + Cartesian) plus
possible reruns.

| Option | Benefits | Drawbacks |
|---|---|---|
| A MPS only | zero cost, no credentials | may limit to one or two training runs before the deadline |
| **B (recommended)** MPS for smoke tests and the joint baseline; one rented CUDA GPU (e.g. a single 24 GB card for ≈ 6–10 h) for the Cartesian run(s) after the 2-step train/save/resume test passes | protects the research-core run; bounded spend | needs the user to provision and set a budget cap |
| C rented GPU for everything | fastest | cost, environment duplication |

**Recommendation:** B. The user must state the budget cap and provider; nothing is rented
without that.

**Approved (user, 2026-10-12): B.** Provider: Modal with USD 30 credit as the first choice;
vast.ai and Kaggle notebooks as further sources. Spend cap = the Modal credit unless the user
raises it. GPU use still requires the 2-step train/save/resume test to pass locally first.

---

## D-010 Dropped iphone frames (episodes 25, 33, 34 suspected)

**Measured (full-decode audit, 2026-10-12).** Decoded iphone frames inside each episode's video
window versus data rows: episode 25 has 2 fewer, episode 33 has 14 fewer, episode 34 has 1
fewer; all other episodes match exactly. The image-versus-joint motion-lag estimate is +2 frames
in the healthy episodes but 0, 0 and −1 in episodes 25, 33 and 34, which is what a drop near
the **start** of the episode produces (the image timeline is shortened, so images appear
earlier relative to the joints for the rest of the episode). LeRobot's timestamp lookup will not
raise: it silently returns neighbouring frames. Episode 34 is in the validation split; 25 and
33 are in the training split.

| Option | Effect | Benefits | Drawbacks |
|---|---|---|---|
| **A (recommended)** exclude episode 33 from training and validation; keep 25 and 34 with a documented ≤ 2-frame (≤ 67 ms) shift | train 29 / val 10 | 14-frame (0.47 s) misalignment removed; 1–2 frame shifts are within the lag spread already present in healthy episodes (0 to +3) | loses 1,093 frames (2.3 %) |
| B exclude all three | train 28 / val 9 | zero known misalignment | loses 3,157 frames (6.7 %) and one validation episode |
| C keep all 40 | train 30 / val 10 | maximum data | episode 33's images lead its actions by up to 0.47 s, which corrupts labels for 1 of 30 training episodes |

Recommendation: A. Changes: `configs/data.yaml` `excluded_episodes`, split regenerated with
the exclusion recorded in the manifest, `docs/data_contract.md`.

**Approved (user, 2026-10-12): B — exclude episodes 25, 33 and 34.** Implementation: the
seed-0 split over all 40 episodes is computed first and the excluded episodes are then removed,
so the remaining assignments are unchanged: train 28, validation 9 (episode 34 leaves validation).

---

## D-011 Architecture and system design documents

**Approved (user, 2026-10-12): both documents.**

`docs/architecture.md` and `docs/system_design.md` were written as **DRAFT — PENDING APPROVAL**
at the user's explicit request on 2026-10-12. They embed the options above rather than locking
them. They become approved designs only when the user approves them in chat; the approved
answers will then be written back into both files and this register.

---

## D-012 Calibration values before measurement

**Approved (user, 2026-10-12): use standard values.** `docs/calibration.md` lists each value with
its source (ITF rules, LeRobot motor normalisation, the operator's own calibration files,
Menagerie joint ranges, typical iPhone main-lens field of view). Every such value is labelled
PROVISIONAL-STANDARD, is never described as measured, and is replaced if the operator measures
it. Ball start and bowl positions will additionally be estimated from the recorded data (FK of
the gripper at grasp and release) at M2; those estimates are labelled DATA-DERIVED.

---

## D-013 Gripper channel of the canonical 7-D action

**Evidence (training split, 10,692 ticks at 10 Hz, `real2sim.data.labels`).** D-003 makes the six
pose dimensions achieved motion. The gripper is different because the ball physically blocks
it. While holding, the leader commands about 0–20 % but the follower reads 25–46 %.

- Follower-state targets pile up between 0.25 and 0.45 (4,656 ticks).
- Leader-command targets spread across the range, with 3,461 ticks below 0.05.

The simulated gripper is a stiff position servo. A target of 0.35 would stop at 0.35 open
whether or not a ball is there, so grip force would depend on the provisional bowl/ball sizes
rather than on the policy.

| Option | Label | Benefits | Drawbacks |
|---|---|---|---|
| **A (recommended)** leader command | `gripper_target = action.gripper[k] / 100` | Encodes intent to squeeze; sim position control then produces a real grasp force on the ball; same semantics on Panda ("close") | Mixes semantics: pose is achieved, gripper is commanded. Must be documented. |
| B follower state | `gripper_target = state.gripper[k+1] / 100` | Fully consistent with D-003 achieved motion | Teaches "stop at 35 % open", which encodes the real ball size; on Panda or a different ball it may not grip |
| C binarised leader command | 0 if leader < 10 % else 1 | Simplest, robust to operator squeeze variation | Loses partial opening used during approach and release |

**Recommendation:** A. Changes: `configs/data.yaml` `labels.cartesian_gripper_source`. Until
this is approved, `prepare --mode cartesian` refuses to run.

**Approved (user, 2026-10-12): A — leader command.**

**Other label evidence, no decision needed:**

- Per-tick translation: median 3.7 mm, p99 18 mm, max 29 mm.
- Per-tick rotation: median 0.015 rad, max 0.17 rad.
- No one-tick jumps above 5 cm or 0.5 rad.
- `|FK(leader) − FK(follower)|`: median 3.2 cm, p95 22 cm, which confirms D-003.

---

## D-014 Joint mapping signs and base height

**Evidence (2026-10-13, FK on 8 sustained-grasp poses and the dataset rest pose).**

| Signs (lift, elbow, wrist_flex) | Rest-pose gripper (x, z) m | Grasp mean (x, y, z) m |
|---|---|---|
| **(+, +, +) current** | 0.159, −0.003 | 0.338, 0.014, 0.047 |
| any single or double flip | gripper 10–30 cm above or behind the base | grasp 12–44 cm above the table |

Only the current signs put the grasp on the table in front of the robot, and the rest pose
with the gripper resting near the table 16 cm in front of the base matches the real frames
(`results/audit/montage/iphone/ep000_first.jpg`). The simulated rest pose, however, pushes the
gripper 2.3 cm into the table (sim M2 report), and the FK grasp height is 1.3 cm above the ball
radius. Both are consistent with the real base origin sitting about 1–2 cm above the table top
(mounting plate / clamp), which the provisional identity `T_task_so101base` ignores.

| Option | Benefits | Drawbacks |
|---|---|---|
| **A (recommended)** keep signs; add `robot_so101.base_z_offset = 0.012 m` (PROVISIONAL) so the base body sits 1.2 cm above the table top; task frame stays = base frame; the start pose then rests just above the table | removes the reset push-out; grasp height lands at ≈ ball radius; one config line | the 1.2 cm is inferred, not measured |
| B keep signs and identity; start closed-loop from a physics-settled pose | no new assumption | every rollout starts with a 2 cm jump; data-derived bowl/ball heights stay 1 cm high |
| C measure the base plate height on the real robot | exact | needs the operator |

Recommendation: A now, C when the operator can measure. Changes: `configs/task.yaml`,
`src/real2sim/sim/model.py` (place base at table top + offset), `docs/calibration.md`.

**Posture check (2026-10-13 00:20).** Two workflow agents reported that the simulated arm looks
elbow-down while the real arm is elbow-up at episode 12 frame 769. Forward kinematics with the
current mapping at that frame puts the elbow at z = 0.203 m and the shoulder at 0.062 m: the
simulated arm is elbow-up too. An offset grid search (lift, elbow, wrist_flex in ±40°) found no
offset that keeps the 8 grasp positions fixed while changing the posture; every candidate moves
the gripper by 1–3 cm for no gain. The visual difference is the camera's weakly constrained
depth (PnP focal refinement was degenerate), not the joints. Signs and zero offsets stand.

**Approved (user, 2026-10-13): A.** `configs/task.yaml robot_so101.base_z_offset = 0.012`.
Supporting evidence from the E4b oracle: the dataset's grasp style (fixed jaw under the ball,
tilt 45–76°) is infeasible in sim with the base on the table because the fixed jaw body extends
2.7 cm below its finger surface; a lift of this size is what lets the fixed jaw get under the ball.
After applying the lift: 248 tests pass; the PnP camera refit with the lowered table planes has
37/43 inliers (was 27/43); the oracle still succeeds 3/3 on the new geometry.

---

## D-015 Panda embodiment engineering defaults (M5)

Implemented in `src/real2sim/panda/` and `configs/sim_panda.yaml`; built by a workflow agent and
reviewed by two lenses with 2-vote refutation. These are controller-level choices that do not
change the shared 7-D contract, so they were applied as defaults; the user may veto.

- **Gravity compensation on.** The unmodified Menagerie Panda position servos sag 3.9 mm under
  gravity at the start pose, and because the contract is relative to the observed pose, five
  zero-delta ticks drifted 10 mm. The env adds the arm's `qfrc_bias` as `qfrc_applied` each
  physics substep (what the real Franka controller does internally); actuator gains unchanged.
  Measured: sag and drift drop to ~1e-15 m.
- **TCP frame.** A `panda_tcp` site injected into the assembled XML string (upstream file
  unchanged) at +0.1034 m along the hand z (fingertip-pad centre), with axes permuted to the
  SO-101 `gripperframe` semantics (x approach, y jaw hinge, z jaw opening).
- **Gripper.** `actuator8` tendon mapping verified from the XML: g in [0,1] → opening 0–0.08 m.
- **Start pose.** TCP at task (0.30, 0, 0.15) pointing down, from IK. It is not the SO-101
  folded rest pose, which the Panda cannot reach; matched seeds match ball starts only.
- **Reach.** 8 of 40 sampled targets at task x ≤ 0.28 m hit joint 4's limit; the ball start
  and bowl centre are reachable. Moving the Panda base behind the task origin would widen reach
  but changes the frame mapping, so it is not done.
- **Domain gap.** Same camera pose; the Panda is about 3× taller and intrudes into the top of
  the frame. Any E3 number must carry this caveat.

---

## D-016 Real-only diagnostics and bounded augmentation/regularisation retrains

**Approved (user, 2026-10-13):** perform the two suggested diagnostics and cheap retraining
remedy, run MPS training in the background, and avoid spending agent tokens waiting. The
alternatives in the user's proposal were more unchanged training steps, modest augmentation /
regularisation, and additional real versus mixed real/simulator data. Selected here: the
diagnostics plus augmentation/regularisation. No choice of a new data source was made.

- Frozen original joint and Cartesian policies: clean, blank-grey, and cross-episode shuffled
  image checks with unchanged proprioception/labels; 32 evenly spaced ticks per episode.
  Also compare clean train/validation under the same inference mode (zero VAE latent).
- Familiar-start diagnostics: existing estimated ball positions for training episodes 0, 5,
  6, 7 only; compare random starts, familiar ball, familiar ball plus recorded initial joints.
  Reset poses are clamped to model limits and the collision/displacement confound is logged.
  These are approximate scene matches, not a calibrated reconstruction or a primary benchmark.
- Separate fresh joint and Cartesian runs keep the approved 512/ResNet-18, batch 16, camera,
  action labels, 10 Hz clock, split, derived arrays, and normalization. No simulator examples.
- Implementation settings for the requested remedy: brightness/contrast ±20%, saturation ±15%,
  reflected image shifts ≤4 pixels on the image content, preserved letterbox padding, train
  only. Dropout 0.1→0.2, AdamW weight decay 1e-4→1e-3. No architecture/backbone change.
- Sequential MPS runs: maximum 6,000 steps each, validation every 1,000, early stopping after
  three checks without ≥0.001 improvement (minimum 3,000 steps). Save every new validation best;
  final checks and 10 closed-loop SO-101 trials per model run automatically in the background.
- Original checkpoints and primary results stay frozen. New results are under
  `results/metrics/remedies/`; configs `configs/act_{joint,cartesian}_augmented.yaml`; training
  outputs `outputs/act_{joint,cartesian}_augmented`; job status `outputs/act_remedies/status.json`.

Limits: one training seed and several regularisation changes together cannot establish which
change caused an improvement. Existing geometry calibration and reset penetration remain
confounds; the previously reported moderate open-loop visual gap does not rule them out.

---

## D-017 Matched ACT reconstruction-loss comparison after D-016

**Approved (user, 2026-10-13):** "setup that experiment as well line it up after the previous
ones", referring to the proposed L1/L2/Huber comparison. Additional real recording is proposed
by the operator; a new wrist-camera training schema or data split has not yet been selected.

- Reuse the two completed D-016 L1 runs as controls. Four fresh sequential MPS runs: joint L2,
  joint Huber, Cartesian L2, Cartesian Huber. No additional unchanged L1 run or larger sweep.
- Copy each control's resolved configuration. Identical model, augmentation, seed 0, batch 16,
  optimizer, learning rate, real data, 10 Hz semantics, camera, normalization and episode split.
  Same maximum 6,000 steps, validation every 1,000, early-stopping rule and validation-MAE
  checkpoint selection. Actual step counts may differ under the shared stopping rule.
- Reconstruction is averaged over valid action components, excluding padding/extrapolation:
  L1 = |e|; L2 = e² (MSE, not vector norm); Huber = 0.5e² for |e|≤1, |e|−0.5 otherwise.
  Huber delta 1 is in per-dimension normalized action units. KL beta stays 10; reconstruction,
  weighted KL and gradient norm are logged separately. These objective scales are different,
  so this first sweep tests the loss substitution at fixed beta, not independently tuned losses.
- Existing/default L1 training delegates to the installed LeRobot 0.6.1 forward method. New
  losses use the same model/VAE forward and KL reduction; inference/checkpoint architecture
  stays identical. Resume rejects switching objectives.
- Compare every selected checkpoint on full held-out MAE, RMSE and absolute-error p95 (also
  raw-unit values per action dimension), plus the same 10 SO-101 simulation seeds 0–9.
  Reuse the completed L1 simulation trials; recompute L1-control open-loop metrics to include
  RMSE/p95. Never compare raw training objective numbers across losses as a quality ranking.
- `scripts/run_act_loss_comparison.py` waits on the remedies manager's process lock until all
  its training/check/evaluation phases complete. A failed predecessor stops the new queue.
  Status/logs/config snapshots: `outputs/act_loss_comparison/`; results:
  `results/metrics/loss_comparison/`. It runs independently of the chat turn with no agent polling.
- New runs retain only their best and last checkpoints to bound disk use; other runs are not
  pruned. The queue checks free space before each model and refuses existing output directories.

Limits: a single seed is exploratory evidence. MAE-based checkpoint selection is common across
the runs but can miss a different checkpoint with better RMSE or task success. Simulation
calibration/reset confounds remain. No hardware motion, data merge, camera-schema change or
new calibration is part of this approval.

## D-018 Hourly background monitoring and evidence-driven follow-up

**Approved (user, 2026-10-13):** check background jobs every hour, fix failures/issues, leave
healthy work running without spending model tokens waiting, and experiment with new suggestions.
Operational implementation: `scripts/monitor_act_jobs.py`, a detached local Python process
checking the two existing managers hourly. It reads status, process locks, active-log freshness,
disk space and newly completed comparisons. Healthy unchanged checks make no model calls.
Issues or new results enqueue a follow-up into this same chat through the installed CLI's
`codex queue`; fixes and experiment decisions retain the current chat's permissions/context.
Repeated unchanged issues/results are deduplicated. Delivery failures are logged and retried
at the next hour. No other person or external messaging service is contacted.

The IDE does not expose native scheduled-task controls; this is a local monitor, not a cloud
schedule. The Mac must be available; sleeping delays checks. The monitor coalesces missed
intervals instead of replaying them. State/logs/config: `outputs/act_monitor/`; create its
`STOP` file to stop. After all queued work and results are reviewed, the follow-up should stop
the monitor unless a justified additional experiment was started and added to its job list.

New follow-ups should be small, based on measured results and recorded with their configurations.
This instruction does not silently change approved camera/action/frame/split/benchmark or
hardware constraints. No new training run was added at monitor setup: existing jobs were healthy.


## D-019 Geometry correction snapshot

**Approved for commit (user, 2026-10-13):** preserve the current corrected camera,
mat, table, ball and bowl configuration, scene-builder appearance support, oracle
adjustments and diagnostics as a separate reviewable commit, message
`Sim geometry changes`, before paired-image training. This records the explicit
request to publish the existing correction; no new measurements are inferred.

Configuration: operator-supplied table/mat sizes and camera-position constraints;
image-fitted camera orientation/focal length, mat placement, and object centres.
See `configs/task.yaml` provenance and the operator-anchored camera snapshot in
`results/audit/camera_fit/operator_anchored_20261013/`. The existing frame and
camera feature schema remain unchanged. Oracle carry changes apply only to the
labelled privileged diagnostic, not to the primary ACT controller.

Primary benchmark results and previous camera audits are historical until rerun.
This commit adds neither synthetic training examples nor a mixed-data checkpoint;
any paired training must retain real-label provenance, training-only pairs,
unchanged held-out episodes and an explicit mixed-appearance training label.


## D-020 Approved paired approach-image pilot

User explicitly approved the proposed priority order: regenerate corrected-geometry
baselines; build verified approach pairs; fine-tune a copy; evaluate early; inspect
command/motion transfer if approach improves but execution fails. User requested
no model polling during training, with a completion-triggered follow-up.

Implementation: preserve original joint 020000; fresh optimizer, weights-only
initialization, 1,000 additional MPS steps, seed 0, batch 16, LR 5e-6 for model and
backbone, original dropout 0.1 / decay 1e-4 / L1+KL10. Validate on unchanged real
held-out episodes every 250 steps and select by real validation MAE. Keep best/last.
`configs/act_joint_paired_approach.yaml` records settings. Mixed real/sim appearance
is explicitly recorded in checkpoint provenance; this is not real-images-only training.

At each eligible training tick, substitute its static simulated image with probability
0.5; labels, state, normalizer, camera schema, 10 Hz and original episode split stay
unchanged. Of 10,692 training ticks, 3,299 have eligible pairs, so expected overall
sim-image fraction is about 15.4%, not 50% of all samples. Remaining images stay real.
The 1,134 validation renders reside in a separate guarded cache and are evaluation-only.

Stationary-ball prefix: stable first five detections; stop at missing detection,
>3 policy-pixel movement, area ratio outside 0.7–1.3, tool–ball distance <8 cm, or
20 seconds. Ball pose is an estimate from the recorded image and fixed-camera
backprojection, not fabricated measured ground truth. Recorded joints are applied
for static FK/rendering without limit clamping or physics steps. Original image/state
timing is retained. Transport/contact trajectories are not invented. Human-supplied
geometry remains approximate; preview review and fixed-camera residuals are recorded.

After training: full real held-out check; parent/pilot comparison of first-two-action
errors on held-out real and paired images; ten unchanged closed-loop seeds. One local
`codex queue` notification after completion or failure; no hourly monitor or LLM wait loop.
Notification setup uses the installed CLI help and OpenAI Docs guidance; previous
same-thread delivery was verified. If approach improves but execution fails, compare
recorded leader-command replay to follower motion before changing labels/controllers.


## D-021 Joint/command diagnostic scope and rejected range correction

2026-10-13, user approved "go for it" for joint-limit/mapping/timing audit, recorded
replay and frozen-policy evaluation. This authorizes isolated sensitivity tests,
not an assertion that unmeasured physical alignment is known. Tested one local
calibration-derived shoulder lower bound (-105.5824 degrees) in a copied model;
primary model/config and hardware untouched. Only joint range/control range and
the corresponding unclamped reset differ. Ten frozen paired-policy trials remain
0/10 and mean closest distance worsens 14.0 → 16.1 cm. Decision: retain original
primary settings; diagnostic range is not an adopted correction. No global lag,
joint-zero or training-label change is supported. Evidence and limitations:
`results/audit/control_contract_20261013/review.json`, `docs/progress.md`.


## D-022 Bounded follower-response experiment (approved diagnostic scope)

2026-10-13 user approved train-only follower-response fitting and held-out replay
before connecting it to ACT. Added a bounded per-joint gain/momentum/delay model
at nominal recording frequency. Internal candidate selection and refit use only
training episodes; primary labels, splits and normalizers remain unchanged.
Full-episode fit worsened replay and was rejected. Adaptive approach-only fit
improved held-out approach replay error 3.06 → 2.20 cm through unchanged MuJoCo,
in eight of nine episodes. Reused diagnostic validation episodes are disclosed;
this does not establish confirmatory generalization or full-task reliability.
Neither model is a production adapter. Any later online test must retain causal
state feedback, explicit diagnostic provenance and nonprivileged phase/horizon
handling. See `results/audit/follower_response_approach_20261013/review.json` and
`docs/progress.md`. No new ACT training or physical hardware action.


## D-023 Frozen ACT online response test (approved isolated diagnostic)

2026-10-13 user approved testing the frozen ACT with the response model and
requested no active waiting if training becomes necessary. The diagnostic wraps
only the paired joint checkpoint 000500. Response model remains frozen from D-022.
Apply it for 78 ticks (7.8 s), ending at the last known state in the shortest
training stationary-ball prefix; then return to the ordinary joint adapter.
This fixed early window is not a learned or privileged phase detector. Use only
current measured joints, velocity estimated from previous joint observation, and
past executed commands held at the 30 Hz internal response clock. No future chunk
actions, simulator object positions, contacts or success state enter the adapter.
Native scene, limits, safety rates, seeds 0–9 and success definitions are unchanged.
No training/label/normalization change is part of this test. Traces, fixed protocol
and implementation hashes: `results/metrics/act_response_20261013/`.
The normal evaluation entry point is unchanged; the runner installs the adapter
only in its local diagnostic process. Verified 17 response-model/adapter tests.


D-023 outcome (14 Oct): 0/10, no aborts; seven seeds approach closer, but handoff
at 7.8 s precedes most closest points and one ball escapes. Adapter remains an
isolated diagnostic, not production. No new training was justified/launched.
See `results/metrics/act_response_20261013/review.json` for all outcomes.


## D-024 Landmark-camera continuation and isolated geometry diagnostics

2026-10-14 user supplied the unfinished landmark-refit transcript and requested
continuation. Fixed float32 finite-difference loss before trusting its joint-offset
comparison. Preserved annotations and old results; repeated fitting to a fresh
directory. Revised annotations/validation exclusions make the scores exploratory.
Camera-only M1 is tested without adopting fitted joint offsets or changing the
primary scene. Object reprojection uses TRAIN-only initial ball observations;
policy inputs, weights, labels, splits, hardware and benchmark scoring are unchanged.
Visual mat yaw is optional and defaults to zero; it has no collision effect.

Separate camera-only and camera-plus-object snapshots distinguish visual changes
from moved physical targets. Completed 60 trials, including the unchanged frozen
paired checkpoint, its 78-tick response diagnostic, original joint ACT, hold and
oracle. Refitted geometry brings paired+response closest distance to 3.85 cm mean,
but still no holds/lifts or task successes. Oracle remains 10/10. Do not promote
these diagnostic scores into primary tables or retrain solely on this result.

Recorded follower-motion replay reveals remaining contact correspondence problems
and an episode-specific bowl-placement error: the shared bowl intersects episode
24's initial ball. Preserved that row with an invalid-setup audit; a separately
labelled approximate bowl-rim annotation removes the overlap but not grasp failure.
Next calibration/pair-generation work must account for per-episode bowl placement
and verify initial contacts before interpreting replay failures. No ACT training
was launched. Evidence: `results/metrics/camera_refit_verified_20261014/review.json`
and `results/audit/camera_landmark_refit_float64_20261014/`.


## D-025 Real-demonstration-only ACT improvement pilot

2026-10-14 user requested further fixes toward ≥6/10 and explicitly rejected the
offered real-plus-oracle demonstration route. Training remains based on recorded
real states/actions; simulated images are paired augmentation only. No synthetic
action targets, privileged policy features, changed split or benchmark rules.

Rejected the fitted M2 joint-offset candidate after worsened recorded-motion
replay. Retain M1 camera in diagnostic snapshots. Correct pair generation with
reviewed per-episode bowl pixels and initial object-overlap checks. Test a separate
near-grasp stationary-ball cache: remove the conservative 8 cm cutoff while keeping
the image-motion/occlusion/area/time gates. Dynamic ball transport is not inferred.
The primary simulator/configuration remains untouched by these scene estimates.

Authorize within the user's requested work one bounded 1,000-step warm-start pilot
from the prior paired checkpoint, unchanged labels and train/validation split,
with real-validation checkpoint selection and matched parent/pilot evaluation on
the same corrected snapshot. Keep the fixed response diagnostic separate from
ordinary ACT. Completion/failure callback resumes review; no active waiting or
automatic long-run escalation. ≥6/10 remains an objective, not an achieved result.
See `outputs/act_refit_pilot_20261014/status.json` and docs/progress.md.
