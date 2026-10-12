# Real2Sim-RaC — Codex implementation plan
**Version:** 1.1 — 12 October 2026 (Australia/Sydney)  

**Repository:** https://github.com/dev2703/real2sim  

**Source demonstrations:** https://huggingface.co/datasets/bottowork/so101_act_pickplace_v1  

**Submission deadline:** Friday 16 October 2026, **23:59 BST** = Saturday 17 October, **09:59 AEDT**. Target a working public submission by **16 October, 18:00 BST** to leave contingency time.  

**Environment:** Apple Silicon M3, 16 GB RAM; optional short rented CUDA GPU; Python + LeRobot + PyTorch + MuJoCo; `uv` for dependency management.  

**Owners:** Codex = integrator and contracts; Claude = independent MuJoCo/geometry worktree; user = real hardware, physical measurements, recording and final verification.

**Approved process decisions (12 Oct):** (1) **Camera architecture = audit first**; do not assume two camera streams are usable. (2) **Agent skills = deferred** until after the first verified ACT checkpoint (G1), then time-boxed and optional. (3) Significant architecture, milestone, or skill-design choices require a human decision using the approval protocol in §0.1.

**Priority:** Verify dataset and train a real-data ACT baseline first. Documentation and skills cannot prevent baseline progress.

## 0. Instructions to the coding agent — read this first
You are implementing an academic robotics demonstration **in the existing** `dev2703/real2sim` repository. Begin by inspecting the checked-out repository, existing commits, the public Hugging Face dataset, and the current environment. **Do not assume files or dataset features exist** based on this plan. The repo README currently describes intentions rather than a demonstrably implemented pipeline; preserve anything already working.

**Implement the project incrementally rather than rewriting this plan.** Run and report tests after every milestone. Keep a short `docs/progress.md` with commits, commands, verified outcomes, blockers and next actions. All commands in this plan labelled **target CLI** are interfaces to implement, not claims that they exist today.

Critical rules:

1. Real demonstrations must materially train the learned policy. A trajectory replay or hand-scripted controller is **not** an ACT rollout.

2. Never fabricate episodes, dataset metadata, benchmark results, images, checkpoints, contact parameters, calibration transforms or claimed success. Unknown measurements remain unknown until obtained.

3. Do not change real SO-101 firmware, PID gains, motor registers, torque, `max_relative_target` or leader/follower calibration. Hardware is operator-controlled; no unapproved autonomous physical motion. Retain existing P=12, I=0, D=2, safety limit=5 for **recording**, subject to the user's validated setup.

4. Do not train on validation/test episodes or report successful-only rollouts as overall success. Preserve failures in evaluation.

5. The primary transfer policy gets **no privileged MuJoCo object poses**. Simulator truth is permitted for an explicitly named oracle/diagnostic ablation and for metric computation, never secretly as primary policy input.

6. Freeze real-trained ACT before training a separate recovery MLP. For Panda **zero-shot** experiments, do not train/change ACT weights on Panda data.

7. Ship a working, reproducible baseline before elaborate rendering, recovery, extra architectures or claims of physical fidelity. Never claim full pose controllability on SO-101: it has five arm-motion joints and one gripper motor.

8. Do not silently change camera keys, action semantics, frame conventions, frequency, normalisation or train/val splits. Capture all changes in config + tests + `docs/decisions.md`.

9. Avoid speculative giant downloads. Pin source versions/commits and license notices for Menagerie, LeRobot, Placo and assets.

10. If a gate fails, execute the documented fallback and disclose its limitation. **Report failed experiments**; do not disguise scripted outputs as trained-policy results.

11. **Camera quality is an unresolved dataset issue.** The user suspects `observation.images.top` is corrupted, believes `observation.images.iphone` may be usable, and reports the saved wrist view may be unusable. Treat these as **hypotheses**, not verified findings. Audit every image feature actually present; only approved, validated camera key(s) may enter training. See §3.2 and G0.

### 0.1 Human-led architecture and documentation approval protocol

**The human decides research design; Codex implements and verifies approved choices.** Before creating or substantially revising **architecture**, **system design**, **milestone design**, or **SKILL.md** documents—or making an unresolved material technical decision—Codex must present in chat:

1. The decision to make, observed evidence and open assumptions.
2. **Two or three** feasible options with benefits, drawbacks, implementation cost, and deadline/reproducibility impact.
3. A recommended option, reason, risks and exact files/contracts that would change.
4. A direct question for the user's choice. **Wait for explicit approval** before locking the design, changing shared schemas, or authoring the corresponding new design document.

Material decisions include camera feature set when multiple valid alternatives exist, baseline versus canonical policy changes, training-label semantics, coordinate frames, gripper conventions, physical contact simplifications that change evaluation, recovery training source, and benchmark success definitions. **Never interpret silence as approval.** Record the decision and alternatives in `docs/decisions.md`, with its date and approval status.

**No unnecessary interruptions:** Codex may immediately perform read-only inspection, data-quality audits, independent unit tests, dependency setup, implementation of already approved contracts, and factual updates to `docs/progress.md`, `docs/data_contract.md`, and experiment logs. These are not new architecture proposals. If a decision awaits the user, keep working on independent P0 tasks and mark the affected branch **BLOCKED — AWAITING DECISION**. Do not replace the user's choice with the agent's preference.

**Conflict resolution order:** latest explicit user approval → this plan's approved contracts → `docs/decisions.md` (approved entries) → implementation. If sources disagree, stop only the affected work, describe the inconsistency and seek approval. Do not silently reconcile by overwriting documents.

### What was actually verified before authoring
- GitHub connector found the public `dev2703/real2sim` repository on `main`; its inspected `README.md` links to the specified dataset and describes ACT, recovery and Panda transfer. A complete working codebase was **not established** by that inspection.

- The Hugging Face dataset URL was provided by the user, but its **live metadata/content were not retrievable in this planning environment**. Codex must audit it before choosing episode counts, camera keys, LeRobot version, feature dimensions or split ratios.

- The user physically validated the **tennis ball**, reliable grasp and full pick-and-place into a nearby **bowl**; recent teleoperation was approximately 30 Hz. The ball can roll. iPhone feed worked during teleoperation, and the wrist camera was physically mounted, **but saved dataset stream quality has not been verified**. The user's newer report suspects `top` corruption and an unusable wrist view; this overrides any assumption that both recorded streams are good.

- Some previous runs contained repeated joint-target clamping, particularly during convergence. Log any action/state mismatch. Never assume all raw `action` values were physically executed without clipping.

## 1. Research objective, scope and deliverable
**Question:** Can demonstrations collected using a low-cost real SO-101 teach an image-conditioned action-chunking policy that acts on a physically simulated SO-101 and—using only embodiment-specific kinematics/gripper adapters—be evaluated on a Panda, with a separately trained specialist improving recovery from bounded disturbances?

**Actual physical task:** pick up a tennis ball from a reproducible start region; carry it above the tabletop; place and release it into a nearby bowl. Grasp, transport, bowl containment, controlled release and post-release settling are separate measurable phases. Grasping a ball that rolls during approach is a perturbation/recovery condition.

**Must ship (P0):** audited personally collected source dataset; reproducible preprocessing; real-data-trained ACT checkpoint; functioning SO-101 MuJoCo task with ball+bowl+camera observations; closed-loop policy evaluation with metrics and a labelled video; public README with commands, sources, failures and limitations. Preserve real-world data provenance. If possible use blender/STL or libero assets for environment creation.

**High priority (P1):** canonical 7D end-effector action training; trajectory retargeting comparison; Panda controller and at least measured zero-shot policy attempts; modest geometry/perception baseline; recovery FSM + small learned MLP where there is real expert/intervention supervision.

**Nice to have (P2):** detailed Blender ball/bowl visual models, visually matched lighting and textures, contact identification sweep, broad domain randomisation, many ablations, advanced learned detection/world models. These must never block P0.

**Explicitly out of scope:** custom VLA/foundation model, RL-from-scratch, physically autonomous deployment, guaranteeing sim-to-real or real-to-sim success, reorienting complex fallen objects, unbounded intervention/replanning loops, and declaring Panda transfer solely from scripted IK replay.

## 2. Proposed minimal repository structure
Retain existing files where appropriate. Implement the following **target layout**:

```text
real2sim/
├── README.md
├── AGENTS.md                       # minimal agent entry point from M0; no skill dependency
├── plan.md                         # this plan
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data.yaml
│   ├── act_joint.yaml              # fallback and comparison
│   ├── act_cartesian.yaml          # research policy
│   ├── task.yaml
│   ├── sim_so101.yaml
│   ├── sim_panda.yaml
│   └── recovery.yaml
├── src/real2sim/
│   ├── data/                     # HF loader, audit, preprocessor, resampling
│   ├── geometry/                 # frames, FK/IK, SE(3), gripper mappings
│   ├── perception/               # nonprivileged ball/goal tracking
│   ├── sim/                      # environments, physics, cameras, controller
│   ├── policy/                   # LeRobot ACT adapters and inference
│   ├── recovery/                 # detector, MLP, finite-state gate
│   └── evaluation/               # trial runner, metrics, recording
├── scripts/                      # thin CLI wrappers if needed
├── tests/unit/
├── tests/integration/
├── assets/                        # license-documented local additions
├── docs/
│   ├── architecture.md             # human-approved system design, when needed
│   ├── decisions.md
│   ├── data_contract.md
│   ├── calibration.md
│   ├── experiment_protocol.md
│   ├── progress.md
│   └── limitations.md
├── .agents/skills/                # DEFER until after G1; add only if approved/time permits
└── results/
    ├── audit/
    ├── configs/
    ├── metrics/
    └── media/
```

Large datasets, video shards, generated meshes and checkpoints stay out of Git; keep links and revision-pinned manifests. Provide `.gitignore`, Ruff, pytest, optional GitHub Actions CPU checks. `uv run python -m real2sim.<module> ...` is the canonical interface unless repo conventions demand otherwise.

## 3. Frozen technical contracts before model training
### 3.1 Raw dataset — preserve, inspect, never rewrite
Source dataset: https://huggingface.co/datasets/bottowork/so101_act_pickplace_v1

Generate `results/audit/dataset_report.json`, `results/audit/camera_quality.json`, and a human-readable `results/audit/camera_montage.html` (or an equivalent portable HTML report with embedded/local preview images). These are **actual generated artifacts**, not screenshots selected to look good.

- dataset repo+immutable revision, `codebase_version`, `robot_type`, actual `fps`, episode count, frame count, task strings, camera key names, each image width/height/encoding, number of joints/action names and units;

- per-episode durations, frame counts, anomalies, completed-task labels (if supplied), duplicate or corrupt frames, unexpected timestamp steps, missing image streams; **audit each discovered image key separately**, with successful frame decode rate, first/middle/last sample previews, black/blank/frozen frame checks, useful ball/bowl visibility, and temporal coverage against timestamps;

- joint/state/action summary (min/max/rate), median/p95 `|action[t]-state[t]|` per joint, action clipping or physical lag **as inferable** rather than asserted;

- source episode IDs, deterministic episode-level train/validation split, hashed version of split manifest, review notes and source data license/status.

If source episodes lack success/failure labels, create an operator-reviewed `annotations/episode_outcomes.csv` manifest with `episode_id,success,reason,start_region,reviewer`; unknown stays `unknown`. Do not infer every episode succeeded just because it was recorded. No frame-random splits. If fewer than ten usable episodes, build the smoke test and request additional data from the user instead of inventing 60 demonstrations.

### 3.2 Observation interface — audit-gated camera selection

**Do not hard-code two cameras, `top`, `wrist`, or `iphone` as guaranteed working data.** Codex must first inspect the source dataset's **actual feature names**, decoded frames and recorded timing. The user's suspicion that `observation.images.top` is corrupted and that the wrist feed is unusable is a diagnostic lead, **not proof**. In particular, a camera that worked live in Rerun is not automatically a usable stored training stream.

Audit all available `observation.images.*` streams independently. For each, record exact key, image dimensions/format, decodability, sampled contact sheets, repeated/blank/corrupt frames, approximate timestamp coverage and whether the ball, bowl and relevant manipulation are actually visible. Label **USABLE / PARTIALLY_USABLE / UNUSABLE** with evidence in `results/audit/camera_quality.json` and show representative **good and bad** frames in `results/audit/camera_montage.html`.

**Selection rule:**

- **No usable image stream:** block learned vision-policy training; report why and request a repaired local recording or alternate source. Continue only schema fixtures and sim/IK work; do not invent success.
- **Exactly one usable image stream:** use that stream (e.g., `observation.images.iphone` **if it passes**). Record the choice as an audit-driven approved rule, keep the real feature name in the manifest, and make the simulated camera reproduce its viewpoint approximately. Single-camera ACT is the P0 baseline.
- **Two or more usable streams:** propose **single-camera reliability baseline versus multi-camera ACT** with compute, latency and transfer trade-offs, then obtain user approval per §0.1 before freezing the model configuration. Multi-camera is optional, not a prerequisite for G0/G1/G4.
- **Partially usable stream:** do not silently mix invalid frames. Consider excluding that camera or affected whole episodes; present impact on data volume, episode splits and missing-data semantics before selecting a design.

Persist the resolved ordered `camera_keys` and per-key aspect-preserving preprocessing in `configs/data.yaml`, then pin the choice, dataset revision and preprocessing version alongside the checkpoint. **Model inputs for training and inference must have the same count, names, shapes and meanings**; never add a new camera channel at evaluation. For more than one stream, letterbox/resize each to a common LeRobot ACT tensor shape (initial candidate `[3,192,256]`), preserving aspect ratios and updated camera intrinsics/padding. Do not require two cameras to share the same native recording resolution.

Base observation = audited image(s) + the approved SO-101 joint-state baseline or canonical EEF proprioception for the Cartesian policy. Optional object/goal geometry must be estimated through an **equivalent observable pipeline** in real and simulation, explicitly included in the **training** feature schema; oracle truth never enters the primary policy. If visual perception is unreliable, train image+state ACT and keep oracle/geometry diagnostics separate. Rerun visualisation does **not** establish hardware-time synchronisation; validate saved timestamps and image/action temporal consistency.

### 3.3 Coordinate systems, kinematics and shared actions
Define frames `world` (MuJoCo), `task` (physical tabletop origin), `so101_base`, `panda_base`, `ee`, **all physically present / selected camera frames discovered by the audit**, `ball`, `bowl`. Document right-handed axis choices, metres, radians internally, camera optical conventions and every transform direction (`T_parent_child`). Calibration must use actual fiducials/reprojection checks; phone height and an approximate 45° tilt do not uniquely determine extrinsics.

Implement LeRobot/Placo SO-101 FK (pinned URDF, order and units checked) and an independent MuJoCo FK consistency test. Use the documented Placo kinematics interface if compatible, with SO-101 position-prioritised IK, soft orientation loss, joint/workspace limits, seed from current joint state, and explicit unreachable-target rejection. Panda uses 7-arm-joint full-pose IK (plus separate gripper mapping); record residual/limit status. Keep **both adapters** on the same frame/action contract.

Canonical policy action per **10-Hz policy tick** (change only through explicit contract update):

```text
[dx_task, dy_task, dz_task, d_rx_task, d_ry_task, d_rz_task, gripper_target]
```

- Translation in **metres** in the shared task frame.

- `d_r*` is the `SO(3)` log/rotation vector of `R_next * R_current^T` in the task frame, in **radians** (not Euler angles, not arbitrary unconstrained SO-101 yaw).

- `gripper_target` is a normalised semantic open/close target in `[0,1]`, with **explicitly verified** sign and robot-specific gap/joint conversions (do not guess).

- Each target is relative to the **current observed** EE pose at that tick. The controller enforces clamps on translation, rotation, workspace, joint rate and solver residual. No integration beyond safe bounds.

- Training labels: compare (A) physical state transitions `FK(q_observed[t+1])` and (B) `FK(q_recorded_action[t])`; choose and document based on action-vs-state audit and control lag. A raw leader goal subject to massive clipping must **not** be mislabeled as achieved motion. The chosen definition must match simulator execution semantics.

- Use a 30-Hz raw recording clock, resample to **10 Hz** for initial policy experiments, and step MuJoCo physics at a higher internal rate with interpolated, rate-limited controller commands. Resampling must preserve episode boundaries and use actual timestamps.

Canonical proprioception recommendation: `ee_xyz[3] + ee_rot6d[6] + ee_linear_velocity[3] + semantic_gripper[1] = 13 floats`; derive from measured joints, standardise using train statistics only. No robot-specific joint vector enters **the canonical cross-embodiment ACT**. A robot-specific raw-joint ACT is allowed as a labelled diagnostic baseline, not as zero-shot Panda policy.

### 3.4 Task environment and observations
The object is a **tennis ball**, roughly spherical and free to roll; the receptacle is a **nearby bowl**. Measure ball diameter/mass, bowl interior radius/depth/rim geometry, tabletop and workspace, starting region, grasp height, bowl pose and physical motor/gripper behaviour. Until measured, use clearly labelled provisional sim parameters and sensitivity sweeps, not claims of a calibrated digital twin.

Build one task for each embodiment from MuJoCo Menagerie SO-101/Panda models (pin exact model versions and obey licenses); do not silently change model collision exclusions to make a grasp work. A known Menagerie SO-101 self-collision issue has been reported around its elbow limits—test reachable poses and document any model limitations rather than deleting collisions indiscriminately.

Ball: spherical **collision** geom with measured radius/mass/inertia, friction/restitution calibrated approximately. Bowl: **nonconvex concave interior** modelled by compound collision geoms or another validated collision approach (a single concave STL is not sufficient for faithful MuJoCo contact). Blender may produce visual meshes for ball/bowl; physics gets tested collision geometries, not visual realism claims.

Render the **approved audited camera semantics** (one or multiple cameras) matched approximately to the selected physical observation(s), with the same camera count, frame identities, preprocessing and ACT input shape as training. Bowl centre and ball target can be estimated by colour/feature detection in both domains where practical. Rendered textures/lighting differences are a real domain gap; record and test them. MuJoCo ball/bowl ground truth is available **only for scoring and explicitly privileged, correctly structured diagnostics** under §7; it is never silently supplied to the primary ACT.

## 4. Milestones and execution order (hard gates)
The dates below are target internal dates, not evidence of completion. The agent can parallelise independent work in separate worktrees, but may not merge broken contracts.

### M0 — 12 October: repository scaffold + ACT source-data audit [P0, BLOCKER]
1. Inspect current Git tree, branch, README, existing asset files, current local Python/LeRobot versions and available GPU. Record commit SHA and audit outcome in `docs/progress.md`.

2. Create `pyproject.toml`, `uv.lock`, package skeleton, Ruff, pytest, `.gitignore`, `AGENTS.md` from §0, basic CI if cheap.

3. Download dataset metadata first; implement canonical `python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 --out results/audit`. Audit images and episode content; write `docs/data_contract.md` from **observed** metadata.

4. Generate a **per-camera** montage / HTML quality report covering every discovered image feature plus an episode-level table. Mark each stream usable/partial/unusable based on the decoded evidence and document usable successful episodes and starting regions. Explicitly investigate `top`, `iphone`, and `wrist` **if those keys exist**.

5. Produce a deterministic whole-episode split (default ~80/20 only if count permits) and keep held-out physical episodes away from training preprocessing and fit stats.

**Gate G0:** verified source revision, per-camera quality report and a working loader yielding real `state`, `action`, `timestamp`, and **at least one audited USABLE image key selected via §3.2**. Two-camera data are **not** required. **If inaccessible:** log exact error, support `--local-root` and continue synthetic interface tests, but **do not claim training was completed**.

**Target checks:** `uv run ruff check .`; `uv run pytest -q tests/unit`; `uv run python -m real2sim.data.audit --help`.

### M0.5 — Minimal documentation & approval workflow [parallel; NOT a gate before M1]

**Purpose:** Keep the existing plan executable without spending the first day on an agent-skills framework. This is **not** a mandatory three-skill build milestone. M0 already creates the minimal `AGENTS.md`, `docs/progress.md`, and factual `docs/data_contract.md` (from the actual audit).

1. Ensure `AGENTS.md` is short and operational: read `plan.md`; observe the human approval protocol (§0.1); never command unapproved physical motion; own interfaces clearly; run `uv`, Ruff and pytest commands; report gate evidence. It must **not** require nonexistent skills or unverified architecture files.
2. Maintain factual `docs/data_contract.md`, `docs/progress.md`, and a lightweight `docs/decisions.md` including pending and approved decisions. These may be updated as work is verified, without architectural speculation.
3. **Before writing `docs/architecture.md` or redesigning milestone/system/skill documents**, present 2–3 alternatives and obtain the user's approval per §0.1. Once approved, document selected sensor pipeline, ACT, 7D action semantics, kinematic adapters, physics, recovery and trade-offs, with explicit provenance. Until then, an outline/approval proposal is sufficient; don't create an invented definitive architecture.
4. Populate `docs/calibration.md`, `docs/experiment_protocol.md`, and `docs/limitations.md` **incrementally when their respective experimental work starts**. Do not write fabricated calibrations, protocols claimed as executed, or results.
5. Keep Ruff, pytest and dataset-smoke verification active from M0; this **testing harness is required** and distinct from the deferred optional repository-local agent skills described in §10.

**Acceptance G0.5 (non-blocking housekeeping):** minimal `AGENTS.md`, `docs/progress.md`, `docs/decisions.md`, and audit-grounded `docs/data_contract.md` exist as appropriate; no conflicting document requirements; pending user approvals are visible. **Do not block G1/M1 on an unapproved architecture narrative or missing `.agents/skills/` directory.**

**Deferred skills:** Only **after G1** (first verified, real-data-trained ACT checkpoint), optionally propose at most three skills under §10. Obtain user approval **before writing any SKILL.md**. If the deadline is tight, skip them and disclose deferral.

### M1 — 13 October: baseline ACT from actual recordings [P0]
1. Build deterministic image transform + episode-preserving data loader for LeRobot ACT. Verify `[B,3,192,256]` **for each selected image stream** (one or multiple according to G0), and correct action sequence indices/masks. No requirement for a nonexistent/unusable second stream.

2. As an **engineering fallback**, train or smoke-test a **joint-space ACT** on actual raw 6-joint motor targets, clearly named `act_joint_so101`. This is SO-101-specific and **not** cross-embodiment. It proves the dataset/model pipeline independent of Placo or MuJoCo complexities.

3. Configure `ACTConfig` using the current, **actual** LeRobot API. Preferred 7D research policy: `dim_model=512`, `n_heads=8`, `n_encoder_layers=4`, `n_decoder_layers=1`, `use_vae=true`, ResNet-18, `chunk_size=20`, `n_action_steps=2` initially; tune based on measured memory. Start `batch_size=4–8`; gradient accumulation allowed. Avoid unbudgeted `batch=64`/`2000 epochs`.

4. Run one real minibatch forward+backward, finite gradient and loss test, checkpoint save/reload, one inference action chunk; measure CPU/MPS/CUDA speed. Validate by episode, not frames.

5. Launch useful training once smoke test passes, save every defined interval, track configuration/dataset commit, loss curves and validation scores. If compute is scarce, train smaller `dim_model=256` as **explicit smaller-model fallback**, retaining the 512 configuration as planned experiment.

**Gate G1:** real-trained ACT checkpoint can be loaded and predicts correctly shaped actions from **the G0-approved image stream(s)**; no false claim of closed-loop task success. **Deliver:** `results/metrics/training_history.csv`, `results/configs/act_joint_resolved.yaml`, checkpoint reference and reproducible training command.

### M2 — 13–14 October (parallel after schema freeze): physics + SO-101 simulation [P0]
1. Pin/load Menagerie SO-101 model. Add measured-size table, tennis ball and bowl, cameras, lights, actuator/controller configuration; include initial state reset with ball on table and bowl in reach.

2. Implement `reset(seed, scenario)` and `step(action)` with Gym-style observation keys and deterministic random seeds; render **only the selected G0 camera channels for the policy** (other views may be diagnostic-only); record RGB and all physical state for metrics.

3. Test ball falls/rests/rolls plausibly, settles inside bowl, cannot pass through bowl, and responds to gripper pads. Repeat contact tests over documented plausible friction/restitution ranges.

4. Calibrate SO-101 real/sim joint name/order/sign/units from source metadata and robot model; test forward kinematics against simulated tool frame. Check remote self-collision report and avoid infeasible elbow regions.

5. Reproduce a **single demonstrated trajectory** in sim via (a) joint setpoint playback and (b) FK→Cartesian→SO-101 IK. Plot tracking error. **Label this `REPLAY`, not learned policy.** Test a scripted controlled lift/release only as a physics smoke test.

**Gate G2:** headless environment loads, deterministic resets, useful RGB images, stable bowl collision, FK consistency, scripted grasp or documented physical limitation, captured replay diagnostic video. **If no stable grasp:** prioritise contact geometry/actuator tuning; lower visual fidelity instead of hardcoding scripted object teleportation.

### M3 — 14 October: canonical real-data Cartesian dataset [P1; P0 if feasible]
1. Implement SO-101 FK for joint observations, unit-tested transform algebra (`SE(3)`, SO(3) log/exp, frame transforms), test inverse transforms and round trips.

2. Convert physical recorded transitions or approved motor targets into 7D task-space action labels. Record control delay assumptions and clipping diagnostics. Transform gripper with empirically checked semantics.

3. Resample to a consistent 10 Hz, preserve episode IDs, use overlapping 20-step chunks **only within episodes** and mask final shortened horizons correctly.

4. Create derived LeRobot-compatible dataset in a separate local directory/repository; retain original recording intact. Generate train-only normalisation stats and a byte/revision manifest.

5. Validate 10 randomly selected frame/action pairs with FK plots, world-frame arrows and manual video inspection; reject implausible one-tick pose jumps.

**Gate G3:** converted sample has **exactly the selected, audited `observation.images.*` key(s)** from G0, canonical `observation.state` of documented size, and 7D actions. A 2-step train+infer checkpoint round trip succeeds. If conversion fails, preserve the verified raw-joint ACT baseline and report exact blocker.

### M4 — 14–15 October: real-trained Cartesian ACT + closed-loop SO-101 [P0 research core]
1. Train ACT **on the actual converted real demonstrations**; log best validation checkpoint, training duration, dataset revision and resolved LeRobot config. Freeze the chosen checkpoint.

2. Build an inference adapter that assembles **live simulated images matching the checkpoint's audited camera key(s)** plus simulated observable EEF state in the **same semantic task frame**, camera ordering, and preprocessing. No simulator ball pose in primary observation unless it was an approved observable geometric feature at training time, and even then use a real-observable estimator, not GT.

3. Execute predicted action chunk with safe rate-limited IK/control, reobserve after `n_action_steps`, drop stale chunk actions on reset, log commanded/achieved EEF pose, infeasible targets, ball contact, gripper status.

4. Run 3 tiny debugging rollouts first. Then at least 10 honest primary SO-101 evaluation trials (target 20+ if time), reporting each seed, success/failure phase, video path, near-collision, ball escape, perception confidence, and total task completion. Never filter failures.

5. Compare to `scripted_controller`/`replay` baselines, clearly distinguished. **Only run an in-place `oracle_perception` replacement with the same frozen ACT if its TRAINING schema already included estimated ball/bowl geometry;** substitute simulator GT for the **same feature slots**, without changing feature count/shape, and label it **privileged**. For RGB-only ACT, **do not inject new GT XYZ features into the checkpoint**; instead report a separately labelled oracle geometry/controller diagnostic (or separately trained geometry policy, if time), not a paired frozen-ACT oracle upper bound. See E4 in §7.

**Gate G4:** one fully instrumented, **learned** closed-loop ACT rollout is possible, even if performance is poor; success rate and failed cases are measured. This is the primary submission evidence. If zero sim successes, still report `0/N` and isolate whether vision, IK, gripping or distribution shift failed. Do not invent a win.

### M5 — 15 October: Panda cross-embodiment test [P1]
1. Pin/load Menagerie Panda and map identical `task` coordinates, ball/bowl object parameters and **the checkpoint's selected audited semantic camera channels**. Mount a simulated wrist camera only **if the frozen ACT was trained with that channel**; document camera view differences and resulting domain gap.

2. Write Panda `CartesianActionAdapter` using full-pose IK and gripper width mapping, distinct from SO-101. Validate ≥10 reachable 7D target poses and log translational/orientation residuals, joint-limit and workspace rejection.

3. Transfer a canonical trajectory from recorded SO-101 data to Panda through IK and replay as a **retargeting engineering baseline**. This does not establish frozen ACT transfer.

4. Run the **same frozen real-trained Cartesian ACT weights** in Panda closed-loop sim, with identical preprocessing, object/goal semantics, action dimensions and no Panda policy fine-tuning. Record every trial; target ≥10 matched seeds if feasible.

5. Compare task success, trajectories, grip/orientation feasibility, contact loss, perception failures against SO-101 under carefully documented embodiment-specific differences.

**Gate G5:** successful Panda controller smoke test and **attempted** real-trained frozen-policy transfer. Report actual outcome, even `0/N`; distinguish IK feasibility from independent closed-loop learned success. If no time, deliver retargeting+IK residuals as partial engineering evidence and mark learned transfer pending.

### M6 — 15–16 October: recovery and bounded perturbations [P1, timeboxed]
**Only start after a frozen ACT checkpoint and G4 instrumentation exist.**

1. Define failures: ball rolls away before grasp; approach misses ball; small end-effector drift while ball remains reachable. If ball is lost, grasp unsafe or collision risk, stop. A dropped/escaped ball is an evaluation failure unless explicit safe recovery data exist.

2. Implement transparent failure detector with measured geometry thresholds, confidence bounds and persistence. Do not train a classifier from fabricated labels.

3. Build a gate finite-state machine: `ACT -> DEVIATION_CONFIRMED -> SPECIALIST -> HANDOFF_CHECK -> ACT`, plus `STOP`. On takeover discard queued ACT actions; on handoff force a fresh observation and new ACT chunk. Handoff requires safe EE clearance, ball visibility, stable state and familiar pre-grasp corridor for multiple consecutive ticks; maximum recovery duration/retries.

4. Collect **actual** expert correction trajectories on-policy in MuJoCo if human teleop interface exists; otherwise label safe **scripted-oracle corrections** separately. Real SO-101 intervention recordings can supplement only when already available and correctly aligned. Keep `controller_source`, `intervention_start/end`, `expert_action`, `base_proposed_action`, `state`, `reason`, `success`, `provenance`.

5. Freeze ACT; train small MLP `input -> 256 -> 256 -> 128 -> 7` with supervised Huber/L1 and correct action bounds. Do not feed hidden privileged object pose to learned specialist in main benchmark.

6. Compare three matched seeded cases: (A) frozen ACT alone, (B) ACT + deterministic recovery using same detector/replan frequency, (C) ACT + learned recovery. Track gate precision, false triggers, handoff, full task success, number of collisions/STOPs. A handoff alone is not task success.

**Gate G6:** finite-state tests and an actual specialist dataset + checkpoint if data permit; paired evaluation with real outcome counts. If specialist can't be trained meaningfully, ship deterministic comparator and explicitly label learned recovery **not completed**. Do not invent human DAgger data.

### M7 — 16 October: evaluation, public packaging and submission [P0]
1. Freeze physical task dimensions, success conditions, configs, checkpoint and evaluation scenarios **before** final tests. Use explicit random seeds and `results/metrics/trials.csv` with one row per attempt, including failures and aborts.

2. Required metrics: N, ball-in-bowl stable success, lift success, bowl delivery, release success, rolling-out failure, trajectory FK/IK error, inference latency, action saturation, detector/handoff metrics when recovery exists. Report real vs simulator dataset provenance separately.

3. Produce labelled proof video: real SO-101 tennis-ball teleoperation and **the audited usable camera view(s)** (show invalid views only as explicitly labelled data issues); training/data diagram; SO-101 simulation learned-policy rollout; Panda if attempted; recovery if implemented; at least one real failure and accurate methodology narration. Caption demos as `REAL DEMO`, `SCRIPTED REPLAY`, `LEARNED ACT`, `PANDA FROZEN ACT`, `ORACLE`, etc.

4. Rewrite README to include task/problem, exactly how real data were collected, dataset link and pinned revision, models/physics assumptions, installation + one-line commands for audit/train/evaluate, evaluation table, video link, failure analysis and limitations. Never claim calibrated contact physics without measured evidence.

5. `uv sync --locked`; run Ruff, unit and headless simulation tests; `uv run python -m real2sim.evaluation.run --help`; smoke-test README from clean checkout. Record tested Mac/MPS/CUDA environment.

6. Publish code and checkpoint/model card where permission/bandwidth allow; verify public GitHub access and HF links from an unauthenticated view if possible; submit the correct URL **well before 16 Oct 23:59 BST**.

**Gate G7 / definition of done:** repository reproducibly audits the real dataset, trains or loads a real-trained ACT, runs an honest simulation ACT inference loop, publishes measured results with unsuccessful attempts included, and has working README instructions. Stretch components must be labelled as implemented/attempted/not implemented.

## 5. Acceptance test matrix — build these alongside code
| Test name / location | What it proves | Required by |
|---|---|---|
| `tests/unit/test_raw_dataset.py` | schema, frame/episode boundaries, no train/val leakage | G0 |
| `tests/unit/test_images.py` | audited single- and multi-camera selection; aspect-preserving transforms produce correct CHW shapes; intrinsics/padding tracked | G0/G1 |
| `tests/unit/test_camera_quality.py` | corrupted/blank/frozen frames and missing streams are flagged; bad second camera never required for G0 | G0 |
| `tests/unit/test_oracle_schema.py` | GT geometry replacement only when an identical trained feature slot exists; RGB-only ACT rejects new GT inputs | G4 |
| `tests/unit/test_so3.py` | SO(3) log/exp, pose composition/inverse, frame sign/units | G3 |
| `tests/unit/test_labels.py` | joint→EEF delta consistency, action alignment, no cross-episode chunks | G3 |
| `tests/integration/test_so101_scene.py` | headless MuJoCo load, cameras, stable ball and bowl | G2 |
| `tests/integration/test_kinematics.py` | FK across URDF/MuJoCo, safe IK residual handling, gripper mapping | G2/G5 |
| `tests/integration/test_act_smoke.py` | genuine real batch forward/backward, finite loss, inference, checkpoint restore | G1/G3 |
| `tests/integration/test_closed_loop.py` | ACT controls sim **by predictions**, not recorded/scripted target injection | G4 |
| `tests/unit/test_recovery_fsm.py` | all transitions, chunk flush, STOP, timeout, handoff, no oracle leakage | G6 |
| `tests/integration/test_reproducibility.py` | fixed-seed reset + experiment config yields logged comparable trials | G7 |

CI runs small CPU/unit/headless integration tests. GPU, live hardware and whole-dataset downloads are opt-in, not required on every pull request. Test commands must **actually run**; do not leave placeholder passing tests.

## 6. Runbook: executable interfaces Codex must implement
```bash
# after checking repo and installing a compatible pinned Python version
uv sync
uv run ruff check .
uv run pytest -q tests/unit
uv run pytest -q tests/integration
# TARGET CLI: these modules are to be implemented, then verified
uv run python -m real2sim.data.audit \
  --repo-id bottowork/so101_act_pickplace_v1 \
  --out results/audit
uv run python -m real2sim.data.prepare \
  --repo-id bottowork/so101_act_pickplace_v1 \
  --config configs/data.yaml \
  --out-dir .cache/derived_cartesian
uv run python -m real2sim.sim.smoke --robot so101 --headless
uv run python -m real2sim.sim.smoke --robot panda --headless
uv run python -m real2sim.policy.train \
  --config configs/act_cartesian.yaml \
  --data .cache/derived_cartesian \
  --smoke-steps 2
uv run python -m real2sim.evaluation.run \
  --robot so101 \
  --checkpoint outputs/act_cartesian_best \
  --config configs/sim_so101.yaml \
  --n-trials 3 \
  --out results/metrics/debug_so101
uv run python -m real2sim.evaluation.run \
  --robot panda \
  --checkpoint outputs/act_cartesian_best \
  --config configs/sim_panda.yaml \
  --n-trials 10 \
  --out results/metrics/final_panda
```

Make `--help` work for every entry point. If real metadata reveal incompatible LeRobot version, pin the matching version and adapt the loader—do not force-upgrade and lose original data schema support. On M3/MPS, profile before choosing training duration. Use cloud GPU only after the two-step train/save/resume test and with bounded spend. Never commit credentials or user raw videos.

## 7. Sim-vs-real transfer and experimental validity

**The difficult part is not rendering a ball; it's bridging observations and dynamics.** The frozen checkpoint's **actual training schema** is authoritative: chosen camera keys, image sizes, proprioception and optional observable geometry are fixed at inference time. Report these experiments distinctly:

| Experiment ID | Data / policy | Simulation input | Claim supported |
|---|---|---|---|
| E0 `raw_replay` | Real recorded joint/EEF trajectory, no learned inference | Direct setpoint or retargeted replay | Calibration/control feasibility only |
| E1 `joint_ACT` | ACT trained on real SO-101 6D joint targets | Audited camera(s) + joints | Real-data ACT integration (SO-101-specific) |
| E2 `cart_ACT_so101` | Frozen ACT trained on real canonical 7D task actions | G0-selected cameras + observable EEF state | Main learned real→sim closed-loop evaluation |
| E3 `cart_ACT_panda` | **Exact same E2 checkpoint**, no Panda training | Matched camera schema + observable canonical pose | Attempted zero-shot cross-embodiment transfer |
| E4a `oracle_perception_same_schema` **conditional** | Same frozen E2 **only if E2 was trained with estimated object/goal geometry slots** | GT replaces **only those existing geometric features**; no new modalities | Privileged perception diagnostic; NOT primary result and not necessarily an upper bound |
| E4b `oracle_geometry_diagnostic` **alternative** | Separate geometry-based scripted controller or separately trained policy, clearly identified | Explicit simulator GT | Feasibility / perception-vs-control diagnostic; **cannot be reported as improvement of frozen E2** |
| E5 `ACT+det_rec` | E2 frozen + deterministic correction | Same observable inputs as E2 | Effect of rules, gating, replanning |
| E6 `ACT+MLP_rec` | E2 frozen + separately trained specialist | Same observable inputs as E2 | Incremental value of learned specialist |

**Oracle validity requirement:** A frozen RGB+proprioception ACT cannot suddenly consume ground-truth XYZ; shape compatibility alone is not enough—feature **meaning, order, units, normalization and train-time presence** must match. If there is no training-time estimated geometry channel, **E4a is NOT APPLICABLE**, and any E4b run is a different model/controller, not an E2 oracle ablation. When E4a is applicable, freeze the model, keep all non-geometric inputs unchanged, disclose source of GT and never mix its results into the non-privileged success rate. A different feature distribution can make oracle estimates worse, so don't assert a theoretical upper bound.

Aim for matched seeds and object initial states where practical. Count every attempt including failed grasp, failed place, loss of vision, unreachable IK, bowl miss, ball rolls out, timeout and explicit STOP. Distinguish camera/perception error from IK/controller error from physical contact failure. Report `successes/N`, not unsupported confidence intervals with tiny N.

## 8. Fallback decision tree (must execute, not just discuss)
- **Dataset access denied/metadata inconsistent:** support local dataset path; create audit-only and unit-test fixture; request user-authenticated download for true training. Never claim data trained.

- **Suspected corrupted `top` or unusable wrist stream:** run independent per-camera quality audit. Select the single reliable image stream if exactly one passes; for 2+ usable options, obtain human approval for single-vs-multi. If none passes, halt vision training and request new data. Never copy a bad camera into the model just to satisfy a two-view schema.

- **Two selected cameras have different native dimensions:** preprocess with aspect-preserving resize/letterbox to the same model-input shape; retain camera calibration transforms. Do not stretch 16:9 into 4:3 silently.

- **Calibration incomplete:** ship canonical FK with a clearly documented assumed task alignment for a controlled virtual setup; do not claim physically calibrated world coordinates.

- **Full 7D Cartesian actions unavailable/unstable:** keep `joint_ACT` as an honestly labelled working baseline; continue task-space conversion with logged IK residuals. Do not call joint policy cross-embodiment.

- **Learned ACT fails on sim due domain gap:** run E4a **only if its feature schema was present during ACT training**; otherwise label an E4b oracle geometry/controller diagnostic separately. Improve camera framing, an observable estimator, or bounded domain randomisation; retain measured `0/N` where applicable.

- **Sim ball physically cannot be grasped:** debug collision, actuator, gripper gap/contact; show script controller only as debug, not policy success.

- **Panda transfer fails:** publish IK retargeting result and failed frozen-policy rollout, diagnose whether gripper, view or reachable-pose issues; no fake transfer claims.

- **Recovery time/data inadequate:** deliver tested FSM/deterministic comparator with experiment labels; mark learned MLP incomplete, prioritise core learned result.

- **Deadline approaching:** stop scope expansion. Public reproducible dataset→ACT→MuJoCo evaluation with accurate failures beats polishing physics meshes or adding new models.

## 9. Definition of completion and agent reporting format
At end of **every milestone**, Codex must respond with:

```text
Milestone: M# — <name>
Status: PASS / PARTIAL / BLOCKED
Files added/changed: <paths>
Verified commands: <exact commands and result>
Dataset/checkpoint revisions: <IDs where applicable>
Metrics/evidence: <actual path(s) and sample counts>
Unverified assumptions: <explicit>
Blockers: <and chosen fallback>
Next milestone: <single highest-priority action>
```

A README/demo claim is acceptable only when traceable to a checked artifact/test/rollout. Maintain a dated change log and **freeze integration interfaces before assigning parallel worktrees**.

## 10. Parallel worktrees and **deferred** agent skills — after G1 only

**Codex owns** dataset interface, ACT setup, camera preprocessing, integration, evaluation metrics, merge decisions, README and releases. **Claude owns** `src/real2sim/sim/**`, `assets/**` and simulation-specific integration tests in a separate worktree, based on a schema reviewed by Codex. No simultaneous edits to `data_contract.md`, camera/action specification or `pyproject.toml`. Human operator exclusively controls live hardware. Interface freeze and ownership rules **apply from M0**, even though custom skills are deferred.

**Required immediately (M0):** minimal `AGENTS.md`, `uv`, Ruff, pytest, dataset audit, tests and progress tracking. This is the **execution/testing harness**, not an autonomous multi-agent framework. GitHub Actions CPU-only CI is helpful but not a gate if time is tight. Do not block ACT training on CI polish or skill-file generation.

**Do NOT create `.agents/skills/` or any `SKILL.md` before G1** (a verified real-data ACT checkpoint can be loaded and can infer from the approved camera schema). **After G1, if time permits**, propose these three compact portable skills, with purpose, invocation conditions, steps, constraints, acceptance commands, and pointers to authoritative tests/docs. Present 2–3 implementation/organization options and receive explicit user approval per §0.1 before writing:

- `.agents/skills/robot-data-contract/SKILL.md` — LeRobot revision/schema, camera audit selection, episode-only splits, image preprocessing and leak prevention.
- `.agents/skills/robot-kinematics/SKILL.md` — coordinate frames, FK/IK, units, underactuation, safety/limits and gripper semantics.
- `.agents/skills/experiment-integrity/SKILL.md` — learned vs scripted vs privileged controllers, frozen checkpoint compatibility, fixed scenarios, failure accounting and evidence.

**If approved**, each skill must be a **repeatable, test-backed procedure**, not a generic essay. Ensure `AGENTS.md` refers only to skills that actually exist. If G1 is delayed or submission risk increases, mark skill creation **DEFERRED / NOT REQUIRED FOR P0** in `docs/progress.md` and continue core research milestones. Optional skills never supersede tests or user decisions.

## 11. Verified technical sources to pin during implementation
- LeRobot ACT config (actual names `dim_model`, `n_heads`, `n_encoder_layers`, `n_decoder_layers`, `use_vae`, `chunk_size`, `n_action_steps`; same-shaped camera inputs): https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py

- LeRobot training workflow: https://github.com/huggingface/lerobot/blob/main/docs/source/il_robots.mdx

- LeRobot Placo RobotKinematics FK/IK, SO-101 weighted orientation: https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py

- MuJoCo Menagerie SO-101/Panda models (each with independent licensing): https://github.com/google-deepmind/mujoco_menagerie

- Original ACT: https://arxiv.org/abs/2304.13705

- DAgger: https://arxiv.org/abs/1011.0686

- Previous RaC paper reference: https://arxiv.org/abs/2509.07953

**Source-version principle:** pin installed LeRobot, MuJoCo, Menagerie and Placo revisions actually used by the implemented project, not the moving `main` branch.

---

**Human control reminder:** The decision-approval protocol in §0.1 applies to architecture, milestones, system design, and skill documents. Do not infer unapproved architecture decisions from this plan.
