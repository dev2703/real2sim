# Progress log

Factual record of commits, commands, verified outcomes, blockers and next actions.
Newest entries at the bottom. Dates are Australia/Sydney.

---

## 2026-10-12 — M0 read-only inspection (no code written yet)

**Repo state at start:** branch `main`, HEAD `d207dfa123b31703a7ad40dcfc6a24d8deb898d9`
("Update README.md"). Tree contained only `README.md`, `plan.md` (ignored by git, see below),
`pyproject.toml` (empty deps, `requires-python >=3.13`), `main.py` (hello world), `.python-version`
(3.13), `.gitignore`, `media_assets/bottowork.png`, empty `docs/`, empty `.venv` (Python 3.13.12,
no packages). No source package, tests, configs or CI exist.

**Housekeeping defect found:** `.gitignore` line `*.md*` with `!README.md` ignores `plan.md` and
every file under `docs/`. Fixed in this session so documentation can be tracked.

### Environment (verified)

| Item | Observed |
|---|---|
| Machine | Apple M3, 16 GB RAM, macOS Darwin 25.6.0, no CUDA |
| `uv` | 0.11.19 |
| Pythons | 3.12.13 (miniforge), 3.13.12 (miniforge), 3.14.x (brew/uv) |
| Repo `.venv` | Python 3.13.12, empty |
| conda env `lerobot` | Python 3.12.13, torch 2.10.0 (MPS available), pyav 15.1.0, opencv 4.13, pandas 3.0.1, pyarrow 23.0.1. Its editable `lerobot 0.5.1` install points to `/Users/deveshvyas/lerobot`, which no longer exists, so `import lerobot` fails in that env. No mujoco, no placo. |
| LeRobot source clone | `/Users/deveshvyas/Downloads/lerobot` at commit `6156e82ad7407fa70be740e42fdc7b8c5bc26d02` (2026-10-10, `v0.6.1-128`, pyproject version 0.6.2, dataset `CODEBASE_VERSION = "v3.0"`) |
| LeRobot on PyPI | latest 0.6.1, `Requires-Python >=3.12`, dataset format v3.0, torch `>=2.7,<2.12`, placo extra `>=0.9.6,<0.9.16`, torchcodec available for darwin-arm64 |
| uv resolution (py3.12 and py3.13) | lerobot 0.6.1, mujoco 3.15.0, torch 2.11.0, numpy 2.2.6, placo 0.9.16 (must be pinned `<0.9.16` to match LeRobot's extra) |
| ffmpeg | 8.1.2 with `libdav1d` AV1 decoder (needed: dataset videos are AV1) |
| MuJoCo Menagerie | `main` at `0059d4335f8156206f63a35662313385f7ad6d74` (2026-10-07). Contains `robotstudio_so101/` (Apache-2.0, derived from TheRobotStudio SO-ARM100 `so101_new_calib.xml`, joint names identical to the dataset's six names, `position` actuators, has a `wrist_cam`) and `franka_emika_panda/`. |

### Dataset (verified metadata, sampled frames only)

- Repo `bottowork/so101_act_pickplace_v1`, public, license apache-2.0, pinned revision
  `fd65a6600b0cd111a434ece1abfa3d44dc7b4d57` (lastModified 2026-10-11T23:15:41Z).
- A full local copy exists at `~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1`.
  Its `meta/info.json` is identical to the pinned revision and all four data parquet SHA-256
  hashes and all 13 video file sizes match the remote. No 1.4 GB re-download is needed.
- Full numbers in `docs/data_contract.md`. Headline: 40 episodes, 47,393 frames, 30 fps,
  6-joint state/action in degrees (gripper 0–100), two image keys `observation.images.top` and
  `observation.images.iphone` (both 1280x720 AV1). **No wrist key exists in the dataset.**
- Sampled frames (6 per camera, over HTTP, not a full decode audit):
  `top` is aimed at the ceiling (light fixture and wall edge visible, Laplacian sharpness
  0.2–2.6) and shows no robot, ball or bowl; `iphone` shows the arm, tennis ball, bowl and mat
  clearly (sharpness 312–441). Evidence: `results/audit/preliminary_frames/`.
- `iphone` video has 47,376 frames versus 47,393 data rows (17 short, in files 005 and 007;
  episode-table timestamps implicate episodes 25, 33, 34). Must be checked at decode time.

### Commands run (all read-only)

```
curl https://huggingface.co/api/datasets/bottowork/so101_act_pickplace_v1
curl .../resolve/fd65a66.../meta/{info.json,stats.json,tasks.parquet,episodes/...}
ffprobe .../videos/observation.images.{top,iphone}/chunk-000/file-00N.mp4
ffmpeg -ss T -i <remote mp4> -frames:v 1  (sampled frames)
pandas analysis of data/chunk-000/*.parquet (action vs state lag, timestamps, gripper)
uv pip compile (py3.12, py3.13) for lerobot/mujoco/placo
```

### Outcome

- Files added: `docs/progress.md`, `docs/data_contract.md`, `docs/decisions.md`,
  `docs/implementation_plan.md`, `docs/architecture.md` (DRAFT), `docs/system_design.md`
  (DRAFT), `docs/calibration.md` (requests only), `AGENTS.md`,
  `results/audit/preliminary_frames/*.jpg`. `.gitignore` fixed.
- No package code, tests or configs written yet (user asked for documents first).
- **Status: M0 inspection PASS; M0 implementation NOT STARTED; decisions D-002..D-010 PENDING.**
- Next action: obtain user answers to the pending decisions, then implement the M0 scaffold and
  the canonical `real2sim.data.audit` CLI against the local dataset copy.

---

## 2026-10-12 (evening) — decisions round + M0 implementation

**Decisions approved in chat** (see `docs/decisions.md`): D-002 A (uv, Python 3.12, LeRobot
0.6.1), D-003 C (joint = leader targets, Cartesian = achieved motion), D-004 dim 512 / batch 16 /
chunk 20 / execute 2, image tensor delegated → `[3,192,256]` letterbox, D-005 A (MLP 256-256-128),
D-006 30/10 split, D-007 10 Hz, D-009 B in principle (provider/budget still missing).
Pending: D-008, D-010, D-011.

**Branch:** `m0-scaffold-audit` (created from `main` at `d207dfa`); commit `e51bd05`
"M0: scaffold, uv environment, dataset audit CLI, docs and decisions", followed by a one-line
docs commit recording this SHA. Not merged to `main`; the user merges after review.

### Environment (built)

`pyproject.toml` + `uv.lock` on uv-managed CPython 3.12 (`~/.local/share/uv/python/cpython-3.12-macos-aarch64-none`):
lerobot 0.6.1 `[dataset]`, torch 2.11.0 (MPS available), torchvision 0.26.0, torchcodec 0.11.1,
mujoco 3.15.0, av 15.1.0, opencv-python-headless, pandas, pyarrow, scipy, pyyaml; dev: ruff,
pytest. `placo` moved to the optional `kinematics` extra because its 0.9.x wheel fails to import
on macOS arm64 (`liburdfdom_sensor.4.0.dylib` missing) — D-002 fallback applies (MuJoCo FK).
Known harmless warning: `objc` duplicate `AVFFrameReceiver` class from av and cv2 both bundling
libavdevice.

### Code added

`src/real2sim/{__init__,config}.py`, `src/real2sim/data/{raw,quality,montage,split,transforms,
synthetic,audit}.py`, `configs/data.yaml`, `annotations/episode_outcomes.csv` (40 × unknown),
`tests/unit/{conftest,test_raw_dataset,test_camera_quality,test_images}.py`,
`.github/workflows/ci.yml`. `main.py` (hello world) removed. `.python-version` → 3.12.

### Verified commands

```
uv sync --locked                 # ok
uv run ruff check .              # All checks passed!
uv run pytest -q tests/unit      # 21 passed in 16.6 s
uv run python -m real2sim.data.audit --help   # ok
uv run python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 \
  --revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 \
  --local-root ~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1 \
  --out results/audit --n-val 10 --seed 0      # 1 min 43 s wall, both cameras, every frame
```

### Audit outcome (artifacts: `results/audit/dataset_report.json`, `camera_quality.json`,
`camera_montage.html`, `montage/` (260 thumbnails), `split.json`, `episodes.csv`)

| Item | Result |
|---|---|
| Frames decoded | top 47,393/47,393; iphone 47,376/47,393; 0 decode failures |
| `top` | UNUSABLE (37/40 episodes, median sharpness 2.3, ceiling view) |
| `iphone` | USABLE in 37/40 episodes; PARTIALLY_USABLE in 25 (−2 frames), 33 (−14), 34 (−1) |
| Timing | all 40 episodes uniform 1/30 s, frame_index contiguous |
| Image/joint lag | +2 frames median on iphone (consistent across healthy episodes) |
| Action vs state | shoulder_lift offset −13.1°, lag 16 frames; details in `docs/data_contract.md` |
| Split | seed 0, 30/10, SHA `bb8bfb57…03c4a`, val = 1,2,3,4,11,18,23,24,27,34 |

Metric fix during the run: `state_max_step_per_frame` originally spanned episode boundaries
(reported a spurious 175° wrist jump); replaced by within-episode max/p99 and the audit re-run.

**Gate G0: PASS conditional on D-010** (which episodes to keep). Unverified: joint units,
episode outcomes, physical measurements. Unexpected: `media_assets/bottowork.png` is deleted in
the working tree by something outside this session; left out of the commit.

Next action: D-010 answer → regenerate split with exclusions → M1 loader/transform + ACT smoke
test on MPS.

---

## 2026-10-12 (late evening) — decisions round 2 + M1 pipeline

**Decisions approved in chat:** D-008 A (13-float proprioception), D-009 B (Modal USD 30, then
vast.ai/Kaggle), D-010 B (exclude episodes 25, 33, 34), D-011 (architecture and system design
approved), D-012 (standard calibration values). The user also supplied bowl dimensions in
`docs/calibration.md`. An image cross-check suggests the bowl is larger (≈12 cm outer diameter,
≈5–6 cm tall) than the supplied 10.4 cm / 3 cm; re-measurement requested, supplied values in use.

**Commits on `m0-scaffold-audit`:** `bf2060f` (decisions, split, calibration), `7bd8691` (M1).

**Incident:** `bf2060f` was staged with `git add -A <dirs>` and swept in another writer's
in-progress simulation files (`src/real2sim/sim/{__init__,env,kinematics,model}.py`,
`configs/sim_so101.yaml`). History was not rewritten while that writer is active. From `7bd8691`
on, only explicit paths are staged. Remaining untracked sim work (`assets/`, `scripts/`,
`configs/task_m2_diagnostic.yaml`, `tests/integration/test_{kinematics,so101_scene}.py`) is
left to its owner; 4 of its scene tests were failing at 22:50.

### Verified

```
uv run python -m real2sim.data.audit ... --exclude-episodes 25,33,34   # split 28/9, SHA 94880378…
uv run python -m real2sim.data.prepare --mode joint --config configs/data.yaml \
  --local-root ~/.cache/huggingface/lerobot/local/so101_act_pickplace_v1 --out-dir .cache/derived_joint
    # 59 s; 14,756 ticks (10,692 train / 4,064 val); images [14756,192,256,3] uint8
uv run pytest -q tests/unit                                         # 28 passed
uv run pytest -q tests/integration/test_act_smoke.py                # 1 passed (real data:
    # 2 steps, save, resume to step 3, reload, predict [2,20,6], select_action [2,6])
```

MPS timing (ACT dim 512, batch 16, 51.6 M params, ResNet-18 ImageNet init): **≈0.31 s/step**,
first step ≈0.6–5 s (warm-up / worker start). CPU: ≈0.85 s/step. Batch 16 fits on MPS without
accumulation.

**Bug found and fixed:** `to_device` used `non_blocking=True` for host→MPS copies of unpinned
tensors that were freed immediately. With in-process loading this produced corrupted batches
(L1 5.9 M at step 2 in one run, NaN in another, a plausible-looking but wrong value in a
third). Copies are now blocking; step-2 loss is identical across repeated runs and loader
settings.

### Running

`outputs/act_joint_so101` — 20,000 steps on MPS, validation every 1,000 steps on all 9 held-out
episodes, checkpoints every 2,000. Log: `outputs/act_joint_so101/train.log`. Expected ≈2 h.

---

## 2026-10-13 (~00:30 AEDT) — M3 labels, G1 reached, Modal blocked

**Decision approved:** D-013 A (gripper target = leader command / 100).

**Commits:** `61711b7` (geometry + FK), `aa1fce4` (labels, cartesian prepare, Modal script), and
this docs commit.

### Gate G1: PASS on an intermediate checkpoint

`outputs/act_joint_so101/checkpoints/002000` (model.safetensors SHA-256 `f788bcbd14a704297c11aa605e194e440da92c5bdcc916659b84767da8459fc4`).
This is a joint-space ACT trained on real recordings, 2,000 of 20,000 steps.

```
uv run python -m real2sim.policy.check --checkpoint outputs/act_joint_so101/checkpoints/002000 \
  --data .cache/derived_joint --out results/metrics/act_joint_g1_check_step2000.json
```

- Contract matches the data: camera `observation.images.iphone`, image [3,192,256], split SHA `94880378…`.
- Predicts finite action chunks of shape [16, 20, 6].
- Open-loop validation L1 on the 9 held-out episodes (4,064 ticks): 0.337 normalised. Per joint
  in degrees / %: pan 3.4, lift 13.8, elbow 12.4, wrist_flex 15.1, wrist_roll 2.5, gripper 9.5.
- Step 1,000 was slightly better (0.325). Watch for overfitting; select by validation.
- **This is offline prediction error on recorded episodes, not closed-loop or task success.**

Deliverables: `results/metrics/act_joint_training_history.csv` (snapshot),
`results/configs/act_joint_resolved.yaml`, `results/metrics/act_joint_g1_check_step2000.json`.

Reproduce:

```
uv run python -m real2sim.data.prepare --mode joint --config configs/data.yaml --local-root <dataset> --out-dir .cache/derived_joint
uv run python -m real2sim.policy.train --config configs/act_joint.yaml --out outputs/act_joint_so101
```

### M3 Cartesian data

`.cache/derived_cartesian`: 14,756 ticks, 13-float state, 7-D action, same split.
Diagnostics: 0 jump flags; 37 extrapolated end-of-episode ticks; leader-vs-follower EE error
median 3.7 cm, p95 27.6 cm. A 2-step Cartesian smoke train and the check CLI both pass.

### FK sanity check (joint mapping)

Gripper FK height at sustained grasp is 0.034 m against the 0.0335 m ball radius (n = 12).
This supports the provisional mapping but does not verify it. See `docs/calibration.md`.

### Blocked

Modal refused GPU functions: "Please add a payment method to use A10G GPU functions."
Only the CPU image build ran. GPU training needs the user to add a card on Modal, or to use
vast.ai/Kaggle. Fallback in use: MPS.

### Running (background, MPS)

1. `outputs/act_joint_so101`: 20k steps, about 0.42 s/step, expected to finish around 02:30 AEDT.
2. `outputs/act_cartesian`: queued by `outputs/act_cartesian/queue.sh`. It starts automatically
   when the joint run writes step 20,000 and takes a similar time.

Skills: G1 is reached, so the three optional skills (plan §10) may now be proposed. They are
deferred because the core milestones (M2 sim, M4 closed loop) matter more.

---

## 2026-10-13 00:50 AEDT — M2 complete, M4 harness, M5/M6 packages (two workflows, 78 agents)

Two orchestrated workflows built and adversarially reviewed the remaining milestones. Every
reported number below comes from a test or a CLI run inside the working tree.

**M2 (SO-101 scene)** — `configs/sim_so101.yaml` now reads all geometry from
`configs/task.yaml` (the diagnostic duplicate was deleted). `So101PickPlaceEnv` gained
real-unit adapters (`real_state`, `step_real_joint_targets`, `ee_pose`, `canonical_proprio`),
`observation.state`, and privileged scoring metrics. Bowl physics measured and recorded in
`results/audit/sim_bowl_physics.json`. Camera: PnP fit, 17.2 px RMS on 27/43 points, applied
with full axes (roll −10.9°). 48 scene/kinematics tests pass.

**E0 replay** (`python -m real2sim.evaluation.replay --episode 1`): joint setpoint tracking
median 0.05° (p95 elbow 5°), EE p95 3.2 cm after the start transient; FK→IK 96 % accepted.
Captions `SCRIPTED REPLAY`. Videos in `results/media/replay/`.

**M4 harness** — `real2sim.policy.adapters` (contract-bound `ObservationAdapter` that refuses
foreign slots: E4a is NOT APPLICABLE), `real2sim.evaluation.run` with `trials.csv`,
`summary.json`, captioned videos, HOLD_POSE null baseline. **E1 debug run: 0/3** (joint ACT
step 2000, seeds 0–2, 300 ticks; phase_failed = approach; latency ≈ 125 ms on a loaded CPU):
`results/metrics/debug_so101_joint/`. This run predates the camera fix and the D-014 fix.

**M5 Panda** — `real2sim.panda`: scene, TCP site, 7-DoF full-pose IK, gravity compensation
(D-015); 12/12 validation poses, residual ≤ 0.22 mm. 13 tests. Retargeting replay and frozen
E2 rollouts pending a Cartesian checkpoint.

**M6 recovery** — `real2sim.recovery`: observable-only detector, FSM, loop, MLP, dataset
loader; 40 unit tests. No correction data exist, so E6 is not completed.

**Joint mapping** — signs verified (D-014): only the current mapping places grasps on the
table; posture is elbow-up in both sim and real. The 2.3 cm start-pose penetration points to a
base height offset; plumbing for `robot_so101.base_z_offset` exists, value awaits approval.

**Verification:** `uv run ruff check .` clean; `uv run pytest -q tests` → **223 passed** in 52 s.

**Training:** joint run at step ~13 000 of 20 000 (best val L1 0.318 at 11 000 and 13 000);
Cartesian run queued.

---

## 2026-10-13 08:10 AEDT — both ACT runs finished overnight; E4b oracle 10/10; D-014 approved

**Joint ACT (act_joint_so101)** — 20 000 steps on MPS, 2 h 11 min. Validation L1 (normalised)
went 0.325 → 0.312; the final checkpoint 020000 is the best (SHA-256 `d27ff21f869804545723d354c112863a399c21857e4bd73ba0e525dbc4411274`).
Per joint on the 9 held-out episodes: pan 2.9°, lift 13.4°, elbow 10.3°, wrist_flex 14.4°,
roll 2.4°, gripper 9.2 %. Evidence: `results/metrics/act_joint_g1_check_step20000.json`,
`results/metrics/act_joint_training_history.csv`. Open-loop numbers only.

**Cartesian ACT (act_cartesian)** — 20 000 steps on MPS. Validation L1 0.528–0.559, best saved
checkpoint 006000 at 0.534 (SHA-256 `0a3b22211c9775ca2c28ade4a11627f0f2b8917959fd2eda501dc1d7acd5bff0`); the loss stops improving after ~3 000 steps.
In raw units: 1.6–1.8 mm per-tick translation error against a 3.7 mm median tick motion, 5–17
mrad rotation, 0.10 gripper. Evidence: `results/metrics/act_cartesian_training_history.csv`,
`results/configs/act_cartesian_resolved.yaml`. Frozen checkpoint for E2/E3 = 006000.

**E4b oracle (scripted, privileged GT, labelled ORACLE_GEOMETRY_DIAGNOSTIC):** 10/10 pick-and-
place successes on seeds 0–9 and across the friction sweep 0.3/0.6/1.0; the simulated gripper
holds the 67 mm ball with both jaws (settles at 43 % opening, matching the real 30–40 %). The
dataset's grasp style (fixed jaw under the ball) was infeasible before the base lift. This is
scene-feasibility evidence, not an upper bound for the learned policy.
`results/metrics/oracle_so101/`, videos `results/media/oracle/`.

**Camera/bowl:** bowl centre re-derived by back-projecting the in-bowl ball detections
through the PnP camera: (0.328, −0.163) m, std 0.8/1.3 cm; start region (0.331, 0.069) m.
The FK-at-release value was 6.3 cm off (it marks where the gripper opened, not the bowl).

**D-014 approved:** `robot_so101.base_z_offset = 0.012` (PROVISIONAL). A workflow is making
the height-dependent tests and back-projection planes offset-aware.

**Panda retarget (E0-Panda):** `python -m real2sim.panda.retarget --episode 1` built; results in
`results/metrics/panda_retarget/`.

**Panda retarget detail** (`results/metrics/panda_retarget/panda_retarget_ep1.json`, label
SCRIPTED_REPLAY): 598 ticks of validation episode 1 retargeted through the Panda 7-D adapter;
IK accepted 567/598 (94.8 %, the 31 rejections all joint-limit), accepted residual p95 0.23 mm /
5 mrad, no robot-table contact. Position error versus the SO-101 FK path grows from 0 to 31 cm
over the episode because deltas are executed relative to the Panda's own TCP from a different
start pose (0.21 m offset) and rejected ticks lose their motion. This is an engineering
baseline for the adapter, not a transfer result.

---

## 2026-10-13 09:10 AEDT — G4/G5 primary experiments (honest 0/N) and failure isolation

Geometry frozen at commit `5a29044` (base lift 0.012 m, camera refit, v3 centres); 248 tests pass.
Checkpoints: joint `outputs/act_joint_so101/checkpoints/020000` (SHA-256 `d27ff21f8698…`),
Cartesian `outputs/act_cartesian/checkpoints/006000` (`0a3b22211c97…`), frozen; no Panda training.

| Experiment | Controller | Robot | Checkpoint | Success | Lifted | Phase failed | Notes |
|---|---|---|---|---|---|---|---|
| E0 `hold_pose_null` | HOLD_POSE (no policy) | SO-101 | — | **0/10** | 0 | approach ×10 | null floor |
| E0 `hold_pose_null` | HOLD_POSE (no policy) | Panda | — | **0/10** | 0 | approach ×10 | null floor |
| E1 `joint_ACT` | LEARNED_ACT | SO-101 | joint 020000 (`d27ff21f8698…`) | **0/10** | 0 | approach ×10 | min EE–ball 14–17 cm; arm rises and holds |
| E2 `cart_ACT_so101` (run 1) | LEARNED_ACT | SO-101 | cartesian 006000 (`0a3b22211c97…`) | **0/10** | 0 | approach ×10 | 4 trials had 227–238 IK rejections from a workspace-floor bug |
| E2 `cart_ACT_so101` (run 2, floor fixed) | LEARNED_ACT | SO-101 | cartesian 006000 | **0/10** | 0 | approach ×10 | 2–3 IK rejections per trial; min EE–ball 15–19 cm |
| E3 `cart_ACT_panda` | LEARNED_ACT (same frozen E2 weights, no Panda training) | Panda | cartesian 006000 | **0/10** | 0 | approach ×10 | 0 IK rejections; min EE–ball 6–12 cm |
| E4b `oracle_geometry_diagnostic` | ORACLE (scripted on simulator GT, privileged) | SO-101 | — | **10/10** | 10 | none | scene feasibility only, not an E2 upper bound |
| E5/E6 recovery | — | — | — | not run | | | detector/FSM/MLP built; no correction data (G6 not completed) |

Every trial is a row in `results/metrics/trials.csv` (70 rows). Seeds 0–9, 300 ticks (30 s) at
10 Hz, success = ball resting inside the bowl and released for 1 s. Captioned videos per trial in
`results/media/` (not tracked in git; regenerate with the commands below).


**Failure isolation (plan M4.4).** The learned policies never reach the ball: minimum
end-effector–ball distance 14–19 cm on the SO-101 and 6–12 cm on the Panda, versus the oracle's
grasp at tick ~30. Evidence for each candidate cause:

- *Gripping / scene physics*: ruled out as the blocker. The scripted oracle grasps, carries and
  places the ball 10/10 with the same gripper, contacts and friction sweep.
- *IK / controller*: the first E2 run rejected 227–238 of 300 ticks in 4 trials because the IK
  workspace floor (z ≥ 0.015) sat above the lifted start pose; with the floor at the table top
  the re-run has 2–3 rejections per trial and the outcome is unchanged (0/10). Not the cause.
- *Vision domain gap*: measured directly by `real2sim.evaluation.domain_gap` on validation
  episode 1 (120 ticks): feeding the frozen policy a simulated render of the recorded state
  instead of the real frame raises the open-loop error from 10.4° to 13.5° (joint policy, ×1.29)
  and from 0.016 to 0.023 (Cartesian, ×1.44), with the gripper channel taking most of the hit
  (7.0→14.0 %, 0.082→0.127). A moderate gap, not a collapse.
- *Open-loop quality and compounding (distribution shift)*: the Cartesian policy's per-tick
  error on *real* frames is 1.5–2 mm against a 3.7 mm median tick motion (relative error ≈ 0.4);
  the joint policy's is 10–21° on the large joints. With 28 training episodes and no
  perturbation data, closed-loop execution drifts off the demonstrated manifold within the
  first second and never recovers. This is the dominant cause on the evidence above.

Latency: 30–70 ms per chunk on CPU (two runs sharing the machine); no timeouts from latency.

Commands (each ≈ 1–5 min on CPU):

```
uv run python -m real2sim.evaluation.run --robot so101 --checkpoint outputs/act_joint_so101/checkpoints/020000 --n-trials 10 --seed-start 0 --out results/metrics/E1_joint_so101 --video
uv run python -m real2sim.evaluation.run --robot so101 --checkpoint outputs/act_cartesian/checkpoints/006000 --n-trials 10 --seed-start 0 --out results/metrics/E2_cart_so101_v2 --video
uv run python -m real2sim.evaluation.run --robot panda --checkpoint outputs/act_cartesian/checkpoints/006000 --config configs/sim_panda.yaml --eval-config configs/eval_panda.yaml --n-trials 10 --seed-start 0 --out results/metrics/E3_cart_panda --video
uv run python -m real2sim.evaluation.run --robot so101 --controller hold --n-trials 10 --out results/metrics/E0_hold_so101
uv run python -m real2sim.evaluation.oracle --config configs/sim_so101.yaml --n-trials 10 --seed-start 0 --out results/metrics/oracle_so101 --video results/media/oracle
uv run python -m real2sim.evaluation.domain_gap --checkpoint outputs/act_cartesian/checkpoints/006000 --data .cache/derived_cartesian --episode 1 --out results/metrics/domain_gap/cart_ep1.json
uv run python scripts/aggregate_trials.py
```

**Gates:** G4 PASS (instrumented learned closed-loop rollouts exist; success 0/10, causes
isolated). G5 PASS as *attempted* transfer (0/10, IK feasible, 0 rejections). G6 NOT COMPLETED
(no correction data; FSM/detector/MLP unit-tested only). Subagent workflows stopped at 08:30 when
the API usage limit was reached; the remaining work was done directly.

---

## 2026-10-13 — Codebase cleanup, logic review and edge-case tests

Three read-only reviewers (data/geometry, policy/evaluation, sim/Panda/recovery) plus direct
fixes. Tests 248 → **375 passed**; coverage 84 % → **87 %**; `ruff check .` clean.

**Bugs fixed (confirmed by running code):**

| Area | Bug | Effect before the fix |
|---|---|---|
| Panda scene | ignored the D-014 base lift and the fitted camera roll | E3 run 1 measured on a mis-registered scene (12 mm proprio shift, rotated image) |
| prepare | aligned video frames to rows by count only | a frame dropped at a segment start shifted every image 33 ms, undetected ~2/3 of the time |
| scoring | phase attribution, "lifted" by a bump, roll-out on successful trials, aborts hidden as `approach` | per-phase failure counts could be wrong; a harness bug could look like a policy failure |
| adapters | runtime guard declared but unused; NaN images passed validation; `n_action_steps=0` silently replaced | guards existed only in tests |
| recovery MLP | per-component box clamp vs the adapters' norm clamp | a diagonal step could reach 1.7× the limit with a changed direction |
| labels / stats | extrapolated end-of-episode labels trained as real targets | 37 placeholder targets in the loss and the action statistics (applies to future training) |
| geometry | `log_so3` lost 1e-4 near π; rot6d / letterbox degenerate inputs gave NaN / crash | latent |
| config / split / audit / check / train | duplicate cameras, null exclusions, empty training set, short video passing audit, re-prepared data passing check, duplicated history rows on resume, non-atomic `last` link | latent |

Re-preparing the real joint dataset with the new alignment checks is byte-identical to the one
the checkpoints were trained on. All primary experiments were regenerated under the final code
(`results/metrics/trials.csv`, 90 rows): E1 0/10, E2 0/10, E3 (corrected scene) 0/10 with two
grasp-phase trials, HOLD 0/10 on both robots, oracle 10/10, 0 aborted trials.

**Cleanup:** one `REPO_ROOT`/`project_path` (was four copies); dead constants removed; videos
untracked (the ignore pattern missed subdirectories); `.coverage`/cache artefacts ignored.

---

## 2026-10-13 — D-016 diagnostics completed; bounded MPS remedies launched

**Milestone:** M1/M4 follow-up. **Status: PARTIAL — diagnostics PASS, retraining RUNNING.**
Base commit `7943dd1`; this session's changes are uncommitted. User explicitly requested the
two diagnostics, augmentation/regularisation retraining, and background execution without
agent polling while waiting. No additional data source was selected.

### Verified findings

All image conditions use inference/eval mode and the zero VAE latent, with the same state,
labels and masks. Samples are 32 evenly spaced ticks per episode: 896 train ticks (28 episodes),
288 validation ticks (9 episodes). Shuffled image donors always come from another validation
episode; their tick mapping is recorded and hashed in each report.

| Frozen checkpoint | Clean train L1 | Clean validation L1 | Blank validation L1 | Shuffled validation L1 |
|---|---|---|---|---|
| joint 020000 | 0.08009 | 0.30049 | 0.45413 | 0.64177 |
| Cartesian 006000 | 0.34690 | 0.52264 | 0.63585 | 0.72288 |

These measurements establish sensitivity to images and an inference-mode generalisation gap.
They do not prove the policy localises the ball: robot appearance or motion phase could be
the relevant visual cue. Comparing the old training loss directly to validation was confounded
by the target-conditioned VAE posterior at training time; the new clean-train check removes it.
Evidence: `results/metrics/remedies/vision_{joint,cartesian}.json`.

Familiar starts use the existing camera-fit ball estimates for training episodes **0, 5, 6, 7**.
For **each** checkpoint, three conditions (4 attempts each) measured **0/4**: random ball,
familiar ball, and familiar ball plus recorded initial joint pose. All 24 attempts failed in
approach, none aborted; no recorded controls are injected after reset. These diagnostic trials
are separate from the primary benchmark. Exact physical camera/scene matching is not claimed.
Evidence: `results/metrics/remedies/familiar_{joint,cartesian}/comparison.json`, condition
`trials.csv` and full `summary.json` histories.

Reset confound: three of the four recorded starts have 9.4–11.4 mm robot/table overlap; episode
0 has none. The first 100 ms hold tick moves the tool by **5.1–9.4 mm** across these starts,
including the contact-free one. This is evidence for transient control/dynamics mismatch as well
as contact push-out; it cannot be attributed to policy error alone. Familiar starts still fail,
so novel ball position alone is insufficient to explain failure. Limited data coverage,
compounding error, visual mismatch and the provisional simulation remain plausible contributors.
The earlier moderate open-loop image-gap ratio does not rule out closed-loop domain shift.

### Changes and verification

- `evaluation/vision_ablation.py`, `evaluation/familiar_starts.py`; optional explicit reset
  scenarios in the existing trial runner, with all primary defaults and scoring unchanged.
- `policy/augmentation.py`: train-only brightness/contrast ±20%, saturation ±15%, reflected
  translation ≤4 px; original letterbox padding retained. Dataset refuses augmentation on
  validation episodes. State, action, masks, arrays, statistics, and source episodes unchanged.
- `policy/train.py`: save every validation best, atomic `best`/`last` links, configurable
  early stopping, final `training_result.json`, and checkpoint augmentation metadata.
- `configs/act_{joint,cartesian}_augmented.yaml`: fresh 512/ResNet-18 runs with dropout 0.2,
  weight decay 1e-3, batch 16, maximum 6,000 steps, full validation each 1,000; patience 3,
  minimum 3,000 steps, minimum significant improvement 0.001. One seed; effects of the separate
  hyperparameter changes are not isolated. No simulator demonstrations or baseline resume.
- `scripts/run_act_remedies.py`: sequential detached MPS training, then best-checkpoint checks
  and 10 learned closed-loop SO-101 trials per model, with persistent status and error logs.

Commands actually run:

```text
uv sync --locked                                      # PASS
uv run ruff check .                                   # PASS
uv run pytest -q tests/unit                           # 277 passed
uv run pytest -q tests/integration                    # 107 passed
uv run python -m real2sim.evaluation.vision_ablation --help   # PASS
uv run python -m real2sim.evaluation.familiar_starts --help   # PASS
uv run python -m real2sim.evaluation.vision_ablation --checkpoint outputs/act_joint_so101/checkpoints/020000 --data .cache/derived_joint --out results/metrics/remedies/vision_joint.json --device cpu --per-episode 32
uv run python -m real2sim.evaluation.vision_ablation --checkpoint outputs/act_cartesian/checkpoints/006000 --data .cache/derived_cartesian --out results/metrics/remedies/vision_cartesian.json --device cpu --per-episode 32
uv run python -m real2sim.evaluation.familiar_starts --checkpoint outputs/act_joint_so101/checkpoints/020000 --data .cache/derived_joint --out results/metrics/remedies/familiar_joint --device cpu
uv run python -m real2sim.evaluation.familiar_starts --checkpoint outputs/act_cartesian/checkpoints/006000 --data .cache/derived_cartesian --out results/metrics/remedies/familiar_cartesian --device cpu
```

The integration suite includes a real-data augmented forward/backward/save/reload check on MPS
and a zero-learning-rate test that verifies early stopping and selection of the earlier best
checkpoint; these are smoke-test artifacts, not a useful trained policy.

**Source/checkpoints:** dataset `fd65a6600b0cd111a434ece1abfa3d44dc7b4d57`; frozen originals
joint SHA-256 `d27ff21f869804545723d354c112863a399c21857e4bd73ba0e525dbc4411274` and Cartesian
`0a3b22211c9775ca2c28ade4a11627f0f2b8917959fd2eda501dc1d7acd5bff0` (also in diagnostic JSONs).
No completed augmented checkpoint or improved task outcome is claimed at launch.

**Background execution:** detached `.venv` Python runs `scripts/run_act_remedies.py`;
manager PID recorded in `outputs/act_remedies/manager.pid`. Status, per-stage logs and eventual
comparisons are in `outputs/act_remedies/`; outputs are separate from the originals. The manager
records errors and stops on failure, refuses existing run directories, and holds an exclusive
job lock. No agent polling is needed for training or follow-up checks to complete.

**Unverified:** benefit of retraining, exact camera/joint calibration, physical matching of
training starts. **Blockers:** none for the authorised retrain; new data route remains undecided.
**Next action:** inspect the manager's final validation and 10-trial comparisons, then choose
additional real demonstrations versus an explicitly labelled mixed-data experiment separately.

## 2026-10-13 — D-017 loss comparison queued behind the remedies

Commit base: `7943dd1bf7f9e2c87bf24057359be3fc2effa0de` plus uncommitted working changes.
User approved the proposed L1/L2/Huber experiment after the current retrains. Existing D-016
L1 runs are controls; four additional runs use L2 and Huber for joint and Cartesian policies.
Each copies its control's resolved configuration, including the maximum 6,000 steps, seed 0,
augmentation, optimizer and validation-MAE stopping/selection. KL beta stays 10, with separate
reconstruction/weighted-KL/gradient-norm logs. This is a fixed-beta, single-seed comparison.

Implemented `policy/losses.py`; opt-in loss selection and best/last retention in `policy/train.py`;
full-validation RMSE and absolute-error p95 alongside existing MAE in `policy/check.py`; the
sequential manager `scripts/run_act_loss_comparison.py` and `configs/act_loss_comparison.yaml`.
Default L1 still calls upstream LeRobot's forward method. Only checkpoint directories created
by the current opted-in training call can be pruned. The manager refuses changed control data
manifests or existing output runs, waits on the predecessor's process lock, and stops if that
predecessor failed. Checkpoints include loss metadata; checks include a weights SHA-256.
It automatically checks and runs 10 simulation trials (seeds 0–9) for each new best checkpoint,
reusing the existing controls' completed trials.

Verified commands:

- `uv sync --locked` — passed.
- `uv run ruff check .` — passed.
- `uv run pytest -q tests/unit` — **297 passed**; formulas/gradients, masked padding, upstream
  L1 parity, unchanged KL, error metrics, retention and actual interprocess queue locking.
- `uv run pytest -q tests/integration` — final run **109 passed**, including two actual
  real-data MPS optimization steps for each of L2 and Huber, save/reload, finite inference and
  validation, and rejection of a resume with a different objective.
- `uv run python scripts/run_act_loss_comparison.py --help` — passed.
- `git diff --check` — passed.

Initial integration run: two new tests incorrectly required tighter agreement than float32
arithmetic/CSV rounding supports when summing reconstruction and weighted KL. Corrected the
assertion to relative tolerance 2e-7 plus absolute 2e-6; no training formula changed. The same
run hit an existing renderer exact-equality failure: 3 of 147,456 image components differed by
1/255 on repeated reset. The unmodified renderer test passed in the full rerun; the cause is
unverified. No simulator geometry, rendering logic or test tolerance changed.

At handoff: loss manager PID **32373** is alive with its own lock held, status
`waiting / waiting_for_remedies`, zero completed phases. The predecessor lock is held and its
phase is `cartesian_train`. Confirmed without an agent polling loop. Full new experiment
results are **pending**, not successful training claims. Status/logs:
`outputs/act_loss_comparison/`; eventual comparisons: `results/metrics/loss_comparison/`.

D-016 interim evidence: joint augmentation training stopped at step 4,000; best step 1,000,
full-validation MAE **0.3500843**, worse than original joint baseline **0.311605**. The completed
joint remedy evaluation is **0/10**. Evidence: `outputs/act_joint_augmented/training_result.json`,
`results/metrics/remedies/joint_augmented_check.json` and
`results/metrics/remedies/joint_augmented_closed_loop/summary.json`. Cartesian result pending.

Source remains `bottowork/so101_act_pickplace_v1` at
`fd65a6600b0cd111a434ece1abfa3d44dc7b4d57`; joint split SHA-256
`948803789de23605f85378917c40680be567764c17df0774979a6f5f50091717` (28 train / 9 validation).
No data, camera schema, normalizer, labels, frequency, original checkpoint or hardware changes.
`docs/recording_checklist.md` contains proposed 30/40-episode budgets, camera pilot checks,
position/recovery coverage and whole-episode holdouts. A new usable wrist stream needs a
separate audit and a subsequent training/simulator camera decision.

Milestone: M1/M4 follow-up — reconstruction-loss experiment setup
Status: PARTIAL — setup verified and queued; full loss training/results pending
Files added/changed: policy loss/train/check code, queue config/script, tests, decision/status
  docs and recording checklist
Verified commands: listed above, 406 tests passed in the final suites
Dataset/checkpoint revisions: unchanged source/split above; L1 controls selected by D-016
Metrics/evidence: D-016 joint 0.3500843 MAE, 0/10; no full L2/Huber result yet
Unverified assumptions: fixed-beta loss-scale effects, one seed, continued sim reset/domain gap
Blockers: none for setup; queue waits for predecessor completion
Next milestone: inspect automatic comparisons when the queued runs finish

## 2026-10-13 — D-018 hourly checks installed

Current managers verified healthy: remedies in `cartesian_train` with fresh log output; loss
comparison waiting with its lock held. About 9.8 GiB free at inspection. No failure repair or
extra training justified at setup. Note that the joint remedy's CPU reload check reports MAE
0.3544541, whereas training-time MPS validation recorded 0.3500843; these are separate measured
values, not interchangeable. The automatic loss comparison evaluates all checkpoints on CPU.
The cause of that difference has not yet been isolated.

Implemented/tested `scripts/monitor_act_jobs.py`: hourly stdlib-only checks; no model calls for
healthy unchanged jobs; deduplicated same-chat follow-ups for issues or newly completed results.
Checks include manager locks, stale phase logs, failed/missing status, predecessor handoff and
free disk space. The monitor never kills or restarts training itself; the existing chat diagnoses
and repairs issues. Stop control: `outputs/act_monitor/STOP`.

Verification: Ruff passed; `uv run pytest -q tests/unit/test_act_monitor.py` — **4 passed**;
`python3 scripts/monitor_act_jobs.py --once` — no issues, both manager locks held. The installed
CLI accepted a same-chat delivery test (exit 0; receipt stored in
`outputs/act_monitor/delivery_test.json`). Processing that queued test is separate from delivery
acceptance; it instructs the receiving turn only to record the acknowledgment and end.

Monitor launched as PID **41501**. First scheduled check **13 Oct 2026 11:21:14 AEDT**
(`2026-10-13T00:21:14Z`), then hourly. State/config/logs under `outputs/act_monitor/`.
No native IDE/cloud schedule was created; this is a local process and requires the Mac to remain
available. Sleeping delays checks; missed intervals are coalesced. CLI handoff requires Codex
to remain available. The monitor stops after a follow-up writes STOP when all work is reviewed.

## 2026-10-13 — D-016/D-017 completed results reviewed

Both managers completed without reported runtime failure. Remedies completed at 10:38 AEDT;
the loss comparison completed at **12:40 AEDT**. All joint runs early-stopped at 4,000 steps
with best checkpoint 1,000; all Cartesian runs at 5,000 with best checkpoint 2,000.
The table uses the common **CPU reload evaluation** over the complete held-out split, not
training-time MPS validation. Values are in normalized action units.

| Policy | Reconstruction | Validation MAE | RMSE | Absolute-error p95 | SO-101 success |
|---|---|---:|---:|---:|---:|
| Joint | L1 | 0.354454 | 0.516133 | 1.111740 | 0/10 |
| Joint | L2 | 0.373012 | 0.519244 | 1.094892 | 0/10 |
| Joint | Huber | 0.367642 | 0.517601 | 1.095051 | 0/10 |
| Cartesian | L1 | 0.566122 | 0.926188 | 2.000558 | 0/10 |
| Cartesian | L2 | 0.599552 | 0.917620 | 1.932990 | 0/10 |
| Cartesian | Huber | 0.596240 | 0.924646 | 1.969469 | 0/10 |

Evidence: `results/metrics/loss_comparison/summary.json`, its six `*_check.json` files,
the four new `*_closed_loop/summary.json` files, and reused L1 controls under
`results/metrics/remedies/`. These are six evaluations of the same ten scenario seeds, not
sixty distinct starting scenarios. Original frozen checkpoints/primary experiments unchanged.

Interpretation: L1 has lower MAE in this fixed-beta, single-seed comparison. L2 reduces
Cartesian RMSE by about 0.9% and p95 by 3.4%, but gives worse MAE and no task-success improvement.
This does not establish that L1 universally outperforms a separately tuned L2/Huber objective.
Final joint reconstruction/weighted-KL values were approximately 0.164/0.223 (L1), 0.059/0.218
(L2), and 0.028/0.220 (Huber); the objective-scale confound remains substantial. No additional
blind training run launched. Data coverage/recovery and simulator alignment remain unresolved.

Hourly monitor recorded six checks through 16:21 AEDT with no job issues and successfully
queued follow-ups. An earlier process-liveness verification was rejected by automatic approval
review because the account hit its usage limit, delaying agent follow-up; the detached jobs
and local monitor continued. Delivery receipt is now acknowledged in
`outputs/act_monitor/delivery_verified.json`. Both queues complete, results reviewed: wrote
`outputs/act_monitor/STOP` to stop further checks, without killing any training process.

Milestone: M1/M4 follow-up — reconstruction-loss experiment
Status: PASS for experiment execution; all learned task outcomes remain 0/10
Files added/changed: implementation status/progress; monitor completion controls
Verified commands: status/result/history inspection; all six train/check/evaluation pipelines
  report completion; prior setup verification remains 297 unit + 109 integration tests
Dataset/checkpoint revisions: unchanged real source fd65a660… and split; exact checkpoint
  SHA-256 values recorded in each *_check.json
Metrics/evidence: table and artifact paths above
Unverified assumptions: one seed, fixed KL beta, backend discrepancy, simulator alignment
Blockers: no job failure; task success still unresolved
Next milestone: audit additional real recordings and target the remaining simulation/data gaps


## 2026-10-13 — frozen joint visual-causality audit

User requested assessment of image-swap/settled-start findings and additional
peculiarities. Added `scripts/diag_visual_causality.py`; no training, hardware,
schema, benchmark or shared simulation changes in this audit. Existing appearance
work in the workspace was preserved and copied into the diagnostic snapshot.
Commit: uncommitted working tree.

Evidence: `results/metrics/debug_so101_joint/visual_causality_20261013_render/`
(`findings.md`, `report.json`, input/source snapshots, input tensors,
`action_limit_audit.json`, `split_end_pose_audit.json`). Original joint checkpoint
020000 SHA-256 d27ff21f869804545723d354c112863a399c21857e4bd73ba0e525dbc4411274;
source revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 and split unchanged.
Six episode starts: train 0/5/6/7, validation 1/3. No fit or parameter selection.

Verified results: real images yield mean first wrist target +80.67 degrees;
current sim images +41.96; real images with five-tick-settled state +82.01.
The previous "no ball" condition still shows 932 native-render ball pixels;
true absence has zero pixels and shifts the first two wrist commands by mean
absolute 4.31 degrees, without restoring real-image behaviour. Only the first
two of 20 actions execute before replanning. Dataset/runtime normalization
matches exactly on the six real inputs. No input file changed during the run.

Data audit: training shoulder labels outside current simulator limits 29.32%;
shoulder command/state absolute gap p95 60.48 degrees train, 70.07 validation.
Stationary-shoulder-with-gap flag (<0.2 degree increment, >10 degree gap, 10 Hz)
is 18.51% of train transitions versus 29.45% validation; longest 20.4 s in ep30.
These flags do not identify a mechanical fault. Final wrist below -90 degrees:
16/28 training episodes, 0/9 validation. This is a distribution difference, not
permission to change the split or proof it causes failed approaches.

Interpretation: images demonstrably cause inappropriate initial joint commands;
exclusive causality of all 0/10 outcomes remains unproven. Existing appearance
sweep did not restore real-image predictions. Prioritize camera/scene structure
and phase-conditioned visual tests, then unchanged E1 evaluation. Do not tune
rendering to held-out actions or treat a matching initial chunk as task success.

Commands verified:
- `.venv/bin/python scripts/diag_visual_causality.py --output results/metrics/debug_so101_joint/visual_causality_20261013_render` — completed;
  initial sandbox attempt failed to create CoreGraphics context, approved
  outside-sandbox offscreen-render run succeeded.
- `.venv/bin/ruff check scripts/diag_visual_causality.py` — passed.
- `UV_CACHE_DIR=/tmp/real2sim-uv-cache uv run --offline pytest -q tests/unit/test_resample_and_policy_data.py tests/unit/test_policy_guards.py` — 54 passed.

No new rollout success claimed; original E1 remains 0/10. Next action: evaluate
camera/scene alignment at approach frames, with executed-prefix comparisons and
an untouched evaluation scenario set; command-to-motion mismatch remains a
separate unresolved transfer limitation.

## 2026-10-13 — Root-cause diagnostics for the 0/10 approach failure; camera and mat refit

All diagnostics use the frozen joint checkpoint `outputs/act_joint_so101/checkpoints/020000`; none
join the primary tables. Scripts: `scripts/diag_settled_start.py`, `diag_first_frame_gap.py`,
`diag_image_brittleness.py`, `diag_appearance_grid.py`, `diag_feature_heatmap.py`,
`diag_camera_fit_robot.py`, `diag_camera_refit_check.py`, `diag_skeleton_overlay.py`.
Outputs: `results/metrics/debug_so101_joint/{settled_start,first_frame_gap,settled_start_refit_camera}`,
`outputs/tmp_appearance/` (images, grid results, fitted camera JSON).

Findings, in order of elimination:
- Start-pose kick ruled out. Every trial began with a contact-driven jump (shoulder lift -100 -> -57 deg
  in tick 1, settling by tick 4 at elbow 88 deg vs 97 recorded). Running 5 hold ticks before the first
  inference changed nothing: 0/10, min ball distance 13-16 cm, same as E1.
- Leader/follower gap measured in the dataset: follower shoulder lift differs from the leader command by
  14.7 deg mean, 64 deg p95 (18 % of ticks > 30 deg, mostly in the second half of episodes; lag up to
  500 ms does not explain it; e.g. ep1 follower lift frozen at 24 deg for ~27 s). Not the approach cause.
- Per-tick trace: in sim the policy never raises shoulder/elbow and folds the wrist to -80 deg, i.e. the
  recorded end-of-episode rest pose (episode-end wrist mean -80 deg; start +73 deg).
- First-frame test (same state, five images): real first frame -> hold at start (wrist +80 -> +70);
  real frame with the simulator's clamped state -> identical (state offsets irrelevant); sim render with
  or without the ball -> wrist +35 -> -13 (return-to-rest); real END frame -> same. The rendered image
  is the cause; the policy does not register the simulated ball.
- Appearance grid: recolouring robot/desk/mat/bowl, lighting, shadows, speckle textures and scenery
  leave the backbone feature distance unchanged (12.1-13.0 for every variant incl. the legacy yellow
  look; real-to-real baseline 3.2-8.5). Mild perturbations of real frames (brightness, tint, noise,
  8 px shift) keep the hold; blur sigma 1.5 and horizontal flip break it.
- Feature-difference heatmap: the gap sits on the robot/mat region and survives 16x low-pass, i.e. it is
  shape/shading, not colour or texture.
- Camera: the 2026-10-12 PnP assumed fx = 924 px (42.6 deg fovy at 0.6 m). The real frames show almost
  no perspective (mat near/far edge ratio 1.13). Joint fit of camera pose, focal length and mat size to
  13 hand-read joint pixels (4 frames of ep5, MJCF link lengths as scale) plus the mat outline:
  fx ~ 2270 px (fovy 18.0 deg), camera at task (-0.045, -0.647, 0.793) m, pitch 47 deg, mat
  0.304 x 0.279 m (was 0.45 x 0.40, unmeasured). Joint RMS 61 px, mat RMS 24 px. The projected sim
  skeleton now lands on the real arm in all six checked frames, so the provisional joint mapping
  (signs +1, offsets 0) is consistent with the images; it is not the cause.
- After the refit + matched appearance: feature distance 11.5-12.6, policy still folds the wrist at
  tick 0; closed loop 0/10, 13-16 cm (`settled_start_refit_camera`).

Config changes (visual/calibration only, no physics): `configs/task.yaml` camera_iphone (old values kept
in comments) and `table.mat.size_xy`; `configs/sim_so101.yaml` new `appearance` section (materials
recoloured in the parsed scene tree; upstream so101.xml untouched); `src/real2sim/sim/model.py`
appearance/texture support; `tests/integration/test_so101_scene.py` roll bound widened to the refit.
Consequences not yet propagated: `ball.start_region`, `bowl.center_task_xy` and
`results/audit/camera_fit/*` were derived with the old camera and need re-estimation; the first-frame
ball back-projection through the new camera gives (0.35, 0.11) m, close to the old (0.332, 0.101).

Milestone: M4 follow-up — approach-failure root cause
Status: cause isolated (rendered-image domain gap); render matching alone does not close it
Verified commands: diagnostics above; scene/camera tests after the roll-bound update
Unverified assumptions: hand-read joint pixels (+-15 px); mat size and camera from one episode;
  principal point at the image centre; no lens distortion
Blockers: flat-shaded render stays outside the policy's visual distribution
Next: paired real/sim-render training (sim render of the same state substituted for the real frame
  with probability p) or heavy render-aware augmentation; user measurements of mat size and camera
  position to confirm the fit; re-estimate ball/bowl positions through the refit camera

## 2026-10-13 — Operator measurements adopted; camera, mat, table, ball and bowl geometry corrected

Operator measurements (+-3 %): camera 24 in horizontally from the robot's nearest edge and 21 in above
the table; mat 12 x 16 in; table 29 in; ball ~15 in from the table edge. With the camera position fixed
from these, focal length, orientation, camera x and mat placement were fitted to hand-read robot joint
pixels and the mat outline (`scripts/diag_camera_fit_measured.py`): fx 1786 px (fovy 22.8 deg), camera
at task (0.091, -0.704, 0.521) m, pitch 33 deg. The mat must be 12 in across the view (x) and 16 in along
it (y) to reproduce its apparent shape at that pitch. Independent check: the ball's apparent diameter in
the eight start frames implies a camera distance within 4 % of the geometric one. Projected skeleton and
mat outline land on the real frames (`outputs/tmp_appearance/measured_camera_overlay.jpg`); the earlier
wrist/tip residuals were hand-reading errors, not model errors.

`configs/task.yaml`: camera_iphone (operator-measured position + fitted orientation/focal), table.height
0.737, mat 0.305 x 0.406 at (0.235, 0.044), ball.start_region (0.356, 0.205) (v4: 8 start frames
back-projected through the new camera, std 1.0/1.6 cm; 10 cm further in y than v3), bowl centre
(0.318, -0.035) (rim centre back-projected; was (0.330, -0.169)). Old values kept in comments.

Oracle: under the corrected geometry the carry solve stalled (`residual_rejected`, wrist_flex at its
95 deg limit, 8 mm residual) and delivered 1/4. `OracleParams` carry phases are now position-only
(orientation tolerance 1.5 rad, weight 0.001, position tolerance 5 mm): **10/10** again
(`outputs/tmp_appearance/oracle_check4`). The E4 oracle result in the primary tables was measured under
the old geometry and should be regenerated before being cited.

Joint ACT under the corrected scene: first-frame test still returns the return-to-rest chunk (feature
distance 12.1-12.7 vs real-to-real 3.2-8.5); closed loop 0/10, min ball distance 22-26 cm
(`results/metrics/debug_so101_joint/settled_start_measured_camera`; larger than before because the ball
start moved 10 cm further away). Scene/oracle/closed-loop/domain-gap/unit tests: 362 passed.

Milestone: M2 calibration correction + M4 follow-up
Status: geometry now operator-anchored; policy failure unchanged (render domain gap)
Unverified assumptions: hand-read mat corners and joint pixels (+-15 px); principal point at the image
  centre; no lens distortion; mat orientation inferred from the image, operator quoted 12 x 16 as height x width
Next: regenerate E0/E4 baselines and the camera-fit audit under the new geometry; paired real/sim-render
  training for the policy


## 2026-10-13 — Geometry snapshot prepared for GitHub

User requested committing the corrected geometry and related diagnostics separately
from paired real/sim training, with commit message `Sim geometry changes`.
Selected files: task/sim configuration, scene-builder appearance support, oracle
carry tolerances, scene integration test and `scripts/diag_*.py`; diagnostic
reports/traces and operator-anchored camera-fit evidence accompany the change.
Unrelated ACT augmentation/loss/monitor work remains uncommitted in the workspace.

The measured table/mat dimensions and camera-position constraints anchor this
configuration; focal length, orientation, mat placement, ball and bowl positions
remain fitted estimates. The oracle retains a weak orientation penalty (it is
not strictly position-only). Fixed contradictory camera/bowl provenance comments
without changing numeric settings. Diagnostic scripts were formatted/linted;
no camera keys, action labels, frequency, split or policy weights changed.

Existing primary hold/oracle/learned tables and legacy camera-fit audit files
predate this correction. They are preserved as historical results and must be
regenerated under this snapshot before being cited as current. The visual input
is implicated in failed initial commands, but exclusive causality of all 0/10
outcomes is unproven; command/dynamics mismatch remains unresolved. No paired
training or primary benchmark rerun was performed as part of the commit request.

Verification: `UV_CACHE_DIR=/tmp/real2sim-uv-cache uv run --offline ruff check .`
passed; `uv run --offline pytest -q tests/unit` 301 passed and
`uv run --offline pytest -q tests/integration` 109 passed (same cache override).
Both suites used approved macOS rendering access; the initial sandboxed unit
run had 300 passes and one CoreGraphics-access failure, then passed on retry.
`uv run --offline python -m real2sim.sim.camera_fit --help` passed.
These counts describe the current workspace, including pending training tests.
Next: regenerate baselines and camera audit under the corrected geometry, then
assess the proposed approach-phase paired-image training as a separate change.


## 2026-10-13 — D-020 baselines and paired-image pilot preparation

Corrected geometry `3988577`: hold 0/10, original joint ACT 0/10, Cartesian ACT 0/10,
privileged oracle 10/10; 40 trials, zero aborts. New isolated artifacts under
`results/metrics/geometry_3988577/`, with config snapshots and hashes; README/protocol
now distinguish these from historical tables. Frozen weights unchanged.
Fixed-camera audit refreshed (no refit): start RMS 19.26 px, bowl 66.73, held 138.09;
held-ball=tool-site and old detections are approximate. Audit under
`results/audit/paired_approach_3988577/`, previous camera audit preserved.

Rendered 3,299 train / 1,134 validation approach pairs, all 28/9 episodes; 2,442
training frames have >5 degrees of arm displacement from their episode start.
Reviewed first/middle/final contact-sheet samples across episodes 0/5/6/7,
36/37/38/39, and 1/2/3/4. Visible residual pose/appearance differences remain;
accepted only as approximate inputs for a bounded pilot. No transport ball poses
invented, no new labels, no split or normalization change. Cache manifests/hashes
and review record are saved. An initial JSON serialization bug was fixed; failed
cache/audit directories retained with `_failed_metadata` suffix.

Verified: Ruff; 9 new unit tests for stationary-prefix exclusion, unchanged labels,
train/validation leak guards, deterministic substitution, checksum rejection and
single notification dispatch; one real-data two-step MPS warm-start/save/reload test
passed (8.97 s), original checkpoint SHA unchanged. Full pilot is a separate run,
not the smoke result. Completion queue will review real validation, paired-image
prefix errors and ten closed-loop trials; no tokens spent waiting for training.

Pilot launched 2026-10-13 11:23:35 UTC, manager PID 35863. Startup log confirms
`device=mps`, 51.6M parameters, 10,692 train / 4,064 validation ticks. The isolated
1,000-step run mixes paired images only on eligible training ticks, then runs real
validation, paired prefix checks and ten closed-loop trials. Status and logs:
`outputs/act_paired_approach/`. One completion/failure callback; no interval polling.
Additional regression verification: 70 existing data/policy/loss/smoke tests passed
(80 relevant tests total with the new unit and MPS smoke tests). Parent held-out
approach-prefix preflight completed: normalized MAE real 0.1320, sim 0.2066,
real/sim prediction gap 0.1647, before fine-tuning. These are sampled prefix
metrics, not full validation or task success.


## 2026-10-13 — D-020 pilot completed and reviewed

All four phases completed without pipeline failure; manager exited, completion
callback delivered successfully. MPS training took 394.3 s; 1,000 steps completed,
checkpoint 500 selected by real validation (SHA256
`8aef203053954598c1b81b7651adfa39686fb370247b6d6bb2f8abb0b0862409`).
Full real validation L1 0.31161 parent → 0.31322 pilot (slightly worse).
144 held-out approach ticks, executed two-action prefixes: real MAE
0.13204 → 0.12992; sim MAE 0.20661 → 0.13821 (33.1% lower); real/sim prediction
gap 0.16469 → 0.09612 (41.6% lower). Same state/labels for image comparisons.

Ten unchanged primary seeds: still 0/10 success, no lifts, approach failure ×10,
zero aborts. Mean closest distance 24.15 → 14.00 cm, median 24.12 → 13.12 cm.
Nine seeds improved; seed 8 worsened 24.20 → 26.39 cm. Pilot range 7.65–26.39 cm.
Thus visual adaptation helps approach but has not solved execution. Do not call
this purely real-trained: this checkpoint used mixed real/sim approach images.
Results and compact comparison: `results/metrics/paired_approach_3988577/review.json`.

Performed the approved recorded-command versus follower-motion diagnostic via
`scripts/diag_paired_command_tracking.py` (Ruff passed, complete headless replay).
Replayed all 1,134 paired held-out approach ticks, nine episodes, twice, using
fixed geometry/dynamics and image-estimated episode ball positions. Compared
post-step simulator state to recorded next tick; same-tick real gaps also saved.
Leader-command replay had model-based tool trajectory error mean 3.06 cm,
p95 8.23 cm. Privileged follower-next-state setpoints reduced this to mean 1.03 cm,
p95 3.80 cm. Moving-prefix elbow MAE decreased 6.62 → 0.55 degrees; shoulder
2.58 → 1.24 degrees. Leader shoulder targets were limit-clipped on 40.9% of ticks;
follower-next targets also clipped on 41.4%. Rate limits affected 5.9% of leader
shoulder targets. Traces/report: `results/audit/paired_command_tracking_3988577/`.
These are SCRIPTED REPLAY diagnostics, no policy inference or task-success claims;
FK uses provisional mapping, not measured physical tool poses. Reset transients
and contact also contribute. This does not prove hardware malfunction or isolate
one exclusive cause of the policy failures.

No training labels, split, normalization, control/geometry settings, benchmark
criteria, physical hardware or original weights changed. No larger training run
launched. Next justified work: audit command timing and joint mapping/limits plus
remaining camera correspondence against existing recordings before choosing a
control/label adjustment; larger training is not yet supported by this pilot.


## 2026-10-13 — Joint/command audit and range sensitivity completed

User approved the audit/replay/evaluation priority sequence. All work was offline
or simulated; no physical robot connection. Original configs, labels, split,
normalization and both parent/paired checkpoint hashes remain unchanged.

Raw audit: 32,050 training and 12,186 held-out rows (37 included episodes).
Exact zero-error raw/derived state and action parity rules out a subsampling row
shift. Encoder-lattice residuals support degrees for arm and percent for gripper;
they do not verify physical zero/sign alignment. All included observed states
fall inside local follower calibration spans (0.1-unit comparison tolerance).
Shoulder is outside native model bounds on 23.88% train / 22.00% validation rows;
wrist flex on 8.94% train / 0% validation rows. No limits changed in primary configs.

Local recording source (checkout 6156e82ad7407fa70be740e42fdc7b8c5bc26d02) clips
follower targets relative to measured position but discards returned `_sent_action`
when recording, saving `action_values`. Exact recording-time version is unknown.
This supports recorded-leader labels and warns against assuming motor execution.
Train shoulder action/state gap p95 60.62 degrees; validation 70.15 degrees.
An extended moving-frame lag audit over -15..30 frames gives train best lags
5/30/8/9/4/14 in joint order; shoulder remains 18.33-degree train / 23.87-degree
validation MAE and its minimum is at the boundary. No global lag correction or
constant joint offset is justified. A stalled follower, servo response, clipping,
load and calibration differences cannot be separated solely by these signals.

Tested one isolated change: shoulder lower joint and actuator bound -100 →
-105.5824 degrees, the local follower calibration endpoint; reset admits the same
requested recorded start. No gains/contact/other joint bounds modified. Verified
XML difference consists of exactly those two attributes. Replayed the same 1,134
held-out pre-contact ticks with recorded leader versus follower-next setpoints:
- Leader replay shoulder MAE 3.61 → 1.71 degrees, but model-based tool discrepancy
  mean 3.06 → 3.08 cm (no improvement), p95 unchanged at 8.23 cm.
- Follower-next replay shoulder MAE 2.60 → 0.63 degrees, tool mean 1.03 → 0.87 cm,
  p95 3.80 → 2.51 cm. Privileged tracking control, not learned success.

Ten frozen paired-policy trials in the isolated range scene: 0/10, approach ×10,
no lifts/aborts. Mean closest distance 14.00 → 16.14 cm; only seeds 2,8,9 improve.
The range change is NOT adopted. This comparison also includes the reset-pose
change induced by unclamping; it does not isolate lower-stop contact alone.

Artifacts: `results/audit/control_contract_20261013/` (audit, calibration-range
scene, normalization, extended timing, provenance, replay traces, review) and
`results/metrics/control_contract_20261013/paired_shoulder_range/` (ten trials).
Reproduction: `scripts/diag_control_contract.py`; invoke
`diag_paired_command_tracking` with its CONFIG/OUT set to the isolated config and
a fresh replay directory; `scripts/diag_recorded_timing.py --out <fresh.json>`;
standard `real2sim.evaluation.run` with isolated config, frozen checkpoint 000500,
seeds 0–9 and diagnostic experiment ID. Ruff passed; actual replay and rollout
completed; original checkpoint SHA assertions passed. No new ACT training.

Next justified experiment: identify a bounded leader-to-follower execution response
using training recordings only and evaluate held-out recorded-command replay
before adopting a different simulator adapter or changing training semantics.
A larger ACT run or blind joint-offset correction is not supported by this audit.


## 2026-10-13 — Bounded follower-response identification and replay

User approved fitting a follower response on training recordings and testing on
held-out replay before any ACT integration. Implemented
`src/real2sim/sim/follower_response.py` and `scripts/fit_follower_response.py`.
At nominal 30 Hz, each joint has bounded target-error gain, momentum and causal
integer delay. Target error and movement are capped at 5 dataset units; position
bounds use local calibration normalization endpoints, not asserted physical MJCF
limits. There are no image/object/contact/force inputs and no ACT weight updates.

Selection uses only the existing 28 training episodes: 21 for coefficient fits,
seven (every fourth in the fixed train list) for free-running episode-balanced
candidate selection. Candidates: gain-only or gain+momentum, delays 0/2/4/6 raw
frames. Refit selected structures on all 28, write/hash model, then load the nine
held-out episodes. Replay gets only the initial observed state and causal recorded
commands; it never feeds later measured follower states back into predictions.
Evaluated full episodes plus the same 1,134 stationary-ball approach ticks. Two
command clocks: all raw 30 Hz targets, or 10 Hz targets held for three raw ticks.
MuJoCo then follows the predicted held10 positions using unchanged physics/limits.
This is a scripted learned-response trajectory diagnostic, not an ACT rollout or
an online adapter. FK errors are model-based, not physical pose measurements.

First, full-episode fitting failed the approach check: tool discrepancy grew from
3.06 cm baseline to 5.54 cm after MuJoCo replay (p95 16.98 cm). It averages regimes
with substantially different shoulder response. Retained failed experiment at
`results/audit/follower_response_20261013/`; no production adoption.

Second, an adaptive follow-up restricted fitting and internal selection to cached
TRAINING stationary-ball approach prefixes. Held-out episodes never selected
coefficients/delays. The same nine diagnostic episodes were reused, so this is
exploratory evidence, not a fresh confirmatory test. Results:

| Replay | Mean model-based tool error | p95 |
|---|---:|---:|
| Original MuJoCo, recorded leader targets | 3.06 cm | 8.23 cm |
| Full-episode response through MuJoCo | 5.54 cm | 16.98 cm |
| Approach response, raw30 commands, numerical only | 1.13 cm | 4.34 cm |
| Approach response, held10 commands, numerical only | 1.19 cm | 4.48 cm |
| Approach response through unchanged MuJoCo | 2.20 cm | 5.80 cm |

Approach response improves MuJoCo replay mean error 28.2%, in eight of nine
episodes. Episode 24 worsens 1.97 → 2.90 cm. All failures remain in pooled results.
Raw30 versus held10 difference is small here; temporal subsampling alone does not
explain the baseline discrepancy. The extra error after MuJoCo indicates remaining
limits/contact/actuator tracking effects; the model does not isolate their causes.
The approach response has no validated transport/grasp behavior; full held-out
rollouts still show shoulder MAE 17.55 degrees and elbow 9.54 degrees with held10
commands. Do not apply it to the whole task without phase-specific validation.

Artifacts: `results/audit/follower_response_approach_20261013/` contains frozen
model, selection scores, per-episode trajectories, comparison.csv, report,
verification and review. Commands: `.venv/bin/python scripts/fit_follower_response.py`
and the same with `--fit-scope approach --out <fresh-output-directory>`.
Verification: 11 unit tests passed (known-dynamics recovery, episode isolation,
causality/delay, bounds, serialization and invalid inputs); Ruff passed. Both
frozen model hashes matched after held-out evaluation. Recomputing all nine saved
trajectories for both experiments was bitwise equal; every trace finite and each
predicted step within bound. Training-only prefix provenance and checksums passed.
No primary configs, hardware settings, datasets, labels or ACT checkpoints changed.
No ACT inference or policy training was launched in this experiment.

Next justified test: a separate frozen-ACT approach diagnostic with a causal online
response adapter and an explicit nonprivileged horizon/phase rule. Do not use
simulator ball pose to gate it, and do not claim new task success from replay.


## 2026-10-14 — Frozen ACT with online approach-response adapter reviewed

D-023 completed: same frozen paired checkpoint 000500, original geometry, seeds
0–9, 300 ticks/trial and success rules. Adapter uses measured joint feedback,
causal executed-action history and the frozen approach-response model for exactly
78 ticks (7.8 s), then ordinary targets. Horizon was fixed from training-prefix
coverage before evaluation. No object poses, contacts or privileged phase labels
enter control. Simulator ground truth is used only for metrics and post-hoc audit.

Result: 0/10 success, no held/lifted ball, no aborts. Mean closest tool–ball distance
14.00 → 11.20 cm (20.0% reduction); median 13.12 → 7.36 cm. Seven seeds improve;
6,7,8 worsen. Five trials reach 4.67–6.87 cm. Seed 5's ball escapes the table.
All primary phase failures remain approach under the unchanged 3 cm definition.
This is diagnostic approach progress, not demonstrated grasp or task success.

Saved-command replay reproduced all ten state/ball trajectories within assertions.
At closest points in seeds 0,1,2,3,4,5,9, target gripper is 48–67 percent and the
tool misses predominantly along negative task y. These minima occur at ticks
82–91, shortly after switching off the adapter at tick 78. The handoff and
terminal alignment are unresolved; do not attribute the outcome solely to model
quality. One gripper/ball contact was sampled at seed 1 tick 90 without holding.
Contacts are sampled at 10 Hz and can miss between-tick collisions; no sampled
contact for escaped seed 5 is not evidence of absence of contact.

Artifacts: `results/metrics/act_response_20261013/` contains fixed protocol,
implementation hashes, all trial rows/history, response traces, contact audit and
review.json. Code: `evaluation/response_adapter.py`, `scripts/eval_act_response.py`,
`scripts/inspect_response_contacts.py`. The normal evaluation entry point and
production simulator are unchanged. Response/ACT model hashes stayed frozen.
Verification: 17 response-model/adapter tests passed, including causal delay,
measured-state feedback, reset isolation and exact ordinary-command fallback;
Ruff passed. Trace audit verified exactly 78 adapted ticks per trial and exact
ordinary targets afterward. Post-hoc replay is SCRIPTED, not a second ACT trial.

No ACT retraining launched: results currently justify diagnosing the fixed-window
handoff and terminal alignment, not an unbounded longer training run. Nothing is
training in the background. User requested completion/failure-only notification
if later justified training is launched; retain that preference.


## 2026-10-14 — Publication review and complete test run

User requested GitHub push, broad verification and retraining only if warranted,
with no active waiting during training. Ran the complete unit/integration suite:
**437 passed in 50.12 s**, including MPS training smoke tests, MuJoCo rendering,
SO-101/Panda evaluation, replay, data guards and the new response diagnostics.
Review found that a malformed paired manifest could omit a required checksum;
now both ticks.npy and images.npy checksums are mandatory. Two new regression
cases plus existing paired tests and real-data warm-start smoke: **12 passed in
7.04 s** (439 distinct tests verified across the full and targeted runs). Valid
experiment manifests already contained both hashes, so existing training weights
and measured results are not invalidated. Ruff and git diff whitespace checks
passed; dataset-audit CLI help passed. Test-generated changes to two historical
Panda reports were restored, preserving the original evidence.

Changed the diagnostic asset symlink to a checkout-relative target; added the
latest online-response result and artifact scope to README. Training caches,
weights and failed-preparation duplicate previews remain local/ignored. Tracked
reports retain historical provenance and local absolute paths; they are evidence,
not a substitute for regenerating local caches on another machine.

Fetched GitHub and preserved upstream commit d46384e deleting plan.md. No new
policy training is justified by these implementation checks: the current failure
requires isolating response-adapter handoff and terminal alignment first. Nothing
is training in the background. Publication includes the augmentation/loss, paired
image, response identification and online-adapter code/tests plus measured results.
