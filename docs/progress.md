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
