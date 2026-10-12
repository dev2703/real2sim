# Implementation plan (execution view of plan.md)

**Version:** 2026-10-12 evening, after the M0 read-only inspection.
**Deadline:** 16 Oct 2026 23:59 BST (17 Oct 09:59 AEDT). Internal target: 16 Oct 18:00 BST.
**Authority:** `plan.md` is the specification; this file is the work breakdown and status board.
It changes only when work is verified or a decision in `docs/decisions.md` is approved.

## 0. Current status board

| Milestone | Gate | Status | Blocking decisions |
|---|---|---|---|
| M0 scaffold + audit | G0 | **PASS**: env, package, audit CLI, full audit artifacts in `results/audit/`, final split 28/9 | none |
| M0.5 docs | G0.5 | **PASS**: all docs exist; architecture and system design approved | none |
| M1 joint ACT baseline | G1 | **G1 PASS** on step-2000 checkpoint (open-loop val L1 0.337); 20k-step MPS run continuing | none |
| M2 SO-101 MuJoCo | G2 | **IN PROGRESS by another writer** in this working tree (`src/real2sim/sim/`, `assets/`, sim tests; 4 sim tests failing at 22:50) | ownership/worktree clarification |
| M3 Cartesian dataset | G3 | **PASS**: 7-D labels and 13-float proprio built, 2-step train/infer round trip OK; full run queued on MPS | joint-map verification at M2 |
| M4 Cartesian ACT + closed loop | G4 | training queued on MPS; closed loop needs the M2 env | Modal needs a payment method (fallback MPS) |
| M5 Panda | G5 | not started | G4 |
| M6 recovery | G6 | not started | D-005, G4 |
| M7 packaging | G7 | not started | all |

Skills under `.agents/skills/`: G1 reached; still **DEFERRED** in favour of M2/M4 (plan §10 allows skipping). Not required for P0.

## 1. Evidence-driven changes to the original schedule

1. **No camera decision is needed.** The audit rule resolves to a single stream (`iphone`).
   Multi-camera work is removed from the plan entirely. The `top` stream is documented as a data
   issue only.
2. **No download step.** The local copy under `~/.cache/huggingface/lerobot/local/` is
   byte-identical to the pinned revision; the audit CLI gets `--local-root` from day one and
   records the pinned SHA from `meta/info.json` plus file hashes.
3. **Label semantics are a real decision**, not a formality: the 13° shoulder offset and 28°
   elbow overshoot in `action` mean raw leader targets differ materially from achieved motion.
4. **Environment must be rebuilt** (empty venv, broken conda editable install).
5. **Three episodes carry an iphone frame deficit** and need a measured alignment check before
   they can be used.

## 2. Work breakdown

### M0 — scaffold + audit (target: finish 13 Oct morning)

| # | Task | Output | Verification |
|---|---|---|---|
| 0.1 | Fix `.gitignore`; commit docs | tracked `plan.md`, `docs/` | `git status` |
| 0.2 | `pyproject.toml` (Python 3.12, lerobot 0.6.1 `[dataset]`, mujoco 3.15, placo <0.9.16, pyav, opencv-headless, pyyaml, pandas, pyarrow, ruff, pytest), `uv.lock`, `src/real2sim/` skeleton, `AGENTS.md` | importable package | `uv sync --locked`, `uv run ruff check .`, `uv run pytest -q tests/unit` |
| 0.3 | `real2sim.data.raw`: thin loader over LeRobot v3.0 (meta, episodes table, parquet, video decode by timestamp) supporting `--local-root` and `--repo-id` + revision pin | `RawDataset` class | `tests/unit/test_raw_dataset.py` on a 2-episode fixture |
| 0.4 | `real2sim.data.audit` CLI: dataset_report.json, camera_quality.json (per key: decode rate, black/blank/frozen/duplicate counts, sharpness stats, timestamp coverage, frame-count deficit per episode, label USABLE/PARTIAL/UNUSABLE), camera_montage.html (first/middle/last per episode per camera, plus worst frames), action/state audit, split manifest with hash | `results/audit/*` | `uv run python -m real2sim.data.audit --local-root ... --out results/audit`; `tests/unit/test_camera_quality.py` with synthetic black/frozen videos |
| 0.5 | `annotations/episode_outcomes.csv` template (all `unknown`) + montage for operator review | file | operator fills it |
| 0.6 | `configs/data.yaml` with `camera_keys`, image transform, split seed, label source placeholders | config | `tests/unit/test_images.py` |
| 0.7 | GitHub Actions: ruff + unit tests on CPU (cheap, optional) | workflow | green run |

**G0 evidence:** audit JSONs, montage, split manifest, loader test, `docs/data_contract.md` updated
from the full audit, `docs/progress.md` entry.

### M1 — joint-space ACT baseline (13 Oct)

| # | Task | Verification |
|---|---|---|
| 1.1 | `real2sim.data.prepare --mode joint`: 10 Hz resample by timestamp, episode-preserving, letterbox transform to `[3,192,256]`, train-only normalisation stats, derived LeRobot-format dataset in `.cache/derived_joint` | `tests/unit/test_images.py`, `tests/unit/test_raw_dataset.py` (no leakage) |
| 1.2 | `real2sim.policy.train` wrapping LeRobot `ACTPolicy`/`ACTConfig` with resolved YAML, `--smoke-steps`, checkpoint save/resume, CSV history | `tests/integration/test_act_smoke.py`: real batch fwd/bwd, finite loss, save/reload, inference shape |
| 1.3 | Measure MPS step time and memory at batch 16 (and 8 with 2× accumulation) for dim 512; record in `docs/progress.md` | timing table |
| 1.4 | Launch `act_joint_so101` training (20 k steps or budget), log validation L1 per 1 k | `results/metrics/training_history.csv`, `results/configs/act_joint_resolved.yaml` |

**G1 evidence:** loadable checkpoint producing `[B, chunk, 6]` actions from iphone + state.

### M2 — SO-101 MuJoCo scene (13–14 Oct, parallel with M1; sim owner's worktree)

| # | Task | Verification |
|---|---|---|
| 2.1 | Vendor Menagerie `robotstudio_so101` + `franka_emika_panda` at commit `0059d43…` with licenses into `assets/menagerie/`; `assets/LICENSES.md` | file hashes |
| 2.2 | `scene_so101_pickplace.xml`: table, mat, ball (sphere, provisional 3.35 cm radius, 58 g), bowl (compound collision: ring of boxes/capsules + base disc, visual mesh optional), iphone-like camera, lights | `tests/integration/test_so101_scene.py` headless load |
| 2.3 | `So101PickPlaceEnv.reset(seed, scenario)` / `step(action)` with Gym-style dict obs, physics 500 Hz, control interpolation, render only `observation.images.iphone` for the policy | determinism test, image shape test |
| 2.4 | Contact tests: ball rests, rolls, settles in bowl, cannot tunnel; friction/restitution sweep | integration test |
| 2.5 | Joint convention calibration: real state ↔ MJCF sign/zero/units; FK consistency test (MuJoCo site vs Placo URDF) | `tests/integration/test_kinematics.py` |
| 2.6 | `REPLAY` of one validation episode by joint setpoints and by FK→IK; tracking-error plot; labelled video | `results/media/replay_*.mp4` |

### M3 — canonical Cartesian dataset (14 Oct)

| # | Task | Verification |
|---|---|---|
| 3.1 | `real2sim.geometry`: SO(3) log/exp, SE(3), rot6d, frame transforms | `tests/unit/test_so3.py` |
| 3.2 | SO-101 FK (Placo URDF + MuJoCo cross-check) | kinematics test |
| 3.3 | 7-D labels per D-003 (achieved motion, one-tick lead), gripper semantic mapping with verified sign, clipping diagnostics, 20-step within-episode chunk masks | `tests/unit/test_labels.py` |
| 3.4 | `real2sim.data.prepare --mode cartesian` → `.cache/derived_cartesian` + train-only stats + manifest | 2-step train/infer round trip |
| 3.5 | Validation plots for 10 random frames (FK arrows on iphone frames) | `results/audit/label_check/` |

### M4 — Cartesian ACT + closed loop (14–15 Oct)

| # | Task | Verification |
|---|---|---|
| 4.1 | Train `act_cartesian` on derived real data (MPS or rented GPU per D-009), freeze best validation checkpoint | history CSV, resolved config |
| 4.2 | Inference adapter: sim iphone render → identical letterbox → policy; sim observable EEF state → identical 13-float; no GT object pose | `tests/unit/test_oracle_schema.py` |
| 4.3 | Cartesian controller: delta pose → clamped target → SO-101 IK → rate-limited joint targets; logs commanded/achieved pose, IK residual, infeasible rejections | kinematics test |
| 4.4 | 3 debug rollouts, then ≥10 seeded trials, `results/metrics/trials.csv`, labelled videos | `tests/integration/test_closed_loop.py` proves actions come from the policy |

### M5 — Panda (15 Oct), M6 — recovery (15–16 Oct), M7 — packaging (16 Oct)

As specified in `plan.md` §4; each starts only when its gate dependency exists. M6 begins with
the deterministic detector and FSM (`tests/unit/test_recovery_fsm.py`), then on-policy correction
collection in MuJoCo with a scripted oracle labelled as such, then the MLP per D-005. If time is
short, M6 ships FSM + deterministic comparator and marks the learned specialist not completed.

## 3. Parallelisation and ownership (from M0)

- Integrator worktree (this session): data, policy, evaluation, docs, merges.
- Simulation worktree: `src/real2sim/sim/**`, `assets/**`, sim integration tests. Starts after
  D-002 is approved and the package skeleton exists, against the frozen observation contract
  (`camera_keys`, image tensor, proprioception vector, 7-D action) in `docs/data_contract.md`.
- Operator: hardware, measurements (`docs/calibration.md`), outcome labels, approvals.
- No simultaneous edits to `docs/data_contract.md`, `configs/data.yaml`, `pyproject.toml`.

## 4. Daily cadence

Each milestone ends with the §9 report block in chat and a `docs/progress.md` entry listing exact
commands and artifacts. Failed experiments are reported as failures. Scripted or replayed motion
is always captioned `SCRIPTED REPLAY` and never presented as learned behaviour.
