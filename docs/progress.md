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
