# Real2Sim-RaC: real SO-101 demonstrations → ACT → MuJoCo evaluation

Status board: `docs/implementation_plan.md`. Progress log with every verified command:
`docs/progress.md`. Decisions and their evidence: `docs/decisions.md`.

## What this is

Forty teleoperated pick-and-place demonstrations were recorded on a real SO-101 arm (tennis
ball into a bowl) and published as
[bottowork/so101_act_pickplace_v1](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v1),
pinned at revision `fd65a6600b0cd111a434ece1abfa3d44dc7b4d57`. This repository audits that
data, trains Action Chunking Transformer (ACT) policies on it, and evaluates them closed-loop in
a MuJoCo simulation of the same task. Nothing here claims sim-to-real or real-to-sim success;
every number is labelled with where it came from.

## Verified so far (2026-10-13)

| Item | Evidence |
|---|---|
| Dataset audit, every frame of both cameras | `results/audit/camera_quality.json`, `camera_montage.html` |
| Camera selection: `top` unusable (points at ceiling), `iphone` usable in 37/40 episodes | `docs/data_contract.md` |
| Episode split 28 train / 9 validation, 3 excluded for dropped frames | `results/audit/split.json` |
| Joint-space ACT trained on real data (gate G1) | `results/metrics/act_joint_g1_check_step2000.json` |
| Canonical 7-D Cartesian labels and 13-float proprioception | `src/real2sim/data/labels.py`, `tests/unit/test_labels.py` |
| SO-101 forward kinematics consistent across two implementations | `tests/integration/test_fk_consistency.py` |

Not yet verified: closed-loop task success in simulation, Panda transfer, recovery. See
`docs/limitations.md`.

## Data collection

Teleoperation with an SO-101 leader/follower pair through LeRobot (dataset format v3.0,
30 Hz, follower PID P=12 I=0 D=2). Two cameras were recorded; only the iPhone side view is
usable. Joint values are degrees, gripper 0–100 %. Full observed schema and the action-versus-
state audit (follower lag 4–16 frames, 13° shoulder offset) are in `docs/data_contract.md`.

## Install

```
uv sync --locked            # Python 3.12, LeRobot 0.6.1, MuJoCo 3.15, torch 2.11
uv run ruff check .
uv run pytest -q tests/unit
```

## Commands

```
# 1. audit the dataset (uses a local copy if given; downloads otherwise)
uv run python -m real2sim.data.audit --repo-id bottowork/so101_act_pickplace_v1 \
  --revision fd65a6600b0cd111a434ece1abfa3d44dc7b4d57 [--local-root <dir>] \
  --out results/audit --n-val 10 --exclude-episodes 25,33,34

# 2. build the 10 Hz derived datasets (train-only statistics, hashed manifest)
uv run python -m real2sim.data.prepare --mode joint     --config configs/data.yaml --local-root <dir> --out-dir .cache/derived_joint
uv run python -m real2sim.data.prepare --mode cartesian --config configs/data.yaml --local-root <dir> --out-dir .cache/derived_cartesian

# 3. train (smoke test first)
uv run python -m real2sim.policy.train --config configs/act_joint.yaml --out outputs/smoke --smoke-steps 2
uv run python -m real2sim.policy.train --config configs/act_joint.yaml --out outputs/act_joint_so101
uv run python -m real2sim.policy.train --config configs/act_cartesian.yaml --out outputs/act_cartesian

# 4. check a checkpoint against held-out episodes (open-loop, not task success)
uv run python -m real2sim.policy.check --checkpoint outputs/act_joint_so101/checkpoints/last --data .cache/derived_joint

# 5. simulation evaluation (being built; see docs/experiment_protocol.md)
uv run python -m real2sim.evaluation.run --help
```

## Models and physics assumptions

- ACT (LeRobot 0.6.1): ResNet-18, `dim_model` 512, chunk 20 at 10 Hz, 2 actions executed per
  inference, batch 16. Joint baseline learns the recorded leader targets; the Cartesian policy
  learns achieved end-effector motion from forward kinematics (decision D-003).
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`, pinned at commit
  `0059d433…`, actuators unmodified. Ball and bowl are compound primitives. Physical values are
  provisional standards, not measurements; see `docs/calibration.md`.

## Evaluation

Not yet run. The protocol, metrics and captions are fixed in `docs/experiment_protocol.md`.
Results will appear in `results/metrics/<experiment>/trials.csv` with one row per attempt,
failures included.

## Failures and limitations

`docs/limitations.md`. Highlights: one usable camera; three episodes dropped; provisional
calibration; the supplied bowl depth is shallower than the ball radius; GPU training blocked
until a Modal payment method exists, so training runs on Apple MPS.

## Layout

`src/real2sim/{data,geometry,policy,sim,evaluation,recovery}`, `configs/`, `tests/`,
`docs/`, `results/` (small artifacts only), `assets/menagerie` (pinned, licensed).

## Licence and sources

Code Apache-2.0. Dataset apache-2.0 (card). Menagerie models under their own licences, see
`assets/LICENSES.md`. LeRobot, MuJoCo and the original ACT paper (Zhao et al. 2023) are the
upstream sources listed in `plan.md` §11.
