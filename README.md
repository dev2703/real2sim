# Real2Sim-RaC: real SO-101 demonstrations → ACT → MuJoCo evaluation

## What this is

Forty teleoperated pick-and-place demonstrations were recorded on a real SO-101 arm (tennis
ball into a bowl) and published at [Hugging Face](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v2). 

The main aim is to train an ACT policy with a recovery MLP head in Mujoco and use the frozen ACT weights for cross embodiment and use trained policy to work on 7DOF panda arm in MujOco


## Data collection

Teleoperation with an SO-101 leader/follower pair through LeRobot (dataset format v3.0,
25 Hz, follower PID P=16 I=0 D=4). Two cameras were recorded; observation.images.top (wrist camera), observation.images.iphone (global top view camera) - both recorded at 1280 * 720, couldn't get iphone to work with 640 * 480 but could be resized easily with pytorch tensor ops

Note every measurement is in inches first number is x axis(length wise) second number y axis (width wise) third z axis (height wise), if single number thats x axis
camera from ball: (30.5, 4.5, 16)
ball & bowl: 12 inches
Ball from robot base: (2,13)
mat: (16,12)
Bowl: 4.5 diameter
Ball: 8 circumference 

Note: there could be +- 5% measurments error.

Run the assisted system with the pinned HF `latest/` checkpoint:

```sh
.venv/bin/python scripts/evaluate_hf_recovery.py --robot so101 --seeds 500 501 502 503 504 \
  --out outputs/v2/hf_recovery_so101_NEXT --video
.venv/bin/python scripts/evaluate_hf_recovery.py --robot panda --seeds 500 501 502 503 504 \
  --out outputs/v2/hf_recovery_panda_NEXT --video
```

Evaluate the unchanged HF policy alone in the measured scene:

```sh
.venv/bin/python scripts/evaluate_hf_act.py --robot so101 --out outputs/v2/hf_only_so101_NEXT
.venv/bin/python scripts/evaluate_hf_act.py --robot panda --out outputs/v2/hf_only_panda_NEXT
```

Run the separate geometric baseline from the configured start:

```sh
.venv/bin/python scripts/run_visual_waypoint.py --robot so101 --seeds 0 1 2 3 4 \
  --config configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml \
  --out outputs/v2/waypoint_so101_NEXT --video
.venv/bin/python scripts/run_visual_waypoint.py --robot panda --seeds 0 1 2 3 4 \
  --config configs/v2/measured_registration_20261028/sim_panda_positive_all_scalars.yaml \
  --out outputs/v2/waypoint_panda_NEXT --video
```

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

# 5. simulation evaluation (see docs/experiment_protocol.md for the full command set)
uv run python -m real2sim.evaluation.run --robot so101 --checkpoint outputs/act_cartesian/checkpoints/006000 --n-trials 10 --out results/metrics/E2_cart_so101_v2 --video
uv run python -m real2sim.evaluation.oracle --n-trials 10 --out results/metrics/oracle_so101
uv run python scripts/aggregate_trials.py
```

## Models and physics assumptions

- ACT (LeRobot 0.6.1): ResNet-18, `dim_model` 256, chunk 20 at 10 Hz, 2 actions executed per inference or chunking 2s of actions before recalculating, batch 16. Joint baseline learns the recorded leader targets; the Cartesian policy
  learns achieved end-effector motion from forward kinematics.
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`. Ball and bowl are compound primitives.
- The trained model has been published [Hugging Face](https://huggingface.co/bottowork/so101-act-h256-b16)

## Experiments
- Replaced ACT VAE with flow matching head
- Replaced ACT resnet18 backbone with DinoV2 or Clip (fine tuned ViT on our dataset)
- Used end effector delta as action state
- Added a separate Cartesian recovery MLP (hidden widths 256→128, seven action outputs), trained with DAgger corrective labels around frozen ACT.
- On going experiments continued... 
