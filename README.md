# Real2Sim-RaC: real SO-101 demonstrations → ACT → MuJoCo evaluation

## What this is

Forty teleoperated pick-and-place demonstrations were recorded on a real SO-101 arm (tennis
ball into a bowl) and published at [Hugging Face](https://huggingface.co/datasets/bottowork/so101_act_pickplace_v1). 

The main aim is to train an ACT policy with a recovery MLP head in Mujoco and use the frozen ACT weights for cross embodiment and use trained policy to work on 7DOF panda arm in MujOco


## Data collection

Teleoperation with an SO-101 leader/follower pair through LeRobot (dataset format v3.0,
30 Hz, follower PID P=12 I=0 D=2). Two cameras were recorded; only the iPhone side view is
usable. 

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
  learns achieved end-effector motion from forward kinematics.
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`. Ball and bowl are compound primitives. Physical values are
  provisional standards not precise measurement 


