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

Latest SO-101 baselines under geometry commit `3988577` (13 Oct): hold **0/10**,
frozen joint ACT **0/10**, frozen Cartesian ACT **0/10**, privileged scripted oracle
**10/10**. All 40 trials completed without aborts. See
[current results](results/metrics/geometry_3988577/summary.json) and
[protocol/history](docs/experiment_protocol.md). Paired real/sim approach-image
fine-tuning is a separate mixed-image experiment: still **0/10**, but mean closest
distance improved from **24.2 to 14.0 cm** (9/10 seeds improved). See the
[pilot review](results/metrics/paired_approach_3988577/review.json); no larger run is queued.

The subsequent frozen-policy response-adapter diagnostic is also **0/10**, with
mean closest distance **11.2 cm** and median **7.4 cm**. Seven trials approach
closer; none grasp, and one ball leaves the table. The adapter remains experimental.
See [online response review](results/metrics/act_response_20261013/review.json).
Training caches and model weights remain local and are excluded from Git.

Diagnostic results retain their original paths, source hashes and configuration
snapshots. Absolute paths inside historical reports describe the recording machine;
reproduction requires the pinned dataset/assets and locally regenerated caches.
Use fresh output directories for new experiments so historical results remain intact.

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

- ACT (LeRobot 0.6.1): ResNet-18, `dim_model` 512, chunk 20 at 10 Hz, 2 actions executed per
  inference, batch 16. Joint baseline learns the recorded leader targets; the Cartesian policy
  learns achieved end-effector motion from forward kinematics.
- Simulation: MuJoCo Menagerie `robotstudio_so101` and `franka_emika_panda`. Ball and bowl are compound primitives. Physical values are
  provisional standards not precise measurement 


