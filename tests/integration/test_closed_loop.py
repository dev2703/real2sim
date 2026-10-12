"""G4: ACT controls the simulator by its own predictions, not by injected recorded targets.

Runs one short trial (15 ticks) with the step-2000 joint checkpoint and spies on
``ACTPolicy.predict_action_chunk`` at the class level: the number of calls must equal
ceil(ticks / n_action_steps), the batch must contain only the contract's two keys (no
``privileged*`` / ``sim.*``), and the trials.csv row must be labelled LEARNED_ACT. The HOLD_POSE
null baseline produces its own labelled row without any policy call. The checkpoint-driven
tests are skipped if the checkpoint is absent; the cartesian-adapter test needs only the scene.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation.run import run_trials

CHECKPOINT = Path("outputs/act_joint_so101/checkpoints/002000")
CONFIG = Path("configs/sim_so101.yaml")
EVAL = Path("configs/eval_so101.yaml")
TICKS = 15
pytestmark = [pytest.mark.integration]
needs_checkpoint = pytest.mark.skipif(not (CHECKPOINT / "model.safetensors").is_file(),
                                      reason="joint checkpoint missing")


def _rows(out: Path) -> list[dict]:
    with (out / "trials.csv").open() as f:
        return list(csv.DictReader(f))


@needs_checkpoint
def test_learned_rollout_is_driven_by_policy_predictions(tmp_path, monkeypatch):
    from lerobot.policies.act.modeling_act import ACTPolicy

    calls, batch_keys = [], set()
    original = ACTPolicy.predict_action_chunk

    def spy(self, batch):
        batch_keys.update(batch.keys())
        calls.append(tuple(batch["observation.state"].shape))
        return original(self, batch)

    monkeypatch.setattr(ACTPolicy, "predict_action_chunk", spy)
    out = tmp_path / "learned"
    summary = run_trials(robot="so101", checkpoint=CHECKPOINT, config=CONFIG, eval_config=EVAL,
                         n_trials=1, seed_start=0, out=out, video=True, controller="learned",
                         device="cpu", max_ticks=TICKS)
    n_action_steps = summary["checkpoint"]["n_action_steps"]
    trial = summary["trials"][0]
    assert trial["abort"] is None, trial["abort"]
    assert trial["ticks"] == TICKS
    assert len(calls) == math.ceil(TICKS / n_action_steps) == trial["policy_calls"]
    assert calls[0] == (1, 6)
    assert batch_keys == {"observation.images.iphone", "observation.state"}
    assert not any(k.startswith(("privileged", "sim.")) for k in batch_keys)
    rows = _rows(out)
    assert len(rows) == 1
    assert rows[0]["controller_source"] == "LEARNED_ACT"
    assert rows[0]["experiment_id"] == "E1_joint_ACT"
    assert len(rows[0]["checkpoint_sha256"]) == 64 and rows[0]["checkpoint_step"] == "2000"
    assert rows[0]["success"] in {"True", "False"}
    assert Path(rows[0]["video_path"]).is_file()
    assert float(rows[0]["mean_latency_ms"]) > 0


@needs_checkpoint
def test_hold_pose_baseline_is_labelled_and_policy_free(tmp_path, monkeypatch):
    from lerobot.policies.act.modeling_act import ACTPolicy

    def forbid(self, batch):
        raise AssertionError("HOLD_POSE must never call the policy")

    monkeypatch.setattr(ACTPolicy, "predict_action_chunk", forbid)
    out = tmp_path / "hold"
    summary = run_trials(robot="so101", checkpoint=None, config=CONFIG, eval_config=EVAL,
                         n_trials=1, seed_start=3, out=out, video=False, controller="hold",
                         device="cpu", max_ticks=TICKS)
    rows = _rows(out)
    assert rows[0]["controller_source"] == "HOLD_POSE"
    assert rows[0]["seed"] == "3" and rows[0]["checkpoint_sha256"] == ""
    assert rows[0]["success"] == "False"
    assert summary["summary"]["successes"] == 0


def test_cartesian_adapter_survives_soft_limit_overshoot():
    """MuJoCo limits are soft: holding the rest pose (shoulder_lift exactly at its lower bound)
    pushes measured qpos ~1e-3 rad past jnt_range. The adapter must clip its IK seed instead of
    letting NativeKinematics raise 'IK seed exceeds native joint limits' and abort the run."""
    from real2sim.policy.adapters import CartesianActionAdapter
    from real2sim.sim.env import So101PickPlaceEnv
    from real2sim.sim.kinematics import NativeKinematics

    env = So101PickPlaceEnv(CONFIG)
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(env.initial_qpos)
        q = env.sim_state()
        assert np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1]), \
            "precondition: rest pose no longer overshoots a joint limit; test needs a new setup"
        ad = CartesianActionAdapter(env, NativeKinematics(env.model, env.config))
        for g in (0.5, 0.0, 1.0):
            for _ in range(3):
                res = ad.apply(np.array([0, 0, 0, 0, 0, 0, g], float))
                assert res.obs is not None and "privileged_metrics" in res.info
    finally:
        env.close()
