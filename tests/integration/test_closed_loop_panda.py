"""E3 harness check: the frozen cartesian checkpoint drives the Panda through the same executor.

One 15-tick trial with the real ``outputs/act_cartesian/checkpoints/006000`` weights on
``--robot panda``: ``predict_action_chunk`` runs ceil(ticks / n_action_steps) times, the policy
batch holds only the contract's two keys (never ``privileged*`` / ``sim.*``), the trials.csv row is
``robot=panda, controller_source=LEARNED_ACT``, the velocity slot of the proprio equals the
one-tick finite difference the SO-101 path uses, the joint checkpoint is refused with a clear
error, and the HOLD_POSE null baseline produces its own labelled, policy-free row. Skipped when
the checkpoint is absent. No claim about task success is made here.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation.run import run_trials
from real2sim.policy.adapters import ContractError

CHECKPOINT = Path("outputs/act_cartesian/checkpoints/006000")
JOINT_CHECKPOINT = Path("outputs/act_joint_so101/checkpoints/020000")
CONFIG = Path("configs/sim_panda.yaml")
EVAL = Path("configs/eval_panda.yaml")
TICKS = 15
pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]
needs_checkpoint = pytest.mark.skipif(not (CHECKPOINT / "model.safetensors").is_file(),
                                      reason="cartesian checkpoint missing")


def _rows(out: Path) -> list[dict]:
    with (out / "trials.csv").open() as f:
        return list(csv.DictReader(f))


@needs_checkpoint
def test_panda_rollout_is_driven_by_frozen_cartesian_policy(tmp_path, monkeypatch):
    from lerobot.policies.act.modeling_act import ACTPolicy

    calls, batch_keys, states = [], set(), []
    original = ACTPolicy.predict_action_chunk

    def spy(self, batch):
        batch_keys.update(batch.keys())
        calls.append(tuple(batch["observation.state"].shape))
        states.append(batch["observation.state"].detach().cpu().numpy()[0].copy())
        return original(self, batch)

    monkeypatch.setattr(ACTPolicy, "predict_action_chunk", spy)
    out = tmp_path / "panda"
    summary = run_trials(robot="panda", checkpoint=CHECKPOINT, config=CONFIG, eval_config=EVAL,
                         n_trials=1, seed_start=0, out=out, video=True, controller="learned",
                         device="cpu", max_ticks=TICKS)
    n_action_steps = summary["checkpoint"]["n_action_steps"]
    trial = summary["trials"][0]
    assert trial["abort"] is None, trial["abort"]
    assert trial["ticks"] == TICKS
    assert len(calls) == math.ceil(TICKS / n_action_steps) == trial["policy_calls"]
    assert calls[0] == (1, 13)
    assert batch_keys == {"observation.images.iphone", "observation.state"}
    assert not any(k.startswith(("privileged", "sim.")) for k in batch_keys)
    assert all(np.all(np.isfinite(s)) for s in states)
    rows = _rows(out)
    assert len(rows) == 1
    assert rows[0]["robot"] == "panda"
    assert rows[0]["controller_source"] == "LEARNED_ACT"
    assert rows[0]["experiment_id"] == "E3_cart_ACT_panda"
    assert summary["checkpoint"]["kind"] == "cartesian"
    assert len(rows[0]["checkpoint_sha256"]) == 64 and rows[0]["checkpoint_step"] == "6000"
    assert rows[0]["success"] in {"True", "False"}
    assert Path(rows[0]["video_path"]).is_file()
    assert float(rows[0]["mean_latency_ms"]) > 0
    # ik_rejections in the row come from the Panda adapter tally (cross-checked by run.py).
    stats = trial["adapter_stats"]
    assert stats["ticks"] == TICKS and stats["rejected"] == int(rows[0]["ik_rejections"])


@needs_checkpoint
def test_panda_proprio_velocity_matches_so101_finite_difference():
    """The adapter's velocity slot is (p_k - p_{k-1}) * hz from advance_tick, as on the SO-101."""
    from real2sim.panda.env import PandaPickPlaceEnv
    from real2sim.policy.act import load_checkpoint
    from real2sim.policy.adapters import PandaObservationAdapter

    _, normalizer, contract = load_checkpoint(CHECKPOINT, "cpu")
    adapter = PandaObservationAdapter(contract, normalizer, device="cpu")
    with PandaPickPlaceEnv(CONFIG, render_images=False) as env:
        env.reset(seed=0)
        adapter.reset()
        first = adapter.state(env)
        np.testing.assert_array_equal(first[9:12], 0.0)  # zeroed at the first tick, as in labels
        prev = env.ee_pose()[:3, 3].copy()
        adapter.advance_tick(env)
        cmd = np.append(env.initial_qpos, env.initial_opening)
        cmd[1] += 0.05  # move so the velocity is non-zero
        env.step(cmd)
        state = adapter.state(env)
        expected = (env.ee_pose()[:3, 3] - prev) * env.control_hz
        np.testing.assert_allclose(state[9:12], expected, atol=1e-5)
        np.testing.assert_allclose(state[9:12], env.canonical_proprio()[9:12], atol=1e-5)
        assert np.linalg.norm(expected) > 1e-3
        # Pure read: repeated calls within a tick do not change the feature.
        np.testing.assert_array_equal(state, adapter.state(env))


@pytest.mark.skipif(not (JOINT_CHECKPOINT / "model.safetensors").is_file(),
                    reason="joint checkpoint missing")
def test_joint_checkpoint_is_refused_on_panda(tmp_path):
    with pytest.raises(ContractError, match="joint-space checkpoint has no meaning on the 7-joint Panda"):
        run_trials(robot="panda", checkpoint=JOINT_CHECKPOINT, config=CONFIG, eval_config=EVAL,
                   n_trials=1, seed_start=0, out=tmp_path / "refused", video=False,
                   controller="learned", device="cpu", max_ticks=TICKS)
    assert not (tmp_path / "refused" / "trials.csv").exists()


def test_panda_hold_pose_baseline_is_labelled_and_policy_free(tmp_path, monkeypatch):
    from lerobot.policies.act.modeling_act import ACTPolicy

    def forbid(self, batch):
        raise AssertionError("HOLD_POSE must never call the policy")

    monkeypatch.setattr(ACTPolicy, "predict_action_chunk", forbid)
    out = tmp_path / "hold"
    summary = run_trials(robot="panda", checkpoint=None, config=CONFIG, eval_config=EVAL,
                         n_trials=1, seed_start=3, out=out, video=False, controller="hold",
                         device="cpu", max_ticks=TICKS)
    rows = _rows(out)
    assert rows[0]["robot"] == "panda"
    assert rows[0]["controller_source"] == "HOLD_POSE"
    assert rows[0]["experiment_id"] == "E0_hold_pose_null_panda"
    assert rows[0]["seed"] == "3" and rows[0]["checkpoint_sha256"] == ""
    assert rows[0]["success"] == "False" and rows[0]["ticks"] == str(TICKS)
    assert summary["summary"]["successes"] == 0
    assert summary["trials"][0]["abort"] is None
