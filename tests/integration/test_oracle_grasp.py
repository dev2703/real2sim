"""E4b / gate G2: the scripted ORACLE_GEOMETRY_DIAGNOSTIC can grasp the simulated ball.

Measured with ``real2sim.evaluation.oracle`` (seed 0, D-014 base lift 1.2 cm, start pose =
clamped dataset rest pose after 30 settling ticks, approach tilt 15 deg, ball pinched near the
fingertips): ``ball_held`` becomes True inside the CLOSE phase with both jaw tip geoms in contact. These tests pin that
result, the mandatory labels, and the absence of any policy batch. Runtime well under 60 s.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from real2sim.evaluation import oracle
from real2sim.evaluation.metrics import TrialTracker
from real2sim.sim.env import So101PickPlaceEnv

CONFIG = Path("configs/sim_so101.yaml")
EVAL = Path("configs/eval_so101.yaml")
pytestmark = [pytest.mark.integration]


class _SpyObs(dict):
    """Observation dict that records every key written to it and forbids every read.

    The oracle must not consume the policy observation at all (it is a privileged-GT diagnostic);
    any ``obs[...]``/``obs.get`` inside ``step`` raises.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.written: list[str] = []

    def __setitem__(self, key, value):
        self.written.append(key)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        raise AssertionError(f"oracle read obs[{key!r}]; it must only use privileged metrics")

    def get(self, key, default=None):
        raise AssertionError(f"oracle read obs.get({key!r}); it must only use privileged metrics")


@pytest.fixture(scope="module")
def env():
    e = So101PickPlaceEnv(CONFIG, render_images=False)
    yield e
    e.close()


def test_oracle_reaches_ball_held_in_close_phase_seed0(env):
    q0, settle = oracle.settle_start_pose(env, 0)
    # Start pose: rest pose lifted by physics settling, zero table contacts (D-014 confound removed).
    assert settle["before"]["robot_table_contacts"] > 0
    assert settle["after"]["robot_table_contacts"] == 0
    assert np.all(q0 >= env.limits[:, 0]) and np.all(q0 <= env.limits[:, 1])
    ctrl = oracle.OracleGeometryController(env)
    obs, _ = env.reset(seed=0, scenario={"joint_positions": q0})
    ctrl.reset()
    held_phase = None
    for _ in range(120):
        result = ctrl.step(obs, env)
        obs = result.obs
        if result.info["privileged_metrics"]["ball_held"] and held_phase is None:
            held_phase = result.detail["phase"]
            both_jaws = env._ball_contacts()
            assert both_jaws & env.gripper_geoms["gripper"], "fixed jaw not touching the ball"
            assert both_jaws & env.gripper_geoms["moving_jaw_so101_v1"], "moving jaw not touching"
        if ctrl.phase == "TRANSPORT":
            break
    assert held_phase == "CLOSE", f"ball_held first seen in phase {held_phase}; log {ctrl.phase_log}"
    assert ctrl._first_held_tick is not None and ctrl._first_held_tick <= 60
    assert result.info["privileged_metrics"]["lifted"], ctrl.phase_log


def test_trials_csv_rows_are_labelled_oracle(tmp_path):
    out = tmp_path / "oracle"
    summary = oracle.run_trials(config=CONFIG, eval_config=EVAL, n_trials=1, seed_start=0, out=out,
                                video_dir=None, max_ticks=120)
    with (out / "trials.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["controller_source"] == "ORACLE_GEOMETRY_DIAGNOSTIC"
    assert rows[0]["experiment_id"] == "E4b_oracle_geometry_diagnostic"
    assert rows[0]["checkpoint_sha256"] == "" and rows[0]["video_path"] == ""
    assert "LEARNED" not in rows[0]["controller_source"]
    assert summary["controller_source"] == "ORACLE_GEOMETRY_DIAGNOSTIC"
    assert summary["caption"].startswith("ORACLE (privileged GT, scripted)")
    assert summary["trials"][0]["policy_calls"] == 0
    assert summary["trials"][0]["abort"] is None
    assert "rolled_out_note" in summary and "ik_rejections_note" in summary
    assert "tracker_rolled_out_raw" not in summary["trials"][0]  # no oracle-side re-scoring
    # Seed 0 delivers the ball and it never leaves the bowl unheld: rolled_out must be False even
    # though the held ball crosses the rim band during LOWER (metrics.TrialTracker semantics).
    assert rows[0]["success"] == "True"
    assert rows[0]["rolled_out"] == "False", rows[0]
    assert summary["summary"]["rolled_out"] == 0
    # The accepted IK solution is clipped to env.limits before env.step, so a joint resting on its
    # IK (jnt_range) limit is not reported as actuator saturation (ctrlrange differs by 3e-6 rad).
    assert float(rows[0]["max_action_saturation"]) == 0.0, rows[0]


def test_oracle_has_no_policy_batch(env, monkeypatch):
    """The oracle never builds a policy batch: no ``observation.*`` key is written anywhere."""
    from real2sim.policy import adapters

    def forbid(*a, **k):
        raise AssertionError("ObservationAdapter.batch must never run in the oracle")

    monkeypatch.setattr(adapters.ObservationAdapter, "batch", forbid)
    q0, _ = oracle.settle_start_pose(env, 0)
    ctrl = oracle.OracleGeometryController(env)
    obs, _ = env.reset(seed=0, scenario={"joint_positions": q0})
    env_obs_keys = set(obs)
    spy = _SpyObs(obs)
    ctrl.reset()
    for _ in range(20):
        result = ctrl.step(spy, env)  # _SpyObs raises on any read: the oracle never looks at obs
        assert not spy.written, spy.written
        assert not any(k.startswith("observation.") for k in result.detail)
        # The returned obs is the env's own dict, untouched: no policy keys were added to it.
        assert set(result.obs) == env_obs_keys
        spy = _SpyObs(result.obs)
    assert ctrl.n_policy_calls == 0
    assert not hasattr(ctrl, "policy")
    assert ctrl.controller_source == "ORACLE_GEOMETRY_DIAGNOSTIC"


def test_reset_clears_per_trial_state_including_latencies(env):
    q0, _ = oracle.settle_start_pose(env, 0)
    ctrl = oracle.OracleGeometryController(env)
    obs, _ = env.reset(seed=0, scenario={"joint_positions": q0})
    ctrl.reset()
    for _ in range(10):
        obs = ctrl.step(obs, env).obs
    assert len(ctrl.latencies_ms) == 10 and ctrl.tick == 10
    ctrl.reset()
    assert ctrl.latencies_ms == [] and ctrl.tick == 0 and ctrl.ik_rejections == 0
    assert ctrl.phase == "PRE_GRASP" and ctrl.phase_log == [] and ctrl._first_held_tick is None


def test_done_phase_holds_pose_without_ik_and_counts_no_rejections(env):
    """After DONE the arm holds its pose: no IK solve, no latency sample, no rejection."""
    q0, _ = oracle.settle_start_pose(env, 0)
    ctrl = oracle.OracleGeometryController(env)
    obs, _ = env.reset(seed=0, scenario={"joint_positions": q0})
    ctrl.reset()
    for _ in range(120):
        obs = ctrl.step(obs, env).obs
        if ctrl.phase == "DONE":
            break
    assert ctrl.phase == "DONE", ctrl.phase_log
    n_lat, n_rej = len(ctrl.latencies_ms), ctrl.ik_rejections
    q_hold = env.sim_state().copy()
    for _ in range(5):
        result = ctrl.step(obs, env)
        obs = result.obs
        assert result.detail["phase"] == "DONE"
        assert result.detail["ik_reason"] == "done_hold" and not result.ik_rejected
        assert result.saturation == 0.0
    assert len(ctrl.latencies_ms) == n_lat and ctrl.ik_rejections == n_rej
    # Target = measured pose each tick, so the servos creep a few mrad under gravity; no motion.
    assert np.allclose(env.sim_state()[:5], q_hold[:5], atol=1e-2), (env.sim_state(), q_hold)


def test_rolled_out_counts_only_unheld_departures():
    """The oracle reports TrialTracker.rolled_out unchanged (real2sim.evaluation.metrics is the
    single scorer): an in_bowl True -> False transition counts only when the ball is unheld on
    both ticks, so a held ball toggling across the rim band during placement is not a roll-out."""
    def pm(in_bowl, held):
        return {"ee_ball_distance_m": 0.0, "lifted": True, "ball_held": held, "ball_in_bowl": in_bowl,
                "ball_resting": False, "ball_xyz": [0, 0, 0.1], "ball_xyz_task": [0, 0, 0.1],
                "tool_xyz_task": [0, 0, 0.1]}

    def rolled_out(sequence):
        t = TrialTracker(success_hold_ticks=1, approach_distance_m=0.03, table_top_z=0.0, ball_radius_m=0.03)
        for in_bowl, held in sequence:
            t.update(pm(in_bowl, held))
        assert len(t.history) == len(sequence)
        return t.rolled_out

    # Held ball toggling across the bowl boundary during placement (seed 0 pattern): not rolled out.
    assert not rolled_out([(True, True), (False, True), (True, True), (True, False)])
    # A release that coincides with leaving the rim band is a release, not a roll-out (the ball
    # was held on the previous tick): metrics.py checks both ticks.
    assert not rolled_out([(True, True), (False, False), (True, False)])
    # Ball resting in the bowl then leaving it unheld: rolled out, and it latches.
    assert rolled_out([(True, False), (True, False), (False, False)])
    assert rolled_out([(True, False), (False, False), (True, False)])
    assert not rolled_out([])
    assert not rolled_out([(True, False)])
    assert not rolled_out([(False, False), (False, False)])
    assert not hasattr(oracle, "rolled_out_unheld")
