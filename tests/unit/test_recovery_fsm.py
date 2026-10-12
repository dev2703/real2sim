"""Recovery detector + gate FSM (plan §5: all transitions, chunk flush, STOP, timeout, handoff,
no oracle leakage). Pure, no MuJoCo.

The ``test_stream_*`` tests run the detector and the gate *together* on a tick-by-tick feature
stream through :class:`RecoveryLoop` (the integration the closed loop uses); the hand-built
``Decision`` tests below only cover the gate's own transitions."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.data.labels import ACTION_NAMES, PROPRIO_NAMES
from real2sim.recovery.detector import (
    FEATURE_NAMES,
    Decision,
    DeviationDetector,
    ObservableFeatures,
)
from real2sim.recovery.fsm import GateState, RecoveryGate
from real2sim.recovery.loop import RecoveryLoop

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recovery.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(CONFIG)


def feats(
    ee=(0.30, 0.0, 0.10),
    ball_rel=(0.04, 0.04, -0.08),
    conf=0.9,
    contact=False,
    vel=(0.0, 0.0, 0.0),
    gripper_cmd=0.8,
    phase=None,
) -> ObservableFeatures:
    act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_cmd])
    return ObservableFeatures(
        ee_pos=np.array(ee, float),
        ee_rot6d=np.array([1, 0, 0, 0, 1, 0], float),
        ee_vel=np.array(vel, float),
        gripper_obs=0.5,
        ball_rel=None if ball_rel is None else np.array(ball_rel, float),
        bowl_rel=np.array([0.03, -0.10, -0.10]),
        ball_conf=conf,
        bowl_conf=0.9,
        gripper_contact=contact,
        last_action=act,
        phase_hint=phase,
    )


NOMINAL = Decision(False, "none", 0.0)
DEV = Decision(True, "ball_rolled_away", 1.0)


# ----------------------------------------------------------------------------- schema
def test_feature_schema_is_29_observable_names_and_no_gt_fields(cfg):
    assert len(FEATURE_NAMES) == 29 == cfg["features"]["input_dim"]
    assert FEATURE_NAMES[:13] == list(PROPRIO_NAMES)
    assert FEATURE_NAMES[21:28] == [f"base_proposed_{a}" for a in ACTION_NAMES]
    assert sum(n for d in cfg["features"]["layout"] for n in d.values()) == 29
    names = [f.name for f in dataclasses.fields(ObservableFeatures)]
    for n in names + FEATURE_NAMES:
        low = n.lower()
        assert "gt" not in low.split("_") and "truth" not in low and "privileged" not in low and "qpos" not in low, n
    v = feats().to_vector()
    assert v.shape == (29,)
    assert v[-1] == 0.0 and v[19] == 0.9  # gripper_contact flag, ball_conf
    invisible = feats(ball_rel=None, conf=0.9).to_vector()
    assert invisible[13:16].tolist() == [0.0, 0.0, 0.0] and invisible[19] == 0.0


def test_decision_reason_is_validated():
    with pytest.raises(ValueError):
        Decision(True, "simulator_said_so", 1.0)


# --------------------------------------------------------------------------- detector
def test_single_tick_blip_does_not_trigger_rolled_away(cfg):
    det = DeviationDetector(cfg)
    base = feats()
    assert det.update(base).deviation is False
    far = feats(ball_rel=(0.04 + 0.10, 0.04, -0.08))
    assert det.update(far).deviation is False  # one tick
    assert det.update(base).deviation is False
    assert det.counters["rolled_away"] == 0  # reset by hysteresis release


def test_rolled_away_persists_and_respects_hysteresis(cfg):
    det = DeviationDetector(cfg)
    det.update(feats())
    n = cfg["detector"]["rolled_away"]["ticks"]
    far = feats(ball_rel=(0.04 + 0.06, 0.04, -0.08))
    decisions = [det.update(far) for _ in range(n)]
    assert [d.deviation for d in decisions[:-1]] == [False] * (n - 1)
    assert decisions[-1].deviation and decisions[-1].reason == "ball_rolled_away"
    assert 0.0 < decisions[-1].confidence <= 1.0
    # between release_m and displacement_m the counter is held, not reset
    mid = feats(ball_rel=(0.04 + 0.03, 0.04, -0.08))
    assert det.update(mid).reason == "ball_rolled_away"
    # held ball -> rule disabled
    assert det.update(feats(ball_rel=(0.14, 0.04, -0.08), contact=True)).deviation is False


def test_approach_miss_requires_closing_no_contact_and_persistence(cfg):
    det = DeviationDetector(cfg)
    m = cfg["detector"]["approach_miss"]["ticks"]
    near_closing = feats(ball_rel=(0.02, 0.0, -0.02), gripper_cmd=0.0)
    for _ in range(m - 1):
        assert det.update(near_closing).deviation is False
    d = det.update(near_closing)
    assert d.deviation and d.reason == "approach_miss"
    det.reset()
    near_open = feats(ball_rel=(0.02, 0.0, -0.02), gripper_cmd=0.9)
    assert all(not det.update(near_open).deviation for _ in range(m + 2))
    det.reset()
    near_contact = feats(ball_rel=(0.02, 0.0, -0.02), gripper_cmd=0.0, contact=True)
    assert all(not det.update(near_contact).deviation for _ in range(m + 2))


def test_ee_drift_outside_corridor(cfg):
    det = DeviationDetector(cfg)
    k = cfg["detector"]["ee_drift"]["ticks"]
    outside = feats(ee=(0.60, 0.0, 0.10))
    for _ in range(k - 1):
        assert det.update(outside).deviation is False
    assert det.update(outside).reason == "ee_drift"
    det.reset()
    assert all(not det.update(feats(ee=(0.60, 0.0, 0.10), contact=True)).deviation for _ in range(k + 1))
    det.reset()
    assert all(not det.update(feats(ee=(0.60, 0.0, 0.10), phase="transport")).deviation for _ in range(k + 1))


def test_ball_lost_recommends_stop(cfg):
    det = DeviationDetector(cfg)
    k = cfg["detector"]["ball_lost_ticks"]
    lost = feats(ball_rel=None, conf=0.0)
    for _ in range(k - 1):
        assert not det.update(lost).stop_recommended
    d = det.update(lost)
    assert d.deviation and d.reason == "ball_lost" and d.stop_recommended
    det.reset()
    low_conf = feats(conf=0.1)
    assert det.counters["ball_lost"] == 0
    for _ in range(k):
        d = det.update(low_conf)
    assert d.reason == "ball_lost"


def test_detector_is_deterministic(cfg):
    seq = [feats(ball_rel=(0.04 + 0.06 * (i % 2), 0.04, -0.08)) for i in range(20)]
    a = DeviationDetector(cfg)
    b = DeviationDetector(cfg)
    assert [a.update(f) for f in seq] == [b.update(f) for f in seq]


# -------------------------------------------------------------------------------- fsm
def run_handoff(gate: RecoveryGate, start_tick: int, cfg) -> list:
    outs = []
    h = cfg["fsm"]["handoff"]["ticks"]
    for i in range(h + 1):
        outs.append(gate.step(start_tick + i, NOMINAL, feats()))
    return outs


def test_full_cycle_with_flush_and_fresh_observation(cfg):
    gate = RecoveryGate(cfg)
    o = gate.step(0, NOMINAL, feats())
    assert o.state is GateState.ACT and not o.use_specialist and not o.flush_chunk

    o = gate.step(1, DEV, feats())
    assert o.state is GateState.DEVIATION_CONFIRMED and o.flush_chunk and o.use_specialist
    assert o.reason == "ball_rolled_away"

    o = gate.step(2, DEV, feats())
    assert o.state is GateState.SPECIALIST and o.use_specialist and not o.flush_chunk

    # still deviating -> keep recovering
    o = gate.step(3, DEV, feats())
    assert o.state is GateState.SPECIALIST

    outs = run_handoff(gate, 4, cfg)
    assert outs[0].state is GateState.HANDOFF_CHECK
    h = cfg["fsm"]["handoff"]["ticks"]
    assert all(x.state is GateState.HANDOFF_CHECK and x.use_specialist for x in outs[: h - 1])
    final = outs[h - 1]
    assert final.state is GateState.ACT and final.force_fresh_observation and not final.use_specialist
    assert not any(x.force_fresh_observation for x in outs[: h - 1])
    assert outs[h].state is GateState.ACT and not outs[h].force_fresh_observation
    states = [s for _, s, _ in gate.transcript]
    assert states == ["DEVIATION_CONFIRMED", "SPECIALIST", "HANDOFF_CHECK", "ACT"]
    assert gate.transcript[0] == (1, "DEVIATION_CONFIRMED", "ball_rolled_away")
    assert gate.transcript[-1][2] == "handoff"


@pytest.mark.parametrize(
    "bad",
    [
        dict(ee=(0.30, 0.0, 0.01)),  # too close to the table
        dict(ball_rel=None, conf=0.0),  # ball not visible
        dict(conf=0.2),  # low confidence
        dict(vel=(0.2, 0.0, 0.0)),  # moving fast
        dict(ee=(0.60, 0.0, 0.10)),  # outside corridor
    ],
)
def test_handoff_check_falls_back_when_a_condition_breaks(cfg, bad):
    gate = RecoveryGate(cfg)
    gate.step(0, DEV, feats())
    gate.step(1, DEV, feats())
    assert gate.step(2, NOMINAL, feats()).state is GateState.HANDOFF_CHECK
    assert gate.step(3, NOMINAL, feats()).state is GateState.HANDOFF_CHECK
    o = gate.step(4, NOMINAL, feats(**bad))
    assert o.state is GateState.SPECIALIST and o.reason == "handoff_failed" and o.use_specialist
    assert gate.handoff_count == 0
    # consecutive-tick requirement restarts from scratch
    h = cfg["fsm"]["handoff"]["ticks"]
    outs = [gate.step(5 + i, NOMINAL, feats()) for i in range(h)]
    assert outs[-1].state is GateState.ACT and outs[-1].force_fresh_observation
    assert all(x.state is GateState.HANDOFF_CHECK for x in outs[:-1])


def test_handoff_blocked_while_detector_still_flags_deviation(cfg):
    gate = RecoveryGate(cfg)
    gate.step(0, DEV, feats())
    gate.step(1, DEV, feats())
    assert gate.step(2, DEV, feats()).state is GateState.SPECIALIST


def test_stop_on_timeout_is_absorbing_until_reset(cfg):
    gate = RecoveryGate(cfg)
    t = 0
    gate.step(t, DEV, feats(ee=(0.60, 0.0, 0.10)))
    bad = feats(ee=(0.60, 0.0, 0.10))  # never satisfies handoff
    max_t = cfg["fsm"]["max_recovery_ticks"]
    o = None
    for t in range(1, max_t + 1):
        o = gate.step(t, DEV, bad)
    assert o.state is GateState.STOP and o.stop and o.reason == "max_recovery_ticks"
    assert gate.transcript[-1] == (max_t, "STOP", "max_recovery_ticks")
    for t in range(max_t + 1, max_t + 20):
        o = gate.step(t, NOMINAL, feats())
        assert o.state is GateState.STOP and o.stop and not o.use_specialist
    gate.reset()
    assert gate.state is GateState.ACT and gate.transcript == [] and gate.retries == 0
    assert gate.step(0, NOMINAL, feats()).state is GateState.ACT


def test_stop_on_max_retries(cfg):
    gate = RecoveryGate(cfg)
    t = 0
    for _ in range(cfg["fsm"]["max_retries"]):
        o = gate.step(t, DEV, feats())
        assert o.state is GateState.DEVIATION_CONFIRMED and o.flush_chunk
        gate.step(t + 1, DEV, feats())
        t += 2
        outs = run_handoff(gate, t, cfg)
        assert outs[-2].state is GateState.ACT
        t += len(outs)
    o = gate.step(t, DEV, feats())
    assert o.state is GateState.STOP and o.reason.startswith("max_retries")
    assert gate.step(t + 1, NOMINAL, feats()).stop


def test_stop_on_detector_recommendation_from_any_state(cfg):
    lost = Decision(True, "ball_lost", 1.0, stop_recommended=True)
    for prelude in ([], [DEV], [DEV, DEV], [DEV, DEV, NOMINAL]):
        gate = RecoveryGate(cfg)
        for i, d in enumerate(prelude):
            gate.step(i, d, feats())
        o = gate.step(len(prelude), lost, feats())
        assert o.state is GateState.STOP and o.reason == "ball_lost" and not o.use_specialist


def test_gate_is_deterministic(cfg):
    seq = [NOMINAL, DEV, DEV, NOMINAL, NOMINAL, DEV, NOMINAL, NOMINAL, NOMINAL, NOMINAL, NOMINAL, NOMINAL]
    a, b = RecoveryGate(cfg), RecoveryGate(cfg)
    oa = [a.step(i, d, feats()) for i, d in enumerate(seq)]
    ob = [b.step(i, d, feats()) for i, d in enumerate(seq)]
    assert oa == ob and a.transcript == b.transcript


def test_config_values_are_marked_provisional():
    text = CONFIG.read_text()
    assert "PROVISIONAL" in text and "replays" in text
    cfg = load_yaml(CONFIG)
    assert cfg["mlp"]["hidden"] == [256, 256, 128]
    assert cfg["mlp"]["huber_delta"] == 0.05 and cfg["mlp"]["lr"] == 3e-4
    assert cfg["mlp"]["action_bounds"]["translation_m"] == 0.03
    assert cfg["mlp"]["action_bounds"]["rotation_rad"] == 0.3


# ------------------------------------------------------------- detector + gate on a stream
def stream_feats(ee, ball_task, vel=(0.0, 0.0, 0.0), conf=0.9, contact=False) -> ObservableFeatures:
    """Features from task-frame EE and ball positions (ball_rel = ball - ee, as the tracker gives)."""
    ee = np.array(ee, float)
    ball = np.array(ball_task, float)
    return feats(ee=tuple(ee), ball_rel=tuple(ball - ee), conf=conf, vel=vel, contact=contact)


def roll_away_stream(cfg, roll_tick=5, hover_tick=8, ball0=(0.34, 0.04, 0.0), ball1=(0.40, 0.04, 0.0), n=120):
    """Ball latched at ball0, rolls 6 cm to ball1 at roll_tick; from hover_tick the specialist
    hovers 5 cm above the *new* ball position, slow, visible, inside the corridor."""
    for t in range(n):
        ball = ball0 if t < roll_tick else ball1
        ee = (0.30, 0.0, 0.10) if t < hover_tick else (ball1[0], ball1[1], 0.05)
        yield t, stream_feats(ee, ball)


def test_stream_rolled_away_takeover_then_handoff_back_to_act(cfg):
    """The confirmed defect: a real roll-away must be able to hand off, not always STOP."""
    loop = RecoveryLoop(cfg)
    outs = []
    for t, f in roll_away_stream(cfg):
        d, o = loop.step(t, f)
        outs.append((t, d, o))
        if o.state is GateState.ACT and o.force_fresh_observation:
            break
    states = [s for _, s, _ in loop.gate.transcript]
    assert states == ["DEVIATION_CONFIRMED", "SPECIALIST", "HANDOFF_CHECK", "ACT"], loop.gate.transcript
    takeover = next(o for _, _, o in outs if o.flush_chunk)
    assert takeover.reason == "ball_rolled_away" and takeover.tick == 7  # 5 + rolled_away.ticks - 1
    assert loop.rearm_ticks == [takeover.tick]
    # reference re-latched at the post-roll ball position, so the specialist's approach is nominal
    assert np.allclose(loop.detector.ball_reference, (0.40, 0.04, 0.0))
    after = [d for t, d, _ in outs if t > takeover.tick]
    assert all(not d.deviation for d in after), [d.reason for d in after]
    final = outs[-1][2]
    h = cfg["fsm"]["handoff"]["ticks"]
    assert final.state is GateState.ACT and final.force_fresh_observation and not final.stop
    # hover starts at 8 (the DEVIATION_CONFIRMED -> SPECIALIST tick); handoff candidate at 9,
    # then H consecutive handoff ticks end at 8 + H
    assert final.tick == 8 + h
    assert loop.state is GateState.ACT and loop.gate.retries == 1


def test_stream_without_rearm_reproduces_the_stop_defect(cfg):
    """Guard: the bare detector+gate pair (no rearm) still ends in STOP, so the re-latch in
    RecoveryLoop is what makes handoff possible, not a change in thresholds."""
    det, gate = DeviationDetector(cfg), RecoveryGate(cfg)
    o = None
    for t, f in roll_away_stream(cfg):
        o = gate.step(t, det.update(f), f)
        if o.stop:
            break
    assert o is not None and o.state is GateState.STOP and o.reason == "max_recovery_ticks"


def test_rearm_clears_counters_and_relatches_reference(cfg):
    det = DeviationDetector(cfg)
    base = stream_feats((0.30, 0.0, 0.10), (0.34, 0.04, 0.0))
    far = stream_feats((0.30, 0.0, 0.10), (0.40, 0.04, 0.0))
    det.update(base)
    for _ in range(cfg["detector"]["rolled_away"]["ticks"]):
        d = det.update(far)
    assert d.reason == "ball_rolled_away" and det.counters["rolled_away"] > 0
    history_len = len(det._c.history)
    det.rearm(far)
    assert det.counters == {"ball_lost": 0, "rolled_away": 0, "approach_miss": 0, "ee_drift": 0}
    assert np.allclose(det.ball_reference, (0.40, 0.04, 0.0))
    assert len(det._c.history) == history_len  # transcript kept
    assert det.update(far).deviation is False
    # rearm without features (or with the ball invisible): next confident observation latches
    det.rearm()
    assert det.ball_reference is None
    det.rearm(stream_feats((0.30, 0.0, 0.10), (0.34, 0.04, 0.0), conf=0.0))
    assert det.ball_reference is None
    det.update(base)
    assert np.allclose(det.ball_reference, (0.34, 0.04, 0.0))


def test_stream_second_roll_during_recovery_blocks_handoff_then_times_out(cfg):
    """A further roll after takeover is measured from the re-latched reference: the specialist
    cannot hand off while the detector flags it, and the recovery times out to STOP."""
    loop = RecoveryLoop(cfg)
    max_t = cfg["fsm"]["max_recovery_ticks"]
    o = None
    for t in range(200):
        # rolls at 5 (takeover at 7) and again at 10, during HANDOFF_CHECK, before handoff completes
        ball = (0.34, 0.04, 0.0) if t < 5 else (0.40, 0.04, 0.0) if t < 10 else (0.46, 0.04, 0.0)
        ee = (0.30, 0.0, 0.10) if t < 8 else (0.40, 0.04, 0.05)  # specialist hovers above the 2nd position only
        d, o = loop.step(t, stream_feats(ee, ball))
        if o.stop:
            break
    assert o.state is GateState.STOP and o.reason == "max_recovery_ticks"
    assert loop.gate.transcript[-1][0] == 7 + max_t
    states = [st for _, st, _ in loop.gate.transcript]
    assert states == ["DEVIATION_CONFIRMED", "SPECIALIST", "HANDOFF_CHECK", "SPECIALIST", "STOP"], loop.gate.transcript
    assert loop.gate.transcript[3][2] == "handoff_failed"
    assert "ball_rolled_away" in loop.detector._c.history[10:]


def test_stream_handoff_then_second_deviation_counts_as_retry_then_stop_on_third(cfg):
    loop = RecoveryLoop(cfg)
    t = 0
    positions = [(0.34, 0.04, 0.0), (0.40, 0.04, 0.0), (0.34, 0.10, 0.0), (0.40, 0.10, 0.0)]
    handoffs = 0
    o = None
    for k in range(1, len(positions)):
        # settle on positions[k-1] for 5 ticks, then roll to positions[k] and hover above it
        for _ in range(5):
            d, o = loop.step(t, stream_feats((0.30, 0.0, 0.10), positions[k - 1]))
            t += 1
        for _ in range(60):
            ee = (positions[k][0], positions[k][1], 0.05)
            d, o = loop.step(t, stream_feats(ee, positions[k]))
            t += 1
            if o.force_fresh_observation:
                handoffs += 1
                break
            if o.stop:
                break
        if o.stop:
            break
    assert handoffs == cfg["fsm"]["max_retries"]
    assert o.state is GateState.STOP and o.reason.startswith("max_retries:ball_rolled_away")
    assert loop.rearm_ticks and len(loop.rearm_ticks) == cfg["fsm"]["max_retries"]


def test_loop_reset_and_determinism(cfg):
    a, b = RecoveryLoop(cfg), RecoveryLoop(cfg)
    ra = [a.step(t, f) for t, f in roll_away_stream(cfg, n=40)]
    rb = [b.step(t, f) for t, f in roll_away_stream(cfg, n=40)]
    assert ra == rb and a.gate.transcript == b.gate.transcript and a.rearm_ticks == b.rearm_ticks
    a.reset()
    assert a.state is GateState.ACT and a.rearm_ticks == [] and a.detector.ball_reference is None
    assert a.detector.counters == {"ball_lost": 0, "rolled_away": 0, "approach_miss": 0, "ee_drift": 0}
