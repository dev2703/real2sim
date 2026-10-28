import pytest

from real2sim.policy.hf_recovery import ApproachMonitor


def test_visual_progress_gets_time_but_stagnant_approach_requests_recovery():
    moving = ApproachMonitor(25)
    stopped = ApproachMonitor(25)
    for tick in range(50):
        assert moving.update(tick, .30 - tick * .001) is None
        assert stopped.update(tick, .30) is None
    assert moving.update(50, .25) is None
    assert "progress" in stopped.update(50, .30)


def test_missing_rgb_uses_bounded_attempt_and_invalid_input_is_rejected():
    monitor = ApproachMonitor(25)
    assert monitor.update(124, None) is None
    assert "elapsed" in monitor.update(125, None)
    with pytest.raises(ValueError, match="finite"):
        monitor.update(126, float("nan"))
