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


@pytest.mark.parametrize("report_missing", [True, False])
def test_old_rgb_samples_expire_during_detection_gaps(report_missing):
    monitor = ApproachMonitor(2, patience_s=2., max_attempt_s=20.)
    assert monitor.update(0, .3) is None
    assert monitor.update(1, .3) is None
    if report_missing:
        for tick in range(2, 9):
            assert monitor.update(tick, None) is None
    # Two old observations plus three fresh ones used to trigger here even
    # though the fresh evidence covers only one second.
    for tick in range(9, 13):
        assert monitor.update(tick, .3) is None
    assert "progress" in monitor.update(13, .3)


def test_missing_rgb_does_not_delay_the_absolute_deadline():
    monitor = ApproachMonitor(2, patience_s=2., max_attempt_s=5.)
    for tick in range(10):
        assert monitor.update(tick, .3 if tick % 2 else None) is None
    assert "elapsed" in monitor.update(10, None)
