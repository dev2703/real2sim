from real2sim.perception.ball_hsv import BALL_HSV


def test_named_ball_hsv_presets_are_distinct_and_ordered():
    assert set(BALL_HSV) == {"audit", "grounding_v1", "metric_targets"}
    audit = BALL_HSV["audit"]
    grounding = BALL_HSV["grounding_v1"]
    metric = BALL_HSV["metric_targets"]
    assert audit.lower == (22, 70, 90) and audit.upper == (50, 255, 255)
    assert grounding.lower == (24, 90, 85) and grounding.upper[0] == 44
    assert metric.lower == (20, 75, 30) and metric.upper[0] == 50
    # The three call sites intentionally disagree; keep that explicit.
    assert len({audit.lower, grounding.lower, metric.lower}) == 3
