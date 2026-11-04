"""Physical scalars must preserve uncertainty and avoid inventing rigid transforms."""

import math
from pathlib import Path

import pytest

from scripts.import_v2_operator_measurements import import_measurements, parse_measurements

SOURCE = """Every measurement is in inches; x lengthwise, y widthwise, z heightwise.
camera from ball: (30.5, 4.5, 16)
ball & bowl: 12 inches
Ball from robot base: (2,13)
mat: (16,12)
Bowl: 4.5 diameter
Ball: 8 circumference
Note: there could be +- 5% measurements error.
"""


def test_circumference_is_not_diameter_and_unknown_registration_stays_unknown(tmp_path):
    source = tmp_path / "physical.md"
    source.write_text(SOURCE)
    measured, task, report = import_measurements(
        source, Path("configs/v2/task.exploratory_20261022.yaml"),
        Path("configs/v2/measurements.template.json"))
    assert task["ball"]["radius"] == pytest.approx(8 * .0254 / (2 * math.pi))
    assert task["ball"]["radius_interval_m"] == pytest.approx([
        task["ball"]["radius"] * .95, task["ball"]["radius"] * 1.05])
    assert measured["bowl"]["inner_diameter_m"] is None
    assert measured["bowl"]["outer_diameter_m"] is None
    assert measured["bowl"]["reported_unspecified_diameter_m"] == pytest.approx(.1143)
    assert measured["operator_frame"]["T_task_operator"] is None
    assert measured["cameras"]["observation.images.iphone"]["T_parent_optical"] is None
    assert task["physical_calibration_accepted"] is False
    assert report["fk_label_changes_required"] is False
    assert report["comparisons"]["ball_radius"]["outside_reported_interval"] is False
    assert report["comparisons"]["ball_bowl_spacing_if_center_to_center"]["outside_reported_interval"] is True
    assert report["mat_axis_order_differs"] is True


@pytest.mark.parametrize("old,new,match", [
    ("inches", "centimetres", "inches"),
    ("8 circumference", "8 diameter", "circumference"),
    ("4.5 diameter", "4.5", "diameter"),
    ("+- 5%", "+- 150%", "percentage"),
    ("(2,13)", "(2,13,5)", "dimensions"),
    ("Ball: 8 circumference", "Ball: -8 circumference", "positive"),
    ("mat: (16,12)", "", "missing"),
])
def test_invalid_or_ambiguous_measurements_are_rejected(old, new, match):
    with pytest.raises(ValueError, match=match):
        parse_measurements(SOURCE.replace(old, new))


def test_repeated_field_cannot_silently_change_the_geometry():
    with pytest.raises(ValueError, match="duplicate"):
        parse_measurements(SOURCE + "\nBall: 9 circumference")


def test_signed_offsets_preserve_directions_and_ordered_uncertainty_bounds():
    parsed = parse_measurements(SOURCE.replace("(30.5, 4.5, 16)", "(-30.5, 0, 16)"))
    camera = parsed["entries"]["camera from ball"]
    assert camera["value_m"] == pytest.approx([-.7747, 0., .4064])
    assert camera["interval_m"][0] == pytest.approx([-.813435, -.735965])
    assert camera["interval_m"][1] == [0., 0.]
