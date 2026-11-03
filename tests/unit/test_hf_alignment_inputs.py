"""Image-swap attribution must expose context dependence and retain cancellations."""

import json

import numpy as np
import pytest

from scripts.diagnose_hf_alignment_inputs import camera_swap_effects, camera_swap_summary


def predictions(wrist, global_view, interaction):
    recorded = np.full((3, 6), 20., np.float32)
    return {"recorded": recorded, "nominal_wrist_only": recorded + wrist,
            "nominal_global_only": recorded + global_view,
            "nominal": recorded + wrist + global_view + interaction}


def test_additive_views_have_zero_interaction_despite_different_action_effects():
    effects = camera_swap_effects(predictions(2., -5., 0.), "nominal", 2)
    assert effects["two_camera_interaction"]["five_arm_joint_mean_abs_effect_degrees"] == 0.
    assert effects["wrist_at_recorded_global"] == effects["wrist_at_rendered_global"]
    assert effects["global_at_recorded_wrist"] == effects["global_at_rendered_wrist"]
    assert effects["global_at_recorded_wrist"]["mean_signed_joint_effect_degree_percent"] == [-5.] * 6
    assert effects["global_at_recorded_wrist"]["mean_abs_joint_effect_degree_percent"] == [5.] * 6


def test_nonadditive_camera_effect_is_measured_before_absolute_aggregation():
    # Global view reverses the wrist effect. Averaging signed deltas or taking
    # absolute values before subtraction would hide or misstate this interaction.
    values = predictions(2., 5., -4.)
    values["nominal"][1, :] += 8.
    effects = camera_swap_effects(values, "nominal", 2)
    interaction = effects["two_camera_interaction"]
    assert interaction["mean_signed_joint_effect_degree_percent"] == [0.] * 6
    assert interaction["mean_abs_joint_effect_degree_percent"] == [4.] * 6
    assert effects["wrist_at_rendered_global"]["mean_abs_joint_effect_degree_percent"] == [4.] * 6


def test_scored_prefix_excludes_future_chunk_predictions():
    values = predictions(1., 3., 0.)
    values["nominal"][2, :] = 1e6
    effects = camera_swap_effects(values, "nominal", 2)
    assert effects["two_camera_interaction"]["gripper_mean_abs_effect_percent"] == 0.
    assert camera_swap_effects(values, "nominal", 3)["two_camera_interaction"][
        "gripper_mean_abs_effect_percent"] > 1000.


@pytest.mark.parametrize("edit", ["shape", "nonfinite", "prefix", "empty"])
def test_incomparable_or_unscored_predictions_are_rejected(edit):
    values, prefix = predictions(1., 2., 0.), 2
    if edit == "shape":
        values["nominal_wrist_only"] = np.zeros((2, 6))
    elif edit == "nonfinite":
        values["recorded"][0, 0] = np.nan
    elif edit == "prefix":
        prefix = 4
    else:
        prefix = 0
    with pytest.raises(ValueError):
        camera_swap_effects(values, "nominal", prefix)


def test_missing_anchors_are_not_silently_credited_with_zero_effect():
    assert camera_swap_summary([], "nominal") == {"cases": 0, "effects": None}
    cases = [{"camera_swap_effects": {"nominal": camera_swap_effects(
        predictions(wrist, 3., 0.), "nominal", 2)}} for wrist in (1., 5.)]
    summary = camera_swap_summary(cases, "nominal")
    assert summary["cases"] == 2
    assert summary["effects"]["wrist_at_recorded_global"][
        "five_arm_joint_mean_abs_effect_degrees"] == 3.
    json.dumps(summary, allow_nan=False)
