"""Robustness plans retain nominal inputs and keep the lighting fixed."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim.config import load_yaml
from scripts.prepare_fixed_lighting_robustness import (
    CONDITIONS,
    SOURCE,
    apply_model_condition,
    prepare,
    scenarios,
)


def test_generated_size_conditions_preserve_source_mass_and_lighting(tmp_path):
    config = load_yaml(SOURCE)
    task = load_yaml(config["task_config"])
    before = copy.deepcopy(task)
    plan = prepare(tmp_path / "plan", seeds=[1000, 1001],
                   conditions=["nominal", "ball_small", "ball_large"])
    for name, condition in plan["conditions"].items():
        variant_sim = load_yaml(condition["source_scene"])
        variant = load_yaml(variant_sim["task_config"])
        assert variant_sim["appearance"]["lighting"] == config["appearance"]["lighting"]
        assert variant["ball"]["mass"] == task["ball"]["mass"]
        assert variant["ball"]["radius"] == pytest.approx(task["ball"]["radius"] * CONDITIONS[name]["radius_scale"])
    assert load_yaml(config["task_config"]) == before
    assert plan["conditions"]["ball_small"]["scenarios"] == plan["conditions"]["ball_large"]["scenarios"]


def test_disc_layouts_are_reproducible_and_share_original_seed_ball_draw():
    task = load_yaml(load_yaml(SOURCE)["task_config"])
    seeds = [1000, 1001]
    nominal = scenarios(task, seeds, CONDITIONS["nominal"])
    jitter = scenarios(task, seeds, CONDITIONS["bowl_jitter"])
    assert nominal == scenarios(task, seeds, CONDITIONS["nominal"])
    for seed in seeds:
        assert nominal[str(seed)]["ball_xy"] == jitter[str(seed)]["ball_xy"]
        assert nominal[str(seed)]["bowl_xy"] != jitter[str(seed)]["bowl_xy"]
    with pytest.raises(ValueError, match="unique"):
        scenarios(task, [1, 1], CONDITIONS["nominal"])


def test_friction_hook_changes_only_collision_sliding_without_touching_lights():
    model = SimpleNamespace(geom_contype=np.array([1, 0, 0]),
        geom_conaffinity=np.array([0, 1, 0]), geom_friction=np.array([[.6, .01, .001]] * 3),
        light_diffuse=np.array([[.4, .4, .4]]))
    before, light = model.geom_friction.copy(), model.light_diffuse.copy()
    metadata = apply_model_condition(SimpleNamespace(model=model), {"collision_sliding_scale": 1.5})
    np.testing.assert_allclose(model.geom_friction[:2, 0], before[:2, 0] * 1.5)
    np.testing.assert_array_equal(model.geom_friction[:, 1:], before[:, 1:])
    np.testing.assert_array_equal(model.geom_friction[2], before[2])
    np.testing.assert_array_equal(model.light_diffuse, light)
    assert metadata["lighting_changed"] is False
    with pytest.raises(ValueError, match="Unknown"):
        apply_model_condition(SimpleNamespace(model=model), {"lighting_scale": .5})
