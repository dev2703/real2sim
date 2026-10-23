"""Contact comparisons must reach compiled physics without changing physical arrays."""

import copy

import mujoco
import numpy as np
import pytest

from real2sim.config import load_yaml
from real2sim.sim.contact_solver import contact_solver_attributes, contact_solver_settings


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_explicit_contact_options_compile_without_changing_physical_parameters(robot):
    if robot == "so101":
        from real2sim.sim.model import scene_xml
    else:
        from real2sim.panda.model import scene_xml
    config = load_yaml(f"configs/sim_{robot}.yaml")
    task = load_yaml(config["task_config"])
    original = mujoco.MjModel.from_xml_string(scene_xml(config, task))
    changed = copy.deepcopy(config)
    changed["contact_solver"] = {"impratio": 100, "noslip_iterations": 3}
    candidate = mujoco.MjModel.from_xml_string(scene_xml(changed, task))
    settings = contact_solver_settings(candidate)
    assert settings["impratio"] == 100
    assert settings["noslip_iterations"] == 3
    for field in ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
                  "geom_solimp", "body_mass", "body_inertia", "jnt_range", "actuator_gainprm",
                  "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(original, field), getattr(candidate, field))
    assert original.opt.solver == candidate.opt.solver
    assert original.opt.cone == candidate.opt.cone
    assert original.opt.timestep == candidate.opt.timestep
    np.testing.assert_array_equal(original.opt.gravity, candidate.opt.gravity)
    assert config.get("contact_solver") is None


@pytest.mark.parametrize("robot", ["so101", "panda"])
def test_absent_contact_options_preserve_pinned_defaults(robot):
    if robot == "so101":
        from real2sim.sim.model import scene_xml
    else:
        from real2sim.panda.model import scene_xml
    config = load_yaml(f"configs/sim_{robot}.yaml")
    task = load_yaml(config["task_config"])
    assembled = mujoco.MjModel.from_xml_string(scene_xml(config, task))
    import xml.etree.ElementTree as ET

    pinned = ET.parse(config["model"]).getroot().find("option")
    assert assembled.opt.impratio == float(pinned.get("impratio", 1))
    assert assembled.opt.noslip_iterations == int(pinned.get("noslip_iterations", 0))


@pytest.mark.parametrize("value", [
    {"impratio": 0}, {"impratio": -1}, {"impratio": float("nan")},
    {"impratio": float("inf")}, {"impratio": True}, {"impratio": "100"},
    {"noslip_iterations": -1}, {"noslip_iterations": 3.5}, {"noslip_iterations": True},
    {"noslip_iterations": "3"}, {"noslip_iterations": None}, {"iterations": 100}, [],
])
def test_invalid_contact_options_are_rejected_before_physics(value):
    with pytest.raises(ValueError, match="contact_solver"):
        contact_solver_attributes({"contact_solver": value})
