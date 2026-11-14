import numpy as np
import pytest

from scripts.train_recovery_mlp import robot_episode_weights


def test_phase_sampling_preserves_robot_episode_mass_despite_long_approach():
    robots = np.array([0] * 6 + [1] * 2)
    episodes = np.array([10] * 5 + [11] + [10] * 2)
    phases = np.array([1, 1, 1, 1, 3, 4, 1, 3])
    weights = robot_episode_weights(robots, episodes, phases).numpy()
    assert weights.sum() == pytest.approx(1.)
    for robot in (0, 1):
        assert weights[robots == robot].sum() == pytest.approx(.5)
    assert weights[(robots == 0) & (episodes == 10)].sum() == pytest.approx(.25)
    assert weights[:4].sum() == pytest.approx(weights[4])
    assert weights[5] == pytest.approx(.25)
    # Phase IDs affect sampling only; repeated source episode IDs across arms
    # must not merge their balancing groups.
    assert weights[6] == pytest.approx(weights[7])


def test_phase_sampling_rejects_misaligned_or_noninteger_labels():
    with pytest.raises(ValueError, match="matching integer"):
        robot_episode_weights(np.zeros(2), np.ones(2), np.array([1]))
    with pytest.raises(ValueError, match="matching integer"):
        robot_episode_weights(np.zeros(2), np.ones(2), np.array([1., 2.]))


def test_stall_collection_rejects_fixed_prefix_and_missing_tracked_rgb(tmp_path):
    from scripts.run_recovery_dagger import collect

    # These validations run before loading a checkpoint or constructing physics.
    with pytest.raises(ValueError, match="not both"):
        collect("so101", tmp_path / "both", [10220], [11220],
                act_prefix_steps=50, act_prefix_until_stall=True)
    with pytest.raises(ValueError, match="tracked RGB"):
        collect("panda", tmp_path / "untracked", [10220], [11220],
                act_prefix_until_stall=True, reference_mode="initial")
