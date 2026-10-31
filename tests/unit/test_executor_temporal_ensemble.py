"""Inference integration checks: temporal settings must affect executed actions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real2sim.policy.adapters import IPHONE_KEY, ChunkExecutor, ObservationAdapter
from tests.unit.test_oracle_schema import FakeNormalizer, joint_contract
from tests.unit.test_policy_guards import (
    FakeEnv,
    IdentityUnnorm,
    RecordingAction,
    good_image,
)


class SequencePolicy:
    def __init__(self, chunks, *, n_action_steps=1, coefficient=None):
        self.chunks = [np.repeat(np.asarray(values, float)[:, None], 6, axis=1)
                       for values in chunks]
        self.config = SimpleNamespace(chunk_size=len(chunks[0]), n_action_steps=n_action_steps,
                                      temporal_ensemble_coeff=coefficient)
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        chunk = self.chunks[self.calls]
        self.calls += 1
        return torch.as_tensor(chunk, dtype=torch.float64)[None]


def executor(policy, **kwargs):
    action = RecordingAction()
    observation = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    instance = ChunkExecutor(policy, IdentityUnnorm(6), observation, action, **kwargs)
    return instance, action


def advance(instance, ticks):
    obs = {IPHONE_KEY: good_image()}
    for _ in range(ticks):
        obs = instance.step(obs, FakeEnv()).obs


def test_config_ensemble_is_executed_at_correct_absolute_offsets_each_tick():
    policy = SequencePolicy([[1., 10., 100.], [2., 20., 200.], [3., 30., 300.]], coefficient=0.)
    instance, action = executor(policy)
    advance(instance, 3)
    # Tick two combines offsets 2, 1, 0 from the three predictions.
    np.testing.assert_allclose(np.asarray(action.applied)[:, 0], [1., 6., 41.])
    assert policy.calls == instance.n_policy_calls == 3
    assert len(instance.latencies_ms) == 3


@pytest.mark.parametrize("coefficient,expected", [(np.log(2.), 22. / 3.),
                                                 (-np.log(2.), 14. / 3.)])
def test_coefficient_sign_favours_documented_prediction_age(coefficient, expected):
    policy = SequencePolicy([[1., 10.], [2., 20.]], coefficient=coefficient)
    instance, action = executor(policy)
    advance(instance, 2)
    # In particular, this must not average the previous first action (1) with 2.
    assert action.applied[1][0] == pytest.approx(expected)


def test_explicit_coefficient_overrides_checkpoint_setting():
    policy = SequencePolicy([[1., 10.], [2., 20.]], coefficient=1.)
    instance, action = executor(policy, temporal_ensemble_coeff=0.)
    advance(instance, 2)
    assert action.applied[1][0] == pytest.approx(6.)


def test_explicit_execution_one_can_override_checkpoint_queue_length():
    policy = SequencePolicy([[1., 10.], [2., 20.]], n_action_steps=2, coefficient=.01)
    instance, action = executor(policy, n_action_steps=1)
    advance(instance, 2)
    assert policy.calls == 2
    assert 2. < action.applied[1][0] < 10.


def test_ensemble_ignores_unsupervised_tail_of_prefix_checkpoint():
    policy = SequencePolicy([[1., 10., 1e30, -1e30], [2., 20., -1e30, 1e30],
                             [3., 30., 1e30, -1e30]], coefficient=0.)
    instance, action = executor(policy, supervised_prefix_steps=2)
    advance(instance, 3)
    np.testing.assert_allclose(np.asarray(action.applied)[:, 0], [1., 6., 11.5])


def test_reset_discards_ensemble_offsets_and_restarts_tick_alignment():
    policy = SequencePolicy([[1., 100.], [2., 200.]], coefficient=0.)
    instance, action = executor(policy)
    advance(instance, 1)
    instance.reset()
    assert policy.resets == 1 and instance.n_policy_calls == 0
    assert instance.latencies_ms == [] and instance.queue == []
    advance(instance, 1)
    np.testing.assert_allclose(np.asarray(action.applied)[:, 0], [1., 2.])


@pytest.mark.parametrize("steps", [2, 3])
def test_temporal_ensemble_refuses_multiple_executed_actions_per_prediction(steps):
    policy = SequencePolicy([[1., 10., 100.]], n_action_steps=steps, coefficient=.01)
    with pytest.raises(ValueError):
        executor(policy)


def test_queue_refuses_execution_beyond_supervised_prefix():
    policy = SequencePolicy([[1., 10., 100.]], n_action_steps=3)
    with pytest.raises(ValueError):
        executor(policy, supervised_prefix_steps=2)


@pytest.mark.parametrize("prefix", [0, -1, 4, True, 1.5])
def test_invalid_supervised_prefix_is_rejected(prefix):
    with pytest.raises(ValueError):
        executor(SequencePolicy([[1., 10., 100.]]), supervised_prefix_steps=prefix)


@pytest.mark.parametrize("coefficient", [None, 0.])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_refused_before_any_action_or_observation_advance(coefficient, bad):
    # Bad values in a tail must not sneak through shape validation, including queue mode.
    policy = SequencePolicy([[1., bad]], coefficient=coefficient)
    instance, action = executor(policy)
    with pytest.raises(ValueError, match="finite"):
        advance(instance, 1)
    assert action.applied == []
    assert instance.obs_adapter._tick == 0


def test_default_queue_execution_keeps_existing_chunk_order():
    policy = SequencePolicy([[1., 10., 100.], [2., 20., 200.]], n_action_steps=2)
    instance, action = executor(policy)
    advance(instance, 3)
    np.testing.assert_allclose(np.asarray(action.applied)[:, 0], [1., 10., 2.])
    assert policy.calls == 2
