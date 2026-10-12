"""Edge cases for the runtime guards between simulator and policy (plan rule 5, section 7).

The guards must reject malformed inputs loudly; a silently-normalised NaN image or an extra
ground-truth key would corrupt a closed-loop result without failing anything.
"""

import numpy as np
import pytest
import torch

from real2sim.config import validate_camera_keys
from real2sim.policy.adapters import (
    IMAGE_SHAPE_CHW,
    IPHONE_KEY,
    STATE_KEY,
    ChunkExecutor,
    ContractError,
    JointActionAdapter,
    ObservationAdapter,
    assert_policy_batch,
    normalize_image_chw,
)
from tests.unit.test_oracle_schema import FakeNormalizer, joint_contract


class FakeEnv:
    def __init__(self, state=None):
        self._state = np.zeros(6) if state is None else np.asarray(state, float)

    def real_state(self):
        return self._state

    def ee_pose(self):
        return np.eye(4)


def good_image():
    return np.full(IMAGE_SHAPE_CHW, 0.5, np.float32)


# ------------------------------------------------------------------ image normalisation
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_pixels_are_rejected(bad):
    img = good_image()
    img[1, 10, 10] = bad
    with pytest.raises(ContractError, match="finite"):
        normalize_image_chw(FakeNormalizer(6), img)


@pytest.mark.parametrize("value", [-1e-6, 1.0 + 1e-6, 255.0])
def test_out_of_range_pixels_are_rejected(value):
    img = good_image()
    img[0, 0, 0] = value
    with pytest.raises(ContractError):
        normalize_image_chw(FakeNormalizer(6), img)


@pytest.mark.parametrize("shape", [(192, 256, 3), (1, 3, 192, 256), (3, 256, 192), (3, 192, 255)])
def test_wrong_image_layout_is_rejected(shape):
    with pytest.raises(ContractError):
        normalize_image_chw(FakeNormalizer(6), np.zeros(shape, np.float32))


def test_boundary_pixels_are_accepted_and_normalised_exactly():
    norm = FakeNormalizer(6)
    for v in (0.0, 1.0):
        out = normalize_image_chw(norm, np.full(IMAGE_SHAPE_CHW, v, np.float32))
        np.testing.assert_allclose(out[:, 0, 0], ((v - norm.img_mean) / norm.img_std).ravel(), rtol=1e-6)


def test_env_image_path_equals_training_uint8_path():
    """Training normalises uint8 HWC; the env path starts from float CHW /255: must be identical."""
    rng = np.random.default_rng(0)
    hwc = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    norm = FakeNormalizer(6)
    via_env = normalize_image_chw(norm, hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)
    np.testing.assert_allclose(via_env, norm.image(hwc), atol=1e-6)


# ------------------------------------------------------------------ batch guard
def test_batch_contains_exactly_the_contract_keys():
    adapter = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    obs = {IPHONE_KEY: good_image(), "sim.joint_position": np.zeros(6), "observation.state": np.zeros(6)}
    batch = adapter.batch(obs, FakeEnv())
    assert set(batch) == {IPHONE_KEY, STATE_KEY}
    assert batch[IPHONE_KEY].shape == (1, *IMAGE_SHAPE_CHW) and batch[STATE_KEY].shape == (1, 6)


@pytest.mark.parametrize("key", ["privileged_metrics", "privileged.ball_xyz"])
def test_observation_with_privileged_keys_is_refused(key):
    adapter = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    with pytest.raises(ContractError, match="privileged"):
        adapter.batch({IPHONE_KEY: good_image(), key: np.zeros(3)}, FakeEnv())


def test_observation_without_camera_is_refused():
    adapter = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    with pytest.raises(ContractError, match=IPHONE_KEY):
        adapter.batch({"observation.state": np.zeros(6)}, FakeEnv())


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_state_is_refused(bad):
    adapter = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    with pytest.raises(ContractError, match="state"):
        adapter.batch({IPHONE_KEY: good_image()}, FakeEnv([0, 0, bad, 0, 0, 0]))


def test_state_of_wrong_length_is_refused():
    adapter = ObservationAdapter(joint_contract(), FakeNormalizer(6))
    with pytest.raises(ContractError, match="length"):
        adapter.batch({IPHONE_KEY: good_image()}, FakeEnv(np.zeros(5)))


@pytest.mark.parametrize("batch", [
    {IPHONE_KEY: 0, STATE_KEY: 0, "sim.joint_position": 0},
    {IPHONE_KEY: 0, STATE_KEY: 0, "privileged_metrics": 0},
    {IPHONE_KEY: 0, STATE_KEY: 0, "observation.ball_xyz": 0},
    {IPHONE_KEY: 0},
    {},
])
def test_assert_policy_batch_rejects_anything_but_the_two_inputs(batch):
    with pytest.raises(ContractError):
        assert_policy_batch(batch)


def test_assert_policy_batch_accepts_the_contract():
    assert_policy_batch({IPHONE_KEY: 0, STATE_KEY: 0})


# ------------------------------------------------------------------ chunk execution
class CountingPolicy:
    def __init__(self, chunk_size=20, n_action_steps=2, dim=6, value=1.0):
        self.config = type("C", (), {"chunk_size": chunk_size, "n_action_steps": n_action_steps})()
        self.calls = 0
        self.resets = 0
        self.dim = dim
        self.value = value

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        assert_policy_batch(batch)
        self.calls += 1
        return torch.full((1, self.config.chunk_size, self.dim), float(self.calls))


class IdentityUnnorm(FakeNormalizer):
    def unaction(self, x):
        return x


class RecordingAction:
    dim = 6

    def __init__(self):
        self.applied = []

    def apply(self, a):
        self.applied.append(a.copy())
        from real2sim.policy.adapters import ApplyResult

        return ApplyResult({IPHONE_KEY: good_image()}, {}, False, 0.0)


@pytest.mark.parametrize("n_steps,ticks,expected_calls", [(1, 5, 5), (2, 5, 3), (2, 4, 2), (20, 21, 2)])
def test_chunk_executor_call_count_and_action_order(n_steps, ticks, expected_calls):
    policy = CountingPolicy(n_action_steps=n_steps)
    act = RecordingAction()
    ex = ChunkExecutor(policy, IdentityUnnorm(6), ObservationAdapter(joint_contract(), FakeNormalizer(6)), act)
    obs = {IPHONE_KEY: good_image()}
    for _ in range(ticks):
        obs = ex.step(obs, FakeEnv()).obs
    assert policy.calls == expected_calls
    # Every executed action comes from the chunk of the call that produced it (value == call id).
    assert [int(a[0]) for a in act.applied] == [1 + i // n_steps for i in range(ticks)]


def test_chunk_executor_reset_drops_queue_and_resets_policy():
    policy = CountingPolicy(n_action_steps=4)
    act = RecordingAction()
    ex = ChunkExecutor(policy, IdentityUnnorm(6), ObservationAdapter(joint_contract(), FakeNormalizer(6)), act)
    obs = {IPHONE_KEY: good_image()}
    ex.step(obs, FakeEnv())
    assert len(ex.queue) == 3
    ex.reset()
    assert ex.queue == [] and policy.resets == 1 and ex.n_policy_calls == 0 and ex.latencies_ms == []
    ex.step(obs, FakeEnv())
    assert policy.calls == 2  # a fresh chunk after reset, never a stale one


@pytest.mark.parametrize("n", [0, -1, 21])
def test_chunk_executor_rejects_invalid_n_action_steps(n):
    with pytest.raises(ValueError):
        ChunkExecutor(CountingPolicy(), IdentityUnnorm(6),
                      ObservationAdapter(joint_contract(), FakeNormalizer(6)), RecordingAction(), n_action_steps=n)


def test_chunk_executor_rejects_dimension_mismatch():
    ex = ChunkExecutor(CountingPolicy(dim=7), IdentityUnnorm(6),
                       ObservationAdapter(joint_contract(), FakeNormalizer(6)), RecordingAction())
    with pytest.raises(ContractError, match="chunk shape"):
        ex.step({IPHONE_KEY: good_image()}, FakeEnv())


# ------------------------------------------------------------------ joint action adapter
@pytest.mark.parametrize("action", [np.zeros(5), np.zeros(7), np.r_[np.zeros(5), np.nan]])
def test_joint_action_adapter_rejects_malformed_actions(action):
    with pytest.raises(ValueError):
        JointActionAdapter(env=None).apply(action)


# ------------------------------------------------------------------ data config
def test_duplicate_camera_keys_are_rejected():
    cfg = {"camera_keys": ["observation.images.iphone", "observation.images.iphone"]}
    with pytest.raises(ValueError, match="more than once"):
        validate_camera_keys(cfg, ["observation.images.iphone"])


def test_empty_camera_keys_are_rejected():
    for cfg in ({}, {"camera_keys": []}, {"camera_keys": None}):
        with pytest.raises(ValueError, match="at least one"):
            validate_camera_keys(cfg, ["observation.images.iphone"])


def test_project_path_resolution(tmp_path):
    from real2sim.config import REPO_ROOT, project_path

    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert project_path("configs/data.yaml") == REPO_ROOT / "configs/data.yaml"
    assert project_path(tmp_path) == tmp_path  # absolute paths pass through
    assert project_path("~/x").is_absolute() and "~" not in str(project_path("~/x"))


# ------------------------------------------------------------------ contract clock / image
class _EnvStub:
    control_hz = 10
    data_config = {"image": {"height": 192, "width": 256}}


def _contract(**over):
    c = {"policy_hz": 10.0, "letterbox": {"src_w": 1280, "src_h": 720, "dst_w": 256, "dst_h": 192}}
    c.update(over)
    return c


def test_contract_timing_and_image_accepts_matching_checkpoint():
    from real2sim.evaluation.run import check_contract_timing_and_image

    check_contract_timing_and_image(_contract(), _EnvStub())


@pytest.mark.parametrize("over,match", [
    ({"policy_hz": 30.0}, "policy_hz"),
    ({"policy_hz": None}, "policy_hz"),
    ({"letterbox": {"src_w": 1280, "src_h": 720, "dst_w": 320, "dst_h": 240}}, "letterbox"),
    ({"letterbox": {"src_w": 640, "src_h": 480, "dst_w": 256, "dst_h": 192}}, "aspect"),
])
def test_contract_timing_and_image_rejects_mismatch(over, match):
    from real2sim.evaluation.run import check_contract_timing_and_image

    with pytest.raises(ContractError, match=match):
        check_contract_timing_and_image(_contract(**over), _EnvStub())


# ------------------------------------------------------------------ checkpoint link
def test_update_last_link_is_atomic_and_repointable(tmp_path):
    from real2sim.policy.train import update_last_link

    for name in ("000002", "000004"):
        (tmp_path / name).mkdir()
        update_last_link(tmp_path, name)
        assert (tmp_path / "last").resolve() == (tmp_path / name).resolve()
    assert not list(tmp_path.glob(".last.tmp.*"))
