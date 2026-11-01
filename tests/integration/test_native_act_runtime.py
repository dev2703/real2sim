"""Cached native weights/processors retain action parity through runtime backends."""

import numpy as np
import pytest
import torch

from real2sim.policy.native_lerobot_act import NativeLeRobotACT, file_sha256
from scripts.evaluate_hf_act_20261027 import SNAPSHOT

pytestmark = [pytest.mark.integration,
    pytest.mark.skipif(not (SNAPSHOT / "model.safetensors").is_file(),
                      reason="optional native checkpoint is not downloaded")]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_native_runtime_parity_and_unchanged_weights(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.set_num_threads(2)
    reference = NativeLeRobotACT(SNAPSHOT, device="cpu")
    optimized = NativeLeRobotACT(SNAPSHOT, device=device, inference_backend="vectorized")
    rng = np.random.default_rng(28)
    state = np.array([0., -70., 80., 60., 0., 45.])
    for _ in range(2):
        images = {key: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
                  for key in reference.camera_keys}
        expected = reference.predict_action_chunk(state, images)
        actual = optimized.predict_action_chunk(state, images)
        np.testing.assert_allclose(actual, expected, rtol=0,
                                   atol=0 if device == "cpu" else 1e-3)
        assert actual.shape == (50, 6)
        assert len(optimized._position_cache) == 1
    assert optimized.runtime_options == {"device": device, "inference_backend": "vectorized"}
    assert file_sha256(SNAPSHOT / "model.safetensors") == reference.weights_sha256 == optimized.weights_sha256


def test_unknown_native_runtime_fails_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="inference_backend"):
        NativeLeRobotACT(tmp_path, inference_backend="unknown")
