"""Use the downloaded native checkpoint's processors exactly once, without a scene."""

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from real2sim.policy.native_lerobot_act import NativeLeRobotACT, file_sha256
from scripts.evaluate_hf_act_20261027 import SNAPSHOT

pytestmark = [pytest.mark.integration,
    pytest.mark.skipif(not (SNAPSHOT / "model.safetensors").is_file(),
                       reason="optional user checkpoint is not downloaded")]


@pytest.fixture(scope="module")
def native():
    torch.set_num_threads(2)
    return NativeLeRobotACT(SNAPSHOT)


def test_native_saved_state_and_image_normalization_are_applied_once(native):
    stats = load_file(SNAPSHOT / "policy_preprocessor_step_3_normalizer_processor.safetensors")
    mean = stats["observation.state.mean"].numpy()
    images = {key: np.zeros((360, 640, 3), np.uint8) for key in native.camera_keys}
    batch = native.batch(mean, images)
    torch.testing.assert_close(batch["observation.state"], torch.zeros(1, 6), atol=1e-6, rtol=0)
    for key in native.camera_keys:
        expected = -stats[f"{key}.mean"] / stats[f"{key}.std"]
        torch.testing.assert_close(batch[key][0, :, :1, :1], expected, atol=1e-6, rtol=1e-6)
    zero_prediction = torch.zeros(1, 50, 6)
    unnormalized = native.postprocessor(zero_prediction)
    torch.testing.assert_close(unnormalized[0, 0], stats["action.mean"], atol=1e-6, rtol=1e-6)


def test_native_prediction_preserves_weights_and_rejects_missing_camera(native):
    images = {key: np.zeros((360, 640, 3), np.uint8) for key in native.camera_keys}
    before = file_sha256(SNAPSHOT / "model.safetensors")
    prediction = native.predict_action_chunk(np.zeros(6), images)
    assert prediction.shape == (50, 6) and np.isfinite(prediction).all()
    assert file_sha256(SNAPSHOT / "model.safetensors") == before == native.weights_sha256
    images.pop(native.camera_keys[0])
    with pytest.raises(ValueError, match="exactly match"):
        native.batch(np.zeros(6), images)
