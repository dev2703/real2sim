"""Load an unmodified native LeRobot ACT and its saved processor state.

Native checkpoints do not use this repository's extra normalization/contract
format. Feed raw degree/percent state and RGB [0,1] images to the saved pipeline
exactly once; do not apply the repository's 192x256 letterbox or normalizer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeLeRobotACT:
    """Frozen ACT, raw observations, and native saved preprocessing/postprocessing."""

    def __init__(self, path, *, device="cpu"):
        self.path = Path(path)
        self.config = PreTrainedConfig.from_pretrained(self.path, local_files_only=True)
        if self.config.type != "act":
            raise ValueError("requires native ACT checkpoint")
        self.config.device = device
        # Constructor initialization is overwritten by every strict-loaded weight.
        # Avoid downloading a temporary pretrained ResNet before this strict load.
        self.config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(self.path, config=self.config,
                                                local_files_only=True, strict=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=self.path,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.camera_keys = list(self.config.image_features)
        self.weights_sha256 = file_sha256(self.path / "model.safetensors")
        self.reset()

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def batch(self, state, images):
        state = np.asarray(state, np.float32)
        expected = tuple(self.config.input_features["observation.state"].shape)
        if state.shape != expected or not np.isfinite(state).all():
            raise ValueError(f"raw state must be finite with shape {expected}")
        if set(images) != set(self.camera_keys):
            raise ValueError("raw image keys must exactly match checkpoint cameras")
        payload = {"observation.state": torch.from_numpy(state.copy())}
        for key in self.camera_keys:
            rgb = np.asarray(images[key])
            chw = tuple(self.config.input_features[key].shape)
            if rgb.dtype != np.uint8 or rgb.shape != (chw[1], chw[2], chw[0]):
                raise ValueError(f"{key} requires raw uint8 RGB {chw[1]}x{chw[2]}")
            payload[key] = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.
        return self.preprocessor(payload)

    @torch.inference_mode()
    def predict_action_chunk(self, state, images):
        output = self.postprocessor(self.policy.predict_action_chunk(self.batch(state, images)))
        actions = output.detach().cpu().numpy()
        expected = (1, self.config.chunk_size, self.config.output_features["action"].shape[0])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError("native ACT returned invalid action chunk")
        return actions[0].astype(np.float64)
