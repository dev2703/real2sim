"""Load an unmodified native LeRobot ACT and its saved processor state.

Native checkpoints do not use this repository's extra normalization/contract
format. Feed raw degree/percent state and RGB [0,1] images to the saved pipeline
exactly once; do not apply the repository's 192x256 letterbox or normalizer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from real2sim.artifacts import sha256_file as file_sha256


def native_scene_contract(config) -> dict:
    """Validate the six-joint, two-camera interface implemented by native runners.

    V2 calls the wrist stream ``top``; v3 calls it ``wrist``. Keep the saved
    feature names and token order while assigning the same physical camera role.
    This validates the software interface, not joint or optical calibration.
    """
    state, action = config.input_features.get("observation.state"), config.output_features.get("action")
    if (tuple(getattr(state, "shape", ())) != (6,)
            or tuple(getattr(action, "shape", ())) != (6,)):
        raise ValueError("native robot runners require six state/action degree-percent channels")
    cameras = list(config.image_features)
    wrist = [key for key in cameras if key in (
        "observation.images.top", "observation.images.wrist")]
    if (len(cameras) != 2 or len(wrist) != 1
            or set(cameras) != {wrist[0], "observation.images.iphone"}):
        raise ValueError("native robot runners require wrist (top or wrist) and iphone cameras")
    shapes = [tuple(config.input_features[key].shape) for key in cameras]
    if (len(shapes[0]) != 3 or shapes[0][0] != 3
            or any(type(value) is not int or value <= 0 for value in shapes[0])
            or any(shape != shapes[0] for shape in shapes[1:])):
        raise ValueError("native cameras require matching positive RGB CHW image shapes")
    if (config.n_obs_steps != 1 or config.temporal_ensemble_coeff is not None
            or not 1 <= config.n_action_steps <= config.chunk_size):
        raise ValueError("native runners require one observation and a non-ensembled action prefix")
    return {"camera_keys": cameras, "wrist_key": wrist[0],
            "image_shape_chw": list(shapes[0]), "policy_hz": 25.,
            "timing_basis": "25 Hz source dataset contract; not inferred from ACT model config"}


def native_checkpoint_identity(path, config) -> dict:
    """Report a requested local checkpoint without assigning it the default model ID."""
    path = Path(path).resolve()
    repo_id, revision = getattr(config, "repo_id", None), None
    basis = "checkpoint_config" if repo_id else "unidentified_local_checkpoint"
    parts = path.parts
    for index, part in enumerate(parts[:-2]):
        if part.startswith("models--") and parts[index + 1] == "snapshots":
            repo_id = part.removeprefix("models--").replace("--", "/")
            revision = parts[index + 2]
            basis = "huggingface_cache_path"
    return {"repo_id": repo_id, "revision": revision, "snapshot": str(path),
            "checkpoint_identity_basis": basis}


class NativeLeRobotACT:
    """Frozen ACT, raw observations, and native saved preprocessing/postprocessing."""

    def __init__(self, path, *, device="cpu"):
        self.path = Path(path)
        self.config = PreTrainedConfig.from_pretrained(self.path, local_files_only=True)
        if self.config.type != "act":
            raise ValueError("requires native ACT checkpoint")
        self.scene_contract = native_scene_contract(self.config)
        self.checkpoint_identity = native_checkpoint_identity(self.path, self.config)
        self.config.device = device
        # Constructor initialization is overwritten by every strict-loaded weight.
        # Avoid downloading a temporary pretrained ResNet before this strict load.
        self.config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(self.path, config=self.config,
                                                local_files_only=True, strict=True)
        self.policy.eval().requires_grad_(False)
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
