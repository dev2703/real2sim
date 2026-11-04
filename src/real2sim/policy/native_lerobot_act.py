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
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

from real2sim.artifacts import sha256_file as file_sha256


def vectorized_act_actions(model, batch, camera_keys, position_cache):
    """Equivalent frozen ACT inference without making one Python tensor per visual token.

    The upstream path unbinds each feature map into hundreds of tensors and
    stacks them again. Concatenating intact sequences preserves camera/token
    order. Spatial positional embeddings depend only on feature-map geometry,
    device and dtype, so reuse them across observations. No image or learned
    feature is cached, and the saved processors still run for every observation.
    """
    if model.training:
        raise ValueError("vectorized ACT is for frozen evaluation only")
    config = model.config
    state = batch[OBS_STATE]
    batch_size = state.shape[0]
    latent = torch.zeros((batch_size, config.latent_dim), dtype=torch.float32,
                         device=state.device)
    tokens = [model.encoder_latent_input_proj(latent).unsqueeze(0)]
    positions = [model.encoder_1d_feature_pos_embed.weight.unsqueeze(1)]
    if config.robot_state_feature:
        tokens.append(model.encoder_robot_state_input_proj(state).unsqueeze(0))
    if config.env_state_feature:
        tokens.append(model.encoder_env_state_input_proj(batch[OBS_ENV_STATE]).unsqueeze(0))
    for key in camera_keys:
        features = model.backbone(batch[key])["feature_map"]
        cache_key = (tuple(features.shape[-2:]), features.device, features.dtype)
        if cache_key not in position_cache:
            position_cache[cache_key] = model.encoder_cam_feat_pos_embed(features).to(
                dtype=features.dtype)
        spatial_positions = position_cache[cache_key]
        projected = model.encoder_img_feat_input_proj(features)
        tokens.append(projected.flatten(2).permute(2, 0, 1))
        positions.append(spatial_positions.flatten(2).permute(2, 0, 1))
    tokens = torch.cat(tokens, dim=0)
    positions = torch.cat(positions, dim=0)
    encoded = model.encoder(tokens, pos_embed=positions)
    decoder_input = torch.zeros((config.chunk_size, batch_size, config.dim_model),
                                dtype=positions.dtype, device=positions.device)
    decoded = model.decoder(decoder_input, encoded, encoder_pos_embed=positions,
                            decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1))
    return model.action_head(decoded.transpose(0, 1))


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

    def __init__(self, path, *, device="cpu", inference_backend="legacy"):
        if inference_backend not in ("legacy", "vectorized"):
            raise ValueError("inference_backend must be legacy or vectorized")
        self.path = Path(path)
        self.inference_backend = inference_backend
        self.runtime_options = {"device": device, "inference_backend": inference_backend}
        self._position_cache = {}
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
        batch = self.batch(state, images)
        if self.inference_backend == "vectorized":
            self.policy.eval()
            prediction = vectorized_act_actions(self.policy.model, batch, self.camera_keys,
                                                self._position_cache)
        else:
            prediction = self.policy.predict_action_chunk(batch)
        output = self.postprocessor(prediction)
        actions = output.detach().cpu().numpy()
        expected = (1, self.config.chunk_size, self.config.output_features["action"].shape[0])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError("native ACT returned invalid action chunk")
        return actions[0].astype(np.float64)
