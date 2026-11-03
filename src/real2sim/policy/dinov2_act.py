"""ACT with HF DINOv2 spatial features and explicitly bounded fine-tuning.

No external weights/network are needed when reloading a saved ACT checkpoint.
The original ACT transformer, VAE, chunk execution and action interface are reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn.functional as F
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from transformers import Dinov2Config, Dinov2Model

from real2sim.policy.attention import validate_attention
from real2sim.policy.vision import SpatialACTPolicy, SpatialBackbone


@PreTrainedConfig.register_subclass("act_dinov2")
@dataclass
class DinoACTConfig(ACTConfig):
    vision_backbone: str = "dinov2"
    pretrained_backbone_weights: str | None = None
    dinov2_model_id: str = "facebook/dinov2-small"
    dinov2_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    dinov2_pretrained_path: str = ""
    dinov2_trainable_blocks: int = 2
    dinov2_config: dict = field(default_factory=dict)
    attention_backend: str = "legacy"
    freeze_backbone: bool = False

    def __post_init__(self):
        if self.vision_backbone != "dinov2" or self.pretrained_backbone_weights is not None:
            raise ValueError("DINO ACT requires vision_backbone=dinov2 and no ResNet weights")
        # Run every upstream timing/feature check, bypassing only its ResNet name restriction.
        self.vision_backbone = "resnet18"
        try:
            super().__post_init__()
        finally:
            self.vision_backbone = "dinov2"
        layers = self.dinov2_config.get("num_hidden_layers", 12)
        if not 0 <= self.dinov2_trainable_blocks <= layers:
            raise ValueError("invalid number of unfrozen DINO blocks")
        if not self.dinov2_config:
            raise ValueError("DINO architecture must be embedded in the saved config")
        validate_attention(self)
        if self.freeze_backbone and self.dinov2_trainable_blocks:
            raise ValueError("freeze_backbone requires dinov2_trainable_blocks=0")


class DinoSpatialBackbone(SpatialBackbone):
    def __init__(self, config: DinoACTConfig, *, initialize_pretrained=False):
        super().__init__()
        hf_config = Dinov2Config(**config.dinov2_config)
        hf_config._attn_implementation = "sdpa"
        if initialize_pretrained:
            if not config.dinov2_pretrained_path:
                raise ValueError("a pinned local DINO snapshot is required for initialization")
            self.dino = Dinov2Model.from_pretrained(
                config.dinov2_pretrained_path, config=hf_config, local_files_only=True,
            )
        else:
            self.dino = Dinov2Model(hf_config)
        self.patch_size = hf_config.patch_size
        self.hidden_size = hf_config.hidden_size
        self.configure_training(config.dinov2_trainable_blocks)

    def _vision_parts(self):
        return (self.dino.embeddings,), self.dino.encoder.layer, self.dino.layernorm

    def forward(self, image):
        # Input is already normalized exactly once with ImageNet mean/std, also
        # used by the official DINO processor. Preserve the whole 192x256 view.
        # Replicate only the bottom/right edges to 196x266 (14x19 patches).
        h, w = image.shape[-2:]
        p = self.patch_size
        padded = F.pad(image, (0, -w % p, 0, -h % p), mode="replicate")
        tokens = self.dino(pixel_values=padded).last_hidden_state[:, 1:]
        rows, cols = padded.shape[-2] // p, padded.shape[-1] // p
        features = tokens.transpose(1, 2).reshape(image.shape[0], self.hidden_size, rows, cols)
        return {"feature_map": features}


class DinoACTPolicy(SpatialACTPolicy):
    config_class = DinoACTConfig
    name = "act_dinov2"
    backbone_class = DinoSpatialBackbone


def warmstart_nonvisual(policy, parent):
    """Copy ALL nonvisual ACT tensors, refusing silent shape/missing-key mismatches."""
    excluded = ("model.backbone.", "model.encoder_img_feat_input_proj.")
    incoming = {k: v for k, v in parent.state_dict().items() if not k.startswith(excluded)}
    expected = {k for k in policy.state_dict() if not k.startswith(excluded)}
    if set(incoming) != expected:
        raise ValueError("parent and DINO policy have different nonvisual ACT schemas")
    result = policy.load_state_dict(incoming, strict=False)
    if result.unexpected_keys or any(not k.startswith(excluded) for k in result.missing_keys):
        raise ValueError("unexpected missing/unmatched ACT warm-start weights")
    return {"copied_tensors": len(incoming), "excluded_prefixes": list(excluded)}
