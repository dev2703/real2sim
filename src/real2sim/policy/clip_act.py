"""CLIP spatial-patch ACT. Saved checkpoints reload without network access."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from transformers import CLIPVisionConfig, CLIPVisionModel

from real2sim.policy.attention import validate_attention
from real2sim.policy.vision import SpatialACTPolicy, SpatialBackbone


@PreTrainedConfig.register_subclass("act_clip")
@dataclass
class ClipACTConfig(ACTConfig):
    vision_backbone: str = "clip"
    pretrained_backbone_weights: str | None = None
    clip_model_id: str = "openai/clip-vit-base-patch16"
    clip_revision: str = "57e6b9622bd450f20321a1e587a7778158ea0610"
    clip_pretrained_path: str = ""
    clip_trainable_blocks: int = 0
    clip_config: dict = field(default_factory=dict)
    attention_backend: str = "legacy"
    freeze_backbone: bool = False

    def __post_init__(self):
        if self.vision_backbone != "clip" or self.pretrained_backbone_weights is not None:
            raise ValueError("CLIP requires vision_backbone=clip and no ResNet weights")
        self.vision_backbone = "resnet18"
        try:
            super().__post_init__()
        finally:
            self.vision_backbone = "clip"
        if not self.clip_config:
            raise ValueError("CLIP vision architecture must be embedded in the saved config")
        if not 0 <= self.clip_trainable_blocks <= self.clip_config["num_hidden_layers"]:
            raise ValueError("invalid number of trainable CLIP blocks")
        if self.freeze_backbone and self.clip_trainable_blocks:
            raise ValueError("freeze_backbone requires clip_trainable_blocks=0")
        validate_attention(self)


class ClipSpatialBackbone(SpatialBackbone):
    def __init__(self, config, *, initialize_pretrained=False):
        super().__init__()
        hf = CLIPVisionConfig(**config.clip_config)
        hf._attn_implementation = "sdpa"
        if initialize_pretrained:
            if not config.clip_pretrained_path:
                raise ValueError("a pinned local CLIP snapshot is required")
            self.clip = CLIPVisionModel.from_pretrained(
                config.clip_pretrained_path, config=hf, local_files_only=True,
                use_safetensors=True,
            )
        else:
            self.clip = CLIPVisionModel(hf)
        self.hidden_size, self.patch_size = hf.hidden_size, hf.patch_size
        # Upstream dataset/inference already applied ImageNet normalization.
        # Convert that tensor to CLIP normalization, preserving the full view.
        for name, values in (
            ("imagenet_mean", [.485, .456, .406]), ("imagenet_std", [.229, .224, .225]),
            ("clip_mean", [.48145466, .4578275, .40821073]),
            ("clip_std", [.26862954, .26130258, .27577711]),
        ):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1), persistent=False)
        self.configure_training(config.clip_trainable_blocks)

    @property
    def vision(self):
        # Transformers 5 flattened CLIPVisionModel; retain older checkpoint support.
        return getattr(self.clip, "vision_model", self.clip)

    def _vision_parts(self):
        vision = self.vision
        return (vision.embeddings, vision.pre_layrnorm), vision.encoder.layers, vision.post_layernorm

    def preprocess(self, image):
        rgb = image * self.imagenet_std + self.imagenet_mean
        image = (rgb - self.clip_mean) / self.clip_std
        h, w = image.shape[-2:]
        return F.pad(image, (0, -w % self.patch_size, 0, -h % self.patch_size), mode="replicate")

    def forward(self, image):
        image = self.preprocess(image)
        tokens = self.clip(pixel_values=image, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        tokens = self.vision.post_layernorm(tokens)
        rows, cols = image.shape[-2] // self.patch_size, image.shape[-1] // self.patch_size
        return {"feature_map": tokens.transpose(1, 2).reshape(len(image), self.hidden_size, rows, cols)}


class ClipACTPolicy(SpatialACTPolicy):
    config_class = ClipACTConfig
    name = "act_clip"
    backbone_class = ClipSpatialBackbone
