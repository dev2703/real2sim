"""Shared ACT wiring and frozen-layer lifecycle for spatial vision encoders."""

import copy

from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn

from real2sim.policy.attention import enable_sdpa


class SpatialBackbone(nn.Module):
    """Keep pretrained embeddings and unselected encoder blocks in evaluation mode."""

    def _vision_parts(self):
        """Return frozen input modules, encoder blocks and the output normalization."""
        raise NotImplementedError

    def configure_training(self, trainable_blocks):
        self.trainable_blocks = trainable_blocks
        self.requires_grad_(False)
        _, blocks, norm = self._vision_parts()
        if trainable_blocks:
            for block in blocks[-trainable_blocks:]:
                block.requires_grad_(True)
            norm.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        frozen, blocks, norm = self._vision_parts()
        for module in (*frozen, *blocks[:len(blocks) - self.trainable_blocks]):
            module.eval()
        if not self.trainable_blocks:
            norm.eval()
        return self


class SpatialACTPolicy(ACTPolicy):
    """Replace ACT's visual modules while preserving its checkpoint parameter paths."""

    backbone_class: type[SpatialBackbone]

    def __init__(self, config, *, initialize_backbone=False, **kwargs):
        # The temporary ResNet must never download weights or enter the saved config.
        temporary = copy.deepcopy(config)
        temporary.vision_backbone = "resnet18"
        temporary.pretrained_backbone_weights = None
        super().__init__(temporary, **kwargs)
        self.config = self.model.config = config
        self.model.backbone = self.backbone_class(config, initialize_pretrained=initialize_backbone)
        self.model.encoder_img_feat_input_proj = nn.Conv2d(
            self.model.backbone.hidden_size, config.dim_model, kernel_size=1,
        )
        if config.attention_backend == "sdpa":
            enable_sdpa(self.model)
