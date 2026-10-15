"""Opt-in SDPA for ACT without changing masks, tensor names or checkpoint weights."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn


class SDPAMultiheadAttention(nn.MultiheadAttention):
    """ACT discards attention weights; disable their materialization to enable SDPA."""

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        return super().forward(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=False,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal,
        )


def enable_sdpa(module: nn.Module) -> int:
    """Replace only MHA modules, retaining parameters, flags and upstream mask semantics."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SDPAMultiheadAttention):
            continue
        if isinstance(child, nn.MultiheadAttention):
            replacement = SDPAMultiheadAttention(
                child.embed_dim, child.num_heads, dropout=child.dropout,
                bias=child.in_proj_bias is not None, add_bias_kv=child.bias_k is not None,
                add_zero_attn=child.add_zero_attn, kdim=child.kdim, vdim=child.vdim,
                batch_first=child.batch_first,
                device="meta", dtype=child.out_proj.weight.dtype,
            )
            # Meta construction consumes no RNG and allocates no temporary weight storage.
            # Called before creating an optimizer; checkpoint names and values are unchanged.
            replacement.load_state_dict(child.state_dict(), assign=True)
            for key, parameter in child.named_parameters():
                replacement.get_parameter(key).requires_grad_(parameter.requires_grad)
            replacement.train(child.training)
            setattr(module, name, replacement)
            count += 1
        else:
            count += enable_sdpa(child)
    return count


def validate_attention(config):
    if config.attention_backend not in {"legacy", "sdpa"}:
        raise ValueError("attention_backend must be legacy or sdpa")
    if config.dim_model % config.n_heads:
        raise ValueError("dim_model must be divisible by n_heads")


@PreTrainedConfig.register_subclass("act_experiment")
@dataclass
class ExperimentACTConfig(ACTConfig):
    attention_backend: str = "sdpa"
    freeze_backbone: bool = False

    def __post_init__(self):
        super().__post_init__()
        validate_attention(self)


class ExperimentACTPolicy(ACTPolicy):
    config_class = ExperimentACTConfig
    name = "act_experiment"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if config.attention_backend == "sdpa":
            enable_sdpa(self.model)
        if config.freeze_backbone:
            self.model.backbone.requires_grad_(False)
            self.model.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.model.backbone.eval()
        return self
