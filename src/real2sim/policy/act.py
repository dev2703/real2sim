"""Build LeRobot ``ACTConfig``/``ACTPolicy`` from our YAML and save/load checkpoints with manifests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from real2sim.policy.dataset import Normalizer

ACT_FIELDS = (
    "chunk_size",
    "n_action_steps",
    "vision_backbone",
    "pretrained_backbone_weights",
    "dim_model",
    "n_heads",
    "dim_feedforward",
    "n_encoder_layers",
    "n_decoder_layers",
    "use_vae",
    "latent_dim",
    "n_vae_encoder_layers",
    "dropout",
    "kl_weight",
    "optimizer_lr",
    "optimizer_lr_backbone",
    "optimizer_weight_decay",
    "temporal_ensemble_coeff",
)


def build_act_config(model_cfg: dict, manifest: dict, device: str) -> ACTConfig:
    h, w, _ = manifest["image_shape_hwc"]
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(len(manifest["state_names"]),)),
    }
    for key in manifest["camera_keys"]:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(len(manifest["action_names"]),))}
    kwargs = {k: model_cfg[k] for k in ACT_FIELDS if k in model_cfg}
    dino_fields = {"dinov2_model_id", "dinov2_revision", "dinov2_pretrained_path",
                   "dinov2_trainable_blocks", "dinov2_config"}
    clip_fields = {k.replace("dinov2", "clip") for k in dino_fields}
    experiment_fields = {"attention_backend", "freeze_backbone"}
    is_dino = model_cfg.get("vision_backbone") == "dinov2"
    is_clip = model_cfg.get("vision_backbone") == "clip"
    unknown = set(model_cfg) - set(ACT_FIELDS) - experiment_fields - (dino_fields if is_dino else clip_fields if is_clip else set())
    if unknown:
        raise ValueError(f"unknown ACT fields in config: {sorted(unknown)}")
    config_class = ACTConfig
    if experiment_fields & model_cfg.keys():
        from real2sim.policy.attention import ExperimentACTConfig

        config_class = ExperimentACTConfig
        kwargs.update({k: model_cfg[k] for k in experiment_fields if k in model_cfg})
    if is_dino:
        from real2sim.policy.dinov2_act import DinoACTConfig

        config_class = DinoACTConfig
        kwargs.update({k: model_cfg[k] for k in dino_fields if k in model_cfg})
    if is_clip:
        from real2sim.policy.clip_act import ClipACTConfig

        config_class = ClipACTConfig
        kwargs.update({k: model_cfg[k] for k in clip_fields if k in model_cfg})
    return config_class(
        input_features=input_features,
        output_features=output_features,
        device=device,
        push_to_hub=False,
        **kwargs,
    )


def build_policy(config, *, initialize_backbone=True):
    if config.vision_backbone == "clip":
        from real2sim.policy.clip_act import ClipACTPolicy

        return ClipACTPolicy(config, initialize_backbone=initialize_backbone)
    if config.vision_backbone == "dinov2":
        from real2sim.policy.dinov2_act import DinoACTPolicy

        return DinoACTPolicy(config, initialize_backbone=initialize_backbone)
    from real2sim.policy.attention import ExperimentACTConfig, ExperimentACTPolicy

    if isinstance(config, ExperimentACTConfig):
        return ExperimentACTPolicy(config)
    return ACTPolicy(config)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(
    out: Path,
    policy: ACTPolicy,
    stats_path: Path,
    manifest: dict,
    run_info: dict,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    shutil.copy(stats_path, out / "stats.json")
    contract = {
        "camera_keys": manifest["camera_keys"],
        "image_shape_chw": [3, *manifest["image_shape_hwc"][:2]],
        "letterbox": manifest["letterbox"],
        "state_names": manifest["state_names"],
        "action_names": manifest["action_names"],
        "label_semantics": manifest["label_semantics"],
        "policy_hz": manifest["policy_hz"],
        "source": manifest["source"],
        "split": manifest["split"],
        "derived_files_sha256": manifest["files_sha256"],
        "prepare_version": manifest["prepare_version"],
    }
    for key in ("observation_variant", "experiment_split", "source_manifest_sha256",
                "contract_version", "action_representation", "controller_semantics",
                "calibration_sha256", "camera_transforms", "timing_basis", "transfer_shared_task_sha256"):
        if key in manifest:
            contract[key] = manifest[key]
    if policy.config.vision_backbone == "dinov2":
        contract["vision_encoder"] = {
            "kind": "dinov2_spatial_patches", "model_id": policy.config.dinov2_model_id,
            "revision": policy.config.dinov2_revision,
            "trainable_final_blocks": policy.config.dinov2_trainable_blocks,
            "preprocessing": "existing ImageNet normalization; right/bottom replicate-pad to patch multiple; no crop",
        }
    elif policy.config.vision_backbone == "clip":
        contract["vision_encoder"] = {
            "kind": "clip_spatial_patches", "model_id": policy.config.clip_model_id,
            "revision": policy.config.clip_revision,
            "trainable_final_blocks": policy.config.clip_trainable_blocks,
            "preprocessing": "ImageNet-to-CLIP normalization; replicate-pad to patch multiple; interpolated positions; no crop",
        }
    if run_info.get("paired_images"):
        contract["training_provenance"] = {
            "paired_images": run_info["paired_images"],
            "initial_checkpoint": run_info.get("initial_checkpoint"),
        }
    (out / "real2sim_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": run_info["step"],
                    "continuation": training_state}, out / "training_state.pt")


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ACTPolicy, Normalizer, dict]:
    path = Path(path)
    config_json = json.loads((path / "config.json").read_text())
    if config_json.get("type") == "small_chunk_head":
        from real2sim.policy.flow_matching import load_flow_checkpoint

        return load_flow_checkpoint(path, device)
    policy_class = ACTPolicy
    if config_json.get("type") == "act_dinov2":
        from real2sim.policy.dinov2_act import DinoACTPolicy

        policy_class = DinoACTPolicy
    elif config_json.get("type") == "act_clip":
        from real2sim.policy.clip_act import ClipACTPolicy

        policy_class = ClipACTPolicy
    elif config_json.get("type") == "act_experiment":
        from real2sim.policy.attention import ExperimentACTPolicy

        policy_class = ExperimentACTPolicy
    policy = policy_class.from_pretrained(str(path), strict=True, local_files_only=True)
    policy.to(device)
    policy.eval()
    contract = json.loads((path / "real2sim_contract.json").read_text())
    return policy, Normalizer.from_json(path / "stats.json"), contract
