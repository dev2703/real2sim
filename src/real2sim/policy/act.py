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
    unknown = set(model_cfg) - set(ACT_FIELDS)
    if unknown:
        raise ValueError(f"unknown ACT fields in config: {sorted(unknown)}")
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        push_to_hub=False,
        **kwargs,
    )


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
    if run_info.get("paired_images"):
        contract["training_provenance"] = {
            "paired_images": run_info["paired_images"],
            "initial_checkpoint": run_info.get("initial_checkpoint"),
        }
    (out / "real2sim_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": run_info["step"]}, out / "training_state.pt")


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ACTPolicy, Normalizer, dict]:
    path = Path(path)
    policy = ACTPolicy.from_pretrained(str(path))
    policy.to(device)
    policy.eval()
    contract = json.loads((path / "real2sim_contract.json").read_text())
    return policy, Normalizer.from_json(path / "stats.json"), contract
