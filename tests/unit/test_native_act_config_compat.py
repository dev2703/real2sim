"""Load 0.6.2's default dtype without weakening native checkpoint validation."""

import json

import draccus
import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from real2sim.policy.native_lerobot_act import load_native_act_config


def write_config(path, **extra):
    payload = draccus.encode(ACTConfig(device="cpu"), PreTrainedConfig)
    payload.update(extra)
    (path / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


@pytest.mark.parametrize("include_null_dtype", [False, True])
def test_native_config_matches_upstream_defaults_without_rewriting_export(tmp_path, include_null_dtype):
    payload = write_config(tmp_path, **({"dtype": None} if include_null_dtype else {}))
    before = (tmp_path / "config.json").read_bytes()
    expected = dict(payload)
    expected.pop("dtype", None)
    loaded = load_native_act_config(tmp_path)
    assert draccus.encode(loaded, PreTrainedConfig) == expected
    assert (tmp_path / "config.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", False, 0])
def test_explicit_dtype_is_rejected_instead_of_silently_changing_semantics(tmp_path, dtype):
    write_config(tmp_path, dtype=dtype)
    before = (tmp_path / "config.json").read_bytes()
    with pytest.raises(ValueError, match="explicit dtype"):
        load_native_act_config(tmp_path)
    assert (tmp_path / "config.json").read_bytes() == before


@pytest.mark.parametrize("extra", [{"unknown_future_option": None},
                                  {"unknown_future_option": None, "dtype": None},
                                  {"dtype": None, "chunk_size": "invalid"}])
def test_compatibility_still_rejects_unknown_or_invalid_saved_semantics(tmp_path, extra):
    write_config(tmp_path, **extra)
    with pytest.raises(draccus.utils.DecodingError):
        load_native_act_config(tmp_path)


def test_dtype_compatibility_does_not_accept_another_policy_type(tmp_path):
    write_config(tmp_path, type="smolvla", dtype=None)
    with pytest.raises(ValueError, match="requires native ACT"):
        load_native_act_config(tmp_path)
