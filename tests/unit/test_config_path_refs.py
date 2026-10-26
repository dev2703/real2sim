"""Refuse canonical config keys that point at missing local contract files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from real2sim.config import REPO_ROOT, project_path

CONFIG_ROOT = REPO_ROOT / "configs"
# Keys that name required inputs (not output dirs or free-form provenance prose).
REQUIRED_PATH_KEYS = {
    "task_config",
    "data_config",
    "model",
    "manifest",
    "fixed_manifest",
    "source_task",
}


def _walk(node, *, key: str | None = None):
    if isinstance(node, dict):
        for child_key, child in node.items():
            yield from _walk(child, key=str(child_key))
    elif isinstance(node, list):
        for child in node:
            yield from _walk(child, key=key)
    elif isinstance(node, str) and key in REQUIRED_PATH_KEYS:
        yield key, node


def _config_files() -> list[Path]:
    return sorted(
        path for path in CONFIG_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}
        and "__pycache__" not in path.parts
    )


def test_required_config_path_keys_resolve():
    missing: list[str] = []
    for path in _config_files():
        raw = path.read_text()
        data = yaml.safe_load(raw) if path.suffix in {".yaml", ".yml"} else json.loads(raw)
        if data is None:
            continue
        for key, value in _walk(data):
            target = project_path(value)
            if not target.exists():
                missing.append(f"{path.relative_to(REPO_ROOT)}:{key} -> {value}")
    assert not missing, "required config path refs must exist:\n" + "\n".join(missing)


def test_v2_canonical_sims_point_at_existing_exploratory_task():
    for name in ("sim_so101.yaml", "sim_panda.yaml", "data.yaml"):
        text = (CONFIG_ROOT / "v2" / name).read_text()
        assert "task_config: configs/v2/task.exploratory_20261022.yaml" in text
        assert project_path("configs/v2/task.exploratory_20261022.yaml").is_file()
