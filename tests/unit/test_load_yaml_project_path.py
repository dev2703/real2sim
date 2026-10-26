from real2sim.config import REPO_ROOT, load_yaml


def test_load_yaml_resolves_repo_relative_paths_from_any_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = load_yaml("configs/task.yaml")
    assert isinstance(data, dict)
    assert "table" in data
    assert (REPO_ROOT / "configs" / "task.yaml").is_file()
