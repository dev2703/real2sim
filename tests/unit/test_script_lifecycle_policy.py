import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
POLICY = ROOT / "docs/script_lifecycle_policy.md"
SCRIPTS = ROOT / "scripts"


def _supported_cli_entries() -> set[str]:
    text = README.read_text()
    section = re.search(r"^## Supported CLIs\s*$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE)
    assert section, "README must include a '## Supported CLIs' section"
    return set(re.findall(r"`(scripts/[^`]+\.py)`", section.group(1)))


def test_policy_document_covers_required_rules():
    text = POLICY.read_text()
    for phrase in (
        "src/real2sim/",
        "tests/unit/",
        "date-stamped",
        "restore-manifest",
        "README.md",
        "Supported CLIs",
    ):
        assert phrase in text


def test_readme_supported_cli_entries_exist():
    entries = _supported_cli_entries()
    assert entries, "README supported CLI list must not be empty"
    missing = [path for path in entries if not (ROOT / path).exists()]
    assert not missing, f"README references missing scripts: {missing}"


def test_script_guardrails_hold():
    files = list(SCRIPTS.glob("*.py"))
    diag_files = [path for path in files if path.name.startswith("diag_")]
    assert len(files) <= 120
    assert len(diag_files) <= 32
