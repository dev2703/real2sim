"""Recover selected pruned experiment inputs from local Git history, without resetting code."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("docs/cleanup_manifest_20261020.json"))
    p.add_argument("--path", required=True, help="original exact path or directory prefix from manifest")
    p.add_argument("--out-dir", type=Path, help="destination root; omit to list selected files only")
    a = p.parse_args()
    prefix = a.path.rstrip("/")
    rows = [r for r in json.loads(a.manifest.read_text())["entries"]
            if r["path"] == prefix or r["path"].startswith(prefix + "/")]
    if not rows:
        raise ValueError("no archived paths match; consult the cleanup manifest")
    if a.out_dir is not None:
        root = a.out_dir.resolve()
        targets = []
        for row in rows:
            target = (root / row["path"]).resolve()
            if not target.is_relative_to(root) or target.exists():
                raise ValueError(f"refusing to overwrite or escape destination: {target}")
            targets.append(target)
        # Verify every requested blob before writing any destination.
        for row in rows:
            blob = subprocess.check_output(["git", "cat-file", "blob", row["git_blob"]])
            if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                raise ValueError(f"archive checksum mismatch: {row['path']}")
        for row, target in zip(rows, targets, strict=True):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(subprocess.check_output(["git", "cat-file", "blob", row["git_blob"]]))
    print(json.dumps({"restored": a.out_dir is not None, "files": len(rows),
                      "bytes": sum(r["bytes"] for r in rows), "paths": [r["path"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
