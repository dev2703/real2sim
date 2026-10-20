"""Recover pruned files from local Git history or a verified local metadata archive."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from contextlib import ExitStack
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
    unavailable = [r["path"] for r in rows if not r.get("git_blob") and not r.get("archive")]
    if a.out_dir is not None:
        if unavailable:
            raise ValueError(f"not archived; see manifest recovery notes: {unavailable}")
        root = a.out_dir.resolve()
        targets = []
        for row in rows:
            target = (root / row["path"]).resolve()
            if not target.is_relative_to(root) or target.exists():
                raise ValueError(f"refusing to overwrite or escape destination: {target}")
            targets.append(target)
        with ExitStack() as stack:
            archives = {}
            for row in rows:
                if row.get("archive") and row["archive"] not in archives:
                    archives[row["archive"]] = stack.enter_context(tarfile.open(row["archive"], "r:gz"))

            def payload(row):
                if row.get("git_blob"):
                    return subprocess.check_output(["git", "cat-file", "blob", row["git_blob"]])
                archive = archives[row["archive"]]
                member = archive.getmember(row["archive_member"])
                if not member.isfile():
                    raise ValueError(f"archive member must be a regular file: {member.name}")
                # Read only the named member; never use tar extraction or follow archive links.
                with archive.extractfile(member) as f:
                    return f.read()

            # Verify all requested contents before writing any destination.
            for row in rows:
                if hashlib.sha256(payload(row)).hexdigest() != row["sha256"]:
                    raise ValueError(f"archive checksum mismatch: {row['path']}")
            for row, target in zip(rows, targets, strict=True):
                data = payload(row)
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"archive changed while restoring: {row['path']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    f.write(data)
    print(json.dumps({"restored": a.out_dir is not None, "files": len(rows),
                      "bytes": sum(r["bytes"] for r in rows), "paths": [r["path"] for r in rows],
                      "unavailable": unavailable}, indent=2))


if __name__ == "__main__":
    main()
