"""Fetch only the pinned model files; verify Git blob IDs and write a SHA-256 manifest.

Run once with network access. Runtime and tests use the local, verified assets only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

REVISION = "0059d4335f8156206f63a35662313385f7ad6d74"
MODELS = ("robotstudio_so101", "franka_emika_panda")


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("assets/menagerie"))
    args = parser.parse_args()
    tree = json.loads(fetch(
        f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{REVISION}?recursive=1"
    ))
    if tree.get("truncated"):
        raise RuntimeError("upstream tree is truncated")
    entries = [
        item for item in tree["tree"]
        if item["type"] == "blob" and item["path"].split("/")[0] in MODELS
        and Path(item["path"]).suffix.lower() not in {".png", ".jpg", ".gif", ".mp4"}
    ]

    def download(item: dict) -> dict:
        path = args.out / item["path"]
        payload = path.read_bytes() if path.exists() else fetch(
            f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/{item['path']}"
        )
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if blob != item["sha"] or len(payload) != item["size"]:
            raise ValueError(f"upstream blob mismatch: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {"path": item["path"], "bytes": len(payload), "git_blob_sha1": blob,
                "sha256": hashlib.sha256(payload).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = sorted(pool.map(download, entries), key=lambda row: row["path"])
    manifest = {"repository": "https://github.com/google-deepmind/mujoco_menagerie",
                "revision": REVISION, "files": files}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} files, {sum(row['bytes'] for row in files)} bytes at {REVISION}")


if __name__ == "__main__":
    main()
