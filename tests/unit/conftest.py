from pathlib import Path

import pytest

from real2sim.data.synthetic import make_synthetic_dataset

CAMERAS = {
    "observation.images.cam_good": "moving",
    "observation.images.cam_black": "black",
    "observation.images.cam_frozen": "frozen",
}


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic") / "ds"
    return make_synthetic_dataset(root, cameras=CAMERAS, episode_lengths=(40, 50, 36))
