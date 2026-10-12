import json
import shutil

import numpy as np

from real2sim.data import audit
from real2sim.data.quality import (
    FrameMetrics,
    frozen_runs,
    label_stream,
    motion_lag,
    summarize_episode,
)
from real2sim.data.synthetic import make_synthetic_dataset


def _frames(kind: str, n: int, rng):
    m = FrameMetrics()
    for i in range(n):
        if kind == "black":
            rgb = np.zeros((180, 320, 3), np.uint8)
        elif kind == "frozen":
            rgb = np.full((180, 320, 3), 120, np.uint8)
            rgb[::7, :] = 40
        else:
            rgb = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
            rgb[60:120, 100:150] = (220, 230, 40)  # ball
        m.add(rgb, i / 30)
    return m


def test_frozen_runs_detects_long_static_segments():
    d = np.array([np.nan] + [0.1] * 20 + [3.0] * 5 + [0.1] * 10)
    assert frozen_runs(d, 0.5, 15) == [(1, 21)]
    assert frozen_runs(d, 0.5, 40) == []


def test_black_episode_is_unusable():
    s = summarize_episode(_frames("black", 30, np.random.default_rng(0)), n_expected=30)
    assert s["label"] == "UNUSABLE"
    assert any("black/blank" in r for r in s["reasons"])


def test_frozen_episode_is_unusable():
    s = summarize_episode(_frames("frozen", 40, np.random.default_rng(0)), n_expected=40)
    assert s["label"] == "UNUSABLE"
    assert s["frozen_fraction"] > 0.8


def test_noisy_moving_episode_is_usable():
    s = summarize_episode(_frames("good", 30, np.random.default_rng(0)), n_expected=30)
    assert s["label"] == "USABLE", s["reasons"]
    assert s["ball_visible_fraction"] == 1.0


def test_frame_deficit_marks_partial():
    s = summarize_episode(_frames("good", 30, np.random.default_rng(0)), n_expected=30, video_frame_deficit=-2)
    assert s["label"] == "PARTIALLY_USABLE"
    assert any("misalignment" in r for r in s["reasons"])


def test_stream_label_rules():
    good = {"label": "USABLE"}
    part = {"label": "PARTIALLY_USABLE"}
    bad = {"label": "UNUSABLE"}
    assert label_stream({0: good, 1: good})[0] == "USABLE"
    assert label_stream({0: good, 1: part})[0] == "PARTIALLY_USABLE"
    assert label_stream({0: bad, 1: bad, 2: good})[0] == "UNUSABLE"
    assert label_stream({})[0] == "UNUSABLE"


def test_motion_lag_recovers_known_shift():
    rng = np.random.default_rng(1)
    b = rng.random(400)
    a = np.r_[np.zeros(4), b[:-4]]  # image motion trails joints by 4 frames
    lag, corr = motion_lag(a, b, max_lag=10)
    assert lag == 4 and corr > 0.9
    assert motion_lag(np.ones(400), b)[0] is None


def test_audit_cli_end_to_end(synthetic_root, tmp_path):
    out = tmp_path / "audit"
    rc = audit.main(
        ["--local-root", str(synthetic_root), "--repo-id", "synthetic/test", "--revision", "abc", "--out", str(out), "--n-val", "1"]
    )
    assert rc == 0
    for name in ("dataset_report.json", "camera_quality.json", "camera_montage.html", "split.json", "episodes.csv"):
        assert (out / name).is_file(), name
    cq = json.loads((out / "camera_quality.json").read_text())
    assert cq["labels"]["observation.images.cam_good"] == "USABLE", cq["streams"]["observation.images.cam_good"]["reasons"]
    assert cq["labels"]["observation.images.cam_black"] == "UNUSABLE"
    assert cq["labels"]["observation.images.cam_frozen"] == "UNUSABLE"
    assert cq["selected_camera_keys"] == ["observation.images.cam_good"]
    good = cq["streams"]["observation.images.cam_good"]
    assert good["frames_decoded"] == good["frames_expected"] == 126
    assert good["video_frame_deficit_total"] == 0
    rep = json.loads((out / "dataset_report.json").read_text())
    assert rep["source"]["revision"] == "abc"
    assert rep["totals"]["episodes"] == 3
    assert rep["timing"]["non_uniform_dt_episodes"] == []
    assert rep["split"]["n_val"] == 1 and rep["split"]["n_train"] == 2
    assert set(rep["state_action"]["per_joint"]) == set(rep["features"]["action"]["names"])
    thumbs = list((out / "montage" / "cam_good").glob("ep000_*.jpg"))
    assert {p.stem.split("_")[1] for p in thumbs} == {"first", "mid", "last"}


def test_missing_video_file_is_flagged(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", cameras={"observation.images.cam": "moving"}, episode_lengths=(20, 20))
    shutil.rmtree(root / "videos")
    out = tmp_path / "out"
    audit.main(["--local-root", str(root), "--out", str(out), "--n-val", "1"])
    cq = json.loads((out / "camera_quality.json").read_text())
    assert cq["labels"]["observation.images.cam"] == "UNUSABLE"
    assert "missing" in cq["streams"]["observation.images.cam"]["episodes"]["0"]["reasons"][0]
    assert cq["selected_camera_keys"] == []
