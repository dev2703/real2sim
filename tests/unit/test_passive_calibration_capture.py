"""Passive capture never invokes calibration, motor writes, action or torque APIs."""

import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.capture_v2_passive_calibration import JOINTS, CaptureConfig, capture, plan


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        self.now += .00001
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeBus:
    def __init__(self, expected, *, mismatch=False, fail_connect=False):
        self.expected = expected
        self.mismatch = mismatch
        self.fail_connect = fail_connect
        self.is_connected = False
        self.calls = []

    def connect(self):
        self.calls.append("connect")
        self.is_connected = True
        if self.fail_connect:
            raise ConnectionError("fake handshake failed")

    def read_calibration(self):
        self.calls.append("read_calibration")
        rows = {key: dict(value) for key, value in self.expected.items()}
        if self.mismatch:
            rows["shoulder_lift"]["homing_offset"] += 1
        return rows

    def sync_read(self, name, *, normalize, num_retry):
        self.calls.append(("sync_read", name, normalize, num_retry))
        return dict.fromkeys(JOINTS, 0.)

    def disconnect(self, *, disable_torque):
        self.calls.append(("disconnect", disable_torque))
        assert disable_torque is False
        self.is_connected = False

    def __getattr__(self, name):
        raise AssertionError(f"unexpected motor operation: {name}")


class FakeCamera:
    def __init__(self, clock, *, fail=False):
        self.clock, self.fail = clock, fail
        self.calls = []

    def connect(self):
        self.calls.append("connect")

    def describe(self):
        return {"reported_width": 8, "reported_height": 6, "reported_fps": 30.}

    def snapshot(self):
        if self.fail:
            raise RuntimeError("fake source frame failed")
        return np.zeros((6, 8, 3), dtype=np.uint8), self.clock()

    def disconnect(self):
        self.calls.append("disconnect")


class FakeVideo:
    def __init__(self, *_args):
        self.released = False
        self.count = 0

    def write(self, _rgb):
        self.count += 1

    def release(self):
        self.released = True


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({name: {"id": index, "drive_mode": 0, "homing_offset": 0,
                                     "range_min": 0, "range_max": 4095}
                                for index, name in enumerate(JOINTS, 1)}))
    return CaptureConfig("explicit_fake_port", 7, 9, path, tmp_path / "capture",
                         duration_s=.11, sample_hz=25., width=8, height=6)


def factories(config, *, mismatch=False, camera_failure=False, connect_failure=False):
    expected, _ = plan(config)
    clock = Clock()
    bus = FakeBus(expected, mismatch=mismatch, fail_connect=connect_failure)
    cameras, videos = [], []

    def camera_factory(_config, _index):
        camera = FakeCamera(clock, fail=camera_failure and len(cameras) == 1)
        cameras.append(camera)
        return camera

    def video_factory(*args):
        video = FakeVideo(*args)
        videos.append(video)
        return video

    return bus, cameras, videos, {"bus_factory": lambda *_: bus,
        "camera_factory": camera_factory, "video_factory": video_factory,
        "clock": clock, "sleep": clock.sleep}


def test_dry_run_creates_no_output_or_hardware_data(config):
    _, report = plan(config)
    assert not config.out.exists()
    assert report["status"] == "dry_run_no_devices_opened"
    assert report["torque_state"] is None
    assert report["camera_exposure_timestamps"] is None
    assert report["physical_calibration_accepted"] is False
    with pytest.raises(ValueError, match="distinct"):
        plan(replace(config, top_camera_index=config.iphone_camera_index))


def test_capture_allows_only_reads_and_closes_without_torque_change(config):
    bus, cameras, videos, args = factories(config)
    report = capture(config, **args)
    assert report["status"] == "captured_not_physically_calibrated"
    assert report["observed_device_calibration_unchanged"]
    assert report["calibration_file_unchanged"]
    assert bus.calls[0] == "connect"
    assert bus.calls[-1] == ("disconnect", False)
    assert all(call == "connect" or call == "read_calibration" or
               call == ("sync_read", "Present_Position", True, 2) or
               call == ("disconnect", False) for call in bus.calls)
    assert all(camera.calls == ["connect", "disconnect"] for camera in cameras)
    assert all(video.count > 0 and video.released for video in videos)
    rows = [json.loads(line) for line in (config.out / "readbacks.jsonl").read_text().splitlines()]
    assert len(rows) == report["readback_rows"] > 0
    assert rows[0]["sent_action"] is None and rows[0]["applied_action"] is None
    assert rows[0]["camera_samples"]["top"]["exposure_timestamp_s"] is None
    assert (config.out / "calibration_snapshot.json").read_bytes() == config.calibration.read_bytes()


@pytest.mark.parametrize("failure", ["calibration", "camera", "connect"])
def test_failures_keep_evidence_and_cleanup_without_writes(config, failure):
    bus, cameras, videos, args = factories(config, mismatch=failure == "calibration",
        camera_failure=failure == "camera", connect_failure=failure == "connect")
    with pytest.raises((ValueError, RuntimeError, ConnectionError)):
        capture(config, **args)
    assert bus.calls[-1] == ("disconnect", False)
    assert not bus.is_connected
    assert all(camera.calls[-1] == "disconnect" for camera in cameras)
    assert all(video.released for video in videos)
    report = json.loads((config.out / "SESSION.json").read_text())
    assert report["status"] == "failed_or_interrupted"
    assert report["physical_calibration_accepted"] is False
    assert report["torque_state"] is None
    if failure in ("calibration", "connect"):
        assert not cameras
        assert not any(isinstance(call, tuple) and call[0] == "sync_read" for call in bus.calls)
