"""Capture two native-resolution views and follower readbacks without motor writes.

Defaults to a dry run. --capture explicitly opens the specified devices. Bypasses
SOFollower.connect/configure, calibration routines, and all action/torque writes.
The user positions the target/robot; this program never requests robot movement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


@dataclass(frozen=True)
class CaptureConfig:
    port: str
    iphone_camera_index: int
    top_camera_index: int
    calibration: Path
    out: Path
    duration_s: float = 30.
    sample_hz: float = 25.
    width: int = 1280
    height: int = 720
    camera_fps: float = 30.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def calibration_records(value):
    result = {}
    for name, record in value.items():
        row = asdict(record) if is_dataclass(record) else record
        result[name] = {key: int(row[key]) for key in CALIBRATION_FIELDS}
    return result


def plan(config):
    if not config.port or config.iphone_camera_index == config.top_camera_index:
        raise ValueError("explicit port and two distinct camera indices required")
    if min(config.iphone_camera_index, config.top_camera_index) < 0:
        raise ValueError("camera indices must be nonnegative")
    if not 0 < config.duration_s <= 300 or not 0 < config.sample_hz <= 60:
        raise ValueError("duration must be in (0,300] seconds and sample rate in (0,60] Hz")
    if min(config.width, config.height) <= 0 or not math.isfinite(config.camera_fps) or config.camera_fps <= 0:
        raise ValueError("positive source dimensions and camera FPS required")
    if config.out.exists():
        raise FileExistsError(config.out)
    expected = calibration_records(json.loads(config.calibration.read_text()))
    if set(expected) != set(JOINTS):
        raise ValueError("calibration must identify exactly the six SO-101 motors")
    for index, name in enumerate(JOINTS, 1):
        row = expected[name]
        if row["id"] != index or row["range_min"] >= row["range_max"] or row["drive_mode"] != 0:
            raise ValueError("requires the existing six-motor, drive-mode-zero SO calibration")
    return expected, {
        "schema": "passive_so101_calibration_capture_v1", "status": "dry_run_no_devices_opened",
        "requested": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "calibration_sha256": digest(config.calibration), "script_sha256": digest(__file__),
        "motor_operations": ["connect: ping/firmware read handshake", "read_calibration", "sync_read Present_Position", "disconnect disable_torque=False"],
        "motor_writes_requested": 0, "robot_commands_requested": 0,
        "physical_calibration_accepted": False, "torque_state": None,
        "camera_exposure_timestamps": None, "hardware_camera_sync": None,
        "clock": "host time.perf_counter; camera timestamps are driver decode/read completion, not exposure",
        "video_clock": "FFV1 MKV uses requested sample_hz as a viewing timeline. JSONL host timestamps are authoritative; CFR FPS does not assert acquisition cadence.",
        "dependency_environment": "Use the existing sibling lerobot/.venv (Feetech SDK and pyserial); real2sim/.venv does not contain those hardware extras.",
    }


def make_bus(config, expected):
    # Importing the motor stack occurs only after explicit --capture.
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {name: Motor(index, "sts3215", MotorNormMode.DEGREES if index < 6 else MotorNormMode.RANGE_0_100)
              for index, name in enumerate(JOINTS, 1)}
    return FeetechMotorsBus(config.port, motors,
                           calibration={name: MotorCalibration(**row) for name, row in expected.items()})


class CameraSource:
    def __init__(self, config, index):
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, width=config.width,
            height=config.height, fps=config.camera_fps, warmup_s=5))

    def connect(self):
        self.camera.connect()

    def snapshot(self):
        # The local driver records frame+host-read-completion timestamp together.
        # Its public read_latest() returns only the image; copying under the same
        # lock preserves the exact timestamp association without another read.
        with self.camera.frame_lock:
            frame, timestamp = self.camera.latest_frame, self.camera.latest_timestamp
            if frame is None or timestamp is None:
                raise RuntimeError("camera has no source frame/timestamp")
            return frame.copy(), float(timestamp)

    def describe(self):
        import cv2

        cap = self.camera.videocapture
        values = {"reported_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  "reported_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                  "reported_fps": cap.get(cv2.CAP_PROP_FPS)}
        return {**{key: float(value) if math.isfinite(value) and value > 0 else None
                   for key, value in values.items()},
                "lens": None, "zoom": None, "focus": None, "stabilization": None,
                "settings_note": "Camera dimensions/FPS requested explicitly; lens/zoom/focus/stabilization not set or independently reported."}

    def disconnect(self):
        if self.camera.is_connected or self.camera.thread is not None:
            self.camera.disconnect()


class LosslessVideo:
    def __init__(self, path, size, fps):
        import cv2

        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError("FFV1 MKV writer unavailable; no lossy fallback was substituted")

    def write(self, rgb):
        import cv2

        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def release(self):
        self.writer.release()


def capture(config, *, bus_factory=make_bus, camera_factory=CameraSource,
            video_factory=LosslessVideo, clock=time.perf_counter, sleep=time.sleep):
    expected, report = plan(config)
    config.out.mkdir(parents=True)
    snapshot = config.calibration.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != report["calibration_sha256"] or calibration_records(json.loads(snapshot)) != expected:
        raise ValueError("calibration file changed during planning; no devices opened")
    (config.out / "calibration_snapshot.json").write_bytes(snapshot)
    (config.out / "executed_capture.py").write_bytes(Path(__file__).read_bytes())
    report.update(status="connecting", created_at_utc=datetime.now(UTC).isoformat(),
                  on_device_calibration_before=None, on_device_calibration_after=None,
                  cameras={}, readback_rows=0, elapsed_capture_host_s=None,
                  failure=None, cleanup_errors=[])
    write_json(config.out / "SESSION.json", report)
    bus, cameras, writers = None, {}, {}
    previous, frame_counts = {}, {name: 0 for name in ("iphone", "top")}
    failure = None
    try:
        bus = bus_factory(config, expected)
        bus.connect()
        actual = calibration_records(bus.read_calibration())
        report["on_device_calibration_before"] = actual
        if actual != expected:
            raise ValueError("device calibration differs from the supplied file; nothing was written")
        for name, index in (("iphone", config.iphone_camera_index), ("top", config.top_camera_index)):
            cameras[name] = camera_factory(config, index)
            cameras[name].connect()
            report["cameras"][name] = {"index": index, "feature": f"observation.images.{name}",
                                       **cameras[name].describe(), "source_size_wh": None,
                                       "frames_written": 0, "video": f"{name}.mkv"}
        started = clock()
        report["status"] = "capturing"
        with (config.out / "readbacks.jsonl").open("w") as handle:
            while clock() - started < config.duration_s:
                tick_start = clock()
                read_start = clock()
                values = bus.sync_read("Present_Position", normalize=True, num_retry=2)
                read_end = clock()
                if set(values) != set(JOINTS) or not all(math.isfinite(float(value)) for value in values.values()):
                    raise ValueError("readback did not contain six finite measured positions")
                row = {"row_index": report["readback_rows"], "tick_start_host_s": tick_start,
                       "readback_start_host_s": read_start, "readback_end_host_s": read_end,
                       "follower_readback_degree_percent": {name: float(values[name]) for name in JOINTS},
                       "raw_encoder_ticks": None, "sent_action": None, "applied_action": None,
                       "camera_samples": {}}
                for name, camera in cameras.items():
                    image, timestamp = camera.snapshot()
                    now = clock()
                    if not math.isfinite(timestamp) or timestamp > now or now - timestamp > .5:
                        raise ValueError(f"{name} source frame timestamp is unavailable or stale")
                    if image.shape != (config.height, config.width, 3):
                        raise ValueError(f"{name} source size differs from the explicitly requested size")
                    if name not in writers:
                        writers[name] = video_factory(config.out / f"{name}.mkv", (config.width, config.height), config.sample_hz)
                        report["cameras"][name]["source_size_wh"] = [image.shape[1], image.shape[0]]
                    if name in previous and timestamp < previous[name]:
                        raise ValueError(f"{name} driver frame timestamp moved backwards")
                    fresh = timestamp != previous.get(name)
                    if fresh:
                        writers[name].write(image)
                        previous[name] = timestamp
                        frame_counts[name] += 1
                    row["camera_samples"][name] = {
                        "driver_frame_host_read_complete_s": timestamp,
                        "sample_retrieved_host_s": now, "frame_age_host_s": now - timestamp,
                        "new_video_frame_written": fresh, "video_frame_index": frame_counts[name] - 1,
                        "exposure_timestamp_s": None}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                report["readback_rows"] += 1
                deadline = started + report["readback_rows"] / config.sample_hz
                sleep(max(0., min(deadline, started + config.duration_s) - clock()))
        report["elapsed_capture_host_s"] = clock() - started
        report["status"] = "captured_not_physically_calibrated"
    except BaseException as exc:
        failure = exc
        report.update(status="failed_or_interrupted", failure=f"{type(exc).__name__}: {exc}")
    finally:
        for writer in writers.values():
            try:
                writer.release()
            except Exception as exc:
                report["cleanup_errors"].append(f"video: {type(exc).__name__}: {exc}")
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception as exc:
                report["cleanup_errors"].append(f"camera: {type(exc).__name__}: {exc}")
        if bus is not None and bus.is_connected:
            try:
                report["on_device_calibration_after"] = calibration_records(bus.read_calibration())
            except Exception as exc:
                report["cleanup_errors"].append(f"calibration read: {type(exc).__name__}: {exc}")
            finally:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception as exc:
                    report["cleanup_errors"].append(f"serial close: {type(exc).__name__}: {exc}")
        for name in report["cameras"]:
            report["cameras"][name]["frames_written"] = frame_counts[name]
        report["calibration_file_sha256_after"] = digest(config.calibration)
        report["calibration_file_unchanged"] = report["calibration_file_sha256_after"] == report["calibration_sha256"]
        report["observed_device_calibration_unchanged"] = (
            None if report["on_device_calibration_before"] is None or report["on_device_calibration_after"] is None
            else report["on_device_calibration_before"] == report["on_device_calibration_after"])
        if report["cleanup_errors"] or not report["calibration_file_unchanged"] or report["observed_device_calibration_unchanged"] is False:
            report["status"] = "failed_or_interrupted"
            if failure is None:
                failure = RuntimeError("cleanup or calibration consistency check failed; inspect SESSION.json")
                report["failure"] = str(failure)
        write_json(config.out / "SESSION.json", report)
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--iphone-camera-index", type=int, required=True)
    parser.add_argument("--top-camera-index", type=int, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=30.)
    parser.add_argument("--sample-hz", type=float, default=25.)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture", action="store_true", help="Explicitly open the specified devices")
    mode.add_argument("--dry-run", action="store_true", help="Validate/print the plan without device access (default)")
    args = parser.parse_args()
    config = CaptureConfig(**{key: value for key, value in vars(args).items() if key not in ("capture", "dry_run")})
    result = capture(config) if args.capture else plan(config)[1]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
