"""Optional buffered command/timing traces; no motor access or per-tick disk I/O."""

import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def position_snapshot(values: dict[str, Any]) -> dict[str, float | None]:
    """Copy scalar joint positions while excluding images and other observations."""
    result = {}
    for name, value in values.items():
        if name.endswith(".pos"):
            value = float(value)
            result[name] = value if math.isfinite(value) else None
    return result


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RecordingTrace:
    """Buffer one capture attempt and keep saved/discarded/interrupted outcomes distinct."""

    def __init__(self, root: str | Path, metadata: dict[str, Any]):
        """Create a unique local session directory and persist collection provenance."""
        self.root = Path(root) / f"session_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.active: dict[str, Any] | None = None
        self.path: Path | None = None
        _write(self.root / "SESSION.json", {
            "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
            "clock": "time.perf_counter_ns; monotonic host clock, not camera exposure time",
            "observation_semantics": "raw position read before this tick's send_action call",
            "sent_semantics": "returned software target after limiter; before bus quantization; not readback",
            "camera_capture_timestamps_available": False, "camera_frame_ids_available": False,
            "frames_semantics": "frame_index is a dataset row index; it is not a camera acquisition ID",
            "metadata": metadata,
        })

    def begin(self, episode_index: int) -> None:
        """Start a new attempt without overwriting a discarded attempt's trace."""
        if self.active is not None:
            raise RuntimeError("finish the current trace attempt before starting another")
        attempt = uuid.uuid4().hex
        self.path = self.root / f"episode_{episode_index:06d}_{attempt}.json"
        self.active = {"episode_index": episode_index, "attempt_id": attempt,
                       "status": "capturing", "frames": []}
        _write(self.root / "RUNNING.json", {"episode_index": episode_index, "attempt_id": attempt})

    def append(self, frame: dict[str, Any]) -> None:
        """Append one successfully recorded row in memory only."""
        if self.active is None:
            raise RuntimeError("begin a trace attempt before appending")
        frame = dict(frame)
        frame["frame_index"] = len(self.active["frames"])
        self.active["frames"].append(frame)

    def capture_finished(self) -> None:
        """Persist captured rows outside the control loop; dataset save is still pending."""
        if self.active is not None:
            self.active["status"] = "captured_pending_dataset_save"
            _write(self.path, self.active)

    def finish(self, status: str) -> None:
        """Associate a completed attempt with the dataset save/discard outcome."""
        if status not in {"saved", "discarded", "interrupted"}:
            raise ValueError(status)
        if self.active is not None:
            self.active["status"] = status
            _write(self.path, self.active)
            self.active = None
            self.path = None
        (self.root / "RUNNING.json").unlink(missing_ok=True)

    def close(self) -> None:
        """Retain partial rows and explicitly flag interruption instead of claiming success."""
        interrupted = self.active is not None
        if interrupted:
            self.finish("interrupted")
        _write(self.root / ("FAILED.json" if interrupted else "COMPLETE.json"), {
            "status": "interrupted" if interrupted else "trace_session_complete",
            "physical_task_success": "not_evaluated",
        })


def make_recording_trace(root: str, cfg: Any, robot: Any, teleop: Any,
                         recorder_path: str) -> RecordingTrace:
    """Record provenance after calibration, before the first control tick."""
    recorder = Path(recorder_path).resolve()
    try:
        commit = subprocess.run(["git", "-C", str(recorder.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    calibration = {}
    for name, device in (("robot", robot), ("teleop", teleop)):
        path = getattr(device, "calibration_fpath", None)
        if path is not None and Path(path).is_file():
            calibration[name] = {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    cameras = {name: {key: getattr(camera, key, None) for key in ("width", "height", "fps")}
               for name, camera in getattr(cfg.robot, "cameras", {}).items()}
    return RecordingTrace(root, {
        "dataset_repo_id": cfg.dataset.repo_id, "dataset_fps": cfg.dataset.fps,
        "robot_id": getattr(cfg.robot, "id", None), "teleop_id": getattr(cfg.teleop, "id", None),
        "robot_use_degrees": getattr(cfg.robot, "use_degrees", None),
        "teleop_use_degrees": getattr(cfg.teleop, "use_degrees", None),
        "max_relative_target": getattr(cfg.robot, "max_relative_target", None),
        "collector_commit": commit, "recorder_sha256": hashlib.sha256(recorder.read_bytes()).hexdigest(),
        "trace_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration": calibration, "cameras": cameras,
    })
