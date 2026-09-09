"""NDI Aurora acquisition and motor-clock-synchronised CSV recording.

The Aurora hardware is allowed to run at its native update rate.  During a
recording, the newest complete NDI frame is sampled once for every accepted
motor sample.  The CSV therefore has exactly the motor sampling cadence while
``em_frame_number``, ``new_em_frame`` and ``em_age_ms`` preserve the true NDI
timing (repeated hardware frames are never presented as new measurements).
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EmToolPose:
    port_handle: int
    tracker_timestamp_s: float
    frame_number: int
    transform: np.ndarray
    quality: float
    valid: bool


@dataclass(frozen=True)
class EmFrame:
    sequence: int
    host_ns: int
    tools: tuple[EmToolPose, ...]
    status: str = "ok"


def list_ndi_serial_ports() -> list[dict[str, str]]:
    """Return serial ports, prioritising ports that identify as NDI hardware."""

    try:
        from serial.tools import list_ports
    except Exception:
        return []
    result = [
        {
            "device": str(port.device),
            "description": str(port.description or "Serial device"),
            "hwid": str(port.hwid or ""),
        }
        for port in list_ports.comports()
    ]
    result.sort(
        key=lambda item: (
            0 if "NDI" in item["description"].upper() or "DA74" in item["hwid"].upper() else 1,
            item["device"],
        )
    )
    return result


def _scalar(values: Any, index: int, default: float = math.nan) -> float:
    try:
        value = values[index]
        array = np.asarray(value, dtype=float).reshape(-1)
        return float(array[0]) if array.size else float(default)
    except Exception:
        return float(default)


def _integer(values: Any, index: int, default: int = -1) -> int:
    value = _scalar(values, index, float(default))
    return int(value) if np.isfinite(value) else int(default)


def _quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to a normalised (x, y, z, w) quaternion."""

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(max(trace + 1.0, 0.0)) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray((qx, qy, qz, qw), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        return math.nan, math.nan, math.nan, math.nan
    quaternion /= norm
    return tuple(float(value) for value in quaternion)


def _put_latest(target_queue, value) -> None:
    """Publish without ever blocking the hardware acquisition loop."""

    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def _ndi_process_main(settings: dict[str, Any], frame_queue, status_queue, stop_event) -> None:
    """Own the serial driver in a child process so it cannot stall camera/UI threads."""

    tracker = None
    failed = False
    try:
        from sksurgerynditracker.nditracker import NDITracker

        tracker = NDITracker(settings)
        tracker.start_tracking()
        _put_latest(status_queue, ("tracking", ""))
        sequence = 0
        while not stop_event.is_set():
            handles, timestamps, frame_numbers, tracking, quality = tracker.get_frame()
            host_ns = time.perf_counter_ns()
            tools: list[EmToolPose] = []
            for index, handle in enumerate(handles or []):
                try:
                    matrix = np.asarray(tracking[index], dtype=float)
                except Exception:
                    matrix = np.full((4, 4), np.nan, dtype=float)
                valid = matrix.shape == (4, 4) and bool(np.isfinite(matrix).all())
                if matrix.shape != (4, 4):
                    matrix = np.full((4, 4), np.nan, dtype=float)
                else:
                    matrix = matrix.copy()
                tools.append(
                    EmToolPose(
                        port_handle=int(handle),
                        tracker_timestamp_s=_scalar(timestamps, index),
                        frame_number=_integer(frame_numbers, index),
                        transform=matrix,
                        quality=_scalar(quality, index),
                        valid=valid,
                    )
                )
            sequence += 1
            _put_latest(frame_queue, EmFrame(sequence, host_ns, tuple(tools)))
    except Exception as exc:
        failed = True
        _put_latest(status_queue, ("error", str(exc)))
    finally:
        if tracker is not None:
            try:
                tracker.stop_tracking()
            except Exception:
                pass
            try:
                tracker.close()
            except Exception:
                pass
        if not failed:
            _put_latest(status_queue, ("disconnected", ""))


class NdiEmSource:
    """Non-blocking owner of an NDI Aurora child process."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._receiver_stop = threading.Event()
        self._receiver_thread: threading.Thread | None = None
        self._process = None
        self._process_stop = None
        self._frame_queue = None
        self._status_queue = None
        self._closing = False
        self._latest: EmFrame | None = None
        self._state = "disconnected"
        self._error: str | None = None
        self._sequence = 0
        self._frame_times_ns: list[int] = []
        self.serial_port = str(config.get("serial_port", ""))
        self.rom_files = [str(item) for item in config.get("rom_files", [])]
        self.baud_rate = int(config.get("baud_rate", 9600))

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def connected(self) -> bool:
        return self.state == "tracking"

    @property
    def active(self) -> bool:
        return self.state in {"connecting", "tracking"}

    def connect(
        self,
        serial_port: str | None = None,
        rom_files: Iterable[str | Path] | None = None,
        baud_rate: int | None = None,
    ) -> None:
        if self.active:
            return
        port = str(serial_port or self.serial_port).strip()
        if not port:
            raise ValueError("请选择NDI串口")
        raw_roms = list(rom_files) if rom_files is not None else list(self.rom_files)
        resolved_roms = [str(Path(item).expanduser().resolve()) for item in raw_roms if str(item).strip()]
        missing = [path for path in resolved_roms if not Path(path).is_file()]
        if not resolved_roms:
            raise ValueError("请选择至少一个NDI工具ROM文件")
        if missing:
            raise FileNotFoundError(f"NDI ROM文件不存在: {missing[0]}")
        self.serial_port = port
        self.rom_files = resolved_roms
        self.baud_rate = int(baud_rate or self.baud_rate)
        self.config.update(
            serial_port=self.serial_port,
            rom_files=list(self.rom_files),
            baud_rate=self.baud_rate,
        )
        self._receiver_stop.clear()
        self._closing = False
        with self._lock:
            self._state = "connecting"
            self._error = None
            self._latest = None
            self._sequence = 0
            self._frame_times_ns = []
        settings = {
            "tracker type": "aurora",
            "serial port": self.serial_port,
            "baud rate": self.baud_rate,
            "tool ports": None,
            "romfiles": list(self.rom_files),
        }
        context = mp.get_context("spawn")
        self._frame_queue = context.Queue(maxsize=4)
        self._status_queue = context.Queue(maxsize=8)
        self._process_stop = context.Event()
        self._process = context.Process(
            target=_ndi_process_main,
            args=(settings, self._frame_queue, self._status_queue, self._process_stop),
            name="ndi-aurora-process",
            daemon=True,
        )
        self._receiver_thread = threading.Thread(
            target=self._receive, name="ndi-aurora-receiver", daemon=True
        )
        self._receiver_thread.start()
        try:
            self._process.start()
        except Exception:
            self._receiver_stop.set()
            self._receiver_thread.join(timeout=1.0)
            self._receiver_thread = None
            raise

    def _receive(self) -> None:
        while not self._receiver_stop.is_set():
            received = False
            if self._status_queue is not None:
                while True:
                    try:
                        state, error = self._status_queue.get_nowait()
                    except queue.Empty:
                        break
                    received = True
                    with self._lock:
                        self._state = str(state)
                        self._error = str(error) if error else None
            if self._frame_queue is not None:
                newest = None
                while True:
                    try:
                        newest = self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
                if newest is not None:
                    received = True
                    with self._lock:
                        self._latest = newest
                        self._sequence = int(newest.sequence)
                        self._frame_times_ns.append(int(newest.host_ns))
                        if len(self._frame_times_ns) > 240:
                            del self._frame_times_ns[:-240]
            process = self._process
            if (
                process is not None and process.pid is not None and not process.is_alive()
                and not self._closing
            ):
                with self._lock:
                    if self._state not in {"error", "disconnected"}:
                        self._state = "error"
                        self._error = f"NDI采集进程异常退出（exit code {process.exitcode}）"
                return
            if not received:
                self._receiver_stop.wait(0.005)

    def latest(self) -> EmFrame | None:
        with self._lock:
            return self._latest

    def device_rate_hz(self) -> float:
        with self._lock:
            times = tuple(self._frame_times_ns)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) * 1e9 / (times[-1] - times[0])

    def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.state == "tracking":
                return True
            if self.state == "error":
                return False
            time.sleep(0.02)
        return self.state == "tracking"

    def close(self) -> None:
        self._closing = True
        if self._process_stop is not None:
            self._process_stop.set()
        process = self._process
        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            process.close()
        self._receiver_stop.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=1.0)
        self._receiver_thread = None
        for output_queue in (self._frame_queue, self._status_queue):
            if output_queue is not None:
                try:
                    output_queue.close()
                    output_queue.join_thread()
                except Exception:
                    pass
        self._frame_queue = None
        self._status_queue = None
        self._process_stop = None
        self._process = None
        with self._lock:
            self._state = "disconnected"
            self._error = None
        self._closing = False


class EmMotorCsvRecorder:
    """Write one EM row for every motor sample accepted by ``AxisSampler``."""

    FILE_NAME = "em_data_100hz.csv"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._axis_sampler = None
        self._source: NdiEmSource | None = None
        self._listener = None
        self._accepting = False
        self._session_dir: Path | None = None
        self.tick_count = 0
        self.row_count = 0
        self.dropped = 0
        self.unique_frame_count = 0
        self.first_motor_ns: int | None = None
        self.last_motor_ns: int | None = None
        self._last_em_sequence = -1
        self._ages_ms: list[float] = []
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._accepting

    @staticmethod
    def fields() -> list[str]:
        fields = [
            "sample_index", "motor_host_ns", "motor_elapsed_s",
            "em_host_ns", "em_age_ms", "em_sequence", "em_frame_number",
            "tracker_timestamp_s", "port_handle", "valid", "new_em_frame",
            "quality", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw",
        ]
        fields.extend(f"m{row}{column}" for row in range(4) for column in range(4))
        fields.extend(("tool_count", "all_tools_json", "status"))
        return fields

    def start(self, session_dir: str | Path, axis_sampler, source: NdiEmSource) -> Path:
        if self.active:
            raise RuntimeError("NDI EM记录已经在运行")
        self._session_dir = Path(session_dir).resolve()
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._axis_sampler = axis_sampler
        self._source = source
        self.tick_count = self.row_count = self.dropped = self.unique_frame_count = 0
        self.first_motor_ns = self.last_motor_ns = None
        self._last_em_sequence = -1
        self._ages_ms = []
        self.error = None
        # EM packets are tiny compared with video frames.  Keep this queue
        # unbounded so disk jitter cannot silently break the one-row-per-motor-
        # tick guarantee; normal 100 Hz sessions only enqueue a few kB/s.
        self._queue = queue.Queue()
        self._accepting = True
        self._thread = threading.Thread(target=self._writer_loop, name="ndi-em-csv-writer", daemon=True)
        self._thread.start()
        self._listener = self._on_motor_sample
        axis_sampler.add_listener(self._listener)
        return self._session_dir / self.FILE_NAME

    def _on_motor_sample(self, motor_sample) -> None:
        if not self._accepting or self._queue is None:
            return
        frame = self._source.latest() if self._source is not None else None
        try:
            self._queue.put_nowait((motor_sample, frame))
        except queue.Full:
            self.dropped += 1

    @staticmethod
    def _tool_payload(tool: EmToolPose) -> dict[str, Any]:
        matrix = np.asarray(tool.transform, dtype=float)
        quaternion = _quaternion_xyzw(matrix[:3, :3]) if tool.valid else (math.nan,) * 4
        return {
            "port_handle": tool.port_handle,
            "tracker_timestamp_s": tool.tracker_timestamp_s,
            "frame_number": tool.frame_number,
            "quality": tool.quality,
            "valid": tool.valid,
            "translation_mm": matrix[:3, 3].tolist(),
            "quaternion_xyzw": list(quaternion),
            "transform_row_major": matrix.reshape(-1).tolist(),
        }

    def _make_row(self, motor_sample, frame: EmFrame | None) -> dict[str, Any]:
        motor_ns = int(motor_sample.host_ns)
        if self.first_motor_ns is None:
            self.first_motor_ns = motor_ns
        self.last_motor_ns = motor_ns
        selected_handle = int(self.config.get("primary_handle", 10))
        tools = () if frame is None else frame.tools
        selected = next((tool for tool in tools if tool.port_handle == selected_handle), None)
        if selected is None and selected_handle < 0 and tools:
            selected = tools[0]
        em_ns = int(frame.host_ns) if frame is not None else 0
        age_ms = (motor_ns - em_ns) * 1e-6 if em_ns else math.nan
        if np.isfinite(age_ms):
            self._ages_ms.append(float(age_ms))
        sequence = int(frame.sequence) if frame is not None else -1
        is_new = sequence >= 0 and sequence != self._last_em_sequence
        if is_new:
            self.unique_frame_count += 1
            self._last_em_sequence = sequence
        matrix = (
            np.asarray(selected.transform, dtype=float)
            if selected is not None else np.full((4, 4), np.nan, dtype=float)
        )
        quaternion = (
            _quaternion_xyzw(matrix[:3, :3])
            if selected is not None and selected.valid else (math.nan,) * 4
        )
        payload = [self._tool_payload(tool) for tool in tools]
        row: dict[str, Any] = {
            "sample_index": self.tick_count,
            "motor_host_ns": motor_ns,
            "motor_elapsed_s": (motor_ns - self.first_motor_ns) * 1e-9,
            "em_host_ns": em_ns,
            "em_age_ms": age_ms,
            "em_sequence": sequence,
            "em_frame_number": selected.frame_number if selected is not None else -1,
            "tracker_timestamp_s": selected.tracker_timestamp_s if selected is not None else math.nan,
            "port_handle": selected.port_handle if selected is not None else selected_handle,
            "valid": int(selected.valid) if selected is not None else 0,
            "new_em_frame": int(is_new),
            "quality": selected.quality if selected is not None else math.nan,
            "tx_mm": matrix[0, 3], "ty_mm": matrix[1, 3], "tz_mm": matrix[2, 3],
            "qx": quaternion[0], "qy": quaternion[1], "qz": quaternion[2], "qw": quaternion[3],
            "tool_count": len(tools),
            "all_tools_json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "status": frame.status if frame is not None else (
                self._source.state if self._source is not None else "no_source"
            ),
        }
        for matrix_row in range(4):
            for matrix_column in range(4):
                row[f"m{matrix_row}{matrix_column}"] = matrix[matrix_row, matrix_column]
        self.tick_count += 1
        return row

    def _writer_loop(self) -> None:
        path = self._session_dir / self.FILE_NAME
        try:
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fields())
                writer.writeheader()
                while True:
                    packet = self._queue.get()
                    try:
                        if packet is None:
                            return
                        writer.writerow(self._make_row(*packet))
                        self.row_count += 1
                        if self.row_count % 100 == 0:
                            stream.flush()
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self.error = str(exc)

    def stop(self) -> Path | None:
        if not self.active:
            return self._session_dir
        self._accepting = False
        if self._axis_sampler is not None and self._listener is not None:
            self._axis_sampler.remove_listener(self._listener)
        self._listener = None
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        rate_hz = 0.0
        if (
            self.tick_count >= 2 and self.first_motor_ns is not None
            and self.last_motor_ns is not None and self.last_motor_ns > self.first_motor_ns
        ):
            rate_hz = (self.tick_count - 1) * 1e9 / (self.last_motor_ns - self.first_motor_ns)
        ages = np.asarray(self._ages_ms, dtype=float)
        summary = {
            "csv_file": self.FILE_NAME,
            "primary_handle": int(self.config.get("primary_handle", 10)),
            "motor_ticks": self.tick_count,
            "csv_rows": self.row_count,
            "motor_clock_rate_hz": rate_hz,
            "unique_ndi_frames": self.unique_frame_count,
            "reused_motor_ticks": max(0, self.tick_count - self.unique_frame_count),
            "ndi_native_rate_hz": self._source.device_rate_hz() if self._source is not None else 0.0,
            "em_age_ms_median": float(np.nanmedian(ages)) if ages.size else None,
            "em_age_ms_p95": float(np.nanpercentile(ages, 95)) if ages.size else None,
            "queue_dropped": self.dropped,
            "error": self.error,
        }
        if self._session_dir is not None:
            (self._session_dir / "em_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self._queue = None
        self._axis_sampler = None
        self._source = None
        return self._session_dir
