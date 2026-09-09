"""Read seven independent Trio axes without conflating DPOS and MPOS."""

from __future__ import annotations

from abc import ABC, abstractmethod
import multiprocessing as mp
import queue
import re
import time
from typing import Callable

import numpy as np

from .models import AxisSample


def map_native_axes(demand: np.ndarray, measured: np.ndarray, config: dict) -> np.ndarray:
    zero = np.asarray(config["zero_native"], dtype=float)
    scale = np.asarray(config["scale_m_per_native"], dtype=float)
    prefer_measured = bool(config.get("prefer_measured_position", True))
    source = (
        np.where(np.isfinite(measured), measured, demand)
        if prefer_measured
        else np.where(np.isfinite(demand), demand, measured)
    )
    displacement = (source - zero) * scale
    control = np.zeros(7, dtype=float)
    neutral = np.asarray(config["neutral_tendon_lengths_m"], dtype=float)
    signs = np.asarray(config.get("tendon_pull_sign", [1] * 6), dtype=float)
    control[:6] = neutral - signs * displacement[:6]
    control[6] = displacement[6]
    return control


class AxisSource(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def sample(self) -> AxisSample:
        raise NotImplementedError

    def empty_sample(self, host_ns: int | None = None, status: str = "unavailable") -> AxisSample:
        values = np.full(7, np.nan)
        return AxisSample(
            host_ns=time.perf_counter_ns() if host_ns is None else int(host_ns),
            demand_native=values,
            measured_native=values,
            control_m=map_native_axes(values, values, self.config),
            demand_valid=np.zeros(7, dtype=bool),
            measured_valid=np.zeros(7, dtype=bool),
            status=status,
        )


class NullAxisSource(AxisSource):
    def sample(self) -> AxisSample:
        return self.empty_sample()


_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _parse_trio_vector(response: str, count: int = 7) -> np.ndarray:
    values = np.asarray(
        [float(value) for value in _FLOAT_PATTERN.findall(str(response))],
        dtype=float,
    )
    if values.size != count:
        raise ValueError(
            f"Trio returned {values.size} values, expected {count}: {response!r}"
        )
    return values


class TrioFastAxisReader:
    """Read MPOS at full rate while periodically refreshing DPOS.

    The Trio 1.1/1.2 Python parameter-block binding rejects parameter names on
    Python 3.12. Two compact read-only BASIC queries avoid fourteen individual
    TCP round trips. MPOS is acquired for every sample; the DPOS cache age is
    explicitly included in ``AxisSample.status``.
    """

    def __init__(self, config: dict) -> None:
        refresh_hz = max(float(config.get("demand_refresh_hz", 20.0)), 1.0)
        self.demand_period_ns = int(1e9 / refresh_hz)
        self.demand = np.full(7, np.nan)
        self.demand_host_ns = 0
        self.demand_query = "?" + ",".join(
            f"DPOS AXIS({axis})" for axis in range(7)
        )
        self.measured_query = "?" + ",".join(
            f"MPOS AXIS({axis})" for axis in range(7)
        )

    def sample(self, connection: object, config: dict) -> AxisSample:
        query = getattr(connection, "ExecuteWithResponse", None)
        if not callable(query):
            return _sample_trio_individual(connection, config)

        errors: list[str] = []
        now_ns = time.perf_counter_ns()
        if self.demand_host_ns == 0 or now_ns - self.demand_host_ns >= self.demand_period_ns:
            try:
                self.demand = _parse_trio_vector(query(self.demand_query))
                self.demand_host_ns = time.perf_counter_ns()
            except Exception as exc:
                errors.append(f"DPOS:{exc}")

        measured = np.full(7, np.nan)
        measured_start_ns = time.perf_counter_ns()
        try:
            measured = _parse_trio_vector(query(self.measured_query))
        except Exception as exc:
            errors.append(f"MPOS:{exc}")
        measured_end_ns = time.perf_counter_ns()
        host_ns = (measured_start_ns + measured_end_ns) // 2
        demand = self.demand.copy()
        demand_age_ms = (
            (host_ns - self.demand_host_ns) * 1e-6
            if self.demand_host_ns
            else float("inf")
        )
        status = f"fast_query;dpos_age_ms={demand_age_ms:.1f}"
        if errors:
            status += ";" + ";".join(errors)
        return AxisSample(
            host_ns=host_ns,
            demand_native=demand,
            measured_native=measured,
            control_m=map_native_axes(demand, measured, config),
            demand_valid=np.isfinite(demand),
            measured_valid=np.isfinite(measured),
            status=status,
        )


class CallbackAxisSource(AxisSource):
    """Adapt an existing main-window Trio connection without owning it."""

    def __init__(self, config: dict, connection_getter: Callable[[], object | None]) -> None:
        super().__init__(config)
        self.connection_getter = connection_getter
        self.reader = TrioFastAxisReader(config)

    def sample(self) -> AxisSample:
        connection = self.connection_getter()
        if connection is None:
            return self.empty_sample(status="trio_disconnected")
        return sample_trio_connection(connection, self.config, self.reader)


def _trio_process_worker(endpoint: str, config: dict, output_queue, stop_event) -> None:
    """Own the Trio extension in another process so it cannot stall video."""
    connection = None
    try:
        import Trio_UnifiedApi as TUA

        def handler(event_type, integer_value, string_value) -> None:
            del event_type, integer_value, string_value

        key = endpoint.upper()
        if key == "PCMCAT":
            connection = TUA.TrioConnectionPCMCAT(handler)
        elif key == "FLEX7":
            connection = TUA.TrioConnectionFlex7(handler)
        else:
            connection = TUA.TrioConnectionTCP(handler, endpoint)
        connection.OpenConnection()
        reader = TrioFastAxisReader(config)
        period_s = 1.0 / max(float(config.get("sample_rate_hz", 100.0)), 1.0)
        deadline = time.perf_counter()
        while not stop_event.is_set():
            sample = reader.sample(connection, config)
            packet = (
                "sample",
                sample.host_ns,
                sample.demand_native,
                sample.measured_native,
                sample.control_m,
                sample.demand_valid,
                sample.measured_valid,
                sample.status,
            )
            try:
                output_queue.put_nowait(packet)
            except queue.Full:
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    output_queue.put_nowait(packet)
                except queue.Full:
                    pass
            deadline += period_s
            stop_event.wait(max(0.0, deadline - time.perf_counter()))
    except Exception as exc:
        try:
            output_queue.put_nowait(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        if connection is not None:
            try:
                connection.CloseConnection()
            except Exception:
                pass


class TrioAxisSource(AxisSource):
    def __init__(self, config: dict, endpoint: str | None = None) -> None:
        super().__init__(config)
        self.endpoint = endpoint or str(config.get("controller", "192.168.0.250"))
        self.connection = None
        self.events: list[str] = []
        self._process = None
        self._queue = None
        self._stop_event = None
        self._latest: AxisSample | None = None

    def connect(self) -> None:
        self.close()
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=256)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_trio_process_worker,
            args=(self.endpoint, dict(self.config), self._queue, self._stop_event),
            name="tdcr-trio-reader",
            daemon=True,
        )
        self._process.start()
        deadline = time.perf_counter() + float(self.config.get("connect_timeout_s", 8.0))
        while time.perf_counter() < deadline:
            try:
                packet = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue
            if packet[0] == "error":
                message = packet[1]
                self.close()
                raise RuntimeError(f"无法连接 Trio 控制器 {self.endpoint}: {message}")
            self._latest = self._packet_to_sample(packet)
            self.connection = True  # Compatibility flag; handle lives in child process.
            return
        self.close()
        raise RuntimeError(f"连接 Trio 控制器 {self.endpoint} 超时")

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._queue is not None:
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass
        self._process = None
        self._queue = None
        self._stop_event = None
        self._latest = None
        self.connection = None

    @staticmethod
    def _packet_to_sample(packet) -> AxisSample:
        return AxisSample(
            host_ns=int(packet[1]),
            demand_native=np.asarray(packet[2], dtype=float),
            measured_native=np.asarray(packet[3], dtype=float),
            control_m=np.asarray(packet[4], dtype=float),
            demand_valid=np.asarray(packet[5], dtype=bool),
            measured_valid=np.asarray(packet[6], dtype=bool),
            status=str(packet[7]),
        )

    def sample(self) -> AxisSample:
        if self.connection is None or self._queue is None:
            return self.empty_sample(status="trio_disconnected")
        try:
            packet = self._queue.get_nowait()
        except queue.Empty:
            return self._latest or self.empty_sample(status="waiting_for_trio_sample")
        if packet[0] == "error":
            return self.empty_sample(status=f"trio_reader_error:{packet[1]}")
        self._latest = self._packet_to_sample(packet)
        return self._latest


def sample_trio_connection(
    connection: object,
    config: dict,
    reader: TrioFastAxisReader | None = None,
) -> AxisSample:
    if reader is not None:
        return reader.sample(connection, config)
    return _sample_trio_individual(connection, config)


def _sample_trio_individual(connection: object, config: dict) -> AxisSample:
    demand = np.full(7, np.nan)
    measured = np.full(7, np.nan)
    demand_valid = np.zeros(7, dtype=bool)
    measured_valid = np.zeros(7, dtype=bool)
    errors: list[str] = []
    for axis in range(7):
        try:
            demand[axis] = float(connection.GetAxisParameter_DPOS(axis))
            demand_valid[axis] = True
        except Exception as exc:
            errors.append(f"DPOS{axis}:{exc}")
        try:
            measured[axis] = float(connection.GetAxisParameter_MPOS(axis))
            measured_valid[axis] = True
        except Exception:
            # MPOS availability depends on the drive and Unified API version.
            pass
    status = "ok" if not errors else ";".join(errors)
    return AxisSample(
        host_ns=time.perf_counter_ns(),
        demand_native=demand,
        measured_native=measured,
        control_m=map_native_axes(demand, measured, config),
        demand_valid=demand_valid,
        measured_valid=measured_valid,
        status=status,
    )


class SyntheticAxisSource(AxisSource):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.start_ns = time.perf_counter_ns()

    def sample(self) -> AxisSample:
        now = time.perf_counter_ns()
        t = (now - self.start_ns) * 1e-9
        native = np.asarray([
            1.2 * np.sin(0.7 * t + phase)
            for phase in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ] + [20.0 + 5.0 * np.sin(0.2 * t)])
        return AxisSample(
            host_ns=now,
            demand_native=native,
            measured_native=native,
            control_m=map_native_axes(native, native, self.config),
            demand_valid=np.ones(7, dtype=bool),
            measured_valid=np.ones(7, dtype=bool),
            status="synthetic",
        )
