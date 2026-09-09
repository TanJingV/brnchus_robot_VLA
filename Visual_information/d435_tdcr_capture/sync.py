"""Clock mapping, axis interpolation and response-delay estimation."""

from __future__ import annotations

from collections import deque
from threading import Event, Lock, Thread
import time
from typing import Callable, Deque

import numpy as np

from .models import AxisSample


class DeviceClockMapper:
    """Map camera milliseconds to host monotonic nanoseconds.

    A rolling affine fit handles device/host clock offset and slow drift. The
    first samples use offset-only mapping so capture works immediately.
    """

    def __init__(self, history: int = 300) -> None:
        self.samples: Deque[tuple[float, float]] = deque(maxlen=history)

    def update(self, device_ms: float, host_arrival_ns: int) -> int:
        device_ns = float(device_ms) * 1e6
        self.samples.append((device_ns, float(host_arrival_ns)))
        if len(self.samples) < 8:
            offsets = [host - device for device, host in self.samples]
            return int(device_ns + float(np.median(offsets)))
        values = np.asarray(self.samples, dtype=float)
        x0 = float(values[-1, 0])
        x = values[:, 0] - x0
        y = values[:, 1]
        if np.unique(x).size < 4 or float(np.ptp(x)) < 1.0:
            offsets = values[:, 1] - values[:, 0]
            return int(device_ns + float(np.median(offsets)))
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        if mad > 0:
            keep = np.abs(residuals - np.median(residuals)) <= 4.0 * 1.4826 * mad
            if np.count_nonzero(keep) >= 8:
                slope, intercept = np.polyfit(x[keep], y[keep], 1)
        return int(intercept)


class AxisSampler:
    """Sample an AxisSource at high rate and interpolate onto camera time."""

    def __init__(self, source, sample_rate_hz: float = 100.0, history: int = 1000) -> None:
        self.source = source
        self.period_s = 1.0 / max(float(sample_rate_hz), 1.0)
        self.samples: Deque[AxisSample] = deque(maxlen=history)
        self.lock = Lock()
        self.listener_lock = Lock()
        self.listeners: list[Callable[[AxisSample], None]] = []
        self.stop_event = Event()
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = Thread(target=self._run, name="tdcr-axis-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            sample = self.source.sample()
            is_new = False
            with self.lock:
                if not self.samples or sample.host_ns != self.samples[-1].host_ns:
                    self.samples.append(sample)
                    is_new = True
            if is_new:
                with self.listener_lock:
                    listeners = tuple(self.listeners)
                for listener in listeners:
                    try:
                        listener(sample)
                    except Exception:
                        # A recorder/UI listener must never stop motor acquisition.
                        continue
            deadline += self.period_s
            self.stop_event.wait(max(0.0, deadline - time.perf_counter()))

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        self.thread = None

    def add_listener(self, listener: Callable[[AxisSample], None]) -> None:
        with self.listener_lock:
            if listener not in self.listeners:
                self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[AxisSample], None]) -> None:
        with self.listener_lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    def interpolate(self, host_ns: int) -> AxisSample:
        with self.lock:
            history = list(self.samples)
        if not history:
            return self.source.empty_sample(host_ns, status="no_axis_samples")
        if len(history) == 1 or host_ns <= history[0].host_ns:
            return history[0]
        if host_ns >= history[-1].host_ns:
            return history[-1]
        times = np.asarray([sample.host_ns for sample in history], dtype=np.int64)
        right = int(np.searchsorted(times, host_ns, side="right"))
        left = right - 1
        a, b = history[left], history[right]
        alpha = float((host_ns - a.host_ns) / max(b.host_ns - a.host_ns, 1))

        def blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            valid = np.isfinite(first) & np.isfinite(second)
            result = np.where(valid, first + alpha * (second - first), np.where(np.isfinite(first), first, second))
            return result

        return AxisSample(
            host_ns=host_ns,
            demand_native=blend(a.demand_native, b.demand_native),
            measured_native=blend(a.measured_native, b.measured_native),
            control_m=blend(a.control_m, b.control_m),
            demand_valid=a.demand_valid & b.demand_valid,
            measured_valid=a.measured_valid & b.measured_valid,
            status="interpolated",
        )


def estimate_delay_ms(
    input_values: np.ndarray,
    response_values: np.ndarray,
    sample_period_s: float,
    maximum_delay_s: float = 0.5,
) -> float:
    """Estimate positive response lag using normalized cross-correlation."""
    u = np.asarray(input_values, dtype=float).reshape(-1)
    y = np.asarray(response_values, dtype=float).reshape(-1)
    count = min(len(u), len(y))
    if count < 8:
        return 0.0
    u = np.gradient(u[:count])
    y = np.gradient(y[:count])
    u -= np.nanmean(u)
    y -= np.nanmean(y)
    u = np.nan_to_num(u)
    y = np.nan_to_num(y)
    corr = np.correlate(y, u, mode="full")
    lags = np.arange(-count + 1, count)
    max_lag = int(maximum_delay_s / max(sample_period_s, 1e-9))
    allowed = (lags >= 0) & (lags <= max_lag)
    if not np.any(allowed):
        return 0.0
    lag = int(lags[allowed][int(np.argmax(corr[allowed]))])
    return 1000.0 * lag * float(sample_period_s)
