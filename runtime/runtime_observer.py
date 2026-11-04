from __future__ import annotations

import atexit
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

from .link_bandwidth import link_bytes_per_s

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSample:
    request_latency_ms: float | None = None
    hbm_utilization: float | None = None
    link_utilization: float | None = None
    timestamp_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a nonnegative integer")
        for name in ("request_latency_ms", "hbm_utilization", "link_utilization"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number or None")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name != "request_latency_ms" and value > 1:
                raise ValueError(f"{name} must be in [0, 1]")


class RuntimeObserver:
    _FIELDS = ("request_latency_ms", "hbm_utilization", "link_utilization")

    def __init__(
        self,
        ema_weight: float = 0.25,
        stale_after_s: float = 5.0,
        sample_interval_s: float = 1.0,
        source=None,
    ):
        if not math.isfinite(ema_weight) or not 0.0 < ema_weight <= 1.0:
            raise ValueError("ema_weight must be finite and in (0, 1]")
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("stale_after_s must be finite and positive")
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be finite and positive")
        self.ema_weight = ema_weight
        self.stale_after_ns = int(stale_after_s * 1e9)
        self.sample_interval_s = sample_interval_s
        self._lock = threading.Lock()
        self._smoothed_measurements: dict[str, tuple[float, int]] = {}
        self._source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, device) -> None:
        if os.getenv("MEMSERVE_AUTO_RATIO", "1").lower() in ("0", "false", "off"):
            return
        if self._thread is not None:
            return
        if self._source is None:
            self._source = _GpmSource(device)
        self._thread = threading.Thread(
            target=self._sample_loop, name="memserve-observer", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)
        logger.info(
            "MemServe hardware sampling started with %s",
            type(self._source).__name__,
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            sample = self._source.sample()
            if self._stop.is_set():
                return
            if sample is not None:
                self.observe(sample)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is None:
            return
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=self.sample_interval_s + 1.0)
        if self._thread.is_alive():
            logger.warning("MemServe sampling thread did not stop; keeping its source open")
            return
        if hasattr(self._source, "close"):
            self._source.close()

    def observe(self, sample: RuntimeSample) -> None:
        if not isinstance(sample, RuntimeSample):
            raise TypeError("sample must be a RuntimeSample")
        if all(getattr(sample, name) is None for name in self._FIELDS):
            raise ValueError("sample must contain at least one measurement")
        with self._lock:
            for name in self._FIELDS:
                value = getattr(sample, name)
                if value is None:
                    continue
                previous = self._smoothed_measurements.get(name)
                if previous is not None and sample.timestamp_ns < previous[1]:
                    continue
                smoothed = (
                    float(value)
                    if previous is None
                    else self.ema_weight * float(value)
                    + (1 - self.ema_weight) * previous[0]
                )
                self._smoothed_measurements[name] = (smoothed, sample.timestamp_ns)

    def snapshot(self, now_ns: int | None = None) -> RuntimeSample:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            values = {
                name: self._fresh_measurement(name, now_ns) for name in self._FIELDS
            }
        return RuntimeSample(**values, timestamp_ns=now_ns)

    def _fresh_measurement(self, name: str, now_ns: int) -> float | None:
        measurement = self._smoothed_measurements.get(name)
        if measurement is None:
            return None
        value, timestamp_ns = measurement
        return value if 0 <= now_ns - timestamp_ns <= self.stale_after_ns else None


class _GpmSource:
    def __init__(self, device, nvml=None):
        if nvml is None:
            import pynvml as nvml
        self._nvml = nvml
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if (device.index or 0) != 0 or "," in uuid or not uuid.startswith(("GPU-", "MIG-")):
            raise ValueError(
                "MemServe requires one GPU/MIG UUID in CUDA_VISIBLE_DEVICES; "
                "numeric CUDA ordinals cannot identify the NVML device"
            )
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        if uuid.startswith("MIG-"):
            self._parent = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
            instance = nvml.nvmlDeviceGetGpuInstanceId(handle)
            self._take_local = lambda sample: nvml.nvmlGpmMigSampleGet(self._parent, instance, sample)
        else:
            self._parent = handle
            self._take_local = lambda sample: nvml.nvmlGpmSampleGet(self._parent, sample)
        if not nvml.nvmlGpmQueryDeviceSupport(self._parent).isSupportedDevice:
            raise RuntimeError("NVML GPM is not supported on this GPU")
        self.link_bytes_per_s = link_bytes_per_s(nvml, self._parent)
        logger.info("MemServe link bandwidth %.1f GB/s", self.link_bytes_per_s / 1e9)
        self._local = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
        self._device = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
        self._take_local(self._local[0])
        nvml.nvmlGpmSampleGet(self._parent, self._device[0])

    def sample(self) -> RuntimeSample | None:
        nvml = self._nvml
        self._take_local(self._local[1])
        nvml.nvmlGpmSampleGet(self._parent, self._device[1])
        hbm = self._metrics(self._local, (nvml.NVML_GPM_METRIC_DRAM_BW_UTIL,))
        link = self._metrics(self._device, (
            nvml.NVML_GPM_METRIC_PCIE_RX_PER_SEC, nvml.NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC,
        ))
        self._local.reverse()
        self._device.reverse()
        if hbm is None and link is None:
            return None
        return RuntimeSample(
            hbm_utilization=None if hbm is None else min(1.0, hbm / 100),
            link_utilization=None if link is None else min(1.0, link * 2**20 / self.link_bytes_per_s),
        )

    def _metrics(self, samples, metric_ids) -> float | None:
        request = self._nvml.c_nvmlGpmMetricsGet_t()
        request.version = self._nvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(metric_ids)
        request.sample1, request.sample2 = samples
        for index, metric_id in enumerate(metric_ids):
            request.metrics[index].metricId = metric_id
        self._nvml.nvmlGpmMetricsGet(request)
        values = [request.metrics[index].value for index in range(len(metric_ids))
                  if request.metrics[index].nvmlReturn == 0]
        return sum(values) if values else None

    def close(self) -> None:
        for sample in self._local + self._device:
            self._nvml.nvmlGpmSampleFree(sample)
        self._nvml.nvmlShutdown()
