from __future__ import annotations

import os
import threading
import weakref

import torch

from .runtime_observer import RuntimeObserver, RuntimeSample
from .ratio_policy import BandwidthBalancePolicy, RatioPolicy
from .ratio_validation import validate_ratio
from .ratio_controller import ControllerStatus


class RatioHandle:

    def __init__(self, runtime: Runtime):
        self._runtime_ref = weakref.ref(runtime)

    @property
    def tensor(self) -> torch.Tensor:
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError("owning Runtime no longer exists")
        runtime._ensure_usable()
        return runtime._ratio_tensor


class Runtime:

    def __init__(
        self,
        device: torch.device | str,
        initial_ratio: float = 0.0,
        *,
        observer: RuntimeObserver | None = None,
        policy: RatioPolicy | None = None,
        update_interval_s: float = 1.0,
    ):
        validated_ratio = validate_ratio(initial_ratio)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Runtime requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this worker")
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._ratio_tensor = torch.tensor(
            [validated_ratio], device=device, dtype=torch.float32
        )
        self.device = self._ratio_tensor.device
        self._ratio = validated_ratio
        self._ratio_handle = RatioHandle(self)

        from .ratio_controller import RatioController

        self._observer = observer if observer is not None else RuntimeObserver()
        self._controller = RatioController(
            self,
            self._observer,
            policy or BandwidthBalancePolicy(),
            update_interval_s=update_interval_s,
        )
        self._observer.start(self.device)

    def _ensure_usable(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("Runtime cannot be used after a process fork")
        if self._closed:
            raise RuntimeError("Runtime is closed")

    def get_ratio_handle(self) -> RatioHandle:
        self._ensure_usable()
        return self._ratio_handle

    def get_ratio(self) -> float:
        with self._lock:
            self._ensure_usable()
            return self._ratio

    def set_ratio(self, value: float, stream: torch.cuda.Stream | None = None) -> None:
        value = validate_ratio(value)
        with self._lock:
            self._ensure_usable()
            self._ensure_stream_matches_device(stream)
            self._write_ratio_tensor(value, stream)
            self._ratio = value

    def _ensure_stream_matches_device(
        self, stream: torch.cuda.Stream | None
    ) -> None:
        if stream is not None and torch.device(stream.device) != self.device:
            raise ValueError("stream must belong to the Runtime CUDA device")

    def _write_ratio_tensor(
        self, value: float, stream: torch.cuda.Stream | None
    ) -> None:
        context = (
            torch.cuda.device(self.device)
            if stream is None
            else torch.cuda.stream(stream)
        )
        with context:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ratio updates must occur outside CUDA Graph capture"
                )
            self._ratio_tensor.fill_(value)

    def observe(self, sample: RuntimeSample) -> None:
        self._ensure_usable()
        self._controller.observe(sample)

    def before_batch(self, stream: torch.cuda.Stream | None = None) -> float:
        self._ensure_usable()
        return self._controller.before_batch(stream)

    def set_manual_ratio(self, value: float) -> None:
        self._ensure_usable()
        self._controller.set_manual_ratio(value)

    def clear_manual_ratio(self) -> None:
        self._ensure_usable()
        self._controller.clear_manual_ratio()

    def status(self) -> ControllerStatus:
        self._ensure_usable()
        return self._controller.status()

    def close(self) -> None:
        self._observer.close()
        with self._lock:
            self._closed = True
