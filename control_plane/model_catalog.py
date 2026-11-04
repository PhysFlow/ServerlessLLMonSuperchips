from collections.abc import Callable, Iterable
from threading import RLock
import time
from typing import NamedTuple

from .errors import ModelDisabled, SchedulableModelNotFound
from .types import WorkerExecutionConfiguration

MAX_CONSECUTIVE_STARTUP_FAILURES = 3
STARTUP_RETRY_DELAY_SECONDS = 30.0


class _StartupFailures(NamedTuple):
    count: int
    reason: str
    failed_at: float


class ModelCatalog:
    def __init__(self, configurations: Iterable[WorkerExecutionConfiguration], *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._configurations: dict[str, WorkerExecutionConfiguration] = {}
        for configuration in configurations:
            if configuration.model_id in self._configurations:
                raise ValueError(f"model {configuration.model_id!r} is configured twice")
            self._configurations[configuration.model_id] = configuration
        self._failures: dict[str, _StartupFailures] = {}
        self._clock = clock
        self._lock = RLock()

    def get_configuration(self, model_id: str) -> WorkerExecutionConfiguration:
        try:
            return self._configurations[model_id]
        except KeyError as exc:
            raise SchedulableModelNotFound(
                "model is not registered", details={"model_id": model_id},
            ) from exc

    def mark_failed(self, model_id: str, reason: str) -> None:
        self.get_configuration(model_id)
        with self._lock:
            previous = self._failures.get(model_id)
            count = previous.count + 1 if previous else 1
            self._failures[model_id] = _StartupFailures(count, reason, self._clock())

    def mark_started(self, model_id: str) -> None:
        with self._lock:
            self._failures.pop(model_id, None)

    def startup_retry_delay(self, model_id: str) -> float:
        with self._lock:
            failures = self._failures.get(model_id)
        if failures is None:
            return 0.0
        return max(0.0, failures.failed_at + STARTUP_RETRY_DELAY_SECONDS - self._clock())

    def require_schedulable(self, model_id: str) -> WorkerExecutionConfiguration:
        configuration = self.get_configuration(model_id)
        with self._lock:
            failures = self._failures.get(model_id)
        if failures is not None and failures.count >= MAX_CONSECUTIVE_STARTUP_FAILURES:
            raise ModelDisabled(
                "model failed to start repeatedly",
                details={"model_id": model_id, "failures": str(failures.count), "reason": failures.reason},
            )
        return configuration
