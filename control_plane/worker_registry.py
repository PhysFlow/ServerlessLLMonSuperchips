from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from threading import RLock

from .errors import WorkerUnavailable
from .model_serving_worker import ModelServingWorker
from .types import WorkerLifecycleState, WorkerStatus

__all__ = ["WorkerRegistry"]


class WorkerRegistry:
    def __init__(self, workers: Mapping[str, ModelServingWorker]) -> None:
        if any(key != worker.instance_id for key, worker in workers.items()):
            raise ValueError("worker registered under another MIG's instance_id")
        self._workers = dict(workers)
        self._lock = RLock()

    def get_worker(self, instance_id: str) -> ModelServingWorker:
        with self._lock:
            try:
                return self._workers[instance_id]
            except KeyError as error:
                raise WorkerUnavailable("worker is not registered") from error

    def stop_all(self) -> None:
        for worker in self._workers.values():
            try:
                worker.stop()
            except Exception:
                continue

    @contextmanager
    def verify_worker(self, instance_id: str):
        with self._lock:
            with self.get_worker(instance_id).verify_status() as status:
                yield None if _stopped(status) else status


def _stopped(status: WorkerStatus) -> bool:
    return status.state is WorkerLifecycleState.STOPPED and status.active_model_id is None
