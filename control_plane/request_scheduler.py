from __future__ import annotations

import asyncio
import logging
import time

from .errors import ControlPlaneError, NoFeasibleModelPlacement, WorkerStateInconsistent
from .inference_response_stream import InferenceResponseStream, settle
from .types import ModelInferenceRequest

__all__ = ["RequestScheduler"]

logger = logging.getLogger("memserve.control_plane")

_MAX_QUEUE_WAIT_BEFORE_REPLAN_SECONDS = 1.0


class RequestScheduler:
    def __init__(self, *, models, placements, workers, placer, changed,
                 queue_timeout: float = 300.0, max_retries: int = 0, retry_delay: float = 0.0) -> None:
        self._models = models
        self._placements = placements
        self._workers = workers
        self._placer = placer
        self._changed = changed
        self._queue_timeout = queue_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._active_requests: set[str] = set()

    async def schedule_and_submit(self, request: ModelInferenceRequest) -> InferenceResponseStream:
        task = asyncio.create_task(asyncio.to_thread(self._schedule, request))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await _close_stream_after_uncancellable_scheduling(task)
            raise

    async def provision(self, model_id: str) -> None:
        await asyncio.to_thread(self._acquire, model_id, None)

    def _schedule(self, request):
        with self._changed:
            if request.request_id in self._active_requests:
                raise WorkerStateInconsistent("request is already being served")
            self._active_requests.add(request.request_id)
        try:
            return self._execute_with_retries(request)
        except BaseException:
            with self._changed:
                self._active_requests.discard(request.request_id)
                self._changed.notify_all()
            raise

    def _execute_with_retries(self, request):
        attempt = 0
        while True:
            try:
                return self._execute(request)
            except ControlPlaneError as error:
                if not error.retryable or attempt >= self._max_retries:
                    raise
                attempt += 1
                logger.warning("retrying request %s after %s (attempt %d of %d)",
                               request.request_id, error.code, attempt, self._max_retries)
                if self._retry_delay:
                    time.sleep(self._retry_delay)

    def _execute(self, request):
        instance_id = self._acquire(request.model_id, request.request_id)
        try:
            source = self._workers.get_worker(instance_id).submit_request(request)
            return InferenceResponseStream(
                source, lambda: self._finish_request(instance_id, request.request_id),
                lambda: self._placements.isolate(instance_id),
            )
        except BaseException as error:
            try:
                self._finish_request(instance_id, request.request_id)
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _acquire(self, model_id, request_id):
        deadline = time.monotonic() + self._queue_timeout
        queued_at = None
        label = request_id or "provisioning"
        while True:
            with self._changed:
                self._models.require_schedulable(model_id)
                decision = None
                if not self._placer.is_loading(model_id):
                    instance_id = self._serving_instance(model_id)
                    if instance_id is not None:
                        if request_id is not None:
                            self._placements.mark_request_started(instance_id, request_id)
                        if queued_at is not None:
                            logger.info("request %s waited %.1fs for %s", label, time.monotonic() - queued_at, model_id)
                        logger.info("request %s -> %s on %s", label, model_id, instance_id)
                        return instance_id
                    if not self._models.startup_retry_delay(model_id):
                        decision = self._placer.place(model_id)
                if decision is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning("request %s for %s timed out after %.1fs in queue",
                                       label, model_id, self._queue_timeout)
                        raise NoFeasibleModelPlacement(
                            "timed out waiting for a MIG", details={"model_id": model_id},
                        )
                    if queued_at is None:
                        queued_at = time.monotonic()
                        logger.info("request %s queued: %s %s", label, model_id, self._queue_reason(model_id))
                    self._changed.wait(min(remaining, _MAX_QUEUE_WAIT_BEFORE_REPLAN_SECONDS))
                    continue
            if queued_at is not None:
                logger.info("request %s waited %.1fs for %s", label, time.monotonic() - queued_at, model_id)
                queued_at = None
            self._placer.load(decision)

    def _queue_reason(self, model_id):
        if self._placer.is_loading(model_id):
            return "is loading"
        if self._models.startup_retry_delay(model_id):
            return "is waiting to retry a failed start"
        return "has no free MIG"

    def _serving_instance(self, model_id):
        snapshot = self._placements.snapshot_placements()
        statuses, _ = self._placements.observe(snapshot)
        serving = [item for item in snapshot.placements if item.model_id == model_id and item.instance_id in statuses]
        if not serving:
            return None
        return min(serving, key=lambda item: (snapshot.requests_on(item.instance_id), item.instance_id)).instance_id

    def _finish_request(self, instance_id, request_id):
        with self._changed:
            self._placements.mark_request_finished(instance_id, request_id)
            self._active_requests.discard(request_id)
            self._changed.notify_all()


async def _close_stream_after_uncancellable_scheduling(task):
    try:
        stream = await settle(task)
    except Exception:
        raise asyncio.CancelledError from None
    await stream.aclose()
