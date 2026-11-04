from __future__ import annotations

import logging
import time

from .errors import NoFeasibleModelPlacement, WorkerStartupFailed

__all__ = ["ModelPlacer"]

logger = logging.getLogger("memserve.control_plane")


class ModelPlacer:
    def __init__(self, *, models, placements, workers, selector, changed) -> None:
        self._models = models
        self._placements = placements
        self._workers = workers
        self._selector = selector
        self._changed = changed
        self._loading: set[str] = set()

    def is_loading(self, model_id: str) -> bool:
        return model_id in self._loading

    def place(self, model_id):
        resources = self._placements.list_mig_resources()
        snapshot = self._placements.snapshot_placements()
        statuses, empty = self._placements.observe(snapshot)
        try:
            decision = self._selector.select_placement(model_id, resources, snapshot, statuses, frozenset(empty))
        except NoFeasibleModelPlacement:
            return None
        self._placements.reserve_placement(decision)
        self._loading.add(model_id)
        return decision

    def recover_isolated(self) -> None:
        for instance_id in self._placements.recoverable():
            try:
                self._workers.get_worker(instance_id).stop()
                self._placements.release_isolated(instance_id)
            except Exception as error:
                logger.warning("recovering isolated %s failed: %s", instance_id, error)
                continue
            logger.info("recovered isolated %s", instance_id)
            with self._changed:
                self._changed.notify_all()

    def load(self, decision) -> None:
        model_id, instance_id = decision.model_id, decision.instance_id
        evicted = self._evicted_model(decision)
        if evicted is None:
            logger.info("loading %s on empty %s", model_id, instance_id)
        else:
            logger.info("evicting %s from %s to load %s", evicted, instance_id, model_id)
        started = time.monotonic()
        try:
            self._activate(decision)
            self._models.mark_started(model_id)
        except BaseException as error:
            reason = str(error.__cause__ or error)
            if isinstance(error, WorkerStartupFailed):
                self._models.mark_failed(model_id, reason)
            logger.error("loading %s on %s failed after %.1fs: %s",
                         model_id, instance_id, time.monotonic() - started, reason)
            raise
        finally:
            with self._changed:
                self._loading.discard(model_id)
                self._changed.notify_all()
        logger.info("loaded %s on %s in %.1fs", model_id, instance_id, time.monotonic() - started)

    def _evicted_model(self, decision):
        if decision.expected_worker_generation is None:
            return None
        return self._current_placement(decision.instance_id).model_id

    def _activate(self, decision):
        instance_id = decision.instance_id
        worker = self._workers.get_worker(instance_id)
        committed = False
        try:
            if decision.expected_worker_generation is not None:
                worker.stop()
            worker.start(self._models.get_configuration(decision.model_id))
            self._placements.commit_placement(instance_id)
            committed = True
        except BaseException as error:
            if not committed:
                try:
                    self._rollback(instance_id, worker)
                except Exception as cleanup_error:
                    raise error from cleanup_error
            raise

    def _rollback(self, instance_id, worker):
        self._placements.isolate(instance_id)
        worker.stop()
        self._placements.rollback_placement(instance_id)

    def _current_placement(self, instance_id):
        return next(item for item in self._placements.snapshot_placements().placements
                    if item.instance_id == instance_id)
