from __future__ import annotations

from contextlib import AbstractContextManager
from threading import RLock
from typing import Callable

from .errors import NoFeasibleModelPlacement, StaleModelPlacementReservation, WorkerStateInconsistent, WorkerUnavailable
from .types import (
    MigResourceCapacity, ModelPlacement, ModelPlacementDecision, ModelPlacementSnapshot,
    WorkerLifecycleState, WorkerStatus,
)

__all__ = ["MigStateTable"]


class MigStateTable:
    def __init__(
        self, resources: tuple[MigResourceCapacity, ...],
        verify_worker: Callable[[str], AbstractContextManager[WorkerStatus | None]],
    ) -> None:
        self._resources = {item.instance_id: item for item in sorted(resources, key=lambda item: item.instance_id)}
        if len(self._resources) != len(resources):
            raise ValueError("duplicate MIG instance ID")
        self._verify_worker = verify_worker
        self._placements: dict[str, ModelPlacement] = {}
        self._switching: dict[str, ModelPlacementDecision] = {}
        self._isolated: set[str] = set()
        self._requests: dict[str, str] = {}
        self._last_used: dict[str, int] = {}
        self._uses = 0
        self._lock = RLock()

    def list_mig_resources(self) -> tuple[MigResourceCapacity, ...]:
        return tuple(self._resources.values())

    def snapshot_placements(self) -> ModelPlacementSnapshot:
        with self._lock:
            counts: dict[str, int] = {}
            for instance_id in self._requests.values():
                counts[instance_id] = counts.get(instance_id, 0) + 1
            return ModelPlacementSnapshot(
                tuple(self._placements[key] for key in sorted(self._placements)),
                tuple(sorted(self._switching)),
                tuple(sorted(counts.items())), tuple(sorted(self._last_used.items())),
            )

    def observe(self, snapshot: ModelPlacementSnapshot) -> tuple[dict[str, WorkerStatus], set[str]]:
        placed = {item.instance_id: item for item in snapshot.placements}
        switching = set(snapshot.switching)
        statuses, empty = {}, set()
        for instance_id in self._resources:
            if instance_id in switching:
                continue
            with self._lock:
                if instance_id in self._isolated:
                    continue
            try:
                with self._verify_worker(instance_id) as status:
                    with self._lock:
                        if instance_id in self._isolated:
                            continue
                        placement = placed.get(instance_id)
                        if placement is None:
                            if status is None:
                                empty.add(instance_id)
                        elif placement.matches(status):
                            statuses[instance_id] = status
                        else:
                            self._isolated.add(instance_id)
            except (WorkerUnavailable, WorkerStateInconsistent):
                continue
        return statuses, empty

    def isolate(self, instance_id: str) -> None:
        with self._lock:
            self._isolated.add(instance_id)

    def recoverable(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(instance_id for instance_id in self._isolated
                                if instance_id not in self._switching and not self._has_requests(instance_id)))

    def release_isolated(self, instance_id: str) -> None:
        with self._verify_worker(instance_id) as status:
            with self._lock:
                if status is not None:
                    raise WorkerStateInconsistent("cannot release MIG while its worker is running")
                if instance_id not in self.recoverable():
                    raise _stale("MIG is no longer recoverable")
                self._placements.pop(instance_id, None)
                self._isolated.discard(instance_id)

    def reserve_placement(self, decision: ModelPlacementDecision) -> str:
        self._require_resource(decision.instance_id)
        with self._verify_worker(decision.instance_id) as status:
            with self._lock:
                self._require_free(decision.instance_id)
                previous = self._placements.get(decision.instance_id)
                if previous is None:
                    if status is not None or decision.expected_worker_generation is not None:
                        raise WorkerStateInconsistent("empty MIG has an unexpected running worker")
                else:
                    if not previous.matches(status):
                        raise WorkerStateInconsistent("worker disagrees with committed placement")
                    if decision.expected_worker_generation != previous.worker_generation:
                        raise _stale("worker generation changed")
                    if decision.model_id != previous.model_id and not decision.requires_restart:
                        raise _stale("changing models requires a restart")
                    if decision.requires_restart and self._has_requests(decision.instance_id):
                        raise _stale("cannot restart a busy worker")
                self._switching[decision.instance_id] = decision
                return decision.instance_id

    def commit_placement(self, instance_id: str) -> None:
        with self._verify_worker(instance_id) as status:
            with self._lock:
                decision = self._switching_decision(instance_id)
                if not _matches_target(decision, self._placements.get(instance_id), status):
                    raise WorkerStateInconsistent("started worker does not match the target model or generation")
                self._placements[instance_id] = ModelPlacement(instance_id, decision.model_id, status.worker_generation)
                self._mark_used(instance_id)
                self._isolated.discard(instance_id)
                del self._switching[instance_id]

    def rollback_placement(self, instance_id: str) -> None:
        with self._verify_worker(instance_id) as status:
            with self._lock:
                self._switching_decision(instance_id)
                previous = self._placements.get(instance_id)
                if status is not None and (previous is None or not previous.matches(status)):
                    raise WorkerStateInconsistent("cannot release MIG while worker state is unresolved")
                if status is None:
                    self._placements.pop(instance_id, None)
                    self._isolated.discard(instance_id)
                del self._switching[instance_id]

    def mark_request_started(self, instance_id: str, request_id: str) -> None:
        self._require_resource(instance_id)
        with self._verify_worker(instance_id) as status:
            with self._lock:
                self._require_free(instance_id)
                if instance_id in self._isolated:
                    raise WorkerUnavailable("MIG is isolated")
                placement = self._placements.get(instance_id)
                if placement is None or not placement.matches(status):
                    raise WorkerStateInconsistent("request requires a matching ready worker")
                if request_id in self._requests:
                    raise _stale("request is already admitted")
                self._requests[request_id] = instance_id
                self._mark_used(instance_id)

    def mark_request_finished(self, instance_id: str, request_id: str) -> None:
        with self._lock:
            owner = self._requests.get(request_id)
            if owner is not None and owner != instance_id:
                raise _stale("request belongs to another MIG")
            if self._requests.pop(request_id, None) is not None:
                self._mark_used(instance_id)

    def _switching_decision(self, instance_id: str) -> ModelPlacementDecision:
        if instance_id not in self._switching:
            raise _stale("MIG is not being switched")
        return self._switching[instance_id]

    def _mark_used(self, instance_id: str) -> None:
        self._uses += 1
        self._last_used[instance_id] = self._uses

    def _require_resource(self, instance_id: str) -> None:
        if instance_id not in self._resources:
            raise NoFeasibleModelPlacement("unknown MIG instance")

    def _require_free(self, instance_id: str) -> None:
        if instance_id in self._switching:
            raise _stale("MIG is being switched")

    def _has_requests(self, instance_id: str) -> bool:
        return instance_id in self._requests.values()


def _matches_target(decision: ModelPlacementDecision, previous: ModelPlacement | None,
                    status: WorkerStatus | None) -> bool:
    if status is None or status.state is not WorkerLifecycleState.READY or status.active_model_id != decision.model_id:
        return False
    if previous is None:
        return True
    if not decision.requires_restart:
        return previous.matches(status)
    return status.worker_generation > previous.worker_generation


def _stale(message: str) -> StaleModelPlacementReservation:
    return StaleModelPlacementReservation(message)
