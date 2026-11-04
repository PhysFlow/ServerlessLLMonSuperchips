from __future__ import annotations

from typing import Mapping

from .errors import NoFeasibleModelPlacement
from .types import (
    MigResourceCapacity, ModelPlacementDecision, ModelPlacementSnapshot, WorkerStatus,
)

__all__ = ["ModelPlacementSelector"]


class ModelPlacementSelector:
    def select_placement(
        self, model_id: str, resources: tuple[MigResourceCapacity, ...],
        snapshot: ModelPlacementSnapshot, statuses: Mapping[str, WorkerStatus], verified_empty: frozenset[str],
    ) -> ModelPlacementDecision:
        usable = {item.instance_id for item in resources} - set(snapshot.switching)
        placements = {item.instance_id: item for item in snapshot.placements if item.instance_id in usable}
        healthy = {instance_id: placement for instance_id, placement in placements.items()
                   if placement.matches(statuses.get(instance_id))}

        empty = sorted((usable & verified_empty) - placements.keys())
        if empty:
            return _decision(model_id, empty[0], None)

        idle = [placement for instance_id, placement in healthy.items()
                if snapshot.requests_on(instance_id) == 0]
        if idle:
            victim = min(idle, key=lambda item: (snapshot.last_used_at(item.instance_id), item.instance_id))
            return _decision(model_id, victim.instance_id, victim, restart=True)

        raise NoFeasibleModelPlacement("no empty MIG or idle worker is available")


def _decision(model_id, instance_id, previous, restart=False):
    return ModelPlacementDecision(
        instance_id, model_id, previous.worker_generation if previous else None, restart,
    )
