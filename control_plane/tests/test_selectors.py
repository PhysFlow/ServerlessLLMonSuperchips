from dataclasses import replace

import pytest

from memserve_control_plane.errors import NoFeasibleModelPlacement
from memserve_control_plane.model_placement_selector import ModelPlacementSelector
from memserve_control_plane.types import (
    MigResourceCapacity, ModelPlacement, ModelPlacementDecision,
    ModelPlacementSnapshot, WorkerLifecycleState, WorkerStatus,
)


def select(placements=(), statuses=None, empty=(), switching=(), resources=None,
           active_requests=(), last_used=()):
    return ModelPlacementSelector().select_placement(
        "target", resources if resources is not None else tuple(
            MigResourceCapacity(i) for i in ("a", "b", "c")
        ), ModelPlacementSnapshot(placements, switching, active_requests, last_used), statuses or {},
        frozenset(empty),
    )


def test_empty_worker_wins_over_eviction_and_is_deterministic():
    placement = ModelPlacement("a", "old", 1)
    status = WorkerStatus("a", WorkerLifecycleState.READY, "old", 1)
    decision = select((placement,), {"a": status}, empty=("c", "b"))
    assert decision.instance_id == "b"
    assert decision.expected_worker_generation is None


@pytest.mark.parametrize("case", ["unverified", "unknown", "busy", "missing_status", "stale", "reserved"])
def test_unusable_candidates_are_rejected(case):
    if case == "unverified":
        args = {}
    elif case == "unknown":
        args = {"empty": ("unknown",)}
    else:
        placement = ModelPlacement("a", "old", 1)
        status = WorkerStatus("a", WorkerLifecycleState.READY, "old", 1)
        if case == "stale":
            status = replace(status, worker_generation=2)
        args = {"placements": (placement,), "statuses": {} if case == "missing_status" else {"a": status}}
        if case == "reserved":
            args["switching"] = ("a",)
        if case == "busy":
            args["active_requests"] = (("a", 1),)
    with pytest.raises(NoFeasibleModelPlacement):
        select(**args)


def test_idle_victim_requires_restart_and_preserves_expected_generations():
    placement = ModelPlacement("a", "old", 3)
    status = WorkerStatus("a", WorkerLifecycleState.READY, "old", 3)
    decision = select((placement,), {"a": status})
    assert decision == ModelPlacementDecision("a", "target", 3, True)


def test_least_recently_used_idle_worker_is_replaced():
    placements = tuple(ModelPlacement(i, "old", 1) for i in ("a", "b"))
    statuses = {i: WorkerStatus(i, WorkerLifecycleState.READY, "old", 1) for i in ("a", "b")}
    decision = select(placements, statuses, last_used=(("a", 2), ("b", 1)))
    assert decision.instance_id == "b"
