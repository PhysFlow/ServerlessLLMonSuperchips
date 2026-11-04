from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event

import pytest

from memserve_control_plane.errors import (
    NoFeasibleModelPlacement, StaleModelPlacementReservation, WorkerStateInconsistent,
    WorkerUnavailable,
)
from memserve_control_plane.mig_state_table import MigStateTable
from memserve_control_plane.types import (
    MigResourceCapacity, ModelPlacementDecision, WorkerLifecycleState, WorkerStatus,
)


class State:
    def __init__(self):
        self.statuses = {"a": None, "b": None}
        self.checked = []
        self.table = MigStateTable(
            tuple(MigResourceCapacity(i) for i in self.statuses),
            self.verify,
        )

    @contextmanager
    def verify(self, instance):
        self.checked.append(instance)
        status = self.statuses[instance]
        if isinstance(status, Exception):
            raise status
        yield status

    def reserve(self, instance="a", model="target", restart=False):
        snapshot = self.table.snapshot_placements()
        previous = next((p for p in snapshot.placements if p.instance_id == instance), None)
        decision = ModelPlacementDecision(
            instance, model, previous.worker_generation if previous else None, restart,
        )
        assert self.table.reserve_placement(decision) == instance
        return decision

    def ready(self, instance="a", model="target", generation=1):
        self.statuses[instance] = WorkerStatus(instance, WorkerLifecycleState.READY, model, generation)

    def committed(self, instance="a"):
        decision = self.reserve(instance)
        self.ready(instance)
        self.table.commit_placement(instance)
        return decision


def test_unknown_resource_rejected_before_worker_io():
    state = State()
    with pytest.raises(NoFeasibleModelPlacement):
        state.reserve("unknown")
    assert state.checked == []
    with pytest.raises(NoFeasibleModelPlacement):
        state.table.mark_request_started("unknown", "request")
    assert state.checked == []


def test_reservation_marks_mig_switching_and_prevents_collision():
    state = State()
    state.reserve()
    assert state.table.snapshot_placements().switching == ("a",)
    with pytest.raises(StaleModelPlacementReservation):
        state.reserve()


def test_commit_uses_observed_generations_and_ends_the_switch():
    state = State()
    state.reserve()
    state.ready(generation=7)
    state.table.commit_placement("a")
    snapshot = state.table.snapshot_placements()
    assert snapshot.placements[0].worker_generation == 7
    assert snapshot.switching == ()
    with pytest.raises(StaleModelPlacementReservation):
        state.table.commit_placement("a")
    with pytest.raises(StaleModelPlacementReservation):
        state.table.rollback_placement("a")


def test_rollback_frees_the_empty_mig():
    state = State()
    state.reserve()
    state.table.rollback_placement("a")
    assert state.table.snapshot_placements().switching == ()
    with pytest.raises(StaleModelPlacementReservation):
        state.table.rollback_placement("a")
    with pytest.raises(StaleModelPlacementReservation):
        state.table.commit_placement("a")
    state.reserve()
    state.table.rollback_placement("a")


def test_invalid_target_cannot_commit_or_release_reservation():
    state = State()
    state.reserve()
    state.ready(model="wrong")
    with pytest.raises(WorkerStateInconsistent):
        state.table.commit_placement("a")
    with pytest.raises(WorkerStateInconsistent):
        state.table.rollback_placement("a")
    assert state.table.snapshot_placements().switching == ("a",)


def test_absent_worker_cannot_commit():
    state = State()
    state.reserve()
    with pytest.raises(WorkerStateInconsistent):
        state.table.commit_placement("a")
    assert state.table.snapshot_placements().switching == ("a",)


def test_commit_requires_the_mig_to_be_switching():
    state = State()
    state.reserve()
    state.ready("b")
    with pytest.raises(StaleModelPlacementReservation):
        state.table.commit_placement("b")
    assert state.table.snapshot_placements().switching == ("a",)


def test_distinct_mig_commit_does_not_invalidate_existing_reservation():
    state = State()
    a, b = state.reserve(), state.reserve("b")
    state.ready("a")
    state.ready("b")
    state.table.commit_placement(a.instance_id)
    state.table.commit_placement(b.instance_id)
    assert len(state.table.snapshot_placements().placements) == 2


def test_restart_is_blocked_by_control_plane_admission():
    state = State()
    state.committed()
    state.table.mark_request_started("a", "request")
    with pytest.raises(StaleModelPlacementReservation):
        state.reserve(model="new", restart=True)


def test_request_admission_checks_reservation_identity_and_owner():
    state = State()
    state.committed()
    state.table.mark_request_started("a", "request")
    with pytest.raises(StaleModelPlacementReservation):
        state.table.mark_request_started("a", "request")
    with pytest.raises(StaleModelPlacementReservation):
        state.table.mark_request_finished("b", "request")
    state.table.mark_request_finished("a", "request")
    state.table.mark_request_finished("a", "request")
    state.reserve()
    with pytest.raises(StaleModelPlacementReservation):
        state.table.mark_request_started("a", "other")
    state.table.rollback_placement("a")
    state.ready(generation=2)
    with pytest.raises(WorkerStateInconsistent):
        state.table.mark_request_started("a", "stale")


@pytest.mark.parametrize("generation", [1, 2])
def test_restart_requires_new_worker_generation(generation):
    state = State()
    state.committed()
    state.reserve(model="new", restart=True)
    state.ready(model="new", generation=generation)
    if generation == 1:
        with pytest.raises(WorkerStateInconsistent):
            state.table.commit_placement("a")
    else:
        state.table.commit_placement("a")
        assert state.table.snapshot_placements().placements[0].model_id == "new"


def test_concurrent_reservations_have_one_winner_on_same_mig():
    state = State()
    barrier = Barrier(2)
    def reserve(_):
        barrier.wait(timeout=3)
        try:
            return state.reserve()
        except StaleModelPlacementReservation:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ("one", "two")))
    assert sum(result is not None for result in results) == 1


def test_worker_verification_does_not_hold_table_lock():
    state = State()
    state.reserve()
    state.ready()
    entered, release = Event(), Event()
    @contextmanager
    def verify(instance):
        entered.set()
        assert release.wait(3)
        yield state.statuses[instance]
    state.table._verify_worker = verify
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(state.table.commit_placement, "a")
        assert entered.wait(3)
        snapshot = pool.submit(state.table.snapshot_placements).result(timeout=3)
        release.set()
        commit.result(timeout=3)
    assert snapshot.switching == ("a",)


def test_mig_resources_are_listed_sorted_and_must_be_unique():
    a, b = MigResourceCapacity("a"), MigResourceCapacity("b")
    assert MigStateTable((b, a), lambda _: None).list_mig_resources() == (a, b)
    with pytest.raises(ValueError, match="duplicate"):
        MigStateTable((a, a), lambda _: None)


def observe(state):
    return state.table.observe(state.table.snapshot_placements())


def test_observe_reports_matching_workers_and_verified_empty_migs():
    state = State()
    state.committed("a")
    statuses, empty = observe(state)
    assert set(statuses) == {"a"} and empty == {"b"}


def test_mismatched_worker_is_isolated_until_a_switch_commits():
    state = State()
    state.committed("a")
    state.ready("a", generation=99)
    assert observe(state) == ({}, {"b"})
    state.ready("a")
    assert observe(state) == ({}, {"b"})
    state.reserve(model="new", restart=True)
    state.ready(model="new", generation=2)
    state.table.commit_placement("a")
    assert set(observe(state)[0]) == {"a"}


def test_isolated_empty_mig_recovers_when_rollback_confirms_the_worker_stopped():
    state = State()
    state.reserve()
    state.table.isolate("a")
    assert observe(state) == ({}, {"b"})
    state.table.rollback_placement("a")
    assert observe(state) == ({}, {"a", "b"})


def test_unreachable_worker_is_skipped_without_isolation():
    state = State()
    state.statuses["b"] = WorkerStateInconsistent("unreachable")
    assert observe(state) == ({}, {"a"})
    state.statuses["b"] = None
    assert observe(state) == ({}, {"a", "b"})


def test_isolation_during_status_read_keeps_the_mig_out_of_observe():
    state = State()
    state.committed("a")
    verify = state.verify

    @contextmanager
    def isolating_verify(instance):
        with verify(instance) as status:
            if instance == "a":
                state.table.isolate("a")
            yield status

    state.table._verify_worker = isolating_verify
    assert observe(state) == ({}, {"b"})


def test_isolated_mig_rejects_new_requests_until_a_switch_commits():
    state = State()
    state.committed("a")
    state.table.isolate("a")
    with pytest.raises(WorkerUnavailable, match="isolated"):
        state.table.mark_request_started("a", "request")
    state.reserve(model="new", restart=True)
    state.ready(model="new", generation=2)
    state.table.commit_placement("a")
    state.table.mark_request_started("a", "request")


def test_only_idle_isolated_migs_outside_a_switch_are_recoverable():
    state = State()
    state.committed("a")
    state.table.mark_request_started("a", "request")
    state.table.isolate("a")
    state.reserve("b")
    state.table.isolate("b")
    assert state.table.recoverable() == ()
    state.table.mark_request_finished("a", "request")
    assert state.table.recoverable() == ("a",)


def test_released_isolated_mig_becomes_empty_only_after_its_worker_stopped():
    state = State()
    state.committed("a")
    state.table.isolate("a")
    with pytest.raises(WorkerStateInconsistent, match="running"):
        state.table.release_isolated("a")
    state.statuses["a"] = None
    state.table.release_isolated("a")
    assert observe(state) == ({}, {"a", "b"})
    assert state.table.snapshot_placements().placements == ()
    with pytest.raises(StaleModelPlacementReservation):
        state.table.release_isolated("a")
